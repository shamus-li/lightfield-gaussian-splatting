from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim

import utils.render as render_mod
from static import eval as static_eval, gaussians as gaussian_lib
from static.covisible import write_covisible_masks
from static.data import ColmapDataset
from static.gaussians import build_schedulers, initialize_default_strategy
from utils.checkpoints import select_model, validation_steps
from utils.cli import parse_args
from utils.io import read_lines, read_yaml, write_json, write_yaml
from utils.runtime import set_seed


@dataclass(frozen=True)
class StaticConfig:
    data_dir: Path
    result_dir: Path
    camera_model: str = "monocular"
    checkpoint: Path | None = None
    train_list: Path | None = None
    eval_list: Path | None = None
    match: str = ""
    alignment: Path | None = None
    covisible_dir: Path | None = None
    max_steps: int = 3_000
    refine_stop_iter: int = 2_600
    test_every: int = 1
    lpips_net: str = "vgg"


def load_static_data(config: StaticConfig) -> tuple[ColmapDataset, ColmapDataset]:
    data_dir = config.data_dir.expanduser().resolve()
    result_dir = config.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    train_names = set(read_lines(config.train_list)) if config.train_list else None
    eval_names = set(read_lines(config.eval_list)) if config.eval_list else None
    normalize_from_train = train_names is not None and config.alignment is None
    dataset = ColmapDataset(data_dir, test_every=config.test_every)
    if not normalize_from_train and config.alignment is None:
        dataset.normalize()
    if normalize_from_train:
        normalization_set = dataset.select(
            "train",
            match_string=config.match,
            selected_images=train_names,
        )
        dataset.normalize(normalization_set.indices)
    if config.alignment is not None:
        dataset.apply_transform(np.load(config.alignment.expanduser()))
    trainset = dataset.select(
        "train",
        match_string=config.match,
        selected_images=train_names,
    )
    valset = dataset.select("val", selected_images=eval_names)
    if config.checkpoint is None:
        alignment_dir = result_dir / "alignments"
        alignment_dir.mkdir(parents=True, exist_ok=True)
        np.save(alignment_dir / "train_normalization.npy", dataset.transform.astype(np.float32))
    return trainset, valset


class CameraOptModule(torch.nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.embeds = torch.nn.Embedding(int(count), 9)
        torch.nn.init.zeros_(self.embeds.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids.to(device=camtoworlds.device, dtype=torch.long))
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        identity = self.identity.to(device=pose_deltas.device, dtype=pose_deltas.dtype)
        rotation = rotation_6d_to_matrix(drot + identity.expand(*batch_dims, -1))
        transform = torch.eye(4, device=pose_deltas.device, dtype=pose_deltas.dtype).repeat(
            (*batch_dims, 1, 1)
        )
        transform[..., :3, :3] = rotation
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def train_static_model(
    config: StaticConfig,
    trainset: ColmapDataset,
    valset: ColmapDataset,
    gaussians: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    render_cfg: render_mod.RenderConfig,
    *,
    device: str,
) -> None:
    max_steps = config.max_steps
    scene_scale = trainset.scene_scale * 1.1
    gaussians.train()
    strategy, strategy_state = initialize_default_strategy(
        scene_scale,
        gaussians,
        optimizers,
        refine_stop_iter=config.refine_stop_iter,
    )
    pose_adjust = CameraOptModule(len(trainset)).to(device)
    pose_optimizer = torch.optim.Adam(
        pose_adjust.parameters(),
        lr=1e-5,
        weight_decay=1e-6,
    )
    schedulers = build_schedulers(
        optimizers,
        pose_optimizer=pose_optimizer,
        max_steps=max_steps,
    )
    # Persistent DataLoader workers draw one seed before RandomSampler shuffles.
    torch.empty((), dtype=torch.int64).random_()
    training_indices: list[int] = []
    eval_steps = {value - 1 for value in validation_steps(config.max_steps) if value > 0}

    for step in range(max_steps):
        index_in_epoch = step % len(trainset)
        if index_in_epoch == 0:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            training_indices = torch.randperm(len(trainset), generator=generator).tolist()
        item = trainset[training_indices[index_in_epoch]]
        target = item.image.to(device=device, dtype=torch.float32)

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        pose_optimizer.zero_grad(set_to_none=True)

        step_render_cfg = replace(
            render_cfg,
            sh_degree=min(step // 1_000, 3),
        )
        camtoworld = item.camtoworld.unsqueeze(0).to(device=device, dtype=torch.float32)
        embed_id = torch.tensor([item.embed_id], device=device, dtype=torch.long)
        adjusted = pose_adjust(camtoworld, embed_id)[0]
        render_item = replace(item, camtoworld=adjusted)
        rendered, render_info = render_mod.render_item(
            gaussians,
            render_item,
            cfg=step_render_cfg,
        )
        strategy.step_pre_backward(
            params=gaussians,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=render_info,
        )
        l1 = F.l1_loss(rendered, target)
        ssim_value = fused_ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            target.permute(2, 0, 1).unsqueeze(0),
            padding="valid",
        )
        scale_loss = 5e-4 * torch.exp(gaussians["scales"]).mean()
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_value) + scale_loss
        loss.backward()
        quaternion_gradients = gaussians["quats"].grad
        if quaternion_gradients is not None:
            scales = gaussians["scales"].detach()
            quaternion_gradients[torch.eq(scales, scales[:, :1]).all(dim=-1)] = 0
        for optimizer in optimizers.values():
            optimizer.step()
        pose_optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        strategy.step_post_backward(
            params=gaussians,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=render_info,
            packed=False,
        )
        if step in eval_steps:
            path = config.result_dir.expanduser().resolve() / "ckpts" / f"ckpt_{step}_rank0.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gaussians.state_dict(), path)
            static_eval.evaluate_static_model(
                config,
                valset,
                gaussians,
                render_cfg,
                step=step,
                device=device,
            )


def run(config: StaticConfig) -> None:
    result_dir = config.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    if config.checkpoint is None:
        write_yaml(
            result_dir / "cfg.yml",
            {
                "camera_model": config.camera_model,
                "match": config.match,
            },
        )

    trainset, valset = load_static_data(config)
    set_seed(42)
    device = "cuda"
    render_cfg = render_mod.RenderConfig(
        sh_degree=gaussian_lib.SH_DEGREE,
        antialiased=True,
        channel_chunk=32,
        tile_size=16,
    )
    if config.checkpoint:
        checkpoint = torch.load(
            config.checkpoint.expanduser().resolve(),
            map_location=device,
            weights_only=True,
        )
        gaussian_state = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(checkpoint[key].to(device=device))
                for key in ("means", "scales", "quats", "opacities", "sh0", "shN")
            }
        )
        metrics = static_eval.evaluate_static_model(
            config,
            valset,
            gaussian_state,
            render_cfg,
            device=device,
        )
        write_json(result_dir / "metrics.json", metrics)
        return
    scene_scale = float(trainset.scene_scale) * 1.1
    gaussian_state = gaussian_lib.init_gaussians(
        trainset.points,
        trainset.point_colors,
        device=device,
    )
    optimizers = gaussian_lib.build_optimizers(gaussian_state, scene_scale=scene_scale)
    train_static_model(
        config,
        trainset,
        valset,
        gaussian_state,
        optimizers,
        render_cfg,
        device=device,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py static",
        description="Train and evaluate a Gaussian-splatting model on a real static scene.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory containing static/shared/.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Directory for checkpoints, renders, and metrics.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("monocular", "iphone", "stereo", "lightfield"),
        default="monocular",
        help=(
            "Input camera design (training default: monocular; "
            "loaded from RESULT_DIR/cfg.yml during evaluation)."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the completed run in RESULT_DIR instead of training.",
    )
    args = parse_args(parser, argv)
    if args.eval:
        config = read_yaml(args.result_dir.expanduser().resolve() / "cfg.yml")
        args.camera_model = config["camera_model"]
    data_root = args.data.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    data_dir = data_root / "static" / "shared"
    train_list = data_dir / "splits" / f"{args.camera_model}.txt"
    eval_camera = "iphone" if args.camera_model == "monocular" else args.camera_model
    eval_list = data_dir / "splits" / f"{eval_camera}_eval.txt"
    alignment = result_dir / "alignments" / "train_normalization.npy"
    support_dir = data_dir / "subsets" / args.camera_model
    eval_dir = data_dir / "subsets" / f"{eval_camera}_eval"
    training_config = StaticConfig(
        data_dir=support_dir,
        result_dir=result_dir,
        camera_model=args.camera_model,
    )
    if args.eval:
        checkpoint = result_dir / "model.pt"
    else:
        run(training_config)
        checkpoint = select_model(result_dir)
    covisible_dir = write_covisible_masks(
        eval_dir,
        support_dir,
        result_dir / "covisible",
        support_test_every=1,
    )
    run(
        StaticConfig(
            data_dir=data_dir,
            result_dir=result_dir,
            camera_model=args.camera_model,
            checkpoint=checkpoint,
            train_list=train_list,
            eval_list=eval_list,
            alignment=alignment,
            covisible_dir=covisible_dir,
        )
    )
    shutil.rmtree(result_dir / "covisible")
