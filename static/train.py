from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import tyro
from fused_ssim import fused_ssim

from static.covisible import write_covisible_masks
from static.data import ColmapDataset
from static.eval import evaluate, save_renders
from utils.camera import compute_scene_scale
from utils.gaussians import (
    build_optimizers,
    build_strategy,
    exponential_decay,
    init_gaussians,
    rasterize,
)
from utils.io import read_yaml, write_config, write_json
from utils.runtime import set_seed

SEED = 42
MAX_STEPS = 3_000
REFINE_STOP_ITER = 2_600
SH_DEGREE = 3
# Steps between raising the rendered SH degree by one.
SH_DEGREE_INTERVAL = 1_000
SSIM_WEIGHT = 0.2
SCALE_REG = 5e-4
POSE_LR = 1e-5
POSE_WEIGHT_DECAY = 1e-6
LPIPS_NET = "vgg"


@dataclass(frozen=True)
class Config:
    # Scene directory containing static/shared/ from prepare-static.
    data: Path
    # Run directory for cfg.yml, model.pt, renders/ and metrics.json.
    result_dir: Path
    # Input camera design; --eval reads it from the run's cfg.yml.
    camera_model: Literal["monocular", "iphone", "stereo", "lightfield"] = "monocular"
    # Evaluate the existing run in result-dir instead of training.
    eval: bool = False


class CameraOptModule(torch.nn.Module):
    """Learned per-image pose corrections: a translation and a 6D rotation delta."""

    identity: torch.Tensor

    def __init__(self, count: int) -> None:
        super().__init__()
        self.embeds = torch.nn.Embedding(count, 9)
        torch.nn.init.zeros_(self.embeds.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rotation = rotation_6d_to_matrix(drot + self.identity.expand(*batch_dims, -1))
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rotation
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def train(
    trainset: ColmapDataset,
    *,
    max_steps: int,
    refine_stop_iter: int,
) -> torch.nn.ParameterDict:
    """Fit Gaussians with pose refinement to the views of a normalized dataset."""
    device = "cuda"
    scene_scale = compute_scene_scale(trainset.camtoworlds) * 1.1
    params = init_gaussians(trainset.points, trainset.point_colors / 255.0, device=device)
    optimizers = build_optimizers(params, scene_scale=scene_scale)
    strategy, strategy_state = build_strategy(
        params, optimizers, scene_scale=scene_scale, refine_stop_iter=refine_stop_iter
    )
    pose_adjust = CameraOptModule(len(trainset)).to(device)
    pose_optimizer = torch.optim.Adam(
        pose_adjust.parameters(), lr=POSE_LR, weight_decay=POSE_WEIGHT_DECAY
    )
    schedulers = [
        exponential_decay(optimizers["means"], max_steps=max_steps),
        exponential_decay(pose_optimizer, max_steps=max_steps),
    ]

    order: list[int] = []
    for step in range(max_steps):
        if step % len(trainset) == 0:
            order = torch.randperm(len(trainset)).tolist()
        view = trainset[order[step % len(trainset)]]
        target = view.image.to(device)
        camtoworld = pose_adjust(
            view.camtoworld[None].to(device), torch.tensor([view.embed_id], device=device)
        )
        renders, _alphas, info = rasterize(
            params,
            camtoworld,
            view.K[None].to(device),
            view.width,
            view.height,
            sh_degree=min(step // SH_DEGREE_INTERVAL, SH_DEGREE),
            packed=False,
            rasterize_mode="antialiased",
        )
        rendered = renders[0]
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        l1 = F.l1_loss(rendered, target)
        ssim = fused_ssim(
            rendered.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], padding="valid"
        )
        loss = (1.0 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1.0 - ssim)
        loss = loss + SCALE_REG * torch.exp(params["scales"]).mean()
        loss.backward()
        for optimizer in [*optimizers.values(), pose_optimizer]:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)
        if (step + 1) % 1_000 == 0 or step + 1 == max_steps:
            print(
                f"step {step + 1}/{max_steps} loss {loss.item():.4f} "
                f"gaussians {len(params['means'])}",
                flush=True,
            )
    return params


def train_and_save(
    train_dir: Path,
    model_path: Path,
    *,
    max_steps: int,
    refine_stop_iter: int,
    holdout_every: int = 0,
    match: str = "",
) -> None:
    """Train on the normalized images of train_dir and save the Gaussians to model_path."""
    trainset = ColmapDataset(train_dir, holdout_every=holdout_every, match=match)
    trainset.normalize()
    set_seed(SEED)
    params = train(trainset, max_steps=max_steps, refine_stop_iter=refine_stop_iter)
    torch.save(params.state_dict(), model_path)


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        Config,
        args=argv,
        prog="train.py static",
        description="Train and evaluate Gaussian splatting on a real static scene.",
    )
    if cfg.eval:
        cfg = replace(cfg, camera_model=read_yaml(cfg.result_dir / "cfg.yml")["camera_model"])
    shared = cfg.data / "static/shared"
    train_dir = shared / "subsets" / cfg.camera_model
    # Every camera design is scored on the same held-out iPhone video frames.
    test_dir = shared / "subsets/iphone_eval"
    if not cfg.eval:
        cfg.result_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg.result_dir / "cfg.yml", cfg)
        train_and_save(
            train_dir,
            cfg.result_dir / "model.pt",
            max_steps=MAX_STEPS,
            refine_stop_iter=REFINE_STOP_ITER,
        )
    mask_dir = cfg.result_dir / "covisible"
    write_covisible_masks(test_dir, train_dir, mask_dir)
    metrics, renders = evaluate(
        cfg.result_dir / "model.pt", train_dir, test_dir, mask_dir, lpips_net=LPIPS_NET
    )
    shutil.rmtree(mask_dir)
    save_renders(cfg.result_dir / "renders", renders)
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)
