from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa
from tqdm import tqdm

from lightfield.camera import mean_group_extent_scale
from lightfield.sensor import build_coordinate_map, multiplex_forward
from utils.camera import CameraView
from utils.io import write_yaml
from utils.metrics import ssim_window, total_variation_2d
from synthetic.config import TrainConfig, parse_args
from synthetic.data import (
    TransformsDataset,
    load_single_view_dataset,
    load_training_group_centers,
    load_training_dataset,
    nerfpp_norm_radius,
    select_adjacent_test_views,
)
from synthetic.evaluate import evaluate_synthetic
from synthetic.gaussians import (
    build_synthetic_optimizers,
    init_synthetic_gaussians,
    make_exponential_lr_schedule,
)
from synthetic.noise import apply_sensor_noise, quantize_14bit
from utils.runtime import set_seed

Rasterizer = Callable[..., tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]


def load_synthetic_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> torch.nn.ParameterDict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["gaussians"]
    return torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(value.detach().to(device=device))
            for name, value in state.items()
        }
    )


def _rasterize_synthetic_views(
    rasterizer: Rasterizer,
    gaussians: torch.nn.ParameterDict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    width: int,
    height: int,
    sh_degree: int,
    absgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    colors = torch.cat([gaussians["sh0"], gaussians["shN"]], dim=1)
    raster_sh_degree: int | None = sh_degree
    if sh_degree == 0:
        colors = (gaussians["sh0"].squeeze(1) * 0.28209479177387814 + 0.5).clamp(min=0.0)
        raster_sh_degree = None
    return rasterizer(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=intrinsics,
        width=width,
        height=height,
        packed=True,
        absgrad=absgrad,
        sparse_grad=False,
        channel_chunk=16,
        tile_size=4,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        sh_degree=raster_sh_degree,
        render_mode="RGB",
    )


def _synthetic_image_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.read_noise > 0.0 or cfg.shot_noise > 0.0:
        pred = quantize_14bit(apply_sensor_noise(pred, cfg.read_noise, cfg.shot_noise))
    l1 = torch.nn.functional.l1_loss(pred, target)
    dssim = 1.0 - ssim_window(pred, target)
    return (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * dssim


def _backward_unseen_tv(
    *,
    rasterizer: Rasterizer,
    gaussians: torch.nn.ParameterDict,
    camtoworlds: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_queue: list[int],
    width: int,
    height: int,
    step: int,
    cfg: TrainConfig,
) -> list[int]:
    if not camera_queue:
        camera_queue = list(range(int(camtoworlds.shape[0])))
        random.shuffle(camera_queue)
    sample_count = min(3, len(camera_queue))
    unseen_tv_loss = torch.zeros((), device=gaussians["means"].device, dtype=torch.float32)
    for _ in range(sample_count):
        camera_index = camera_queue.pop()
        unseen_render, _, _ = _rasterize_synthetic_views(
            rasterizer,
            gaussians,
            camtoworlds[camera_index : camera_index + 1],
            intrinsics[camera_index : camera_index + 1],
            width=width,
            height=height,
            sh_degree=min(step // cfg.sh_degree_interval, cfg.sh_degree),
            absgrad=False,
        )
        unseen_tv_loss = unseen_tv_loss + total_variation_2d(unseen_render[0].permute(2, 0, 1))
    unseen_tv_loss = unseen_tv_loss / float(sample_count)
    weighted_unseen_tv = cfg.tv_unseen_weight * unseen_tv_loss
    weighted_unseen_tv.backward()
    return camera_queue


def _backward_multiplexed_step(
    *,
    rasterizer: Rasterizer,
    gaussians: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy: DefaultStrategy,
    strategy_state: dict[str, Any],
    groups: list[CameraView],
    targets: dict[int, torch.Tensor],
    coord_map: torch.Tensor,
    lens_size: tuple[int, int],
    weights: torch.Tensor,
    step: int,
    cfg: TrainConfig,
) -> list[dict[str, Any]]:
    sample = groups[0]
    width = sample.width
    height = sample.height
    sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
    num_groups = len(groups)
    all_raster_infos: list[dict[str, Any]] = []
    for group in groups:
        rendered_parts: list[torch.Tensor] = []
        raster_infos: list[dict[str, Any]] = []
        for camera in group:
            rendered_chunk, _, raster_info = _rasterize_synthetic_views(
                rasterizer,
                gaussians,
                camera.camtoworld[None],
                camera.K[None],
                width=width,
                height=height,
                sh_degree=sh_degree,
                absgrad=strategy.absgrad,
            )
            strategy.step_pre_backward(
                params=gaussians,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=raster_info,
            )
            rendered_parts.append(rendered_chunk)
            raster_infos.append(raster_info)
        all_raster_infos.extend(raster_infos)
        rendered = torch.cat(rendered_parts, dim=0)
        pred_subviews = rendered.permute(0, 3, 1, 2)
        group_tv = (
            torch.stack([total_variation_2d(image) for image in pred_subviews]).mean()
            if cfg.tv_weight > 0
            else torch.zeros((), device=gaussians["means"].device)
        )
        prediction = multiplex_forward(
            pred_subviews,
            coord_map,
            lens_size,
            weights,
        )
        target = targets[int(group.image_id)]
        group_loss = _synthetic_image_loss(
            prediction.unsqueeze(0),
            target.unsqueeze(0),
            cfg,
        )
        micro_loss = group_loss / float(num_groups)
        if cfg.tv_weight > 0:
            micro_loss = micro_loss + cfg.tv_weight * group_tv / float(num_groups)
        micro_loss.backward()
    return all_raster_infos


def _backward_multiview_step(
    *,
    rasterizer: Rasterizer,
    gaussians: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy: DefaultStrategy,
    strategy_state: dict[str, Any],
    cameras: list[CameraView],
    step: int,
    cfg: TrainConfig,
) -> dict[str, Any]:
    sample = cameras[0]
    width = sample.width
    height = sample.height
    sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
    train_camtoworlds = torch.stack([camera.camtoworld for camera in cameras])
    train_intrinsics = torch.stack([camera.K for camera in cameras])
    gt_images = torch.stack([camera.image for camera in cameras])
    rendered, _, raster_info = _rasterize_synthetic_views(
        rasterizer,
        gaussians,
        train_camtoworlds,
        train_intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        absgrad=strategy.absgrad,
    )
    strategy.step_pre_backward(
        params=gaussians,
        optimizers=optimizers,
        state=strategy_state,
        step=step,
        info=raster_info,
    )
    predictions = rendered.permute(0, 3, 1, 2)
    targets = gt_images.permute(0, 3, 1, 2)
    image_loss = torch.stack(
        [
            _synthetic_image_loss(
                predictions[index : index + 1],
                targets[index : index + 1],
                cfg,
            )
            for index in range(len(cameras))
        ]
    ).mean()
    tv_loss = torch.zeros((), device=gaussians["means"].device)
    if cfg.tv_weight > 0:
        tv_loss = (predictions[:, :, 1:] - predictions[:, :, :-1]).square().mean()
        tv_loss = tv_loss + (predictions[:, :, :, 1:] - predictions[:, :, :, :-1]).square().mean()
    loss = image_loss + cfg.tv_weight * tv_loss
    loss.backward()
    return raster_info


def _prepare_training_groups(
    dataset: TransformsDataset,
    *,
    multiplexed: bool,
    device: torch.device,
) -> list[CameraView]:
    groups: list[CameraView] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample.image.to(dtype=torch.float32)
        if not multiplexed:
            image = image.to(device=device)
        groups.append(
            CameraView(
                K=sample.K.to(device=device, dtype=torch.float32),
                camtoworld=sample.camtoworld.to(device=device, dtype=torch.float32),
                image=image,
                image_id=sample.image_id,
                embed_id=sample.embed_id,
            )
        )
    return groups


def _build_multiplexed_targets(
    groups: list[CameraView],
    camera_model: str,
) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor, dict[int, torch.Tensor]]:
    sample = groups[0]
    device = sample.K.device
    coord_map_array, lens_size = build_coordinate_map(
        sample.num_views,
        12 if camera_model == "lightfield" else 20,
        sample.height,
        sample.width,
    )
    coord_map = torch.from_numpy(coord_map_array).to(device=device)
    weights = (coord_map[..., 0] >= 0).to(dtype=coord_map.dtype)
    weights /= weights.sum(dim=0).max()
    targets: dict[int, torch.Tensor] = {}
    for group in groups:
        images = group.image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        targets[int(group.image_id)] = multiplex_forward(
            images,
            coord_map,
            lens_size,
            weights,
        ).detach()
    return coord_map, lens_size, weights, targets


def _build_training_strategy(
    cfg: TrainConfig,
    gaussians: torch.nn.ParameterDict,
    *,
    scene_scale: float,
    width: int,
    height: int,
) -> tuple[
    dict[str, torch.optim.Optimizer],
    Callable[[int], float],
    DefaultStrategy,
    dict[str, Any],
]:
    optimizers = build_synthetic_optimizers(
        gaussians,
        scene_scale=scene_scale,
        means_lr=cfg.means_lr_init,
        scales_lr=cfg.scales_lr,
        quats_lr=cfg.quats_lr,
        opacities_lr=cfg.opacities_lr,
        sh0_lr=cfg.sh0_lr,
        shN_lr=cfg.shN_lr,
    )
    means_lr_schedule = make_exponential_lr_schedule(
        lr_init=cfg.means_lr_init * scene_scale,
        lr_final=cfg.means_lr_final * scene_scale,
        max_steps=cfg.means_lr_max_steps,
    )
    max_dimension = max(width, height)
    prune_scale2d = cfg.size_threshold / max_dimension
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=cfg.densify_grad_threshold,
        grow_scale3d=cfg.percent_dense,
        grow_scale2d=1.0,
        prune_scale3d=0.1,
        prune_scale2d=prune_scale2d,
        refine_scale2d_stop_iter=(cfg.densify_until_iter if cfg.size_threshold > 0 else 0),
        refine_start_iter=cfg.densify_from_iter,
        refine_stop_iter=cfg.densify_until_iter,
        refine_every=cfg.densification_interval,
        reset_every=cfg.opacity_reset_interval,
        verbose=False,
    )
    strategy.check_sanity(gaussians, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    return optimizers, means_lr_schedule, strategy, state


def _run_training(
    *,
    cfg: TrainConfig,
    gaussians: torch.nn.ParameterDict,
    train_groups: list[CameraView],
    unseen_cameras: list[CameraView],
    run_dir: Path,
    scene_scale: float,
    densification_scale: float,
    rasterizer: Rasterizer,
) -> None:
    multiplexed_inputs = (
        _build_multiplexed_targets(
            train_groups,
            cfg.camera_model,
        )
        if cfg.camera_model in {"lightfield", "multiplexed"}
        else None
    )

    sample = train_groups[0]
    optimizers, means_lr_schedule, strategy, strategy_state = _build_training_strategy(
        cfg,
        gaussians,
        scene_scale=scene_scale,
        width=sample.width,
        height=sample.height,
    )

    unseen_queue: list[int] = []
    unseen_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    if cfg.tv_unseen_weight > 0.0:
        unseen_inputs = (
            torch.stack(
                [camera.camtoworld for camera in unseen_cameras],
            ).to(device=gaussians["means"].device, dtype=torch.float32),
            torch.stack(
                [camera.K for camera in unseen_cameras],
            ).to(device=gaussians["means"].device, dtype=torch.float32),
        )
        unseen_queue = list(range(len(unseen_cameras)))

    progress = tqdm(range(1, cfg.steps + 1))
    for step in progress:
        optimizers["means"].param_groups[0]["lr"] = float(means_lr_schedule(step))

        if unseen_inputs is not None:
            unseen_camtoworlds, unseen_intrinsics = unseen_inputs
            unseen_queue = _backward_unseen_tv(
                rasterizer=rasterizer,
                gaussians=gaussians,
                camtoworlds=unseen_camtoworlds,
                intrinsics=unseen_intrinsics,
                camera_queue=unseen_queue,
                width=unseen_cameras[0].width,
                height=unseen_cameras[0].height,
                step=step,
                cfg=cfg,
            )

        if multiplexed_inputs is not None:
            groups = list(train_groups)
            random.shuffle(groups)
            coord_map, lens_size, weights, targets = multiplexed_inputs
            raster_infos = _backward_multiplexed_step(
                rasterizer=rasterizer,
                gaussians=gaussians,
                optimizers=optimizers,
                strategy=strategy,
                strategy_state=strategy_state,
                groups=groups,
                targets=targets,
                coord_map=coord_map,
                lens_size=lens_size,
                weights=weights,
                step=step,
                cfg=cfg,
            )
        else:
            cameras = [
                group[view_index]
                for view_index in range(train_groups[0].num_views)
                for group in train_groups
            ]
            random.shuffle(cameras)
            raster_infos = [
                _backward_multiview_step(
                    rasterizer=rasterizer,
                    gaussians=gaussians,
                    optimizers=optimizers,
                    strategy=strategy,
                    strategy_state=strategy_state,
                    cameras=cameras,
                    step=step,
                    cfg=cfg,
                )
            ]

        strategy_state["scene_scale"] = densification_scale
        if step < cfg.steps:
            for index, raster_info in enumerate(raster_infos):
                strategy.step_post_backward(
                    params=gaussians,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step if index == len(raster_infos) - 1 else 1,
                    info={**raster_info, "n_cameras": 1},
                    packed=True,
                )
            if step % strategy.reset_every == 0:
                reset_opa(
                    params=gaussians,
                    optimizers=optimizers,
                    state=strategy_state,
                    value=strategy.prune_opa * 2.0,
                )
        strategy_state["scene_scale"] = scene_scale

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    checkpoint = run_dir / "ckpts" / f"ckpt_{cfg.steps:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"gaussians": {key: value.detach().cpu() for key, value in gaussians.items()}},
        checkpoint,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    root = cfg.data_root.expanduser().resolve()
    out_dir = cfg.result_dir.expanduser().resolve()
    checkpoint = out_dir / "ckpts" / f"ckpt_{cfg.steps:06d}.pt"
    test_cameras = load_single_view_dataset(root)

    device = torch.device("cuda:0")
    if cfg.evaluate:
        gaussians = load_synthetic_checkpoint(checkpoint, device=device)
        train_group_centers = load_training_group_centers(cfg) if cfg.num_exposures == 1 else []
    else:
        set_seed(cfg.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(
            out_dir / "cfg.yml",
            {
                "camera_model": cfg.camera_model,
                "num_exposures": cfg.num_exposures,
            },
        )
        gaussians = init_synthetic_gaussians(sh_degree=cfg.sh_degree, device=device)
        dataset, base_centers, object_center = load_training_dataset(cfg)
        train_groups = _prepare_training_groups(
            dataset,
            multiplexed=cfg.camera_model in {"lightfield", "multiplexed"},
            device=device,
        )
        scene_scale = nerfpp_norm_radius(base_centers)
        densification_scale = (
            float(
                scene_scale
                * mean_group_extent_scale(
                    train_groups,
                    object_center=object_center,
                )
            )
            if cfg.camera_model in {"lightfield", "multiplexed"}
            else scene_scale
        )
        _run_training(
            cfg=cfg,
            gaussians=gaussians,
            train_groups=train_groups,
            unseen_cameras=test_cameras,
            run_dir=out_dir,
            scene_scale=scene_scale,
            densification_scale=densification_scale,
            rasterizer=rasterization,
        )
        train_group_centers = [
            group.camtoworld[..., :3, 3].detach().cpu().numpy() for group in train_groups
        ]

    evaluation_cameras = (
        select_adjacent_test_views(
            train_group_centers,
            test_cameras,
            camera_model=cfg.camera_model,
            max_neighbors=6,
        )
        if cfg.num_exposures == 1
        else test_cameras
    )
    evaluate_synthetic(
        gaussians=gaussians,
        cameras=evaluation_cameras,
        cfg=cfg,
        run_dir=out_dir,
    )
