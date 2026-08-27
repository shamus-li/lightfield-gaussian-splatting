from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import tyro
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa
from torch import Tensor
from tqdm import tqdm

from synthetic.config import (
    LENS_SENSOR_DISTANCE,
    SH_DEGREE,
    Config,
    Hyperparameters,
    hyperparameters,
)
from synthetic.data import (
    evaluation_views,
    focus_point,
    group_extent_scale,
    load_test_views,
    load_training_groups,
    scene_radius,
    training_dir,
)
from synthetic.evaluate import evaluate
from synthetic.sensor import build_coordinate_map, multiplex_forward
from utils.camera import CameraView
from utils.gaussians import (
    LEARNING_RATES,
    MAX_INIT_POINTS,
    SH_C0,
    build_optimizers,
    build_strategy,
    exponential_decay,
    init_gaussians,
    rasterize,
)
from utils.io import read_yaml, write_config, write_json
from utils.metrics import ssim_padded, total_variation_2d
from utils.runtime import set_seed

SEED = 0
STEPS = 3_000
# Steps between raising the rendered SH degree by one.
SH_DEGREE_INTERVAL = 500
SCALES_LR = 0.005
# The means learning rate decays over 30k steps while training stops at 3k, so it covers only
# a tenth of its log-space path to the final rate. The paper's numbers were produced this way.
MEANS_LR_DECAY_STEPS = 30_000
# Gaussians whose screen-space radius exceeds this many pixels are pruned.
PRUNE_RADIUS_PIXELS = 150
# DefaultStrategy settings shared by every camera model; grow_scale2d=1.0 disables
# screen-space splitting.
STRATEGY = {"prune_opa": 0.005, "grow_scale3d": 0.06, "grow_scale2d": 1.0, "prune_scale3d": 0.1}
# Test poses rendered per step for the unseen-view total-variation loss.
UNSEEN_VIEWS_PER_STEP = 3
SENSOR_LEVELS = 2**14 - 1


def random_ball_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over the unit ball with near-gray colors."""
    radius = np.random.random(count) ** (1.0 / 3.0)
    theta = np.random.uniform(0.0, 2.0 * math.pi, count)
    phi = np.random.uniform(0.0, math.pi, count)
    points = np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        axis=-1,
    )
    colors = 0.5 + SH_C0 * np.random.random((count, 3)) / 255.0
    return points.astype(np.float32), colors.astype(np.float32)


def add_sensor_noise(image: Tensor, read_noise: float, shot_noise: float) -> Tensor:
    """Read and shot noise followed by 14-bit quantization with a straight-through gradient."""
    noisy = (image + torch.randn_like(image) * (read_noise + shot_noise * image)).clamp(0.0, 1.0)
    quantized = torch.round(noisy * SENSOR_LEVELS) / SENSOR_LEVELS
    return noisy + (quantized - noisy).detach()


def image_loss(pred: Tensor, target: Tensor, cfg: Config, hp: Hyperparameters) -> Tensor:
    """L1 + D-SSIM between [B, 3, H, W] images after simulating sensor noise on the prediction."""
    if cfg.read_noise > 0.0 or cfg.shot_noise > 0.0:
        pred = add_sensor_noise(pred, cfg.read_noise, cfg.shot_noise)
    dssim = 1.0 - ssim_padded(pred, target)
    return (1.0 - hp.lambda_dssim) * torch.nn.functional.l1_loss(pred, target) + (
        hp.lambda_dssim * dssim
    )


def multiview_losses(
    params: torch.nn.ParameterDict,
    views: CameraView,
    sh_degree: int,
    cfg: Config,
    hp: Hyperparameters,
) -> Iterator[tuple[Tensor, dict[str, Any]]]:
    """One loss over all training views, rendered as a single batch."""
    rendered, _, info = rasterize(
        params, views.camtoworld, views.K, views.width, views.height, sh_degree=sh_degree
    )
    predictions = rendered.permute(0, 3, 1, 2)
    loss = image_loss(predictions, views.image.permute(0, 3, 1, 2), cfg, hp)
    yield loss + hp.tv_weight * total_variation_2d(predictions), info


def multiplexed_losses(
    params: torch.nn.ParameterDict,
    groups: Sequence[CameraView],
    measurements: Sequence[Tensor],
    sensor: tuple[Tensor, tuple[int, int], Tensor],
    sh_degree: int,
    cfg: Config,
    hp: Hyperparameters,
) -> Iterator[tuple[Tensor, dict[str, Any]]]:
    """Per-capture losses between multiplexed renders and the multiplexed ground truth."""
    for index in random.sample(range(len(groups)), len(groups)):
        group = groups[index]
        rendered, _, info = rasterize(
            params, group.camtoworld, group.K, group.width, group.height, sh_degree=sh_degree
        )
        views = rendered.permute(0, 3, 1, 2)
        prediction = multiplex_forward(views, *sensor)
        loss = image_loss(prediction[None], measurements[index][None], cfg, hp)
        yield (loss + hp.tv_weight * total_variation_2d(views)) / len(groups), info


def multiplexing_sensor(
    groups: Sequence[CameraView], camera_model: str, device: torch.device
) -> tuple[Tensor, tuple[int, int], Tensor]:
    """Microlens coordinate map, lens size and per-lens weights for multiplex_forward."""
    sample = groups[0]
    coord_map, lens_size = build_coordinate_map(
        sample.num_views, LENS_SENSOR_DISTANCE[camera_model], sample.height, sample.width
    )
    coords = torch.from_numpy(coord_map).to(device)
    weights = (coords[..., 0] >= 0).float()
    return coords, lens_size, weights / weights.sum(dim=0).max()


def densification_scale(groups: Sequence[CameraView], scene_scale: float) -> float:
    """Scene scale for densification, widened by the angular extent of light-field captures."""
    camtoworlds = torch.stack([group.camtoworld for group in groups]).cpu().numpy()
    focus = focus_point(camtoworlds.reshape(-1, 4, 4))
    return scene_scale * group_extent_scale(camtoworlds, focus)


def refine(
    strategy: DefaultStrategy,
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
    step: int,
    infos: list[dict[str, Any]],
) -> None:
    """Densify and prune from the gradient statistics of this step's renders."""
    for index, info in enumerate(infos):
        # Earlier renders pass step=1 so gsplat only accumulates their gradient statistics.
        # n_cameras=1 skips gsplat's batch-size gradient scaling; grow_grad2d was tuned so.
        strategy.step_post_backward(
            params,
            optimizers,
            state,
            step if index == len(infos) - 1 else 1,
            {**info, "n_cameras": 1},
            packed=True,
        )
    # gsplat 1.5.3 never resets opacities: `step % reset_every == 0 & step > 0` is always false.
    if step % strategy.reset_every == 0:
        reset_opa(params, optimizers, state, value=strategy.prune_opa * 2.0)


def train(cfg: Config, unseen_views: Sequence[CameraView]) -> torch.nn.ParameterDict:
    """Fit Gaussians to the simulated captures of cfg.camera_model."""
    device = torch.device("cuda")
    hp = hyperparameters(cfg.camera_model)
    multiplexed = cfg.camera_model in LENS_SENSOR_DISTANCE
    groups = [
        replace(group, K=group.K.to(device), camtoworld=group.camtoworld.to(device))
        for group in load_training_groups(
            training_dir(cfg.data, cfg.camera_model, cfg.num_exposures), cfg.camera_model
        )
    ]
    if multiplexed:
        sensor = multiplexing_sensor(groups, cfg.camera_model, device)
        measurements = [
            multiplex_forward(group.image.to(device).permute(0, 3, 1, 2), *sensor)
            for group in groups
        ]
    else:
        all_views = CameraView(
            K=torch.cat([group.K for group in groups]),
            camtoworld=torch.cat([group.camtoworld for group in groups]),
            image=torch.cat([group.image for group in groups]).to(device),
            image_id=0,
            embed_id=0,
        )

    scene_scale = scene_radius(cfg.data)
    params = init_gaussians(*random_ball_points(MAX_INIT_POINTS), device=device)
    optimizers = build_optimizers(
        params,
        scene_scale=scene_scale,
        learning_rates={
            **LEARNING_RATES,
            "means": hp.means_lr,
            "scales": SCALES_LR,
            "opacities": hp.opacities_lr,
        },
        eps=1e-15,
    )
    means_scheduler = exponential_decay(
        optimizers["means"],
        max_steps=MEANS_LR_DECAY_STEPS,
        final_ratio=hp.means_lr_final / hp.means_lr,
    )
    strategy, state = build_strategy(
        params,
        optimizers,
        scene_scale=densification_scale(groups, scene_scale) if multiplexed else scene_scale,
        refine_stop_iter=hp.refine_stop_iter,
        **STRATEGY,
        grow_grad2d=hp.grow_grad2d,
        prune_scale2d=PRUNE_RADIUS_PIXELS / max(groups[0].width, groups[0].height),
        refine_start_iter=hp.refine_start_iter,
        refine_every=hp.refine_every,
        reset_every=hp.reset_every,
    )
    unseen_camtoworlds = torch.stack([view.camtoworld for view in unseen_views]).to(device)
    unseen_Ks = torch.stack([view.K for view in unseen_views]).to(device)
    unseen_queue: list[int] = []

    for step in tqdm(range(1, STEPS + 1)):
        sh_degree = min(step // SH_DEGREE_INTERVAL, SH_DEGREE)
        if hp.tv_unseen_weight > 0.0:
            if not unseen_queue:
                unseen_queue = random.sample(range(len(unseen_views)), len(unseen_views))
            picked = unseen_queue[-UNSEEN_VIEWS_PER_STEP:]
            del unseen_queue[-UNSEEN_VIEWS_PER_STEP:]
            unseen = rasterize(
                params,
                unseen_camtoworlds[picked],
                unseen_Ks[picked],
                unseen_views[0].width,
                unseen_views[0].height,
                sh_degree=sh_degree,
            )[0]
            (hp.tv_unseen_weight * total_variation_2d(unseen.permute(0, 3, 1, 2))).backward()

        if multiplexed:
            losses = multiplexed_losses(params, groups, measurements, sensor, sh_degree, cfg, hp)
        else:
            losses = multiview_losses(params, all_views, sh_degree, cfg, hp)
        infos = []
        for loss, info in losses:
            strategy.step_pre_backward(params, optimizers, state, step, info)
            loss.backward()
            infos.append(info)
        # Skip the last step so its opacity reset cannot wipe the finished model.
        if step < STEPS:
            refine(strategy, params, optimizers, state, step, infos)
        if sh_degree == 0:
            # shN only receives zeros before it is rendered; keep Adam from counting those steps.
            params["shN"].grad = None
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_scheduler.step()
    return params


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        Config,
        args=argv,
        prog="train.py synthetic",
        description="Train and evaluate Gaussian splatting on a simulated synthetic capture.",
    )
    if cfg.eval:
        saved = read_yaml(cfg.result_dir / "cfg.yml")
        cfg = replace(cfg, camera_model=saved["camera_model"], num_exposures=saved["num_exposures"])
    test_views = load_test_views(cfg.data)
    model_path = cfg.result_dir / "model.pt"
    if not cfg.eval:
        cfg.result_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg.result_dir / "cfg.yml", cfg)
        set_seed(SEED)
        torch.save(train(cfg, test_views).state_dict(), model_path)
    params = torch.load(model_path, map_location="cuda", weights_only=True)
    views = evaluation_views(
        test_views,
        training_dir(cfg.data, cfg.camera_model, cfg.num_exposures),
        cfg.camera_model,
        cfg.num_exposures,
    )
    metrics = evaluate(params, views, cfg.result_dir / "renders")
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)
