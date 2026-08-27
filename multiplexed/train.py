"""Reconstruct a scene from one multiplexed sensor image in volume, surface and refit stages.

volume: 3D Gaussians from the calibration points, with densification.
surface: the volume Gaussians as 2D surfels, with densification and surface regularizers.
refit: the surfels inside a tighter hull, with fixed topology and frozen sensor gain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from fused_ssim import fused_ssim
from gsplat.strategy import DefaultStrategy
from torch import Tensor

from multiplexed.render import export
from multiplexed.sensor import (
    Sensor,
    lens_batches,
    load_sensor,
    predict_sensor,
    random_cameras,
    render,
    sensor_image,
)
from multiplexed.support import Hull, build_support, inside_hull, project_hull_
from multiplexed.surfels import SurfaceStrategy, convert_to_surface, surface_regularizers
from utils.color import linear_to_srgb, srgb_to_linear
from utils.gaussians import (
    SH_C0,
    build_optimizers,
    build_strategy,
    exponential_decay,
    init_gaussians,
)
from utils.io import write_config, write_json
from utils.runtime import set_seed

SEED = 0
STAGE_STEPS = {"volume": 3_000, "surface": 3_000, "refit": 500}
REFINE_STOP_ITER = 2_400
SSIM_WEIGHT = 0.2
GAIN_LR = 1e-3
TV_WEIGHT = 0.1
# Surface-stage regularizers switch on at these steps; the refit stage uses them throughout.
DISTORTION_WEIGHT = 0.01
DISTORTION_START = 300
NORMAL_WEIGHT = 0.05
NORMAL_START = 700
# Random cameras near the lens array rendered per step for the regularizers alone.
NOVEL_VIEWS = 3
# The refit stage keeps and confines surfels to this fraction of the hull padding.
REFIT_ALLOWANCE = 0.25
# Scene scale: this quantile of calibration-point distances from their median.
EXTENT_QUANTILE = 0.95
LOG_EVERY = 100


@dataclass(frozen=True)
class Config:
    # Folder with measurement.png and calibration.npz.
    data: Path
    # Run directory for cfg.yml, model.pt, renders/ and metrics.json.
    result_dir: Path


def image_tv(images: Tensor) -> Tensor:
    """Mean absolute difference between neighboring pixels of [B, H, W, C] images."""
    return (images[:, 1:] - images[:, :-1]).abs().mean() + (
        images[:, :, 1:] - images[:, :, :-1]
    ).abs().mean()


@torch.no_grad()
def project_nonnegative_sh0(sh0: Tensor) -> None:
    """Clamp SH0 in place so the DC radiance is nonnegative."""
    sh0.clamp_min_(-0.5 / SH_C0)


def as_parameters(tensors: Mapping[str, Tensor]) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value) for name, value in tensors.items()}
    )


def sensor_loss(linear: Tensor, gain: Tensor, measurement: Tensor) -> tuple[Tensor, Tensor]:
    """L1 + D-SSIM between the sRGB sensor prediction and the measurement, and the prediction."""
    prediction = linear_to_srgb(linear * gain)
    l1 = (prediction - measurement).abs().mean()
    ssim = fused_ssim(prediction.permute(2, 0, 1)[None], measurement.permute(2, 0, 1)[None])
    return (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - ssim), prediction


def surface_weights(stage: str, step: int) -> tuple[float, float]:
    """Distortion and normal regularizer weights."""
    if stage == "volume":
        return 0.0, 0.0
    refit = stage == "refit"
    return (
        DISTORTION_WEIGHT if refit or step >= DISTORTION_START else 0.0,
        NORMAL_WEIGHT if refit or step >= NORMAL_START else 0.0,
    )


def regularizer(
    views: Tensor,
    info: dict[str, Any],
    gain: Tensor,
    normalizer: float,
    weights: tuple[float, float],
) -> Tensor:
    """TV of the sRGB views plus the weighted surface regularizers."""
    loss = TV_WEIGHT * image_tv(linear_to_srgb(views * gain / normalizer))
    if any(weights):
        distortion, normal = surface_regularizers(info)
        loss = loss + weights[0] * distortion + weights[1] * normal
    return loss


def backward(
    params: torch.nn.ParameterDict,
    sensor: Sensor,
    log_gain: Tensor,
    surface: bool,
    weights: tuple[float, float],
    novel_target: Tensor,
    gradient_key: str | None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Accumulate all loss gradients.

    Returns the sensor loss, the sensor prediction and, when `gradient_key` names the 2D means
    used for densification, the info dict for the strategy's step_post_backward.
    """
    # Rendering every lens with gradients does not fit in memory. Instead, take the loss gradient
    # with respect to a detached sensor image, then backpropagate it one lens batch at a time.
    linear = predict_sensor(params, sensor, surface).requires_grad_(True)
    loss, prediction = sensor_loss(linear, log_gain.exp(), sensor.measurement)
    loss.backward()
    grads2d, radii = [], []
    for cameras, footprints in lens_batches(sensor):
        views, info = render(params, cameras, surface)
        if gradient_key is not None:
            info[gradient_key].retain_grad()
        sensor_image(views, footprints, sensor.normalizer).backward(linear.grad, retain_graph=True)
        if gradient_key is not None:
            # Densification sees only the sensor-loss gradient, not the regularizers'.
            grads2d.append(info[gradient_key].grad.clone())
            radii.append(info["radii"])
        fraction = len(cameras) / len(sensor.lenses)
        (fraction * regularizer(views, info, log_gain.exp(), sensor.normalizer, weights)).backward()
    novel = random_cameras(sensor.lenses, novel_target, NOVEL_VIEWS)
    views, info = render(params, novel, surface)
    regularizer(views, info, log_gain.exp(), sensor.normalizer, weights).backward()
    if gradient_key is None:
        return loss.detach(), prediction.detach(), {}
    means2d = torch.empty_like(torch.cat(grads2d))
    # gsplat multiplies 2D gradients by n_cameras to undo a loss averaged over cameras; the
    # sensor loss is not averaged, so divide that factor back out.
    means2d.grad = torch.cat(grads2d) / len(sensor.lenses)
    densify_info = {
        "width": sensor.lenses.width,
        "height": sensor.lenses.height,
        "n_cameras": len(sensor.lenses),
        gradient_key: means2d,
        "radii": torch.cat(radii),
        "gaussian_ids": None,
    }
    return loss.detach(), prediction.detach(), densify_info


def fit_stage(
    stage: str, params: torch.nn.ParameterDict, sensor: Sensor, hull: Hull, gain: float | None
) -> float:
    """Fit `params` in place to the measurement; return the sensor gain.

    Without a `gain`, it starts at the least-squares fit and is optimized; otherwise it is frozen.
    """
    steps = STAGE_STEPS[stage]
    surface = stage != "volume"
    center = sensor.points.median(dim=0).values
    extent = float(torch.quantile((sensor.points - center).norm(dim=-1), EXTENT_QUANTILE))
    project_nonnegative_sh0(params["sh0"])
    project_hull_(params, hull, surface)

    optimizers = build_optimizers(params, scene_scale=extent)
    means_decay = exponential_decay(optimizers["means"], max_steps=steps)
    strategy, state, gradient_key = None, {}, None
    if stage != "refit":
        gradient_key = "gradient_2dgs" if surface else "means2d"
        strategy, state = build_strategy(
            params,
            optimizers,
            SurfaceStrategy if surface else DefaultStrategy,
            scene_scale=extent,
            refine_stop_iter=REFINE_STOP_ITER,
            key_for_gradient=gradient_key,
        )
    if gain is None:
        linear = predict_sensor(params, sensor, surface)
        gain = float((linear * srgb_to_linear(sensor.measurement)).sum() / linear.square().sum())
    log_gain = torch.tensor(math.log(gain), device=sensor.measurement.device)
    steppers = list(optimizers.values())
    if stage != "refit":
        log_gain.requires_grad_(True)
        steppers.append(torch.optim.Adam([log_gain], lr=GAIN_LR))

    for step in range(steps):
        loss, prediction, densify_info = backward(
            params, sensor, log_gain, surface, surface_weights(stage, step), center, gradient_key
        )
        for optimizer in steppers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_decay.step()
        project_nonnegative_sh0(params["sh0"])
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, state, step, densify_info, packed=False)
        project_hull_(params, hull, surface)
        if step % LOG_EVERY == 0 or step + 1 == steps:
            psnr = -10 * math.log10(float((prediction - sensor.measurement).square().mean()))
            print(
                f"{stage} step {step + 1}/{steps} points {len(params['means'])} "
                f"loss {float(loss):.4f} sensor PSNR {psnr:.2f} gain {float(log_gain.exp()):.4f}",
                flush=True,
            )
    return float(log_gain.exp())


def train(sensor: Sensor) -> tuple[torch.nn.ParameterDict, float]:
    """Run the volume, surface and refit stages; return the surfels and the sensor gain."""
    hull = build_support(sensor.points)
    set_seed(SEED)
    points = sensor.points.cpu().numpy()
    params = init_gaussians(
        points, np.full_like(points, 0.5), sh_degree=0, device=sensor.points.device
    )
    fit_stage("volume", params, sensor, hull, gain=None)

    set_seed(SEED)
    params = as_parameters(convert_to_surface(params))
    gain = fit_stage("surface", params, sensor, hull, gain=None)

    set_seed(SEED)
    hull = replace(hull, allowance=hull.allowance * REFIT_ALLOWANCE)
    keep = inside_hull(params, hull, tangent_only=True)
    params = as_parameters({name: value.detach()[keep] for name, value in params.items()})
    gain = fit_stage("refit", params, sensor, hull, gain=gain)
    return params, gain


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(Config, args=argv, prog="train.py multiplexed", description=__doc__)
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    write_config(cfg.result_dir / "cfg.yml", cfg)
    sensor = load_sensor(cfg.data, torch.device("cuda"))
    params, gain = train(sensor)
    torch.save({"params": params.state_dict(), "gain": gain}, cfg.result_dir / "model.pt")
    metrics = export(params, gain, sensor, cfg.result_dir / "renders")
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)
