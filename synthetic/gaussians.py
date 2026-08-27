from __future__ import annotations

import math
from collections.abc import Callable
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

INITIAL_POINT_COUNT = 100_000


def init_synthetic_gaussians(
    *,
    sh_degree: int,
    device: torch.device,
) -> torch.nn.ParameterDict:
    """Initialize the Gaussian state used by the synthetic trainer."""

    radius = np.random.random(INITIAL_POINT_COUNT).astype(np.float32) ** (1.0 / 3.0)
    theta = np.random.uniform(0.0, 2.0 * math.pi, INITIAL_POINT_COUNT).astype(np.float32)
    phi = np.random.uniform(0.0, math.pi, INITIAL_POINT_COUNT).astype(np.float32)
    points = np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        axis=-1,
    ).astype(np.float32)
    sh0 = np.random.random((INITIAL_POINT_COUNT, 3)).astype(np.float32) / 255.0

    neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
    neighbors.fit(points.astype(np.float64))
    distances, _indices = neighbors.kneighbors(
        points.astype(np.float64),
        n_neighbors=2,
        return_distance=True,
    )
    nearest_distance = np.maximum(distances[:, 1].astype(np.float32), 1e-7)
    scales = np.log(nearest_distance)[:, None].repeat(3, axis=1)
    quaternions = np.zeros((INITIAL_POINT_COUNT, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    opacity_logit = np.log(0.1 / 0.9)
    opacities = opacity_logit * np.ones((INITIAL_POINT_COUNT,), dtype=np.float32)
    colors = np.zeros((INITIAL_POINT_COUNT, (sh_degree + 1) ** 2, 3), dtype=np.float32)
    colors[:, 0, :] = sh0
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(points)),
            "scales": torch.nn.Parameter(torch.from_numpy(scales)),
            "quats": torch.nn.Parameter(torch.from_numpy(quaternions)),
            "opacities": torch.nn.Parameter(torch.from_numpy(opacities)),
            "sh0": torch.nn.Parameter(torch.from_numpy(colors[:, :1, :])),
            "shN": torch.nn.Parameter(torch.from_numpy(colors[:, 1:, :])),
        }
    ).to(device)


def build_synthetic_optimizers(
    gaussians: torch.nn.ParameterDict,
    *,
    scene_scale: float,
    means_lr: float,
    scales_lr: float,
    quats_lr: float,
    opacities_lr: float,
    sh0_lr: float,
    shN_lr: float,
) -> dict[str, torch.optim.Optimizer]:
    """Build the synthetic Adam optimizers."""

    parameters = (
        ("means", gaussians["means"], means_lr * scene_scale),
        ("scales", gaussians["scales"], scales_lr),
        ("quats", gaussians["quats"], quats_lr),
        ("opacities", gaussians["opacities"], opacities_lr),
        ("sh0", gaussians["sh0"], sh0_lr),
        ("shN", gaussians["shN"], shN_lr),
    )
    return {
        name: torch.optim.Adam(
            [
                {
                    "params": parameter,
                    "lr": learning_rate,
                    "name": name,
                }
            ],
            eps=1e-15,
        )
        for name, parameter, learning_rate in parameters
    }


def make_exponential_lr_schedule(
    lr_init: float,
    lr_final: float,
    *,
    max_steps: int = 1_000_000,
) -> Callable[[int], float]:
    """Return the exponential schedule used by synthetic training."""

    def schedule(step: int) -> float:
        interpolation = float(np.clip(step / max_steps, 0.0, 1.0))
        log_lerp = math.exp(
            math.log(lr_init) * (1.0 - interpolation) + math.log(lr_final) * interpolation
        )
        return float(log_lerp)

    return schedule
