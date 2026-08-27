"""Static Gaussian initialization, optimization, scheduling, and densification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy
from sklearn.neighbors import NearestNeighbors

SH_DEGREE = 3


def initialize_default_strategy(
    scene_scale: float,
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    refine_stop_iter: int,
) -> tuple[DefaultStrategy, dict[str, Any]]:
    strategy = DefaultStrategy(
        prune_opa=0.006,
        grow_grad2d=3.5e-4,
        grow_scale3d=0.012,
        grow_scale2d=0.05,
        prune_scale3d=0.22,
        prune_scale2d=0.12,
        refine_scale2d_stop_iter=refine_stop_iter,
        refine_start_iter=500,
        refine_stop_iter=refine_stop_iter,
        reset_every=100_000,
        refine_every=100,
    )
    strategy.check_sanity(params, optimizers)
    return strategy, strategy.initialize_state(scene_scale=scene_scale)


def build_schedulers(
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    pose_optimizer: torch.optim.Optimizer,
    max_steps: int,
) -> list[torch.optim.lr_scheduler.LRScheduler]:
    gamma = 0.01 ** (1.0 / float(max_steps))
    return [
        torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=gamma),
        torch.optim.lr_scheduler.ExponentialLR(pose_optimizer, gamma=gamma),
    ]


def init_gaussians(
    points: np.ndarray,
    point_colors: np.ndarray,
    *,
    device: str,
) -> torch.nn.ParameterDict:
    """Initialize the repository-owned static Gaussian model."""

    initial_points = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(point_colors.astype("float32") / 255.0).float()
    if len(initial_points) > 100_000:
        selected = torch.randperm(len(initial_points))[:100_000]
        initial_points = initial_points[selected]
        rgbs = rgbs[selected]
    points_np = initial_points.detach().cpu().numpy()
    neighbors = NearestNeighbors(
        n_neighbors=min(4, len(initial_points)),
        metric="euclidean",
    ).fit(points_np)
    distances = neighbors.kneighbors(points_np)[0]
    nearest = torch.from_numpy(distances[:, 1:]).to(
        device=initial_points.device,
        dtype=initial_points.dtype,
    )
    scales = torch.log(torch.sqrt((nearest**2).mean(dim=1)).clamp_min(1e-7))
    scales = scales.unsqueeze(-1).repeat(1, 3)
    count = int(initial_points.shape[0])

    colors = torch.zeros((count, (SH_DEGREE + 1) ** 2, 3), dtype=torch.float32)
    colors[:, 0, :] = (rgbs - 0.5) / 0.28209479177387814
    quats = torch.rand((count, 4), dtype=torch.float32)
    opacities = torch.logit(torch.full((count,), 0.1, dtype=torch.float32))
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(initial_points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    ).to(device)


def build_optimizers(
    gaussian_state: torch.nn.ParameterDict,
    *,
    scene_scale: float,
) -> dict[str, torch.optim.Optimizer]:
    params = [
        ("means", gaussian_state["means"], 1.2e-4 * scene_scale),
        ("scales", gaussian_state["scales"], 3e-3),
        ("quats", gaussian_state["quats"], 1e-3),
        ("opacities", gaussian_state["opacities"], 5e-2),
        ("sh0", gaussian_state["sh0"], 2.5e-3),
        ("shN", gaussian_state["shN"], 2.5e-3 / 20.0),
    ]
    return {
        name: torch.optim.Adam(
            [{"params": tensor, "lr": lr, "name": name}],
            # Keep rasterizer-backward roundoff from becoming full sign-only Adam steps.
            eps=1e-8,
        )
        for name, tensor, lr in params
    }
