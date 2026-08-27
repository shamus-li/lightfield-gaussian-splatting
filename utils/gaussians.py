from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

SH_C0 = 0.28209479177387814
MAX_INIT_POINTS = 100_000
# Means learning rate before scaling by the scene scale.
LEARNING_RATES = {
    "means": 1.2e-4,
    "scales": 3e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
}
# DefaultStrategy settings that differ from gsplat's defaults; reset_every past the end
# of training disables opacity resets.
STRATEGY: dict[str, Any] = {
    "prune_opa": 0.006,
    "grow_grad2d": 3.5e-4,
    "grow_scale3d": 0.012,
    "prune_scale3d": 0.22,
    "prune_scale2d": 0.12,
    "reset_every": 100_000,
}


def init_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    sh_degree: int = 3,
    opacity: float = 0.1,
    device: torch.device | str,
) -> torch.nn.ParameterDict:
    """Gaussians at up to MAX_INIT_POINTS `points` with RGB `colors` in [0, 1] and kNN scales."""
    means = torch.from_numpy(points).float()
    rgbs = torch.from_numpy(colors).float()
    if len(means) > MAX_INIT_POINTS:
        selected = torch.randperm(len(means))[:MAX_INIT_POINTS]
        means = means[selected]
        rgbs = rgbs[selected]
    distances = NearestNeighbors(n_neighbors=3).fit(means.numpy()).kneighbors()[0]
    scales = torch.from_numpy(np.sqrt((distances**2).mean(axis=1))).float()
    scales = torch.log(scales.clamp_min(1e-7))[:, None].repeat(1, 3)
    count = len(means)
    sh = torch.zeros((count, (sh_degree + 1) ** 2, 3))
    sh[:, 0] = (rgbs - 0.5) / SH_C0
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(torch.rand((count, 4))),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((count,), opacity))),
            "sh0": torch.nn.Parameter(sh[:, :1]),
            "shN": torch.nn.Parameter(sh[:, 1:]),
        }
    ).to(device)


def build_optimizers(
    params: torch.nn.ParameterDict,
    *,
    scene_scale: float,
    learning_rates: Mapping[str, float] = LEARNING_RATES,
    eps: float = 1e-8,
) -> dict[str, torch.optim.Optimizer]:
    """One Adam optimizer per parameter; the means learning rate is multiplied by `scene_scale`."""
    return {
        name: torch.optim.Adam(
            [
                {
                    "params": params[name],
                    "lr": lr * scene_scale if name == "means" else lr,
                    "name": name,
                }
            ],
            eps=eps,
        )
        for name, lr in learning_rates.items()
    }


def build_strategy(
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_class: type[DefaultStrategy] = DefaultStrategy,
    /,
    *,
    scene_scale: float,
    refine_stop_iter: int,
    **settings: Any,
) -> tuple[DefaultStrategy, dict[str, Any]]:
    """`strategy_class` with the STRATEGY settings updated by `settings`, and its state."""
    strategy = strategy_class(
        **{**STRATEGY, **settings},
        refine_stop_iter=refine_stop_iter,
        refine_scale2d_stop_iter=refine_stop_iter,
    )
    strategy.check_sanity(params, optimizers)
    return strategy, strategy.initialize_state(scene_scale=scene_scale)


def exponential_decay(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    final_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.ExponentialLR:
    """Scheduler that decays the learning rate by `final_ratio` over `max_steps` steps."""
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=final_ratio ** (1.0 / max_steps))


def rasterize(
    params: Mapping[str, Tensor] | torch.nn.ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    **kwargs: Any,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Render Gaussians from [C, 4, 4] camera-to-world poses with gsplat's rasterization."""
    return rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], dim=1),
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        **kwargs,
    )


def save_ply(params: Mapping[str, Tensor] | torch.nn.ParameterDict, path: Path) -> None:
    """Write Gaussians as a 3DGS-format PLY file."""
    arrays = [
        params["means"],
        torch.zeros_like(params["means"]),
        params["sh0"][:, 0],
        params["shN"].transpose(1, 2).flatten(1),
        params["opacities"][:, None],
        params["scales"],
        params["quats"],
    ]
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(params["shN"].shape[1] * 3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    values = torch.cat(arrays, dim=1).detach().cpu().numpy()
    vertices = np.empty(len(values), dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def load_ply(path: Path, device: torch.device | str) -> dict[str, Tensor]:
    """Read Gaussians from a 3DGS-format PLY file."""
    vertex = PlyData.read(path)["vertex"]
    names = vertex.data.dtype.names or ()
    rest = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.removeprefix("f_rest_")),
    )

    def columns(fields: list[str]) -> Tensor:
        return torch.tensor(
            np.column_stack([vertex[name] for name in fields]), dtype=torch.float32, device=device
        )

    return {
        "means": columns(["x", "y", "z"]),
        "scales": columns([f"scale_{i}" for i in range(3)]),
        "quats": columns([f"rot_{i}" for i in range(4)]),
        "opacities": columns(["opacity"])[:, 0],
        "sh0": columns([f"f_dc_{i}" for i in range(3)])[:, None],
        "shN": (
            columns(rest).reshape(-1, 3, len(rest) // 3).transpose(1, 2).contiguous()
            if rest
            else torch.empty((len(vertex), 0, 3), device=device)
        ),
    }
