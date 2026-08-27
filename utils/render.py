from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from gsplat.rendering import rasterization

from utils.camera import CameraView


@dataclass(frozen=True)
class RenderConfig:
    sh_degree: int = 3
    antialiased: bool = False
    packed: bool = False
    channel_chunk: int = 16
    tile_size: int = 4


def render_item(
    gaussians: torch.nn.ParameterDict,
    item: CameraView,
    *,
    cfg: RenderConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = gaussians["means"].device
    camtoworlds = torch.stack(
        [view.camtoworld.to(device=device, dtype=torch.float32) for view in item]
    )
    intrinsics = torch.stack([view.K.to(device=device, dtype=torch.float32) for view in item])
    colors = torch.cat([gaussians["sh0"], gaussians["shN"]], dim=1)
    renders, _alphas, info = rasterization(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=intrinsics,
        width=item.width,
        height=item.height,
        packed=cfg.packed,
        channel_chunk=cfg.channel_chunk,
        tile_size=cfg.tile_size,
        rasterize_mode="antialiased" if cfg.antialiased else "classic",
        sh_degree=cfg.sh_degree,
    )
    return (renders[0] if item.num_views == 1 else renders), info
