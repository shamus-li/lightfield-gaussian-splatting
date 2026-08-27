from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image

from synthetic.config import SH_DEGREE
from utils.camera import CameraView
from utils.gaussians import rasterize
from utils.metrics import load_lpips, psnr, ssim_padded


@torch.no_grad()
def evaluate(
    params: Mapping[str, Tensor], views: list[CameraView], render_dir: Path
) -> dict[str, float]:
    """Mean PSNR, SSIM and LPIPS (VGG) of renders of `views`, saved to render_dir."""
    device = params["means"].device
    render_dir.mkdir(parents=True, exist_ok=True)
    lpips = load_lpips("vgg", device, spatial=False)
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for index, view in enumerate(views):
        rendered = rasterize(
            params,
            view.camtoworld[None].to(device),
            view.K[None].to(device),
            view.width,
            view.height,
            sh_degree=SH_DEGREE,
        )[0]
        prediction = rendered.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        target = view.image.to(device).permute(2, 0, 1)[None]
        save_image(prediction[0], render_dir / f"{index:04d}.png")
        totals["psnr"] += psnr(prediction, target).item()
        totals["ssim"] += ssim_padded(prediction, target).item()
        totals["lpips"] += lpips(prediction * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()
    return {name: total / len(views) for name, total in totals.items()}
