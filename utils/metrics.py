from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import lpips as lpips_module
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2

_LPIPS_MODELS: dict[tuple[str, str, bool], torch.nn.Module] = {}


def _mask(mask: Tensor | None, pred: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(pred[:, :1])
    return mask.to(pred)


def psnr(pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean PSNR of [B, C, H, W] images in [0, 1], inside an optional [B, 1, H, W] mask."""
    mask = _mask(mask, pred)
    squared_error = ((pred - gt.to(pred)) ** 2 * mask).sum(dim=(1, 2, 3))
    mse = squared_error / (mask.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp(min=1e-10)
    return (-10.0 * torch.log10(mse.clamp(min=1e-10))).mean()


def _gaussian_filter(device: torch.device) -> Tensor:
    coords = torch.arange(SSIM_WINDOW, dtype=torch.float32, device=device) - SSIM_WINDOW // 2
    filt = torch.exp(-0.5 * (coords / SSIM_SIGMA) ** 2)
    return filt / filt.sum()


def _partial_conv1d(
    image: Tensor, mask: Tensor, filt: Tensor, *, horizontal: bool
) -> tuple[Tensor, Tensor]:
    channels = image.shape[1]
    kernel = filt.view(1, 1, 1, -1) if horizontal else filt.view(1, 1, -1, 1)
    values = F.conv2d(image * mask, kernel.repeat(channels, 1, 1, 1), groups=channels)
    coverage = F.conv2d(mask, torch.ones_like(kernel))
    filtered = torch.where(
        coverage != 0,
        values * SSIM_WINDOW / coverage,
        torch.zeros_like(values),
    )
    return filtered, (coverage != 0).to(image)


def _partial_filter(image: Tensor, mask: Tensor, filt: Tensor) -> Tensor:
    image, mask = _partial_conv1d(image, mask, filt, horizontal=True)
    return _partial_conv1d(image, mask, filt, horizontal=False)[0]


def ssim(pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> Tensor:
    """DyCheck SSIM of [B, C, H, W] images, filtering only pixels inside an optional mask."""
    gt = gt.to(pred)
    mask = _mask(mask, pred)
    filt = _gaussian_filter(pred.device)
    mu_pred = _partial_filter(pred, mask, filt)
    mu_gt = _partial_filter(gt, mask, filt)
    sigma_pred_sq = (_partial_filter(pred**2, mask, filt) - mu_pred**2).clamp_min(0)
    sigma_gt_sq = (_partial_filter(gt**2, mask, filt) - mu_gt**2).clamp_min(0)
    sigma_pred_gt = _partial_filter(pred * gt, mask, filt) - mu_pred * mu_gt
    sigma_pred_gt = torch.sign(sigma_pred_gt) * torch.minimum(
        torch.sqrt(sigma_pred_sq * sigma_gt_sq),
        torch.abs(sigma_pred_gt),
    )
    ssim_map = ((2 * mu_pred * mu_gt + SSIM_C1) * (2 * sigma_pred_gt + SSIM_C2)) / (
        (mu_pred**2 + mu_gt**2 + SSIM_C1) * (sigma_pred_sq + sigma_gt_sq + SSIM_C2)
    )
    return ssim_map.mean()


def ssim_padded(pred: Tensor, gt: Tensor) -> Tensor:
    """SSIM of [B, C, H, W] images with a zero-padded Gaussian window, as in the 3DGS loss."""
    channels = pred.shape[1]
    filt = _gaussian_filter(pred.device).to(pred).unsqueeze(1)
    window = (filt @ filt.t()).expand(channels, 1, SSIM_WINDOW, SSIM_WINDOW).contiguous()

    def blur(image: Tensor) -> Tensor:
        return F.conv2d(image, window, padding=SSIM_WINDOW // 2, groups=channels)

    mu_pred = blur(pred)
    mu_gt = blur(gt)
    sigma_pred_sq = blur(pred.square()) - mu_pred.square()
    sigma_gt_sq = blur(gt.square()) - mu_gt.square()
    sigma_pred_gt = blur(pred * gt) - mu_pred * mu_gt
    return (
        ((2 * mu_pred * mu_gt + SSIM_C1) * (2 * sigma_pred_gt + SSIM_C2))
        / ((mu_pred.square() + mu_gt.square() + SSIM_C1) * (sigma_pred_sq + sigma_gt_sq + SSIM_C2))
    ).mean()


def load_lpips(net: str, device: torch.device, *, spatial: bool = True) -> torch.nn.Module:
    """Cached frozen LPIPS network; spatial models return a per-pixel distance map."""
    key = (net, str(device), spatial)
    if key not in _LPIPS_MODELS:
        torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / "models/hub"))
        model = lpips_module.LPIPS(net=net, spatial=spatial).to(device).eval()
        model.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


@torch.no_grad()
def lpips(pred: Tensor, gt: Tensor, mask: Tensor | None = None, net: str = "vgg") -> Tensor:
    """DyCheck LPIPS of [B, C, H, W] images in [0, 1], averaged inside an optional mask."""
    mask = _mask(mask, pred)
    # DyCheck masks in image space before mapping [0, 1] to [-1, 1], so excluded pixels are -1.
    dist_map = load_lpips(net, pred.device)(pred * mask * 2.0 - 1.0, gt.to(pred) * mask * 2.0 - 1.0)
    mask = F.interpolate(mask, size=dist_map.shape[2:], mode="nearest")
    return (dist_map * mask).sum() / mask.sum()


def compute_pair_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    lpips_net: str = "vgg",
) -> dict[str, float]:
    """PSNR, SSIM and LPIPS of one [H, W, 3] image pair inside an optional [H, W] bool mask."""
    pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None]
    gt_t = torch.from_numpy(gt).permute(2, 0, 1)[None]
    mask_t = None if mask is None else torch.from_numpy(mask)[None, None]
    return {
        "psnr": psnr(pred_t, gt_t, mask_t).item(),
        "ssim": ssim(pred_t, gt_t, mask_t).item(),
        "lpips": lpips(pred_t, gt_t, mask_t, net=lpips_net).item(),
    }


def summarize_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric over per-image rows."""
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def total_variation_2d(images: Tensor) -> Tensor:
    """Mean squared difference between neighboring pixels of [..., H, W] images."""
    return (images[..., 1:, :] - images[..., :-1, :]).square().mean() + (
        images[..., :, 1:] - images[..., :, :-1]
    ).square().mean()
