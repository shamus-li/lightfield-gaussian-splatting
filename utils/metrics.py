from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import lpips as lpips_module
import torch
import torch.nn.functional as F
from torch import Tensor

_LPIPS_MODELS: dict[tuple[str, str, bool], torch.nn.Module] = {}


def _mask(
    mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = mask.to(device=device, dtype=torch.float32)
    return (mask > 0.5).to(dtype=torch.float32)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt = gt.to(pred)
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    return (-10.0 * torch.log10(mse.clamp(min=1e-10))).mean()


def mpsnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    gt = gt.to(pred)
    mask_bchw = _mask(
        mask,
        device=pred.device,
    )
    valid = mask_bchw.sum(dim=(1, 2, 3))
    expanded_mask = mask_bchw.expand(-1, pred.shape[1], -1, -1)
    mse = ((pred - gt) ** 2 * expanded_mask).sum(dim=(1, 2, 3)) / (valid * pred.shape[1]).clamp(
        min=1e-10
    )
    return (-10.0 * torch.log10(mse.clamp(min=1e-10))).mean()


def gaussian_filter(filter_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    half_width = filter_size // 2
    shift = (2 * half_width - filter_size + 1) / 2
    coords = torch.arange(filter_size, dtype=torch.float32, device=device)
    filt = torch.exp(-0.5 * (((coords - half_width + shift) / sigma) ** 2))
    return filt / filt.sum()


def partial_conv1d(
    image: torch.Tensor,
    mask: torch.Tensor,
    filt: torch.Tensor,
    *,
    horizontal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = image.shape[1]
    filter_size = int(filt.numel())
    kernel = filt.view(1, 1, 1, filter_size) if horizontal else filt.view(1, 1, filter_size, 1)
    values = F.conv2d(image * mask, kernel.repeat(channels, 1, 1, 1), groups=channels)
    coverage = F.conv2d(mask, torch.ones_like(kernel))
    filtered = torch.where(
        coverage != 0,
        values * float(filter_size) / coverage,
        torch.zeros_like(values),
    )
    return filtered, (coverage != 0).to(dtype=image.dtype)


def partial_filter(
    image: torch.Tensor,
    mask: torch.Tensor,
    filt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, mask = partial_conv1d(image, mask, filt, horizontal=True)
    return partial_conv1d(image, mask, filt, horizontal=False)


def mssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    filter_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    gt = gt.to(pred)
    mask_bchw = _mask(
        mask,
        device=pred.device,
    )
    filt = gaussian_filter(filter_size, sigma, pred.device)

    mu_pred = partial_filter(pred, mask_bchw, filt)[0]
    mu_gt = partial_filter(gt, mask_bchw, filt)[0]
    mu_pred_sq = mu_pred**2
    mu_gt_sq = mu_gt**2
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = partial_filter(pred**2, mask_bchw, filt)[0] - mu_pred_sq
    sigma_gt_sq = partial_filter(gt**2, mask_bchw, filt)[0] - mu_gt_sq
    sigma_pred_gt = partial_filter(pred * gt, mask_bchw, filt)[0] - mu_pred_gt
    sigma_pred_sq = torch.maximum(torch.zeros_like(sigma_pred_sq), sigma_pred_sq)
    sigma_gt_sq = torch.maximum(torch.zeros_like(sigma_gt_sq), sigma_gt_sq)
    variance_product = sigma_pred_sq * sigma_gt_sq
    covariance_limit = torch.sqrt(variance_product)
    sigma_pred_gt = torch.sign(sigma_pred_gt) * torch.minimum(
        covariance_limit,
        torch.abs(sigma_pred_gt),
    )
    c1 = k1**2
    c2 = k2**2
    ssim_map = ((2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    )
    return ssim_map.mean()


def ssim_dycheck(
    pred: torch.Tensor,
    gt: torch.Tensor,
    filter_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    gt = gt.to(pred)
    mask = torch.ones(
        pred.shape[0],
        1,
        pred.shape[2],
        pred.shape[3],
        device=pred.device,
    )
    return mssim(
        pred,
        gt,
        mask,
        filter_size=filter_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
    )


def ssim_window(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    channel = int(img1.size(1))
    filt = gaussian_filter(window_size, 1.5, img1.device).to(dtype=img1.dtype).unsqueeze(1)
    window = (filt @ filt.t()).expand(channel, 1, window_size, window_size).contiguous()
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1.square(), window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.square(), window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    c1 = 0.01**2
    c2 = 0.03**2
    return (
        ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
        / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ).mean()


def get_lpips_model(
    net: str,
    device: torch.device,
    *,
    spatial: bool = True,
) -> torch.nn.Module:
    key = (net, f"{device.type}:{device.index if device.index is not None else 0}", spatial)
    model = _LPIPS_MODELS.get(key)
    if model is not None:
        return model

    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / "models/hub"))
    model = lpips_module.LPIPS(net=net, spatial=spatial).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _LPIPS_MODELS[key] = model
    return model


def lpips(pred: torch.Tensor, gt: torch.Tensor, net: str = "vgg") -> torch.Tensor:
    gt = gt.to(pred)
    mask = torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3], device=pred.device)
    return mlpips(pred, gt, mask, net=net)


def mlpips(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    net: str = "vgg",
) -> torch.Tensor:
    gt = gt.to(pred)
    mask_bchw = _mask(
        mask,
        device=pred.device,
    )
    expanded_mask = mask_bchw.expand_as(pred)
    # DyCheck masks in image space before LPIPS normalizes [0, 1] to [-1, 1].
    # Consequently excluded pixels are -1, not 0.
    pred_lp = pred * expanded_mask * 2.0 - 1.0
    gt_lp = gt * expanded_mask * 2.0 - 1.0
    with torch.no_grad():
        dist_map = get_lpips_model(net, pred.device)(pred_lp, gt_lp)
    mask_down = F.interpolate(mask_bchw, size=dist_map.shape[2:], mode="nearest")
    valid_sum = mask_down.sum()
    return (dist_map * mask_down).sum() / valid_sum


def compute_pair_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    lpips_net: str = "vgg",
) -> dict[str, float]:
    """Compute masked image metrics for one RGB image pair."""

    pred_t = torch.from_numpy(pred.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    gt_t = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    mask_bool = np.ones(pred.shape[:2], dtype=bool) if mask is None else mask > 0
    mask_t = torch.from_numpy(mask_bool.astype(np.float32))[None, None]
    result = {
        "psnr": float(mpsnr(pred_t, gt_t, mask_t).item()),
        "ssim": float(mssim(pred_t, gt_t, mask_t).item()),
    }
    result["lpips"] = float(mlpips(pred_t, gt_t, mask_t, net=lpips_net).item())
    return result


def summarize_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips"):
        values = [row[key] for row in rows]
        summary[key] = float(np.mean(values))
    return summary


def total_variation_2d(image_chw: Tensor) -> Tensor:
    return (image_chw[:, 1:, :] - image_chw[:, :-1, :]).square().mean() + (
        image_chw[:, :, 1:] - image_chw[:, :, :-1]
    ).square().mean()
