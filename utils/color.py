from __future__ import annotations

import torch


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    """Decode sRGB values in [0, 1] to linear radiance."""
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Encode linear radiance as sRGB values."""
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.clamp_min(0.0031308).pow(1 / 2.4) - 0.055,
    )
