from __future__ import annotations

import torch
from torch import Tensor


def apply_sensor_noise(image: Tensor, lambda_read: float, lambda_shot: float) -> Tensor:
    if lambda_read <= 0.0 and lambda_shot <= 0.0:
        return image
    noise = torch.randn_like(image) * (lambda_read + lambda_shot * image)
    return (image + noise).clamp(0.0, 1.0)


def quantize_14bit(image: Tensor) -> Tensor:
    levels = (1 << 14) - 1
    quantized = torch.round(image * levels) / levels
    return image + (quantized - image).detach()
