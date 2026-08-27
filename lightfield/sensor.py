from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def build_coordinate_map(
    num_lens: int, d_lens_sensor: int, height: int, width: int
) -> tuple[np.ndarray, tuple[int, int]]:
    grid_size = int(math.sqrt(num_lens))

    base_microlens_size = min(height // grid_size, width // grid_size) // 12
    microlens_h = int(base_microlens_size * d_lens_sensor)
    microlens_h = microlens_h - (microlens_h % 2)
    microlens_w = microlens_h

    coord_map = -np.ones((num_lens, height, width, 2), dtype=np.float32)

    y_positions = np.linspace(microlens_h // 2, height - microlens_h // 2, grid_size)
    x_positions = np.linspace(microlens_w // 2, width - microlens_w // 2, grid_size)

    for i in range(num_lens):
        row, col = i // grid_size, i % grid_size
        center_y, center_x = int(y_positions[row]), int(x_positions[col])
        start_y = int(max(0, center_y - microlens_h // 2))
        end_y = int(min(height, center_y + microlens_h // 2))
        start_x = int(max(0, center_x - microlens_w // 2))
        end_x = int(min(width, center_x + microlens_w // 2))
        coord_map[i, start_y:end_y, start_x:end_x, 0] = np.arange(end_y - start_y)[:, None]
        coord_map[i, start_y:end_y, start_x:end_x, 1] = np.arange(end_x - start_x)[None, :]

    return coord_map, (microlens_h, microlens_w)


def multiplex_forward(
    images: Tensor,
    coord_map: Tensor,
    lens_size: tuple[int, int],
    microlens_weights: Tensor,
) -> Tensor:
    num_lens, height, width = coord_map.shape[:3]
    grid_size = int(math.sqrt(num_lens))
    image_device = images.device
    idx = torch.arange(grid_size, device=image_device)
    grid_i, grid_j = torch.meshgrid(idx, idx, indexing="ij")
    mapping = ((grid_size - 1 - grid_i) + (grid_size - 1 - grid_j) * grid_size).reshape(-1)

    selected_images = images[mapping]
    resized = F.interpolate(
        selected_images,
        size=lens_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    resized_linear = torch.where(
        resized <= 0.04045,
        resized / 12.92,
        torch.pow((resized + 0.055) / 1.055, 2.4),
    )

    coord_map = coord_map.to(image_device)
    output_linear = torch.zeros(3, height, width, device=image_device, dtype=torch.float32)
    microlens_weights = microlens_weights.to(image_device)
    for i in range(num_lens):
        y_coords = coord_map[i, :, :, 0]
        x_coords = coord_map[i, :, :, 1]
        valid_mask = y_coords >= 0
        y_indices, x_indices = torch.where(valid_mask)
        y_src = y_coords[valid_mask].long()
        x_src = x_coords[valid_mask].long()
        weights = microlens_weights[i, y_indices, x_indices]
        output_linear[:, y_indices, x_indices] += resized_linear[
            i, :, y_src, x_src
        ] * weights.unsqueeze(0)

    output_linear = output_linear.clamp(0.0, 1.0)
    output_srgb = torch.where(
        output_linear <= 0.0031308,
        output_linear * 12.92,
        1.055 * torch.pow(output_linear.clamp(min=0.0031308), 1.0 / 2.4) - 0.055,
    )
    return output_srgb.clamp(0.0, 1.0)
