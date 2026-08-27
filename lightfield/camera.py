from __future__ import annotations

import numpy as np

from utils.camera import CameraView


def mean_group_extent_scale(
    groups: list[CameraView],
    *,
    object_center: tuple[float, float, float],
) -> float:
    scale_values: list[float] = []
    object_center_array = np.asarray(object_center, dtype=np.float32)
    for group in groups:
        camtoworlds = group.camtoworld.detach().cpu().numpy()
        centers = camtoworlds[:, :3, 3]
        x_axis = camtoworlds[0, :3, 0]
        offsets = (centers - centers[0]) @ (x_axis / np.linalg.norm(x_axis))
        left_index = int(np.argmin(offsets))
        right_index = int(np.argmax(offsets))

        left = centers[left_index] - object_center_array
        right = centers[right_index] - object_center_array
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
        angle_degrees = float(np.degrees(np.arccos(cosine)))
        scale_values.append(float(np.clip(angle_degrees / 10.0, 0.7, 1.3)))
    return float(np.mean(scale_values))
