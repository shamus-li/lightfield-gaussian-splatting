from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def image_camtoworld(image: Any) -> np.ndarray:
    transform = image.cam_from_world
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.asarray(transform.rotation.matrix(), dtype=np.float64)
    world_to_camera[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
    return np.linalg.inv(world_to_camera)


def compute_scene_scale(camtoworlds: np.ndarray) -> float:
    centers = camtoworlds[:, :3, 3]
    center = np.mean(centers, axis=0)
    return float(np.max(np.linalg.norm(centers - center, axis=1)))


@dataclass(frozen=True)
class CameraView:
    """Image and camera tensors shared by every train/evaluation path.

    Images are float tensors in ``[0, 1]``. A single view uses ``image [H,W,3]``,
    ``camtoworld [4,4]``, and ``K [3,3]``. A synchronized group uses
    ``image [V,H,W,3]``, ``camtoworld [V,4,4]``, and ``K [V,3,3]``.
    """

    K: torch.Tensor
    camtoworld: torch.Tensor
    image: torch.Tensor
    image_id: int
    embed_id: int

    @property
    def num_views(self) -> int:
        return 1 if self.image.ndim == 3 else int(self.image.shape[0])

    @property
    def height(self) -> int:
        return int(self.image.shape[-3])

    @property
    def width(self) -> int:
        return int(self.image.shape[-2])

    def __getitem__(self, index: int) -> CameraView:
        if self.num_views == 1:
            return self
        return CameraView(
            K=self.K[index],
            camtoworld=self.camtoworld[index],
            image=self.image[index],
            image_id=self.image_id,
            embed_id=self.embed_id,
        )

    def __iter__(self) -> Iterator[CameraView]:
        return (self[index] for index in range(self.num_views))
