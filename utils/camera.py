from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def image_camtoworld(image: Any) -> np.ndarray:
    """Camera-to-world matrix of a pycolmap image."""
    transform = image.cam_from_world()
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.asarray(transform.rotation.matrix(), dtype=np.float64)
    world_to_camera[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
    return np.linalg.inv(world_to_camera)


def compute_scene_scale(camtoworlds: np.ndarray) -> float:
    """Largest distance from a camera center to the mean camera center."""
    centers = camtoworlds[:, :3, 3]
    center = np.mean(centers, axis=0)
    return float(np.max(np.linalg.norm(centers - center, axis=1)))


def camera_match(camera_model: str, method: str) -> str:
    """Image-name token of the single stream a monocular method trains on; empty for multiview."""
    if method == "multiview":
        return ""
    return "wide" if camera_model == "iphone" else "right"


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """Apply a similarity transform to [N, 4, 4] camera-to-world poses with unit-scale rotations."""
    transformed = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    return transformed


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to [N, 3] points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def similarity_from_cameras(camtoworlds: np.ndarray) -> np.ndarray:
    """Similarity that aligns the mean camera up with +z, centers and unit-scales the cameras."""
    t = camtoworlds[:, :3, 3]
    rotations = camtoworlds[:, :3, :3]
    ups = np.sum(rotations * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        rotation_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        rotation_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotations = rotation_align @ rotations
    forwards = np.sum(rotations * np.array([0.0, 0.0, 1.0]), axis=-1)
    translated = (rotation_align @ t[..., None])[..., 0]
    nearest = translated + (forwards * -translated).sum(-1)[:, None] * forwards
    translate = -np.median(nearest, axis=0)

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = rotation_align
    transform[:3, :] /= np.median(np.linalg.norm(translated + translate, axis=-1))
    return transform


def align_principal_axes(points: np.ndarray) -> np.ndarray:
    """Rigid transform that centers points at their median and aligns their principal axes."""
    centroid = np.median(points, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points - centroid, rowvar=False))
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


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
