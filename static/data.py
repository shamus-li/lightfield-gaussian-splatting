from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import torch

from utils.camera import (
    CameraView,
    align_principal_axes,
    image_camtoworld,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
from utils.io import read_image


def select_images(names: list[str], *, holdout_every: int = 0, match: str = "") -> list[int]:
    """Indices of `names` outside every `holdout_every`-th one that contain the token `match`."""
    indices = [
        index for index in range(len(names)) if not holdout_every or index % holdout_every != 0
    ]
    if match:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(match)}(?![A-Za-z0-9])", flags=re.IGNORECASE
        )
        indices = [index for index in indices if pattern.search(names[index])]
    return indices


def _intrinsics(camera: Any) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    K = np.array(
        [
            [camera.focal_length_x, 0.0, camera.principal_point_x],
            [0.0, camera.focal_length_y, camera.principal_point_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    model = camera.model.name
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return K, None
    if model != "SIMPLE_RADIAL":
        raise ValueError(f"Unsupported COLMAP camera model {model}.")
    distortion = np.array([camera.params[3], 0.0, 0.0, 0.0], dtype=np.float32)
    size = (camera.width, camera.height)
    # With alpha=0 the undistorted image keeps the full size and contains only valid pixels.
    undistorted_K, _roi = cv2.getOptimalNewCameraMatrix(K, distortion, size, 0)
    maps = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), undistorted_K, size, cv2.CV_32FC1)
    return undistorted_K.astype(np.float32), maps


class ColmapDataset:
    """Undistorted views of data_dir/images posed by the COLMAP model in data_dir/sparse."""

    def __init__(self, data_dir: Path, *, holdout_every: int = 0, match: str = "") -> None:
        reconstruction = pycolmap.Reconstruction(str(data_dir / "sparse"))
        images = sorted(reconstruction.images.values(), key=lambda image: image.name)
        points = sorted(reconstruction.points3D.items())
        self.image_names = [image.name for image in images]
        self.camtoworlds = np.stack([image_camtoworld(image) for image in images])
        self.points = np.stack([point.xyz for _point_id, point in points])
        self.point_colors = np.stack([point.color for _point_id, point in points])
        self.indices = select_images(self.image_names, holdout_every=holdout_every, match=match)
        self._image_paths = [data_dir / "images" / name for name in self.image_names]
        self._camera_ids = [image.camera_id for image in images]
        self._cameras = {
            camera_id: _intrinsics(reconstruction.cameras[camera_id])
            for camera_id in set(self._camera_ids)
        }
        self._image_cache: dict[int, np.ndarray] = {}

    def apply_transform(self, matrix: np.ndarray) -> None:
        """Move every camera and point by a 4x4 similarity transform."""
        self.camtoworlds = transform_cameras(matrix, self.camtoworlds)
        self.points = transform_points(matrix, self.points)

    def normalize(self) -> np.ndarray:
        """Center, orient and scale the scene from all cameras and points; return the transform."""
        transform = similarity_from_cameras(self.camtoworlds)
        transform = align_principal_axes(transform_points(transform, self.points)) @ transform
        self.apply_transform(transform)
        return transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> CameraView:
        index = self.indices[position]
        K, maps = self._cameras[self._camera_ids[index]]
        if index not in self._image_cache:
            image = read_image(self._image_paths[index], "RGB")
            if maps is not None:
                image = cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)
            self._image_cache[index] = image
        return CameraView(
            K=torch.from_numpy(K),
            camtoworld=torch.from_numpy(self.camtoworlds[index]).float(),
            image=torch.from_numpy(self._image_cache[index]).float() / 255.0,
            image_id=index,
            embed_id=position,
        )
