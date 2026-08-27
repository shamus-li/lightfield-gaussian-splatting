from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any, Iterable

import cv2
import numpy as np
import pycolmap  # ty: ignore[unresolved-import]
import torch

from utils.camera import CameraView, compute_scene_scale, image_camtoworld
from utils.io import read_image
from static.alignment import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def select_dataset_indices(
    image_names: list[str],
    *,
    split: str,
    test_every: int,
    match_string: str | None = None,
    selected_images: Iterable[str] | None = None,
) -> list[int]:
    if selected_images is not None:
        names = set(selected_images)
        indices = [index for index, name in enumerate(image_names) if name in names]
    elif test_every <= 1:
        indices = list(range(len(image_names)))
    else:
        indices = [
            index
            for index in range(len(image_names))
            if (index % test_every != 0) == (split == "train")
        ]

    if match_string:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(match_string)}(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
        indices = [index for index in indices if pattern.search(image_names[index])]
    return indices


class ColmapDataset:
    """Own a COLMAP reconstruction and load a selected split as ``CameraView``."""

    def __init__(
        self,
        data_dir: Path,
        test_every: int,
    ) -> None:
        root = data_dir.expanduser().resolve()
        reconstruction = pycolmap.Reconstruction(str(root / "sparse"))
        cameras = {int(camera_id): camera for camera_id, camera in reconstruction.cameras.items()}
        images = sorted(reconstruction.images.items(), key=lambda item: str(item[1].name))
        self._image_names = [str(image.name) for _image_id, image in images]
        self._image_paths = [root / "images" / name for name in self._image_names]
        self._camtoworlds = np.stack(
            [image_camtoworld(image) for _image_id, image in images]
        ).astype(np.float64)
        self._camera_ids = [int(image.camera_id) for _image_id, image in images]
        self._intrinsics: dict[int, np.ndarray] = {}
        distortion: dict[int, np.ndarray] = {}
        camera_types: dict[int, str] = {}
        image_sizes: dict[int, tuple[int, int]] = {}
        for camera_id in dict.fromkeys(self._camera_ids):
            K, params, camera_type, size = _camera_parameters(cameras[camera_id])
            self._intrinsics[camera_id] = K
            distortion[camera_id] = params
            camera_types[camera_id] = camera_type
            image_sizes[camera_id] = size
        self._points, self._point_colors = _read_points(reconstruction)
        _scale_intrinsics(
            self._intrinsics,
            image_sizes,
            self._camera_ids,
            self._image_paths,
        )
        (
            self._undistort_maps_x,
            self._undistort_maps_y,
            self._undistort_rois,
        ) = _undistortion_maps(
            self._intrinsics,
            distortion,
            camera_types,
            image_sizes,
        )
        self._test_every = test_every
        self._transform = np.eye(4, dtype=np.float64)
        self._scene_scale = compute_scene_scale(self._camtoworlds)
        self._image_cache: dict[int, np.ndarray] = {}
        self.indices = list(range(len(images)))

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def point_colors(self) -> np.ndarray:
        return self._point_colors

    @property
    def transform(self) -> np.ndarray:
        return self._transform

    @property
    def scene_scale(self) -> float:
        return self._scene_scale

    def image_name(self, image_id: int) -> str:
        return self._image_names[int(image_id)]

    def apply_transform(self, transform: np.ndarray) -> None:
        matrix = np.asarray(transform, dtype=self._camtoworlds.dtype)
        self._camtoworlds = transform_cameras(matrix, self._camtoworlds)
        self._points = transform_points(matrix, self._points)
        self._transform = matrix @ self._transform
        self._scene_scale = compute_scene_scale(self._camtoworlds)

    def select(
        self,
        split: str,
        *,
        match_string: str | None = None,
        selected_images: Iterable[str] | None = None,
    ) -> ColmapDataset:
        selected = copy.copy(self)
        selected.indices = select_dataset_indices(
            self._image_names,
            split=split,
            test_every=self._test_every,
            match_string=match_string,
            selected_images=selected_images,
        )
        # Selections address the same immutable source images, so one cache is sufficient.
        selected._image_cache = self._image_cache
        return selected

    def normalize(self, indices: Iterable[int] | None = None) -> None:
        subset = np.asarray(
            list(indices) if indices is not None else list(range(len(self._camtoworlds))),
            dtype=np.int64,
        )
        self.apply_transform(similarity_from_cameras(self._camtoworlds[subset]))
        self.apply_transform(align_principal_axes(self._points))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> CameraView:
        index = int(self.indices[item])
        camera_id = int(self._camera_ids[index])
        image = self.load_image(index)
        K = self._intrinsics[camera_id].copy()
        camtoworld = self._camtoworlds[index].copy()

        if camera_id in self._undistort_maps_x:
            image = cv2.remap(
                image,
                self._undistort_maps_x[camera_id],
                self._undistort_maps_y[camera_id],
                cv2.INTER_LINEAR,
            )
            x, y, width, height = self._undistort_rois[camera_id]
            image = image[y : y + height, x : x + width]

        return CameraView(
            K=torch.from_numpy(K).float(),
            camtoworld=torch.from_numpy(camtoworld).float(),
            image=torch.from_numpy(image.astype(np.float32) / 255.0).float(),
            image_id=index,
            embed_id=int(item),
        )

    def load_image(self, index: int) -> np.ndarray:
        cached = self._image_cache.get(index)
        if cached is not None:
            return cached
        path = self._image_paths[index]
        image = read_image(path, "RGB")
        image.setflags(write=False)
        self._image_cache[index] = image
        return image


def _camera_parameters(camera: Any) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    model_name = str(camera.model.name)
    params = np.asarray(camera.params, dtype=np.float32)
    K = np.asarray(
        [
            [float(camera.focal_length_x), 0.0, float(camera.principal_point_x)],
            [0.0, float(camera.focal_length_y), float(camera.principal_point_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = {
        "SIMPLE_PINHOLE": (),
        "PINHOLE": (),
        "SIMPLE_RADIAL": (3,),
        "RADIAL": (3, 4),
        "OPENCV": (4, 5, 6, 7),
        "OPENCV_FISHEYE": (4, 5, 6, 7),
    }[model_name]
    values = [params[index] for index in indices]
    distortion = np.empty(0, dtype=np.float32)
    if values:
        distortion = np.zeros(4, dtype=np.float32)
        distortion[: len(values)] = values
    return (
        K,
        distortion,
        "fisheye" if model_name == "OPENCV_FISHEYE" else "perspective",
        (int(camera.width), int(camera.height)),
    )


def _scale_intrinsics(
    intrinsics: dict[int, np.ndarray],
    image_sizes: dict[int, tuple[int, int]],
    camera_ids: list[int],
    image_paths: list[Path],
) -> None:
    actual_sizes: dict[int, tuple[int, int]] = {}
    for camera_id, image_path in zip(camera_ids, image_paths):
        if camera_id not in actual_sizes:
            height, width = read_image(image_path, "RGB").shape[:2]
            actual_sizes[camera_id] = (int(width), int(height))

    for camera_id, K in intrinsics.items():
        width, height = actual_sizes[camera_id]
        colmap_width, colmap_height = image_sizes[camera_id]
        K[0, :] *= width / float(colmap_width)
        K[1, :] *= height / float(colmap_height)
        image_sizes[camera_id] = (width, height)


def _undistortion_maps(
    intrinsics: dict[int, np.ndarray],
    distortion: dict[int, np.ndarray],
    camera_types: dict[int, str],
    image_sizes: dict[int, tuple[int, int]],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, tuple[int, int, int, int]],
]:
    map_x: dict[int, np.ndarray] = {}
    map_y: dict[int, np.ndarray] = {}
    rois: dict[int, tuple[int, int, int, int]] = {}
    for camera_id, params in distortion.items():
        if params.size == 0:
            continue
        K = intrinsics[camera_id]
        width, height = image_sizes[camera_id]
        if camera_types[camera_id] == "perspective":
            undistorted_K, roi = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
            x_map, y_map = cv2.initUndistortRectifyMap(
                K,
                params,
                np.eye(3, dtype=np.float32),
                undistorted_K,
                (width, height),
                cv2.CV_32FC1,
            )
            roi = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        else:
            x_map, y_map, undistorted_K, roi = _fisheye_maps(K, params, width=width, height=height)
        map_x[camera_id] = x_map
        map_y[camera_id] = y_map
        rois[camera_id] = roi
        intrinsics[camera_id] = undistorted_K
        image_sizes[camera_id] = (roi[2], roi[3])
    return map_x, map_y, rois


def _fisheye_maps(
    K: np.ndarray,
    params: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image_size = (width, height)
    rotation = np.eye(3, dtype=np.float64)
    distortion = np.asarray(params, dtype=np.float64).reshape(4, 1)
    undistorted_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        np.asarray(K, dtype=np.float64),
        distortion,
        image_size,
        rotation,
        balance=0.0,
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        np.asarray(K, dtype=np.float64),
        distortion,
        rotation,
        undistorted_K,
        image_size,
        cv2.CV_32FC1,
    )
    valid = (map_x >= 0) & (map_y >= 0) & (map_x < width - 1) & (map_y < height - 1)
    y_indices, x_indices = np.nonzero(valid)
    y_min, y_max = int(y_indices.min()), int(y_indices.max() + 1)
    x_min, x_max = int(x_indices.min()), int(x_indices.max() + 1)
    roi = (x_min, y_min, x_max - x_min, y_max - y_min)
    cropped_K = undistorted_K.astype(np.float32)
    cropped_K[0, 2] -= x_min
    cropped_K[1, 2] -= y_min
    return map_x, map_y, cropped_K, roi


def _read_points(reconstruction: Any) -> tuple[np.ndarray, np.ndarray]:
    point_items = sorted(reconstruction.points3D.items())
    points = np.stack([np.asarray(point.xyz, dtype=np.float64) for _point_id, point in point_items])
    colors = np.stack([np.asarray(point.color, dtype=np.uint8) for _point_id, point in point_items])
    return points, colors
