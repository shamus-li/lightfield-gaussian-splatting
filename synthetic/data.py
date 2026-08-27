from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from synthetic.config import capture
from utils.camera import CameraView, compute_scene_scale
from utils.io import read_image, read_json

# Test views scored around each training group when training on a single exposure.
ADJACENT_TEST_VIEWS = 6
VIEWS_PER_EXPOSURE = {"monocular": 1, "stereo": 2, "iphone": 3, "lightfield": 16}
# The three simulated iPhone apertures sit asymmetrically around the exposure center; these
# weights recover the original camera position from them.
IPHONE_CENTER_WEIGHTS = (0.25, 0.25, 0.5)


def view_id(frame: dict[str, Any]) -> int:
    """View number in a NeRF-synthetic frame's file name, e.g. 12 for ./test/r_12."""
    return int(Path(frame["file_path"]).name.split("_")[-1])


def image_path(root: Path, frame: dict[str, Any]) -> Path:
    return (root / frame["file_path"]).with_suffix(".png")


def camera_pose(frame: dict[str, Any]) -> np.ndarray:
    """OpenCV camera-to-world matrix of a NeRF-synthetic (OpenGL) frame."""
    camtoworld = np.array(frame["transform_matrix"], dtype=np.float64)
    camtoworld[:3, 1:3] *= -1.0
    return camtoworld


def intrinsics(camera_angle_x: float, width: int, height: int) -> Tensor:
    """Pinhole intrinsics with a centered principal point."""
    focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def load_image(path: Path) -> Tensor:
    """RGBA render composited on black as an 8-bit [H, W, 3] image in [0, 1]."""
    rgba = read_image(path).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:]
    return torch.from_numpy((rgb * 255.0).astype(np.uint8).astype(np.float32) / 255.0)


def load_frame(root: Path, frame: dict[str, Any], camera_angle_x: float) -> CameraView:
    """Image, pose and intrinsics of one frame under root."""
    image = load_image(image_path(root, frame))
    height, width = image.shape[:2]
    return CameraView(
        K=intrinsics(camera_angle_x, width, height),
        camtoworld=torch.from_numpy(camera_pose(frame)).float(),
        image=image,
        image_id=view_id(frame),
        embed_id=view_id(frame),
    )


def load_test_views(data: Path) -> list[CameraView]:
    transforms = read_json(data / "transforms_test.json")
    return [load_frame(data, frame, transforms["camera_angle_x"]) for frame in transforms["frames"]]


def training_dir(data: Path, camera_model: str, num_exposures: int) -> Path:
    return data / capture(camera_model) / f"{num_exposures}views" / "train"


def training_groups(train_dir: Path, camera_model: str) -> list[list[dict[str, Any]]]:
    """Training frames grouped by synchronized capture and ordered by view."""
    frames = read_json(train_dir / "transforms_train.json")["frames"]
    views = VIEWS_PER_EXPOSURE[capture(camera_model)]
    assert len(frames) % views == 0, f"{len(frames)} frames for {views}-view exposures"
    exposures = len(frames) // views
    # Light-field files list each exposure's lenses together; the others list each view
    # across all exposures together.
    if capture(camera_model) == "lightfield":
        return [frames[group * views : (group + 1) * views] for group in range(exposures)]
    return [frames[group::exposures] for group in range(exposures)]


def load_training_groups(train_dir: Path, camera_model: str) -> list[CameraView]:
    """Training captures as [V]-view CameraViews."""
    camera_angle_x = read_json(train_dir / "transforms_train.json")["camera_angle_x"]
    groups = []
    for index, frames in enumerate(training_groups(train_dir, camera_model)):
        views = [load_frame(train_dir, frame, camera_angle_x) for frame in frames]
        groups.append(
            CameraView(
                K=torch.stack([view.K for view in views]),
                camtoworld=torch.stack([view.camtoworld for view in views]),
                image=torch.stack([view.image for view in views]),
                image_id=index,
                embed_id=index,
            )
        )
    return groups


def scene_radius(data: Path) -> float:
    """NeRF++ normalization radius of the reference training cameras."""
    frames = read_json(data / "transforms_train.json")["frames"]
    return compute_scene_scale(np.stack([camera_pose(frame) for frame in frames])) * 1.1


def focus_point(camtoworlds: np.ndarray) -> np.ndarray:
    """Least-squares intersection of the optical axes of [N, 4, 4] cameras."""
    centers = camtoworlds[:, :3, 3]
    directions = camtoworlds[:, :3, 2] / np.linalg.norm(camtoworlds[:, :3, 2], axis=1)[:, None]
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    return np.linalg.solve(projectors.sum(axis=0), np.einsum("nij,nj->i", projectors, centers))


def group_extent_scale(camtoworlds: np.ndarray, focus: np.ndarray) -> float:
    """Mean over [G, V, 4, 4] groups of the angle their outermost views subtend at `focus`,
    in units of 10 degrees and clipped to [0.7, 1.3]."""
    scales = []
    for group in camtoworlds:
        centers = group[:, :3, 3]
        x_axis = group[0, :3, 0] / np.linalg.norm(group[0, :3, 0])
        offsets = (centers - centers[0]) @ x_axis
        left = centers[np.argmin(offsets)] - focus
        right = centers[np.argmax(offsets)] - focus
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        scales.append(np.clip(angle / 10.0, 0.7, 1.3))
    return float(np.mean(scales))


def evaluation_views(
    test_views: list[CameraView], train_dir: Path, camera_model: str, num_exposures: int
) -> list[CameraView]:
    """All test views, or for single-exposure runs those nearest to each training group."""
    if num_exposures != 1:
        return test_views
    test_centers = torch.stack([view.camtoworld[:3, 3] for view in test_views]).numpy()
    selected: set[int] = set()
    for frames in training_groups(train_dir, camera_model):
        centers = np.stack([camera_pose(frame)[:3, 3] for frame in frames])
        weights = IPHONE_CENTER_WEIGHTS if camera_model == "iphone" else None
        center = np.average(centers, axis=0, weights=weights)
        distances = np.sum((test_centers - center) ** 2, axis=1)
        selected.update(np.argsort(distances)[:ADJACENT_TEST_VIEWS].tolist())
    return [test_views[index] for index in sorted(selected)]
