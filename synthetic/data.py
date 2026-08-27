from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from utils.camera import CameraView
from utils.io import read_image, read_json
from synthetic.config import TrainConfig


def _read_transforms(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transforms: dict[str, Any] = read_json(path)
    return transforms, transforms["frames"]


def _camera_pose(frame: dict[str, Any]) -> np.ndarray:
    camtoworld = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    camtoworld[:3, 1:3] *= -1.0
    return camtoworld


def _load_rgb_image(img_path: Path) -> tuple[np.ndarray, int, int]:
    img = read_image(img_path)
    if img.shape[2] == 4:
        rgb = img[..., :3].astype(np.float32) / 255.0
        a = img[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * a
        img = (rgb * 255.0).astype(np.uint8)
    img = img[..., :3]
    H, W = img.shape[:2]
    return img, int(H), int(W)


def _camera_tensors_from_frame(
    frame: dict[str, Any],
    *,
    camera_angle_x: float,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor]:
    w = int(image_width)
    h = int(image_height)
    fl_x = 0.5 * w / math.tan(0.5 * camera_angle_x)
    cx = w / 2.0
    cy = h / 2.0
    K = torch.tensor(
        [[float(fl_x), 0.0, float(cx)], [0.0, float(fl_x), float(cy)], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return torch.from_numpy(_camera_pose(frame).astype(np.float32)), K


def _group_frames(
    frames: list[dict[str, Any]],
    *,
    camera_model: str,
) -> dict[int, list[dict[str, Any]]]:
    if camera_model == "monocular":
        return {group_id: [frame] for group_id, frame in enumerate(frames)}

    group_ids = sorted({int(frame["group_id"]) for frame in frames})
    return {
        group_id: sorted(
            [frame for frame in frames if int(frame["group_id"]) == group_id],
            key=lambda frame: int(frame["view_id"]),
        )
        for group_id in group_ids
    }


class TransformsDataset:
    def __init__(
        self,
        root: Path,
        *,
        camera_model: str,
    ):
        self.root = root

        data, frames = _read_transforms(root / "transforms_train.json")

        self.camera_angle_x = float(data["camera_angle_x"])

        self.group_frames = _group_frames(
            frames,
            camera_model=camera_model,
        )
        self.groups = list(self.group_frames)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> CameraView:
        gid = int(self.groups[int(idx)])
        frames = self.group_frames[gid]

        images: list[Tensor] = []
        camtoworlds: list[Tensor] = []
        Ks: list[Tensor] = []

        for fr in frames:
            img_path = (self.root / str(fr["file_path"]).removeprefix("./")).with_suffix(".png")
            img, H, W = _load_rgb_image(img_path)
            img_t = torch.from_numpy(img.astype(np.float32) / 255.0)
            images.append(img_t)

            camtoworld, K = _camera_tensors_from_frame(
                fr,
                camera_angle_x=self.camera_angle_x,
                image_width=W,
                image_height=H,
            )
            camtoworlds.append(camtoworld)
            Ks.append(K)

        camtoworld = torch.stack(camtoworlds, dim=0)
        K = torch.stack(Ks, dim=0)
        image_tensor = torch.stack(images, dim=0)
        return CameraView(
            K=K,
            camtoworld=camtoworld,
            image=image_tensor,
            image_id=gid,
            embed_id=gid,
        )


def load_single_view_dataset(
    root: Path,
) -> list[CameraView]:
    data, frames = _read_transforms(root / "transforms_test.json")

    cam_angle_x = float(data["camera_angle_x"])

    samples: list[CameraView] = []
    for fr in frames:
        file_path = str(fr["file_path"])
        img_path = (root / file_path.removeprefix("./")).with_suffix(".png")
        img, H, W = _load_rgb_image(img_path)

        camtoworld, K = _camera_tensors_from_frame(
            fr,
            camera_angle_x=cam_angle_x,
            image_width=W,
            image_height=H,
        )

        uid = int(Path(file_path).stem.split("_")[1])
        samples.append(
            CameraView(
                K=K,
                camtoworld=camtoworld,
                image=torch.from_numpy(img.astype(np.float32) / 255.0),
                image_id=uid,
                embed_id=uid,
            )
        )
    return samples


def nerfpp_norm_radius(centers: np.ndarray) -> float:
    center = centers.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(centers - center, axis=1, keepdims=False)
    return float(np.max(dist) * 1.1)


def select_adjacent_test_views(
    train_group_centers: Sequence[np.ndarray],
    test_cameras: Sequence[CameraView],
    *,
    camera_model: str,
    max_neighbors: int = 6,
) -> list[CameraView]:
    test_centers = np.stack([cam.camtoworld[:3, 3].cpu().numpy() for cam in test_cameras], axis=0)
    selected_ids: list[int] = []
    for centers in train_group_centers:
        if camera_model == "iphone":
            # The three simulated iPhone apertures are asymmetric around the
            # exposure center. These weights recover the original camera pose.
            train_center = np.average(centers, axis=0, weights=(0.25, 0.25, 0.5))
        else:
            train_center = centers.mean(axis=0)
        diffs = test_centers - train_center[None, :]
        dist_sq = np.sum(diffs * diffs, axis=1)
        order = np.argsort(dist_sq)
        neighbors = [int(test_cameras[i].image_id) for i in order[:max_neighbors]]
        selected_ids.extend(neighbors)
    selected_ids = sorted(set(selected_ids))
    id_to_cam = {int(cam.image_id): cam for cam in test_cameras}
    return [id_to_cam[i] for i in selected_ids]


def load_training_group_centers(
    cfg: TrainConfig,
) -> list[np.ndarray]:
    root = cfg.data_root.expanduser().resolve()
    capture = "lightfield" if cfg.camera_model == "multiplexed" else cfg.camera_model
    transforms_path = (
        root / capture / f"{cfg.num_exposures}views" / "train" / "transforms_train.json"
    )
    _data, frames = _read_transforms(transforms_path)
    groups = _group_frames(
        frames,
        camera_model=cfg.camera_model,
    )
    return [
        np.stack([_camera_pose(frame)[:3, 3] for frame in group]).astype(np.float32)
        for group in groups.values()
    ]


def _focus_point_from_transforms(
    frames: Sequence[dict[str, Any]],
) -> np.ndarray:
    centers: list[np.ndarray] = []
    projectors: list[np.ndarray] = []
    eye = np.eye(3, dtype=np.float64)
    for frame in frames:
        camtoworld = _camera_pose(frame)
        direction = camtoworld[:3, 2]
        direction = direction / np.linalg.norm(direction)
        centers.append(camtoworld[:3, 3])
        projectors.append(eye - np.outer(direction, direction))
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for center, projector in zip(centers, projectors):
        lhs += projector
        rhs += projector @ center
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def load_training_dataset(
    cfg: TrainConfig,
) -> tuple[TransformsDataset, np.ndarray, tuple[float, float, float]]:
    root = cfg.data_root.expanduser().resolve()
    capture = "lightfield" if cfg.camera_model == "multiplexed" else cfg.camera_model
    train_root = root / capture / f"{cfg.num_exposures}views" / "train"
    transforms_path = train_root / "transforms_train.json"
    _transforms, frames = _read_transforms(transforms_path)

    reference_path = root / "transforms_train.json"
    _reference, reference_frames = _read_transforms(reference_path)

    camera_centers = np.stack([_camera_pose(frame)[:3, 3] for frame in reference_frames]).astype(
        np.float32
    )
    object_center = _focus_point_from_transforms(frames)

    dataset = TransformsDataset(
        train_root,
        camera_model=cfg.camera_model,
    )
    return (
        dataset,
        camera_centers,
        (
            float(object_center[0]),
            float(object_center[1]),
            float(object_center[2]),
        ),
    )
