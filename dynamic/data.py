from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap  # ty: ignore[unresolved-import]

from utils.io import write_point_cloud


def write_model_data(
    dataset_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Write the files consumed by the dynamic-model loader."""
    root = dataset_dir.expanduser().resolve()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    reference_camera = min(str(record["camera_name"]) for record in records)
    frame_indices = sorted(
        int(record["frame_index"])
        for record in records
        if str(record["camera_name"]) == reference_camera
    )
    test_indices = set(frame_indices[7::8])
    train_records = [record for record in records if int(record["frame_index"]) not in test_indices]
    write_point_cloud(root / "points3D_multipleview.ply", points, colors)
    _write_colmap_reconstruction(
        train_records,
        points=points,
        colors=colors,
        output_dir=root / "sparse_train",
    )
    # The model loader derives time from list position, so render the full timeline
    # and select every eighth frame during evaluation.
    _write_colmap_reconstruction(
        records,
        points=points,
        colors=colors,
        output_dir=root / "sparse_test",
    )
    _write_poses_bounds(records, points=points, output_path=root / "poses_bounds_multipleview.npy")


def _write_colmap_reconstruction(
    records: Sequence[Mapping[str, Any]],
    *,
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
) -> None:
    reconstruction = pycolmap.Reconstruction()
    camera_ids: dict[str, int] = {}
    for record in records:
        camera_name = str(record["camera_name"])
        if camera_name in camera_ids:
            continue
        camera_id = len(camera_ids) + 1
        camera_ids[camera_name] = camera_id
        fl_x = float(record["fl_x"])
        fl_y = float(record["fl_y"])
        if abs(fl_x - fl_y) <= 1e-6:
            camera = pycolmap.Camera(
                camera_id=camera_id,
                model="SIMPLE_PINHOLE",
                width=int(record["width"]),
                height=int(record["height"]),
                params=[fl_x, float(record["cx"]), float(record["cy"])],
            )
        else:
            camera = pycolmap.Camera(
                camera_id=camera_id,
                model="PINHOLE",
                width=int(record["width"]),
                height=int(record["height"]),
                params=[fl_x, fl_y, float(record["cx"]), float(record["cy"])],
            )
        reconstruction.add_camera(camera)

    for image_id, record in enumerate(records, start=1):
        c2w = np.asarray(record["camtoworld"], dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        camera_name = str(record["camera_name"])
        image = pycolmap.Image(
            id=image_id,
            name=str(Path(camera_name) / Path(str(record["image_path"])).name),
            camera_id=camera_ids[camera_name],
            cam_from_world=pycolmap.Rigid3d(w2c[:3, :4]),
        )
        reconstruction.add_image(image)
        reconstruction.register_image(image_id)

    if points.shape[0] <= 5_000:
        point_indices = np.arange(points.shape[0])
    else:
        point_indices = np.linspace(0, points.shape[0] - 1, 5_000).astype(np.int64)
    for index in point_indices:
        reconstruction.add_point3D(
            np.asarray(points[index], dtype=np.float64),
            pycolmap.Track(),
            np.asarray(colors[index], dtype=np.uint8),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_dir))


def _write_poses_bounds(
    records: Sequence[Mapping[str, Any]],
    *,
    points: np.ndarray,
    output_path: Path,
) -> None:
    point_cloud = points.T
    rows: list[np.ndarray] = []
    for record in records:
        c2w = np.asarray(record["camtoworld"], dtype=np.float32).reshape(4, 4)
        pose = np.zeros((3, 5), dtype=np.float32)
        pose[:, :3] = c2w[:3, :3]
        pose[:, 3] = c2w[:3, 3]
        pose[:, 4] = np.asarray(
            [int(record["height"]), int(record["width"]), float(record["fl_x"])],
            dtype=np.float32,
        )
        world_to_camera = np.linalg.inv(c2w.astype(np.float64))
        depths = (world_to_camera[:3, :3] @ point_cloud + world_to_camera[:3, 3:4])[2]
        positive = depths[depths > 0]
        near = float(np.percentile(positive, 1) * 0.9)
        far = float(np.percentile(positive, 99) * 1.1)
        rows.append(np.concatenate([pose.reshape(-1), np.asarray([near, far], dtype=np.float32)]))
    np.save(output_path, np.stack(rows))
