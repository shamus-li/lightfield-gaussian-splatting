from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from static.prepare_data import (
    link_file,
    load_reconstruction,
    write_point_cloud_assets,
    write_sparse_subset,
)
from utils.io import read_lines


def selected_images(reconstruction, names: set[str]) -> list:
    images = [image for image in reconstruction.images.values() if str(image.name) in names]
    return sorted(images, key=lambda image: str(image.name))


def baseline_coordinate_scale(source_path: Path, split_path: Path) -> float:
    """Scale tiny COLMAP reconstructions above the rasterizer's 0.2 near plane."""
    reconstruction = load_reconstruction(source_path / "sparse")
    selected_names = set(read_lines(split_path))
    selected = selected_images(reconstruction, selected_names)
    xyz = np.asarray(
        [point.xyz for point in reconstruction.points3D.values()],
        dtype=np.float32,
    ).reshape(-1, 3)
    if xyz.shape[0] > 100_000:
        xyz = xyz[:: int(np.ceil(xyz.shape[0] / 100_000))]
    positive_depths: list[np.ndarray] = []
    for image in selected:
        rotation = np.asarray(image.cam_from_world.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(image.cam_from_world.translation, dtype=np.float64)
        depths = (xyz @ rotation[2, :].T) + float(translation[2])
        positive_depths.append(depths[np.isfinite(depths) & (depths > 0.0)])
    median_depth = float(np.median(np.concatenate(positive_depths)))
    return 1.0 if median_depth >= 0.5 else 1.0 / median_depth


def write_fsgs_bounds(
    camera_sparse: Path,
    point_sparse: Path,
    names: set[str],
    output: Path,
    *,
    coordinate_scale: float,
) -> None:
    cameras = load_reconstruction(camera_sparse, coordinate_scale=coordinate_scale)
    points = load_reconstruction(point_sparse, coordinate_scale=coordinate_scale)
    images = selected_images(cameras, names)
    xyz = np.asarray(
        [point.xyz for point in points.points3D.values()],
        dtype=np.float64,
    ).reshape(-1, 3)
    rows = []
    for image in images:
        camera = cameras.cameras[image.camera_id]
        rotation = np.asarray(image.cam_from_world.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(image.cam_from_world.translation, dtype=np.float64)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation
        camera_to_world = np.linalg.inv(world_to_camera)[:3, :4]
        depths = (xyz @ rotation[2, :].T) + translation[2]
        depths = depths[np.isfinite(depths) & (depths > 1e-4)]
        near = float(max(0.01, np.percentile(depths, 0.5)))
        far = float(max(near + 1e-3, np.percentile(depths, 99.5)))
        focal = float(camera.params[0])
        pose = np.concatenate(
            [
                camera_to_world,
                np.array([[float(camera.height)], [float(camera.width)], [focal]]),
            ],
            axis=1,
        )
        rows.append(np.concatenate([pose.reshape(-1), np.array([near, far])]))
    np.save(output, np.stack(rows).astype(np.float32))


def prepare_baseline_dataset(
    *,
    method: str,
    source_path: Path,
    split_path: Path,
    destination: Path,
    coordinate_scale: float = 1.0,
    training: bool,
) -> None:
    names = sorted(read_lines(split_path))
    destination.mkdir(parents=True)
    (destination / "images").mkdir(parents=True)
    for name in names:
        source = source_path / "images" / name
        destination_image = destination / "images" / name
        if training:
            link_file(source, destination_image)
        else:
            shutil.copy2(source, destination_image)
    sparse_zero = destination / "sparse" / "0"
    write_sparse_subset(
        combined_sparse=source_path / "sparse",
        keep_names=set(names),
        output_sparse=sparse_zero,
        coordinate_scale=coordinate_scale,
    )
    point_cloud_sparse = source_path / "sparse"
    write_point_cloud_assets(
        source_sparse=point_cloud_sparse,
        output_sparse=sparse_zero,
        coordinate_scale=coordinate_scale,
    )
    if method == "fsgs":
        fused = destination / "0_views" / "dense" / "fused.ply"
        link_file(sparse_zero / "points3D.ply", fused)
        write_fsgs_bounds(
            source_path / "sparse",
            point_cloud_sparse,
            set(names),
            destination / "poses_bounds.npy",
            coordinate_scale=coordinate_scale,
        )
