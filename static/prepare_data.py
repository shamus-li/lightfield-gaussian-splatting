from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pycolmap  # ty: ignore[unresolved-import]

from utils.io import write_lines


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def load_reconstruction(sparse_dir: Path, *, coordinate_scale: float = 1.0) -> Any:
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    if coordinate_scale != 1.0:
        reconstruction.transform(pycolmap.Sim3d(scale=coordinate_scale))
    return reconstruction


def write_sparse_subset(
    *,
    combined_sparse: Path,
    keep_names: set[str],
    output_sparse: Path,
    coordinate_scale: float = 1.0,
) -> None:
    source = load_reconstruction(combined_sparse)
    selected = {
        int(image_id): image
        for image_id, image in source.images.items()
        if str(image.name) in keep_names
    }
    reconstruction = pycolmap.Reconstruction()
    camera_ids = {int(image.camera_id) for image in selected.values()}
    for camera_id in sorted(camera_ids):
        reconstruction.add_camera(copy.deepcopy(source.cameras[camera_id]))
    for image_id, source_image in sorted(selected.items()):
        image = copy.deepcopy(source_image)
        for point_index in range(image.num_points2D()):
            if image.point2D(point_index).has_point3D():
                image.reset_point3D_for_point2D(point_index)
        reconstruction.add_image(image)
        reconstruction.register_image(image_id)

    selected_ids = set(selected)
    for _point_id, point in sorted(source.points3D.items()):
        track = [
            copy.deepcopy(element)
            for element in point.track.elements
            if int(element.image_id) in selected_ids
        ]
        if track:
            reconstruction.add_point3D(
                point.xyz,
                pycolmap.Track(elements=track),
                point.color,
            )
    if coordinate_scale != 1.0:
        reconstruction.transform(pycolmap.Sim3d(scale=coordinate_scale))

    output_sparse.mkdir(parents=True)
    reconstruction.write(str(output_sparse))


def write_point_cloud_assets(
    *,
    source_sparse: Path,
    output_sparse: Path,
    coordinate_scale: float = 1.0,
) -> None:
    reconstruction = load_reconstruction(
        source_sparse,
        coordinate_scale=coordinate_scale,
    )
    output_sparse.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colmap_points_") as temp_dir:
        reconstruction.write(temp_dir)
        shutil.copy2(Path(temp_dir) / "points3D.bin", output_sparse / "points3D.bin")

    points_path = output_sparse / "points3D.ply"
    reconstruction.export_PLY(str(points_path))


def materialize_subset_dirs(
    *,
    output_dir: Path,
    subsets: Mapping[str, Sequence[str]],
    combined_sparse: Path,
) -> None:
    subset_root = output_dir / "subsets"
    subset_root.mkdir(parents=True)

    for name, image_names in subsets.items():
        subset_dir = subset_root / name
        (subset_dir / "images").mkdir(parents=True)
        for image_name in image_names:
            source = output_dir / "images" / image_name
            link_file(source, subset_dir / "images" / image_name)

        write_sparse_subset(
            combined_sparse=combined_sparse,
            keep_names=set(image_names),
            output_sparse=subset_dir / "sparse",
        )


def write_split_lists(output_dir: Path, subsets: Mapping[str, Sequence[str]]) -> None:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True)
    for name, image_names in subsets.items():
        write_lines(splits_dir / f"{name}.txt", image_names)
