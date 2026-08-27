from __future__ import annotations

import copy
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pycolmap  # ty: ignore[unresolved-import]

from utils.io import write_point_cloud


def prepare_scene(
    *,
    scene_dir: Path,
    source_images: Path,
    source_masks: Path,
    source: pycolmap.Reconstruction,
    scene_names: Sequence[str],
    point_ids: Sequence[int],
    track_image_ids: set[int],
) -> None:
    images_dir = scene_dir / "images"
    masks_dir = scene_dir / "calibration_masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for name in scene_names:
        shutil.copy2(source_images / name, images_dir / name)
        mask = f"{Path(name).stem}.png"
        shutil.copy2(source_masks / mask, masks_dir / mask)
    sparse_out = scene_dir / "sparse"
    sparse_out.mkdir(parents=True, exist_ok=True)
    names = set(scene_names)
    images = {
        int(image_id): image
        for image_id, image in source.images.items()
        if str(image.name) in names
    }
    reconstruction = pycolmap.Reconstruction()
    for camera_id in sorted({int(image.camera_id) for image in images.values()}):
        camera = source.cameras[camera_id]
        if camera.model.name == "SIMPLE_PINHOLE":
            focal, cx, cy = camera.params
            camera = pycolmap.Camera(
                camera_id=camera_id,
                model="PINHOLE",
                width=camera.width,
                height=camera.height,
                params=[focal, focal, cx, cy],
            )
        else:
            camera = copy.deepcopy(camera)
        reconstruction.add_camera(camera)
    for image_id, source_image in sorted(images.items()):
        image = copy.deepcopy(source_image)
        reconstruction.add_image(image)
        reconstruction.register_image(image_id)
    for point_id in point_ids:
        point = copy.deepcopy(source.points3D[point_id])
        # Both scenes initialize from points supported by at least two training views.
        point.track = pycolmap.Track(
            elements=[
                copy.deepcopy(element)
                for element in point.track.elements
                if int(element.image_id) in track_image_ids
            ]
        )
        reconstruction.points3D[point_id] = point
    reconstruction.write_text(str(sparse_out))


def prepare(asset_dir: Path, output_dir: Path) -> None:
    asset_dir = asset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_images = asset_dir / "images"
    calibrated_sparse = asset_dir / "sparse"
    masks = asset_dir / "masks"

    all_names = sorted(path.name for path in source_images.glob("*.JPG"))
    training_views = [name for index, name in enumerate(all_names) if index % 7 != 3]
    reconstruction = pycolmap.Reconstruction(str(calibrated_sparse))
    training_names = set(training_views)
    training_image_ids = {
        int(image_id)
        for image_id, image in reconstruction.images.items()
        if str(image.name) in training_names
    }
    point_ids = [
        int(point_id)
        for point_id, point in reconstruction.points3D.items()
        if sum(int(element.image_id) in training_image_ids for element in point.track.elements) >= 2
    ]
    output_dir.mkdir(parents=True)
    train_scene = output_dir / "dataset" / "train_scene"
    eval_scene = output_dir / "dataset" / "eval_scene"
    initial_point_cloud = output_dir / "dataset" / "initial_points.ply"
    points = [reconstruction.points3D[point_id] for point_id in point_ids]
    initial_point_cloud.parent.mkdir(parents=True, exist_ok=True)
    write_point_cloud(
        initial_point_cloud,
        np.asarray([point.xyz for point in points]),
        np.asarray([point.color for point in points], dtype=np.uint8),
    )
    prepare_scene(
        scene_dir=train_scene,
        source_images=source_images,
        source_masks=masks,
        source=reconstruction,
        scene_names=training_views,
        point_ids=point_ids,
        track_image_ids=training_image_ids,
    )
    prepare_scene(
        scene_dir=eval_scene,
        source_images=source_images,
        source_masks=masks,
        source=reconstruction,
        scene_names=all_names,
        point_ids=point_ids,
        track_image_ids=training_image_ids,
    )
    print(f"Prepared multiplexed data at {output_dir / 'dataset'}")
