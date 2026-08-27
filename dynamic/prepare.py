from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pycolmap  # ty: ignore[unresolved-import]
from PIL import Image

from dynamic.data import write_model_data
from utils.camera import image_camtoworld
from utils.io import write_image


def prepare_dynamic_dataset(
    *,
    videos: Sequence[Path],
    output_dir: Path,
    modality: str,
) -> None:
    output_dir = output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True)
    frame_step = 3 if modality == "iphone" else 2
    for index, path in enumerate(videos, start=1):
        video = Path(path).expanduser().resolve()
        camera_name = f"{video.stem}_{index:03d}"
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True)
        frame_count = 0
        for source_index, frame_array in enumerate(iio.imiter(video)):
            if source_index % frame_step:
                continue
            frame_count += 1
            write_image(camera_dir / f"frame_{frame_count:05d}.png", np.asarray(frame_array))

    images_dir = output_dir / "tmp_vggt/images"
    images_dir.mkdir(parents=True)
    for camera_name, frames in camera_frames(output_dir).items():
        frame = frames[0]
        target = images_dir / f"{camera_name}_{frame.name}"
        shutil.copy2(frame, target)


def camera_frames(dataset_dir: Path) -> dict[str, list[Path]]:
    root = dataset_dir.expanduser().resolve()
    return {
        directory.name: sorted(directory.glob("frame_*.png"))
        for directory in sorted(root.iterdir())
        if (directory / "frame_00001.png").is_file()
    }


def vggt_camera_calibration(
    dataset_dir: Path,
    reconstruction: pycolmap.Reconstruction,
) -> dict[str, dict[str, Any]]:
    root = dataset_dir.expanduser().resolve()
    images = {Path(str(image.name)).name: image for image in reconstruction.images.values()}
    calibration: dict[str, dict[str, Any]] = {}
    for camera_name, frames in camera_frames(root).items():
        image = images[f"{camera_name}_{frames[0].name}"]
        camera = reconstruction.cameras[int(image.camera_id)]
        params = np.asarray(camera.params, dtype=np.float64)
        calibration[camera_name] = {
            "camtoworld": image_camtoworld(image),
            "fl_x": float(params[0]),
            "fl_y": float(params[1]),
            "cx": float(params[2]),
            "cy": float(params[3]),
        }
    return calibration


def prepare_vggt_model(dataset_dir: Path) -> None:
    root = dataset_dir.expanduser().resolve()
    reconstruction = pycolmap.Reconstruction(str(root / "tmp_vggt/sparse"))
    calibration = vggt_camera_calibration(root, reconstruction)
    records: list[dict[str, Any]] = []
    for camera_name, frames in camera_frames(root).items():
        camera = calibration[camera_name]
        with Image.open(frames[0]) as image:
            width, height = image.size
        for frame_index, frame in enumerate(frames, start=1):
            records.append(
                {
                    "camera_name": camera_name,
                    "camtoworld": camera["camtoworld"].tolist(),
                    "cx": camera["cx"],
                    "cy": camera["cy"],
                    "fl_x": camera["fl_x"],
                    "fl_y": camera["fl_y"],
                    "frame_index": frame_index,
                    "height": height,
                    "image_path": str(frame.relative_to(root)),
                    "width": width,
                }
            )
    records.sort(key=lambda record: (int(record["frame_index"]), str(record["camera_name"])))
    points = [point for _point_id, point in sorted(reconstruction.points3D.items())]
    write_model_data(
        root,
        records=records,
        points=np.asarray([point.xyz for point in points], dtype=np.float64).reshape(-1, 3),
        colors=np.asarray([point.color for point in points], dtype=np.uint8).reshape(-1, 3),
    )
