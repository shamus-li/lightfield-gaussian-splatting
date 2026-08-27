from __future__ import annotations

import copy
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import imageio_ffmpeg
import pycolmap

from utils.vggt import run_vggt


def link_file(source: Path, destination: Path) -> None:
    """Symlink `destination` to `source` through a relative path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def extract_frames(video: Path, output_dir: Path, *, fps: float, prefix: str) -> list[Path]:
    """Write RGB frames of `video` sampled at `fps` to output_dir/<prefix>frame_NNNNNN.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-i",
        str(video),
        "-vf",
        f"fps={fps:g}",
        "-pix_fmt",
        "rgb24",
        str(output_dir / f"{prefix}frame_%06d.png"),
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return sorted(output_dir.glob(f"{prefix}frame_*.png"))


def add_posed_image(
    reconstruction: pycolmap.Reconstruction, image: pycolmap.Image, cam_from_world: pycolmap.Rigid3d
) -> None:
    """Register `image` as the only view of a new frame on its camera's single-camera rig."""
    sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=image.camera_id)
    if not reconstruction.exists_rig(image.camera_id):
        rig = pycolmap.Rig(rig_id=image.camera_id)
        rig.add_ref_sensor(sensor)
        reconstruction.add_rig(rig)
    frame = pycolmap.Frame(frame_id=image.image_id, rig_id=image.camera_id)
    frame.add_data_id(pycolmap.data_t(sensor_id=sensor, id=image.image_id))
    frame.rig_from_world = cam_from_world
    reconstruction.add_frame(frame)
    image.frame_id = image.image_id
    reconstruction.add_image(image)
    reconstruction.register_frame(image.image_id)


def sparse_subset(
    source: pycolmap.Reconstruction,
    names: Iterable[str],
    *,
    all_points: bool = False,
) -> pycolmap.Reconstruction:
    """Images in `names` with their cameras and observed points, or all points if `all_points`."""
    keep = set(names)
    selected = {image_id: image for image_id, image in source.images.items() if image.name in keep}
    subset = pycolmap.Reconstruction()
    for camera_id in sorted({image.camera_id for image in selected.values()}):
        subset.add_camera(copy.deepcopy(source.cameras[camera_id]))
    for _image_id, source_image in sorted(selected.items()):
        image = copy.deepcopy(source_image)
        image.reset_camera_ptr()
        image.reset_frame_ptr()
        for point_index in range(image.num_points2D()):
            if image.point2D(point_index).has_point3D():
                image.reset_point3D_for_point2D(point_index)
        add_posed_image(subset, image, source_image.cam_from_world())
    for _point_id, point in sorted(source.points3D.items()):
        track = [
            copy.deepcopy(element)
            for element in point.track.elements
            if element.image_id in selected
        ]
        if track or all_points:
            subset.add_point3D(point.xyz, pycolmap.Track(elements=track), point.color)
    return subset


def write_subsets(root: Path, subsets: Mapping[str, Sequence[str]]) -> None:
    """Write root/subsets/<name>/{images,sparse} for each named list of images in root."""
    source = pycolmap.Reconstruction(str(root / "sparse"))
    for name, image_names in subsets.items():
        subset_dir = root / "subsets" / name
        for image_name in image_names:
            link_file(root / "images" / image_name, subset_dir / "images" / image_name)
        (subset_dir / "sparse").mkdir(parents=True)
        sparse_subset(source, image_names).write(str(subset_dir / "sparse"))


def vggt_sparse(root: Path) -> None:
    """Reconstruct root/images into root/sparse with VGGT and bundle adjustment."""
    run_vggt(root, "--use_ba")
    (root / "sparse/points.ply").unlink()
