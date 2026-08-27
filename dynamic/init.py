"""Camera poses and initial points for the 4DGaussians multi-view loader.

VGGT calibrates the rig from the first frame of every camera. With the vggt initializer the rig
stays where VGGT put it; with the megasam initializer MegaSaM tracks the reference camera through
the video and the other cameras follow it at their VGGT offsets.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.camera import camera_match, image_camtoworld
from utils.colmap import add_posed_image, link_file
from utils.io import read_image
from utils.vggt import run_vggt

# Frames with index % TEST_EVERY == TEST_EVERY - 1 are held out of training.
TEST_EVERY = 8
VGGT_CONFIDENCE = 3.0
VGGT_VISIBILITY = 0.05
VGGT_MIN_INLIERS = 16
# Depth back-projection samples twice POINT_LIMIT points over the frames (at least
# MIN_FRAME_POINTS each) and keeps a random POINT_LIMIT of them.
POINT_LIMIT = 300_000
MIN_FRAME_POINTS = 50
MIN_DEPTH = 1e-3
SEED = 0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Rig:
    """VGGT calibration of every camera at its first frame, and VGGT's points."""

    intrinsics: dict[str, Intrinsics]
    camtoworld: dict[str, np.ndarray]  # [4, 4] per camera
    points: np.ndarray  # [N, 3]
    colors: np.ndarray  # [N, 3] uint8


@dataclass(frozen=True)
class Scene:
    """Intrinsics and per-frame poses of every camera, and the initial point cloud."""

    intrinsics: dict[str, Intrinsics]
    camtoworlds: dict[str, np.ndarray]  # [frames, 4, 4] per camera
    points: np.ndarray  # [N, 3]
    colors: np.ndarray  # [N, 3] uint8


def is_test_frame(index: int) -> bool:
    return index % TEST_EVERY == TEST_EVERY - 1


def reference_camera(names: Iterable[str], camera_model: str) -> str:
    """The wide (iPhone) or right (stereo) camera, matched the way 4DGaussians' --match_string is."""
    token = camera_match(camera_model, "monocular")
    pattern = re.compile(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", flags=re.IGNORECASE)
    (name,) = [name for name in names if pattern.search(name)]
    return name


def calibrate_rig(workdir: Path, frames: Mapping[str, list[Path]]) -> Rig:
    """Run VGGT with bundle adjustment on the first frame of every camera."""
    for name, paths in frames.items():
        link_file(paths[0], workdir / "images" / f"{name}.png")
    run_vggt(
        workdir,
        "--stage",
        "both",
        "--use_ba",
        "--conf_thres_value",
        str(VGGT_CONFIDENCE),
        "--vis_thresh",
        str(VGGT_VISIBILITY),
        "--min_inlier_per_frame",
        str(VGGT_MIN_INLIERS),
    )
    reconstruction = pycolmap.Reconstruction(str(workdir / "sparse"))
    shutil.rmtree(workdir)
    images = {Path(image.name).stem: image for image in reconstruction.images.values()}
    intrinsics = {}
    camtoworld = {}
    for name, paths in frames.items():
        image = images[name]
        camera = reconstruction.cameras[image.camera_id]
        with Image.open(paths[0]) as first:
            width, height = first.size
        intrinsics[name] = Intrinsics(
            camera.focal_length_x,
            camera.focal_length_y,
            camera.principal_point_x,
            camera.principal_point_y,
            width,
            height,
        )
        camtoworld[name] = image_camtoworld(image)
    points = [point for _, point in sorted(reconstruction.points3D.items())]
    return Rig(
        intrinsics,
        camtoworld,
        points=np.array([point.xyz for point in points]),
        colors=np.array([point.color for point in points], dtype=np.uint8),
    )


def static_scene(rig: Rig, frames: Mapping[str, list[Path]]) -> Scene:
    """Every camera stays at its VGGT pose; VGGT's points initialize the model."""
    camtoworlds = {
        name: np.repeat(rig.camtoworld[name][None], len(paths), axis=0)
        for name, paths in frames.items()
    }
    return Scene(rig.intrinsics, camtoworlds, rig.points, rig.colors)


def tracked_scene(
    rig: Rig,
    frames: Mapping[str, list[Path]],
    reference: str,
    *,
    depths: np.ndarray,
    trajectory: np.ndarray,
    K: np.ndarray,
) -> Scene:
    """Move the rig along the reference camera's MegaSaM trajectory and back-project its depths.

    depths [F, h, w], trajectory [F, 4, 4] and K [3, 3] are MegaSaM's outputs at depth resolution.
    """
    calibration = rig.intrinsics[reference]
    scale_x = calibration.width / depths.shape[2]
    scale_y = calibration.height / depths.shape[1]
    intrinsics = dict(rig.intrinsics)
    intrinsics[reference] = Intrinsics(
        K[0, 0] * scale_x,
        K[1, 1] * scale_y,
        K[0, 2] * scale_x,
        K[1, 2] * scale_y,
        calibration.width,
        calibration.height,
    )
    world_to_reference = np.linalg.inv(rig.camtoworld[reference])
    camtoworlds = {
        name: trajectory[: len(paths)] @ (world_to_reference @ rig.camtoworld[name])
        for name, paths in frames.items()
    }
    points, colors = backproject(frames[reference], depths, trajectory, intrinsics[reference])
    return Scene(intrinsics, camtoworlds, points, colors)


def backproject(
    frames: list[Path], depths: np.ndarray, camtoworlds: np.ndarray, intrinsics: Intrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Colored world points at random valid pixels of each frame's depth map."""
    rng = np.random.default_rng(SEED)
    quota = max(MIN_FRAME_POINTS, int(2 * POINT_LIMIT / len(frames)))
    size = (intrinsics.width, intrinsics.height)
    all_points = []
    all_colors = []
    for frame, depth_map, camtoworld in zip(frames, depths, camtoworlds, strict=True):
        depth = np.asarray(Image.fromarray(depth_map).resize(size, Image.Resampling.BILINEAR))
        valid = np.flatnonzero(np.isfinite(depth) & (depth > MIN_DEPTH))
        if valid.size > quota:
            valid = rng.choice(valid, quota, replace=False)
        ys, xs = np.unravel_index(valid, depth.shape)
        zs = depth[ys, xs]
        camera_points = np.stack(
            [
                (xs - intrinsics.cx) / intrinsics.fx * zs,
                (ys - intrinsics.cy) / intrinsics.fy * zs,
                zs,
            ],
            axis=1,
        )
        all_points.append(camera_points @ camtoworld[:3, :3].T + camtoworld[:3, 3])
        all_colors.append(read_image(frame, "RGB")[ys, xs])
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    if len(points) > POINT_LIMIT:
        keep = rng.choice(len(points), POINT_LIMIT, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors


def write_dataset(dataset_dir: Path, frames: Mapping[str, list[Path]], scene: Scene) -> None:
    """Write the point cloud and COLMAP models read by the 4DGaussians multi-view loader."""
    write_point_cloud(dataset_dir / "points3D_multipleview.ply", scene.points, scene.colors)
    write_colmap(
        dataset_dir / "sparse_train", frames, scene, lambda index: not is_test_frame(index)
    )
    # 4DGaussians derives time from frame order, so the test model holds the full timeline and
    # evaluation picks the held-out frames from its renders.
    write_colmap(dataset_dir / "sparse_test", frames, scene, lambda index: True)


def write_colmap(
    output_dir: Path,
    frames: Mapping[str, list[Path]],
    scene: Scene,
    include: Callable[[int], bool],
) -> None:
    """COLMAP cameras and images (no points) of the frames whose index passes `include`."""
    reconstruction = pycolmap.Reconstruction()
    camera_ids = {name: camera_id for camera_id, name in enumerate(frames, start=1)}
    for name, camera_id in camera_ids.items():
        intrinsics = scene.intrinsics[name]
        if abs(intrinsics.fx - intrinsics.fy) <= 1e-6:
            model = "SIMPLE_PINHOLE"
            params = [intrinsics.fx, intrinsics.cx, intrinsics.cy]
        else:
            model = "PINHOLE"
            params = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
        reconstruction.add_camera(
            pycolmap.Camera(
                camera_id=camera_id,
                model=model,
                width=intrinsics.width,
                height=intrinsics.height,
                params=params,
            )
        )
    views = sorted(
        (index, name)
        for name, paths in frames.items()
        for index in range(len(paths))
        if include(index)
    )
    for image_id, (index, name) in enumerate(views, start=1):
        world_to_camera = np.linalg.inv(scene.camtoworlds[name][index])
        image = pycolmap.Image(
            name=f"{name}/{frames[name][index].name}",
            camera_id=camera_ids[name],
            image_id=image_id,
        )
        add_posed_image(reconstruction, image, pycolmap.Rigid3d(world_to_camera[:3]))
    output_dir.mkdir()
    reconstruction.write(str(output_dir))


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """PLY with float positions, zero normals and uint8 colors."""
    vertices = np.zeros(
        len(points),
        dtype=[(axis, "f4") for axis in ("x", "y", "z", "nx", "ny", "nz")]
        + [(channel, "u1") for channel in ("red", "green", "blue")],
    )
    for i, axis in enumerate("xyz"):
        vertices[axis] = points[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        vertices[channel] = colors[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
