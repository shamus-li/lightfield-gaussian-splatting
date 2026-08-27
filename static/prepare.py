from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import tyro

from utils.colmap import extract_frames, link_file, vggt_sparse, write_subsets

EVAL_FPS = 10
# Intrinsics group of each image name; images of one physical lens share one camera.
LENS_GROUPS = {
    "mono__wide.png": "iphone_wide",
    "iphone__wide.png": "iphone_wide",
    "iphone__tele.png": "iphone_tele",
    "iphone__ultrawide.png": "iphone_ultrawide",
    "stereo__stereo_left.png": "stereo_left",
    "stereo__stereo_right.png": "stereo_right",
}


@dataclass(frozen=True)
class Config:
    # Scene directory containing static/ and iphone-eval/.
    data: Path


def lens_group(image_name: str) -> str:
    if image_name.startswith("eval__"):
        return "iphone_wide"
    if image_name.startswith("lf__"):
        return "lightfield"
    return LENS_GROUPS[image_name]


def share_lens_intrinsics(sparse_dir: Path) -> None:
    """Give the cameras of all images of one lens the median of their intrinsics."""
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    groups: dict[str, set[int]] = {}
    for image in reconstruction.images.values():
        groups.setdefault(lens_group(image.name), set()).add(image.camera_id)
    for group, camera_ids in sorted(groups.items()):
        cameras = [reconstruction.cameras[camera_id] for camera_id in sorted(camera_ids)]
        params = np.median([camera.params for camera in cameras], axis=0)
        for camera in cameras:
            camera.params = params
        print(f"Shared median intrinsics across {len(cameras)} cameras of '{group}'")
    reconstruction.write(str(sparse_dir))


def prepare(scene_dir: Path) -> None:
    """Build static/shared/{images,sparse,subsets} from the captures and the iPhone eval video."""
    static_dir = scene_dir / "static"
    shared = static_dir / "shared"
    images = shared / "images"
    sources = {
        "monocular": ("mono", [static_dir / "wide.png"]),
        "iphone": (
            "iphone",
            [static_dir / name for name in ("wide.png", "tele.png", "ultrawide.png")],
        ),
        "stereo": ("stereo", [static_dir / "stereo_left.png", static_dir / "stereo_right.png"]),
        "lightfield": ("lf", sorted((static_dir / "lightfield").glob("*.png"))),
    }
    subsets: dict[str, list[str]] = {}
    for subset, (prefix, files) in sources.items():
        subsets[subset] = [f"{prefix}__{path.name}" for path in files]
        for path, name in zip(files, subsets[subset]):
            link_file(path, images / name)
    video = min(
        path for path in (scene_dir / "iphone-eval").iterdir() if path.suffix.lower() == ".mov"
    )
    frames = extract_frames(video, images, fps=EVAL_FPS, prefix="eval__")
    subsets["iphone_eval"] = [path.name for path in frames]

    vggt_sparse(shared)
    share_lens_intrinsics(shared / "sparse")
    write_subsets(shared, subsets)
    print(f"Prepared static dataset at {shared}")


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        Config,
        args=argv,
        prog="train.py prepare-static",
        description="Prepare a real static capture for training and evaluation.",
    )
    prepare(cfg.data)
