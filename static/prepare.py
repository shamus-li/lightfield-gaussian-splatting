from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pycolmap  # ty: ignore[unresolved-import]

from static.prepare_data import (
    link_file,
    materialize_subset_dirs,
    write_split_lists,
)
from utils.cli import parse_args
from utils.vggt import run_vggt


def consolidate_stream_intrinsics(sparse_dir: Path) -> None:
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    camera_map = reconstruction.cameras
    groups: dict[str, list[int]] = {}
    for image in reconstruction.images.values():
        image_name = str(image.name)
        if image_name.startswith("eval__") or image_name in {
            "mono__wide.png",
            "iphone__wide.png",
        }:
            group = "iphone_wide"
        elif image_name.startswith("lf__"):
            group = "lightfield"
        else:
            group = {
                "iphone__tele.png": "iphone_tele",
                "iphone__ultrawide.png": "iphone_ultrawide",
                "stereo__stereo_left.png": "stereo_left",
                "stereo__stereo_right.png": "stereo_right",
            }[image_name]
        groups.setdefault(group, []).append(int(image.camera_id))

    remap: dict[int, int] = {}
    replacement: dict[int, np.ndarray] = {}
    for group, ids in sorted(groups.items()):
        camera_ids = sorted(set(ids))
        primary_id = camera_ids[0]
        if len(camera_ids) > 1:
            replacement[primary_id] = np.median(
                np.stack([np.asarray(camera_map[camera_id].params) for camera_id in camera_ids]),
                axis=0,
            )
            print(
                f"Consolidated intrinsic group '{group}' from {len(camera_ids)} cameras "
                f"-> camera {primary_id}"
            )
        for camera_id in camera_ids:
            remap[camera_id] = primary_id

    for camera_id, params in replacement.items():
        camera_map[camera_id].params = params
    for image in reconstruction.images.values():
        image.camera_id = remap[int(image.camera_id)]
    for camera_id, primary_id in remap.items():
        if camera_id != primary_id:
            del camera_map[camera_id]
    reconstruction.write(str(sparse_dir))


def prepare(data_root: Path) -> None:
    scene_dir = data_root.expanduser().resolve()
    output_dir = (scene_dir / "static" / "shared").resolve()
    static_dir = scene_dir / "static"
    lightfield_images_root = static_dir / "lightfield/inner_02/images"
    eval_dir = scene_dir / "iphone-eval"
    eval_images_dir = eval_dir / "images"
    video = sorted(path for path in eval_dir.iterdir() if path.suffix.lower() == ".mov")[0]
    eval_images_dir.mkdir(parents=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=10",
        str(eval_images_dir / "frame_%05d.png"),
    ]
    print("$", " ".join(command))
    subprocess.run(command, check=True)
    eval_frames = sorted(eval_images_dir.glob("*.png"))
    output_dir.mkdir(parents=True)
    sources = {
        "monocular": (static_dir, "mono", [static_dir / "wide.png"]),
        "iphone": (
            static_dir,
            "iphone",
            [static_dir / name for name in ("wide.png", "tele.png", "ultrawide.png")],
        ),
        "stereo": (
            static_dir,
            "stereo",
            [static_dir / name for name in ("stereo_left.png", "stereo_right.png")],
        ),
        "lightfield": (
            lightfield_images_root,
            "lf",
            sorted(lightfield_images_root.glob("*.png")),
        ),
        "iphone_eval": (eval_images_dir, "eval", eval_frames),
    }
    subsets: dict[str, list[str]] = {}
    (output_dir / "images").mkdir()
    for name, (source_dir, prefix, files) in sources.items():
        image_names = [
            f"{prefix}__{path.relative_to(source_dir).as_posix().replace('/', '__')}"
            for path in files
        ]
        subsets[name] = image_names
        for path, image_name in zip(files, image_names):
            link_file(path, output_dir / "images" / image_name)

    run_vggt(output_dir, "--use_ba")
    (output_dir / "sparse/points.ply").unlink()

    sparse_dir = output_dir / "sparse"
    consolidate_stream_intrinsics(sparse_dir)
    materialize_subset_dirs(
        output_dir=output_dir,
        subsets=subsets,
        combined_sparse=sparse_dir,
    )
    write_split_lists(output_dir, subsets)
    print(f"Prepared static dataset at {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py prepare-static",
        description="Prepare a real static capture for training and evaluation.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory containing static/ and iphone-eval/.",
    )
    args = parse_args(parser, argv)
    prepare(args.data)
