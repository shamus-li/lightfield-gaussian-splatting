from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pycolmap  # ty: ignore[unresolved-import]
from PIL import Image

from dynamic.data import write_model_data
from dynamic.prepare import camera_frames, vggt_camera_calibration
from utils.io import read_image
from utils.runtime import torch_env


def prepare_model(
    dataset_dir: Path,
    *,
    modality: str,
    dataset_name: str,
) -> None:
    """Estimate reference-camera motion with MegaSaM and propagate the VGGT rig."""
    root = dataset_dir.expanduser().resolve()
    cameras = camera_frames(root)
    keyword = "wide" if modality == "iphone" else "right"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){keyword}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    reference_camera = next(name for name in cameras if pattern.search(name))
    frame_dir = cameras[reference_camera][0].parent
    npz_path = _run_megasam(
        frame_dir=frame_dir,
        scene_key=dataset_name,
        dataset_dir=root,
    )
    records, points, colors = hybrid_records_from_megasam(
        root,
        npz_path=npz_path,
        reference_camera=reference_camera,
    )
    write_model_data(
        root,
        records=records,
        points=points,
        colors=colors,
    )


def hybrid_records_from_megasam(
    dataset_dir: Path,
    *,
    npz_path: Path,
    reference_camera: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Combine MegaSaM temporal poses with VGGT relative multi-camera calibration."""
    root = dataset_dir.expanduser().resolve()
    cameras = camera_frames(root)
    reference_frames = cameras[reference_camera]
    calibration = vggt_camera_calibration(
        root,
        pycolmap.Reconstruction(str(root / "tmp_vggt/sparse")),
    )
    depths, cam_c2w, intrinsic = _load_megasam_arrays(
        npz_path,
        frame_count=len(reference_frames),
    )
    records, points, colors = _reference_records_and_points(
        root,
        reference_camera=reference_camera,
        reference_frames=reference_frames,
        depths=depths,
        cam_c2w=cam_c2w,
        intrinsic=intrinsic,
    )
    records.extend(
        _propagated_rig_records(
            root,
            cameras,
            calibration=calibration,
            reference_camera=reference_camera,
            cam_c2w=cam_c2w,
        )
    )
    records.sort(key=lambda record: (int(record["frame_index"]), str(record["camera_name"])))
    return records, points, colors


def _load_megasam_arrays(
    npz_path: Path, *, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        depths = np.asarray(data["depths"], dtype=np.float32)
        cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
        intrinsic = np.repeat(
            np.asarray(data["intrinsic"], dtype=np.float64)[None],
            frame_count,
            axis=0,
        )
    return depths, cam_c2w, intrinsic


def _reference_records_and_points(
    dataset_dir: Path,
    *,
    reference_camera: str,
    reference_frames: list[Path],
    depths: np.ndarray,
    cam_c2w: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    point_limit = 300_000
    frame_count = len(reference_frames)
    samples_per_frame = max(50, int(2.0 * point_limit / frame_count))
    reference_records: list[dict[str, Any]] = []
    sampled_points: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    for index, frame in enumerate(reference_frames):
        depth = depths[index]
        record = _megasam_reference_record(
            dataset_dir,
            frame,
            camera_name=reference_camera,
            frame_index=index + 1,
            camtoworld=cam_c2w[index],
            intrinsic=intrinsic[index],
            depth_shape=depth.shape,
        )
        reference_records.append(record)
        points, colors = _sample_points_from_depth(
            dataset_dir,
            record=record,
            depth_map=depth,
            sample_quota=samples_per_frame,
            rng=rng,
        )
        if points.size:
            sampled_points.append(points)
            sampled_colors.append(colors)
    points = np.concatenate(sampled_points)
    colors = np.concatenate(sampled_colors)
    if points.shape[0] > point_limit:
        indices = rng.choice(points.shape[0], point_limit, replace=False)
        points = points[indices]
        colors = colors[indices]
    return reference_records, points, colors


def _megasam_reference_record(
    dataset_dir: Path,
    frame: Path,
    *,
    camera_name: str,
    frame_index: int,
    camtoworld: np.ndarray,
    intrinsic: np.ndarray,
    depth_shape: tuple[int, ...],
) -> dict[str, Any]:
    with Image.open(frame) as image:
        width, height = image.size
    scale_x = width / depth_shape[1]
    scale_y = height / depth_shape[0]
    return {
        "camera_name": camera_name,
        "camtoworld": camtoworld.tolist(),
        "cx": float(intrinsic[0, 2] * scale_x),
        "cy": float(intrinsic[1, 2] * scale_y),
        "fl_x": float(intrinsic[0, 0] * scale_x),
        "fl_y": float(intrinsic[1, 1] * scale_y),
        "frame_index": frame_index,
        "height": height,
        "image_path": str(frame.relative_to(dataset_dir)),
        "width": width,
    }


def _propagated_rig_records(
    dataset_dir: Path,
    cameras: Mapping[str, list[Path]],
    *,
    calibration: Mapping[str, Mapping[str, Any]],
    reference_camera: str,
    cam_c2w: np.ndarray,
) -> list[dict[str, Any]]:
    reference_vggt = np.asarray(calibration[reference_camera]["camtoworld"], dtype=np.float64)
    reference_inverse = np.linalg.inv(reference_vggt)
    records: list[dict[str, Any]] = []
    for camera_name, frames in cameras.items():
        if camera_name == reference_camera:
            continue
        camera_calibration = calibration[camera_name]
        camera_to_reference = reference_inverse @ np.asarray(
            camera_calibration["camtoworld"], dtype=np.float64
        )
        with Image.open(frames[0]) as image:
            width, height = image.size
        for index, frame in enumerate(frames):
            records.append(
                {
                    "camera_name": camera_name,
                    "camtoworld": (cam_c2w[index] @ camera_to_reference).tolist(),
                    "cx": float(camera_calibration["cx"]),
                    "cy": float(camera_calibration["cy"]),
                    "fl_x": float(camera_calibration["fl_x"]),
                    "fl_y": float(camera_calibration["fl_y"]),
                    "frame_index": index + 1,
                    "height": height,
                    "image_path": str(frame.relative_to(dataset_dir)),
                    "width": width,
                }
            )
    return records


def _run_megasam(
    *,
    frame_dir: Path,
    scene_key: str,
    dataset_dir: Path,
) -> Path:
    repo = Path(__file__).resolve().parents[1]
    megasam_root = repo / "submodules/mega-sam"
    megasam_python = repo / ".envs/dynamic/bin/python"
    python_paths = [
        megasam_root / "UniDepth",
        megasam_root / "base",
        megasam_root / "base/droid_slam",
        megasam_root / "base/thirdparty/lietorch",
        megasam_root / "cvd_opt/core",
    ]
    storage = dataset_dir / "megasam"
    output_path = storage / "outputs_cvd" / f"{scene_key}_sgd_cvd_hr.npz"
    depth_root = storage / "Depth-Anything"
    unidepth_root = storage / "UniDepth"
    outputs_root = storage / "outputs_cvd"
    depth_output = depth_root / scene_key
    commands = [
        [
            str(megasam_python),
            str(megasam_root / "Depth-Anything/run_videos.py"),
            "--encoder",
            "vitl",
            "--load-from",
            str(repo / "models/mega-sam/depth-anything/checkpoints/depth_anything_vitl14.pth"),
            "--localhub",
            "--img-path",
            str(frame_dir),
            "--outdir",
            str(depth_output),
        ],
        [
            str(megasam_python),
            str(megasam_root / "UniDepth/scripts/demo_mega-sam.py"),
            "--img-path",
            str(frame_dir),
            "--scene-name",
            scene_key,
            "--outdir",
            str(unidepth_root),
        ],
        [
            str(megasam_python),
            str(megasam_root / "camera_tracking_scripts/test_demo.py"),
            "--datapath",
            str(frame_dir),
            "--weights",
            str(repo / "models/mega-sam/megasam_final.pth"),
            "--scene_name",
            scene_key,
            "--mono_depth_path",
            str(depth_root),
            "--metric_depth_path",
            str(unidepth_root),
            "--disable_vis",
        ],
        [
            str(megasam_python),
            str(megasam_root / "cvd_opt/preprocess_flow.py"),
            "--datapath",
            str(frame_dir),
            "--model",
            str(repo / "models/dycheck/raft-things.pth"),
            "--scene_name",
            scene_key,
            "--mixed_precision",
        ],
        [
            str(megasam_python),
            str(megasam_root / "cvd_opt/cvd_opt.py"),
            "--scene_name",
            scene_key,
            "--output_dir",
            str(outputs_root),
        ],
    ]
    work_root = storage / "work"
    storage.mkdir()
    for path in (depth_root, unidepth_root, outputs_root, work_root):
        path.mkdir()
    (work_root / "torchhub").mkdir()
    (work_root / "torchhub/facebookresearch_dinov2_main").symlink_to(
        repo / "submodules/dinov2",
        target_is_directory=True,
    )
    env = torch_env(*python_paths)
    env["HF_HOME"] = str(repo / "models/huggingface")
    for command in commands:
        print("[dynamic:megasam]", " ".join(command), flush=True)
        subprocess.run(command, cwd=work_root, env=env, check=True)
    for path in (depth_root, unidepth_root, work_root):
        shutil.rmtree(path)
    return output_path


def _sample_points_from_depth(
    dataset_dir: Path,
    *,
    record: Mapping[str, Any],
    depth_map: np.ndarray,
    sample_quota: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    size = (int(record["width"]), int(record["height"]))
    depth_image = Image.fromarray(np.asarray(depth_map, dtype=np.float32), mode="F")
    depth = np.asarray(depth_image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    valid_indices = np.flatnonzero(np.isfinite(depth) & (depth > 0.001))
    if valid_indices.size > sample_quota:
        valid_indices = rng.choice(valid_indices, sample_quota, replace=False)
    ys, xs = np.unravel_index(valid_indices, depth.shape)
    zs = depth[ys, xs]
    xs_camera = (xs - float(record["cx"])) / float(record["fl_x"]) * zs
    ys_camera = (ys - float(record["cy"])) / float(record["fl_y"]) * zs
    camera_points = np.stack([xs_camera, ys_camera, zs], axis=1)
    camtoworld = np.asarray(record["camtoworld"], dtype=np.float64).reshape(4, 4)
    points = (camtoworld[:3, :3] @ camera_points.T + camtoworld[:3, 3:4]).T.astype(np.float32)
    image_path = dataset_dir / str(record["image_path"])
    rgb = read_image(image_path, "RGB")
    return points, rgb[ys, xs]
