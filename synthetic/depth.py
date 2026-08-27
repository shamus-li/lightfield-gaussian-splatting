"""Render and evaluate expected depth from a trained synthetic model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import OpenEXR  # ty: ignore[unresolved-import]
import torch
from gsplat import rasterization

from utils.camera import CameraView
from utils.io import read_image, read_json, read_yaml, write_json
from synthetic.config import TrainConfig
from synthetic.data import (
    load_single_view_dataset,
    load_training_group_centers,
    select_adjacent_test_views,
)
from synthetic.train import load_synthetic_checkpoint
from utils.cli import parse_args


def render_depth(
    gaussians: torch.nn.ParameterDict,
    camera: CameraView,
) -> tuple[np.ndarray, np.ndarray]:
    colors = torch.cat((gaussians["sh0"], gaussians["shN"]), dim=1)
    rendered, alpha, _ = rasterization(
        means=gaussians["means"],
        quats=gaussians["quats"],
        scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camera.camtoworld.cuda())[None],
        Ks=camera.K.cuda()[None],
        width=camera.width,
        height=camera.height,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        channel_chunk=16,
        tile_size=4,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        sh_degree=3,
        render_mode="ED",
    )
    return (
        rendered[0, ..., 0].cpu().numpy().astype(np.float32),
        alpha[0, ..., 0].cpu().numpy().astype(np.float32),
    )


def load_ground_truth(
    data_root: Path,
    frame: dict[str, Any],
    view_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    components = []
    for component in "xyz":
        path = data_root / "positions" / f"position_{component}_{view_id:04d}.exr"
        with OpenEXR.File(str(path)) as exr:
            components.append(exr.channels()["V"].pixels)
    position_world = np.stack(components, axis=2).astype(np.float64)
    world_to_camera = np.linalg.inv(np.asarray(frame["transform_matrix"], dtype=np.float64))
    position_camera = (
        np.einsum(
            "ij,hwj->hwi",
            world_to_camera[:3, :3],
            position_world,
            optimize=True,
        )
        + world_to_camera[:3, 3]
    )
    depth = -position_camera[..., 2].astype(np.float32)
    alpha = read_image(
        (data_root / str(frame["file_path"]).removeprefix("./")).with_suffix(".png")
    )[..., 3]
    foreground = cv2.erode(
        (alpha >= 250).astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)
    mask = (
        foreground & np.isfinite(position_world).all(axis=2) & np.isfinite(depth) & (depth > 0.01)
    )
    return depth, mask


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py depth",
        description="Render and evaluate expected depth for a completed synthetic run.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory, for example data/synthetic/drums.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Completed synthetic training directory.",
    )
    args = parse_args(parser, argv)
    data_root = args.data.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    values = read_yaml(result_dir / "cfg.yml")
    config = TrainConfig(
        data_root=data_root,
        result_dir=result_dir,
        camera_model=values["camera_model"],
        num_exposures=values["num_exposures"],
    )
    cameras = load_single_view_dataset(data_root)
    if config.num_exposures == 1:
        cameras = select_adjacent_test_views(
            load_training_group_centers(config),
            cameras,
            camera_model=config.camera_model,
        )
    checkpoint = result_dir / "ckpts" / f"ckpt_{config.steps:06d}.pt"
    gaussians = load_synthetic_checkpoint(checkpoint, device=torch.device("cuda:0"))
    transforms = read_json(data_root / "transforms_test.json")
    frames = {
        int(Path(str(frame["file_path"])).name.split("_")[-1]): frame
        for frame in transforms["frames"]
    }
    output = result_dir / "depth"
    output.mkdir(parents=True, exist_ok=True)

    abs_rel = []
    rmse = []
    with torch.no_grad():
        for camera in cameras:
            view_id = int(camera.image_id)
            predicted, opacity = render_depth(gaussians, camera)
            target, mask = load_ground_truth(data_root, frames[view_id], view_id)
            mask &= np.isfinite(predicted) & (opacity > 1e-4)
            difference = predicted[mask] - target[mask]
            np.save(output / f"depth_{view_id:04d}.npy", predicted)
            abs_rel.append(float(np.mean(np.abs(difference) / target[mask])))
            rmse.append(float(np.sqrt(np.mean(difference**2))))

    metrics = {
        "abs_rel": float(np.mean(abs_rel)),
        "rmse": float(np.mean(rmse)),
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def prepare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py prepare-depth",
        description="Render world-position ground truth for a synthetic scene.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory, for example data/synthetic/drums.",
    )
    args = parse_args(parser, argv)
    data_root = args.data.expanduser().resolve()
    blend = data_root.parent / "blender" / f"{data_root.name}.blend"
    subprocess.run(
        [
            "blender",
            str(blend),
            "--background",
            "--python",
            str(Path(__file__).resolve().parents[1] / "scripts/render_blender_position.py"),
            "--",
            "--data",
            str(data_root),
        ],
        check=True,
    )
