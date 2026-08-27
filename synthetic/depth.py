"""Depth evaluation of synthetic runs against Blender world-position renders."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import OpenEXR  # ty: ignore[import-not-found]
import torch
import tyro

from synthetic.config import SH_DEGREE
from synthetic.data import evaluation_views, image_path, load_test_views, training_dir, view_id
from utils.gaussians import rasterize
from utils.io import read_image, read_json, read_yaml, write_json


@dataclass(frozen=True)
class DepthConfig:
    # Scene directory with positions/ from prepare-depth, for example data/synthetic/drums.
    data: Path
    # Completed synthetic run directory.
    result_dir: Path


@dataclass(frozen=True)
class PrepareConfig:
    # Scene directory, for example data/synthetic/drums; reads ../blender/<scene>.blend.
    data: Path


def ground_truth_depth(data: Path, frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space depth of a test frame and the mask of eroded foreground pixels to score."""
    components = []
    for component in "xyz":
        path = data / "positions" / f"position_{component}_{view_id(frame):04d}.exr"
        with OpenEXR.File(str(path)) as exr:
            components.append(exr.channels()["V"].pixels)
    position_world = np.stack(components, axis=2).astype(np.float64)
    world_to_camera = np.linalg.inv(np.array(frame["transform_matrix"], dtype=np.float64))
    position_camera = position_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    # Blender cameras look down -z.
    depth = -position_camera[..., 2].astype(np.float32)
    alpha = read_image(image_path(data, frame))[..., 3]
    foreground = cv2.erode((alpha >= 250).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask = (
        foreground & np.isfinite(position_world).all(axis=2) & np.isfinite(depth) & (depth > 0.01)
    )
    return depth, mask


@torch.no_grad()
def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        DepthConfig,
        args=argv,
        prog="train.py depth",
        description="Render and evaluate expected depth for a completed synthetic run.",
    )
    saved = read_yaml(cfg.result_dir / "cfg.yml")
    views = evaluation_views(
        load_test_views(cfg.data),
        training_dir(cfg.data, saved["camera_model"], saved["num_exposures"]),
        saved["camera_model"],
        saved["num_exposures"],
    )
    params = torch.load(cfg.result_dir / "model.pt", map_location="cuda", weights_only=True)
    frames = {
        view_id(frame): frame for frame in read_json(cfg.data / "transforms_test.json")["frames"]
    }
    abs_rel = []
    rmse = []
    for view in views:
        rendered, alpha, _ = rasterize(
            params,
            view.camtoworld[None].cuda(),
            view.K[None].cuda(),
            view.width,
            view.height,
            sh_degree=SH_DEGREE,
            render_mode="ED",
        )
        predicted = rendered[0, ..., 0].cpu().numpy()
        target, mask = ground_truth_depth(cfg.data, frames[view.image_id])
        mask &= np.isfinite(predicted) & (alpha[0, ..., 0].cpu().numpy() > 1e-4)
        difference = predicted[mask] - target[mask]
        abs_rel.append(np.mean(np.abs(difference) / target[mask]))
        rmse.append(np.sqrt(np.mean(difference**2)))
    metrics = {"abs_rel": float(np.mean(abs_rel)), "rmse": float(np.mean(rmse))}
    (cfg.result_dir / "depth").mkdir(exist_ok=True)
    write_json(cfg.result_dir / "depth/metrics.json", metrics)
    print(metrics)


def prepare_main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        PrepareConfig,
        args=argv,
        prog="train.py prepare-depth",
        description="Render world-position ground truth for a synthetic scene with Blender.",
    )
    data = cfg.data.resolve()
    blend = data.parent / "blender" / f"{data.name}.blend"
    script = Path(__file__).with_name("blender_position.py")
    subprocess.run(
        ["blender", str(blend), "--background", "--python", str(script), "--", str(data)],
        check=True,
    )
