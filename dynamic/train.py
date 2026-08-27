from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from dynamic import fourdgs
from dynamic.frames import FRAME_STEP, camera_frames, extract_frames
from dynamic.init import (
    calibrate_rig,
    is_test_frame,
    reference_camera,
    static_scene,
    tracked_scene,
    write_dataset,
)
from dynamic.megasam import run_megasam
from utils.camera import camera_match
from utils.io import read_image, read_yaml, write_config, write_json
from utils.metrics import compute_pair_metrics, summarize_metric_rows

LPIPS_NET = "vgg"


@dataclass(frozen=True)
class Config:
    # Scene directory containing iphone/ and stereo/ videos.
    data: Path
    # Run directory for cfg.yml, prepared/, model/ and metrics.json.
    result_dir: Path
    # Capture device; --eval reads it from the run's cfg.yml.
    camera_model: Literal["iphone", "stereo"] = "iphone"
    # Train on every synchronized stream or only the wide (iPhone) or right (stereo) one.
    method: Literal["multiview", "monocular"] = "multiview"
    # Camera poses: VGGT's static rig, or the rig moved along MegaSaM's reference-camera track.
    initializer: Literal["vggt", "megasam"] = "megasam"
    # Evaluate the existing run in result-dir instead of training.
    eval: bool = False


def prepare(cfg: Config, prepared: Path) -> dict[str, list[Path]]:
    """Extract frames, estimate cameras and points, and write the 4DGaussians dataset."""
    videos = sorted(
        path for path in (cfg.data / cfg.camera_model).iterdir() if path.suffix.lower() == ".mov"
    )
    frames = extract_frames(videos, prepared, step=FRAME_STEP[cfg.camera_model])
    reference = reference_camera(frames, cfg.camera_model)
    rig = calibrate_rig(prepared / "vggt", frames)
    if cfg.initializer == "megasam":
        depths, trajectory, K = run_megasam(prepared / reference, prepared / "megasam")
        scene = tracked_scene(rig, frames, reference, depths=depths, trajectory=trajectory, K=K)
    else:
        scene = static_scene(rig, frames)
    write_dataset(prepared, frames, scene)
    return frames


def evaluate(render_dir: Path, frames: Sequence[Path]) -> dict[str, float]:
    """Score the held-out frames against 4DGaussians' renders of the same camera's timeline."""
    renders = sorted(render_dir.glob("*.png"))
    if len(renders) != len(frames):
        raise RuntimeError(f"{render_dir} has {len(renders)} renders for {len(frames)} frames")
    rows = [
        compute_pair_metrics(
            read_image(renders[index], "RGB").astype(np.float32) / 255.0,
            read_image(frames[index], "RGB").astype(np.float32) / 255.0,
            lpips_net=LPIPS_NET,
        )
        for index in range(len(frames))
        if is_test_frame(index)
    ]
    return summarize_metric_rows(rows)


def train(cfg: Config, prepared: Path, model_dir: Path, frames: dict[str, list[Path]]) -> None:
    model_dir.mkdir()
    reference = reference_camera(frames, cfg.camera_model)
    monocular = cfg.method == "monocular"
    fourdgs.write_config(
        model_dir / "config.py",
        # The config template tunes for names containing "iphone" or "stereo".
        dataset_name=f"{cfg.data.name}_{cfg.camera_model}",
        camera_count=len(frames),
        frame_count=len(frames[reference]),
        monocular=monocular,
    )
    fourdgs.train(
        prepared, model_dir, match=camera_match(cfg.camera_model, "monocular"), monocular=monocular
    )


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        Config,
        args=argv,
        prog="train.py dynamic",
        description="Train and evaluate 4DGaussians on a synchronized multi-camera video capture.",
    )
    cfg = replace(cfg, data=cfg.data.resolve(), result_dir=cfg.result_dir.resolve())
    prepared = cfg.result_dir / "prepared"
    model_dir = cfg.result_dir / "model"
    if cfg.eval:
        saved = read_yaml(cfg.result_dir / "cfg.yml")
        cfg = replace(cfg, camera_model=saved["camera_model"], method=saved["method"])
        frames = camera_frames(prepared)
    else:
        cfg.result_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg.result_dir / "cfg.yml", cfg)
        frames = prepare(cfg, prepared)
        train(cfg, prepared, model_dir, frames)
    reference = reference_camera(frames, cfg.camera_model)
    render_dir = fourdgs.render_test(prepared, model_dir, fourdgs.latest_iteration(model_dir))
    metrics = evaluate(render_dir, frames[reference])
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)
