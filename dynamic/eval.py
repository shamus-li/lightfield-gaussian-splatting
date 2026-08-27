from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dynamic.prepare import camera_frames
from utils.io import read_image
from utils.metrics import compute_pair_metrics, summarize_metric_rows


def evaluate_dynamic_renders(
    model_dir: Path,
    *,
    dataset_dir: Path,
    match_string: str,
    iteration: int,
) -> dict[str, float]:
    method_dir = model_dir / "test" / f"ours_{iteration}"
    render_dir = method_dir / "renders"
    render_paths = sorted(render_dir.glob("*.png"))

    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(match_string)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    candidates = [
        (frame_index, path)
        for camera_name, frames in camera_frames(dataset_dir).items()
        if pattern.search(camera_name)
        for frame_index, path in enumerate(frames, start=1)
    ]

    rows: list[dict[str, float]] = []
    for (frame_index, source_path), render_path in zip(candidates, render_paths):
        if frame_index % 8:
            continue
        rows.append(
            compute_pair_metrics(
                read_image(render_path, "RGB").astype(np.float32) / 255.0,
                read_image(source_path, "RGB").astype(np.float32) / 255.0,
                lpips_net="vgg",
            )
        )
    return summarize_metric_rows(rows)


def load_iteration(model_dir: Path) -> int:
    model = next((model_dir / "point_cloud").iterdir())
    return int(model.name.removeprefix("iteration_"))
