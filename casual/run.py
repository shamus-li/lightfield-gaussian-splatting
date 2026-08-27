from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import tyro

from casual.prepare import prepare_scene
from static.covisible import write_covisible_masks
from static.eval import evaluate, save_renders
from static.train import train_and_save
from utils.camera import camera_match
from utils.io import read_yaml, write_config, write_json

MAX_STEPS = 30_000
REFINE_STOP_ITER = 26_000
# Every eighth training frame is held out of training, as in Difix3D's trainer.
HOLDOUT_EVERY = 8
LPIPS_NET = "alex"


@dataclass(frozen=True)
class Config:
    # Scene directory containing <camera-model>-train/ and <camera-model>-eval/ videos.
    data: Path
    # Run directory for cfg.yml, prepared/, model.pt, renders/ and metrics.json.
    result_dir: Path
    # Capture device; --eval reads it from the run's cfg.yml.
    camera_model: Literal["iphone", "stereo"] = "iphone"
    # Train on every synchronized stream or only the wide (iPhone) or right (stereo) one.
    method: Literal["monocular", "multiview"] = "multiview"
    # Camera pose estimator for the extracted frames.
    initializer: Literal["colmap", "vggt"] = "colmap"
    # Evaluate the existing run in result-dir instead of training.
    eval: bool = False


# train(result_dir, train_dir, match) writes result_dir/model.pt.
Train = Callable[[Path, Path, str], None]
# evaluate(result_dir, train_dir, test_dir, mask_dir) writes renders/ and returns metrics.
Evaluate = Callable[[Path, Path, Path, Path], dict[str, float]]


def run(
    argv: Sequence[str], *, prog: str, description: str, train: Train, evaluate: Evaluate
) -> None:
    """Prepare a casual capture, train a model on it and score the model on the eval video."""
    cfg = tyro.cli(Config, args=argv, prog=prog, description=description)
    cfg = replace(cfg, data=cfg.data.resolve(), result_dir=cfg.result_dir.resolve())
    if cfg.eval:
        saved = read_yaml(cfg.result_dir / "cfg.yml")
        cfg = replace(cfg, camera_model=saved["camera_model"], method=saved["method"])
    prepared = cfg.result_dir / "prepared"
    train_dir = prepared / "subsets/train"
    test_dir = prepared / "subsets/test"
    match = camera_match(cfg.camera_model, cfg.method)
    if not cfg.eval:
        cfg.result_dir.mkdir(parents=True, exist_ok=True)
        write_config(cfg.result_dir / "cfg.yml", cfg)
        prepare_scene(
            cfg.data, prepared, camera_model=cfg.camera_model, initializer=cfg.initializer
        )
        train(cfg.result_dir, train_dir, match)
    mask_dir = cfg.result_dir / "covisible"
    write_covisible_masks(test_dir, train_dir, mask_dir, holdout_every=HOLDOUT_EVERY, match=match)
    metrics = evaluate(cfg.result_dir, train_dir, test_dir, mask_dir)
    shutil.rmtree(mask_dir)
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)


def train_gsplat(result_dir: Path, train_dir: Path, match: str) -> None:
    train_and_save(
        train_dir,
        result_dir / "model.pt",
        max_steps=MAX_STEPS,
        refine_stop_iter=REFINE_STOP_ITER,
        holdout_every=HOLDOUT_EVERY,
        match=match,
    )


def evaluate_gsplat(
    result_dir: Path, train_dir: Path, test_dir: Path, mask_dir: Path
) -> dict[str, float]:
    metrics, renders = evaluate(
        result_dir / "model.pt", train_dir, test_dir, mask_dir, lpips_net=LPIPS_NET
    )
    save_renders(result_dir / "renders", renders)
    return metrics


def main(argv: Sequence[str]) -> None:
    run(
        argv,
        prog="train.py casual",
        description="Prepare, train and evaluate Gaussian splatting on a casual video capture.",
        train=train_gsplat,
        evaluate=evaluate_gsplat,
    )
