from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from utils.cli import parse_args as parse_cli_args
from utils.io import read_yaml

_CAMERA_DEFAULTS: dict[str, dict[str, float | int]] = {
    "monocular": {},
    "iphone": {},
    "stereo": {
        "means_lr_init": 2.4e-4,
        "means_lr_final": 8.0e-6,
        "densify_grad_threshold": 1.3e-5,
        "densification_interval": 35,
        "densify_from_iter": 220,
        "densify_until_iter": 2_600,
        "opacity_reset_interval": 600,
        "lambda_dssim": 0.16,
        "tv_weight": 2.5e-3,
        "tv_unseen_weight": 3.0e-4,
        "opacities_lr": 0.07,
    },
    "lightfield": {
        "means_lr_init": 2.7e-4,
        "means_lr_final": 8.0e-6,
        "densify_grad_threshold": 1.1e-5,
        "densification_interval": 30,
        "densify_from_iter": 190,
        "densify_until_iter": 2_550,
        "opacity_reset_interval": 540,
        "lambda_dssim": 0.17,
        "tv_weight": 3.5e-3,
        "tv_unseen_weight": 3.5e-4,
    },
}


@dataclass
class TrainConfig:
    data_root: Path
    result_dir: Path
    camera_model: str = "monocular"
    num_exposures: int = 1
    read_noise: float = 0.01
    shot_noise: float = 0.01
    evaluate: bool = False

    steps: int = 3_000
    seed: int = 0
    lambda_dssim: float = 0.1
    tv_weight: float = 0.0
    tv_unseen_weight: float = 0.0

    sh_degree: int = 3
    sh_degree_interval: int = 500
    densify_from_iter: int = 300
    densify_until_iter: int = 3_000
    densification_interval: int = 50
    densify_grad_threshold: float = 1.5e-5
    percent_dense: float = 0.06
    size_threshold: int = 150
    opacity_reset_interval: int = 1_000

    means_lr_init: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    means_lr_max_steps: int = 30_000
    scales_lr: float = 0.005
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    sh0_lr: float = 0.0025
    shN_lr: float = 0.0025 / 20.0


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        prog="train.py synthetic",
        description="Train or evaluate a synthetic light-field Gaussian model.",
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
        help="Directory for checkpoints, renders, and metrics.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("monocular", "stereo", "iphone", "lightfield", "multiplexed"),
        default="monocular",
        help=(
            "Camera design to simulate (training default: monocular; "
            "loaded from RESULT_DIR/cfg.yml during evaluation)."
        ),
    )
    parser.add_argument(
        "--num-exposures",
        type=int,
        choices=(1, 3),
        default=1,
        help=(
            "Number of captured training exposures (training default: 1; "
            "loaded from RESULT_DIR/cfg.yml during evaluation)."
        ),
    )
    parser.add_argument(
        "--read-noise",
        type=float,
        default=0.01,
        help="Read-noise standard deviation (training default: 0.01).",
    )
    parser.add_argument(
        "--shot-noise",
        type=float,
        default=0.01,
        help="Shot-noise standard deviation (training default: 0.01).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the completed run in RESULT_DIR instead of training.",
    )
    args = parse_cli_args(parser, argv)
    config = TrainConfig(
        data_root=args.data,
        result_dir=args.result_dir,
        camera_model=args.camera_model,
        num_exposures=args.num_exposures,
        read_noise=args.read_noise,
        shot_noise=args.shot_noise,
        evaluate=args.eval,
    )
    if args.eval:
        recorded = read_yaml(args.result_dir.expanduser().resolve() / "cfg.yml")
        config = replace(
            config,
            camera_model=recorded["camera_model"],
            num_exposures=recorded["num_exposures"],
        )
    profile = "lightfield" if config.camera_model == "multiplexed" else config.camera_model
    return replace(config, **_CAMERA_DEFAULTS[profile])
