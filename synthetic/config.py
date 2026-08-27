from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

CameraModel = Literal["monocular", "stereo", "iphone", "lightfield", "multiplexed"]
SH_DEGREE = 3
# Camera models that multiplex the rendered light-field views onto one sensor, with the
# microlens footprint in twelfths of a grid cell: 12 tiles the sensor, 20 overlaps neighbors.
LENS_SENSOR_DISTANCE = {"lightfield": 12, "multiplexed": 20}


def capture(camera_model: str) -> str:
    """Name of the rendered capture a camera model trains on."""
    return "lightfield" if camera_model in LENS_SENSOR_DISTANCE else camera_model


@dataclass(frozen=True)
class Config:
    # Scene directory, for example data/synthetic/drums.
    data: Path
    # Run directory for cfg.yml, model.pt, renders/ and metrics.json.
    result_dir: Path
    # Simulated camera design; --eval reads it from the run's cfg.yml.
    camera_model: CameraModel = "monocular"
    # Number of captured training exposures; --eval reads it from the run's cfg.yml.
    num_exposures: Literal[1, 3] = 1
    # Training only: read-noise standard deviation added to rendered measurements.
    read_noise: float = 0.01
    # Training only: shot-noise standard deviation per unit intensity.
    shot_noise: float = 0.01
    # Evaluate the existing run in result-dir instead of training.
    eval: bool = False


@dataclass(frozen=True)
class Hyperparameters:
    """Training settings tuned per capture; the defaults are the monocular and iPhone ones."""

    # Means learning rates before scaling by the scene radius.
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    opacities_lr: float = 0.05
    lambda_dssim: float = 0.1
    # Total-variation weights on the training renders and on unseen test poses.
    tv_weight: float = 0.0
    tv_unseen_weight: float = 0.0
    # gsplat DefaultStrategy settings.
    grow_grad2d: float = 1.5e-5
    refine_start_iter: int = 300
    refine_stop_iter: int = 3_000
    refine_every: int = 50
    reset_every: int = 1_000


CAPTURE_OVERRIDES: dict[str, dict[str, float]] = {
    "stereo": {
        "means_lr": 2.4e-4,
        "means_lr_final": 8.0e-6,
        "opacities_lr": 0.07,
        "lambda_dssim": 0.16,
        "tv_weight": 2.5e-3,
        "tv_unseen_weight": 3.0e-4,
        "grow_grad2d": 1.3e-5,
        "refine_start_iter": 220,
        "refine_stop_iter": 2_600,
        "refine_every": 35,
        "reset_every": 600,
    },
    "lightfield": {
        "means_lr": 2.7e-4,
        "means_lr_final": 8.0e-6,
        "lambda_dssim": 0.17,
        "tv_weight": 3.5e-3,
        "tv_unseen_weight": 3.5e-4,
        "grow_grad2d": 1.1e-5,
        "refine_start_iter": 190,
        "refine_stop_iter": 2_550,
        "refine_every": 30,
        "reset_every": 540,
    },
}


def hyperparameters(camera_model: str) -> Hyperparameters:
    """Training settings for a camera model."""
    return replace(Hyperparameters(), **CAPTURE_OVERRIDES.get(capture(camera_model), {}))
