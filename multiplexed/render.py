"""Render lens views, the predicted sensor image and an orbit video of a multiplexed run."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import Tensor

from multiplexed.sensor import (
    Cameras,
    Params,
    Sensor,
    lens_batches,
    load_sensor,
    orbit_cameras,
    predict_sensor,
    render,
)
from utils.color import linear_to_srgb
from utils.gaussians import save_ply
from utils.io import read_yaml, write_image

FRAMES = 240
FPS = 60
# Frames also saved as PNGs: one at each quarter turn and the last.
STILLS = {*range(0, FRAMES, FRAMES // 4), FRAMES - 1}
# Orbit half-width as a multiple of the standard deviation of lens positions.
ORBIT_AMPLITUDE = 0.5
# Zoom so that the central FRAMING_QUANTILES of the scene fill at most FRAME_FILL of every
# frame, checked from FRAMING_VIEWS points along the orbit.
FRAMING_QUANTILES = (0.02, 0.98)
FRAME_FILL = 0.70
FRAMING_VIEWS = 8
# gsplat's default near plane: closer points are not rendered, so framing ignores them.
NEAR_PLANE = 0.01


@dataclass(frozen=True)
class Config:
    # Run directory written by `train.py multiplexed`.
    result_dir: Path
    # Directory for the renders.
    output_dir: Path


def to_uint8(image: Tensor) -> np.ndarray:
    return (image.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def orbit_zoom(cameras: Cameras, points: Tensor) -> float:
    """Largest zoom that keeps the central FRAMING_QUANTILES of `points` within FRAME_FILL."""
    quantiles = points.new_tensor(FRAMING_QUANTILES)
    size = points.new_tensor([cameras.width, cameras.height])
    zooms = []
    for pose, K in zip(cameras.camtoworld, cameras.K, strict=True):
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        local = local[local[:, 2] > NEAR_PLANE]
        pixels = local[:, :2] / local[:, 2:] * K.diagonal()[:2]
        radius = torch.quantile(pixels, quantiles, dim=0).abs().max(dim=0).values.clamp_min(1)
        zooms.append(float((FRAME_FILL * size / (2 * radius)).min()))
    return min(zooms)


@torch.no_grad()
def export(params: Params, gain: float, sensor: Sensor, output: Path) -> dict[str, Any]:
    """Write the renders of trained surfels to `output` and return the sensor fit metrics.

    Lens views are shown at the brightness they would add to a sensor pixel fully covered by
    that lens alone; footprint views are the same views masked by each lens's footprint, and
    they sum to the sensor prediction.
    """
    for folder in ("views", "footprint_views"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    save_ply(params, output / "point_cloud.ply")
    exposure = gain / sensor.normalizer
    names = iter(sensor.names)
    for cameras, footprints in lens_batches(sensor):
        views = render(params, cameras, surface=True)[0] * exposure
        for view, footprint in zip(views, footprints, strict=True):
            stem = Path(next(names)).stem
            write_image(output / "views" / f"{stem}.png", to_uint8(linear_to_srgb(view)))
            weighted = linear_to_srgb(view * footprint[..., None])
            write_image(output / "footprint_views" / f"{stem}.png", to_uint8(weighted))

    prediction = linear_to_srgb(predict_sensor(params, sensor, surface=True) * gain)
    residual = prediction - sensor.measurement
    write_image(output / "sensor_prediction.png", to_uint8(prediction))
    write_image(output / "sensor_residual.png", to_uint8(residual.abs()))

    # Frame the orbit on the more opaque half of the surfels.
    opaque = params["means"][params["opacities"] >= params["opacities"].median()]
    target = opaque.median(dim=0).values
    zoom = orbit_zoom(orbit_cameras(sensor.lenses, target, FRAMING_VIEWS, ORBIT_AMPLITUDE), opaque)
    orbit = orbit_cameras(sensor.lenses, target, FRAMES, ORBIT_AMPLITUDE, zoom)
    with cast(
        Any, imageio.get_writer(output / "orbit.mp4", fps=FPS, macro_block_size=1, quality=8)
    ) as writer:
        for index, camera in enumerate(orbit.split(1)):
            frame = to_uint8(linear_to_srgb(render(params, camera, surface=True)[0][0] * exposure))
            writer.append_data(frame)
            if index in STILLS:
                write_image(output / f"orbit_{index:04d}.png", frame)

    mse = float(residual.square().mean())
    return {"sensor_mse": mse, "sensor_psnr": -10 * math.log10(mse), "points": len(params["means"])}


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(Config, args=argv, prog="train.py multiplexed-render", description=__doc__)
    data = Path(read_yaml(cfg.result_dir / "cfg.yml")["data"])
    device = torch.device("cuda")
    checkpoint = torch.load(cfg.result_dir / "model.pt", map_location=device, weights_only=True)
    print(
        export(checkpoint["params"], checkpoint["gain"], load_sensor(data, device), cfg.output_dir)
    )
