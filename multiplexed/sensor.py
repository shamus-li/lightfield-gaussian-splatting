"""Calibrated lenses and the forward model from Gaussians to the multiplexed sensor image."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization_2dgs
from torch import Tensor

from utils.gaussians import rasterize
from utils.io import read_image

# Lenses rendered together; all 42 at once do not fit in GPU memory with gradients.
CAMERA_BATCH = 6
# Footprints store, per sensor pixel, how many of 16 subpixel samples each lens covers.
FOOTPRINT_SAMPLES = 16

Params = Mapping[str, Tensor] | torch.nn.ParameterDict


@dataclass(frozen=True)
class Cameras:
    """Pinhole cameras that share one image size."""

    K: Tensor  # [N, 3, 3]
    camtoworld: Tensor  # [N, 4, 4]
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.K)

    def split(self, size: int) -> list[Cameras]:
        return [
            Cameras(K, camtoworld, self.width, self.height)
            for K, camtoworld in zip(self.K.split(size), self.camtoworld.split(size), strict=True)
        ]


@dataclass(frozen=True)
class Sensor:
    """The lens array, its footprints on the sensor and the measured sensor image."""

    lenses: Cameras
    names: list[str]
    # [N, H, W] fraction of each sensor pixel that each lens covers.
    footprints: Tensor
    # Largest total lens coverage of any sensor pixel.
    normalizer: float
    # [H, W, 3] sRGB values in [0, 1].
    measurement: Tensor
    # [P, 3] triangulated calibration points.
    points: Tensor


def load_sensor(data: Path, device: torch.device) -> Sensor:
    """Read calibration.npz and measurement.png, rendering at the footprint resolution."""
    with np.load(data / "calibration.npz") as calibration:
        sensor_width, sensor_height = calibration["sensor_size"].tolist()
        names = [str(name) for name in calibration["names"]]
        K = torch.tensor(calibration["K"], dtype=torch.float32)
        camtoworld = torch.tensor(calibration["camtoworld"], dtype=torch.float32)
        footprints = torch.from_numpy(calibration["footprints"].astype(np.float32))
        points = torch.from_numpy(calibration["points"])
    footprints /= FOOTPRINT_SAMPLES
    height, width = footprints.shape[1:]
    measurement = read_image(data / "measurement.png", "RGB")
    if measurement.shape[:2] != (height, width):
        raise ValueError(
            f"measurement.png is {measurement.shape[1]}x{measurement.shape[0]}, "
            f"but the calibration footprints are {width}x{height}."
        )
    K[:, 0] *= width / sensor_width
    K[:, 1] *= height / sensor_height
    return Sensor(
        lenses=Cameras(K.to(device), camtoworld.to(device), width, height),
        names=names,
        footprints=footprints.to(device),
        normalizer=float(footprints.sum(0).max()),
        measurement=torch.from_numpy(measurement).to(device, torch.float32) / 255,
        points=points.to(device),
    )


def render(params: Params, cameras: Cameras, surface: bool) -> tuple[Tensor, dict[str, Any]]:
    """Linear RGB [C, H, W, 3] of 3D Gaussians, or of 2D surfels with their surface statistics."""
    if not surface:
        rgb, _alpha, info = rasterize(
            params,
            cameras.camtoworld,
            cameras.K,
            cameras.width,
            cameras.height,
            sh_degree=0,
            packed=False,
        )
        return rgb, info
    rgb_depth, alpha, normals, depth_normals, distortion, _median, info = rasterization_2dgs(
        means=params["means"],
        quats=F.normalize(params["quats"], dim=-1),
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shN"]), dim=1),
        viewmats=torch.linalg.inv(cameras.camtoworld),
        Ks=cameras.K,
        width=cameras.width,
        height=cameras.height,
        sh_degree=0,
        packed=False,
        render_mode="RGB+ED",
        distloss=True,
        depth_mode="expected",
    )
    info.update(
        alpha=alpha,
        normals=normals,
        # gsplat squeezes the camera dimension of depth normals when C=1.
        depth_normals=depth_normals.reshape_as(normals),
        distortion=distortion,
    )
    return rgb_depth[..., :3], info


def lens_batches(sensor: Sensor) -> list[tuple[Cameras, Tensor]]:
    """Lens cameras and their footprints in groups of CAMERA_BATCH."""
    return list(
        zip(
            sensor.lenses.split(CAMERA_BATCH),
            sensor.footprints.split(CAMERA_BATCH),
            strict=True,
        )
    )


def sensor_image(views: Tensor, footprints: Tensor, normalizer: float) -> Tensor:
    """Sum of linear lens views [N, H, W, 3] weighted by their footprints, before gain."""
    return (views * footprints[..., None]).sum(0) / normalizer


@torch.no_grad()
def predict_sensor(params: Params, sensor: Sensor, surface: bool) -> Tensor:
    """Linear sensor image [H, W, 3] before gain."""
    total = torch.zeros_like(sensor.measurement)
    for cameras, footprints in lens_batches(sensor):
        total += sensor_image(render(params, cameras, surface)[0], footprints, sensor.normalizer)
    return total


def _lens_plane(lenses: Cameras) -> tuple[int, Tensor, Tensor, Tensor]:
    """Central lens index, lens-array center, the central lens's right and down axes [2, 3],
    and every lens's offset from the center along those axes [N, 2]."""
    origins = lenses.camtoworld[:, :3, 3]
    center = origins.mean(0)
    central = int((origins - center).square().sum(-1).argmin())
    axes = lenses.camtoworld[central, :3, :2].T
    return central, center, axes, (origins - center) @ axes.T


def _look_at(
    lenses: Cameras, central: int, positions: Tensor, target: Tensor, zoom: float
) -> Cameras:
    """Cameras at `positions` aimed at `target`, with the central lens's zoomed intrinsics."""
    down = lenses.camtoworld[central, :3, 1].expand_as(positions)
    forward = F.normalize(target - positions, dim=-1)
    right = F.normalize(torch.linalg.cross(down, forward), dim=-1)
    camtoworld = torch.eye(4, device=positions.device).repeat(len(positions), 1, 1)
    camtoworld[:, :3, :3] = torch.stack((right, torch.linalg.cross(forward, right), forward), -1)
    camtoworld[:, :3, 3] = positions
    K = lenses.K[central].clone()
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    K[0, 2], K[1, 2] = lenses.width / 2, lenses.height / 2
    return Cameras(K.expand(len(positions), 3, 3), camtoworld, lenses.width, lenses.height)


def random_cameras(lenses: Cameras, target: Tensor, count: int) -> Cameras:
    """Cameras spread uniformly over one standard deviation of the lens array, aimed at `target`."""
    central, center, axes, offsets = _lens_plane(lenses)
    sampled = (2 * torch.rand(count, 2, device=target.device) - 1) * offsets.std(0)
    return _look_at(lenses, central, center + sampled @ axes, target, 1.0)


def orbit_cameras(
    lenses: Cameras, target: Tensor, count: int, amplitude: float, zoom: float = 1.0
) -> Cameras:
    """An elliptical orbit from the central lens, `amplitude` lens-offset deviations wide."""
    central, _center, axes, offsets = _lens_plane(lenses)
    phase = torch.arange(count, device=target.device) * (2 * torch.pi / count)
    path = (
        torch.stack((phase.sin(), phase.cos() - 1), -1) * amplitude * offsets.std(0, unbiased=False)
    )
    return _look_at(lenses, central, lenses.camtoworld[central, :3, 3] + path @ axes, target, zoom)
