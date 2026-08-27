"""Export the final cleaned and refit surface model with its calibrated image model."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch
from imageio_ffmpeg import count_frames_and_secs, read_frames
from PIL import Image
from plyfile import PlyData

from multiplexed.train import (
    linear_to_srgb,
    load_cameras,
    load_footprints,
    load_model,
    measurement,
    render_batch,
    save_image,
    sensor_contribution,
)

FRAMES = 240
FPS = 60


def orbit_cameras(
    cameras: list[dict[str, Any]], count: int, target: torch.Tensor, zoom: float = 1.0
) -> list[dict[str, Any]]:
    """Use camera position spread to orbit a fixed reconstruction-derived target."""
    poses = torch.stack([camera["camtoworld"] for camera in cameras])
    origins = poses[:, :3, 3]
    center = origins.mean(0)
    nearest = int((origins - center).square().sum(-1).argmin())
    anchor = poses[nearest]
    right, down = anchor[:3, 0], anchor[:3, 1]
    offsets = origins - center
    horizontal = (offsets @ right).std(unbiased=False) * 0.5
    vertical = (offsets @ down).std(unbiased=False) * 0.5
    K = cameras[nearest]["K"].clone()
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    result = []
    for phase in torch.arange(count, device=poses.device) * (2 * torch.pi / count):
        position = anchor[:3, 3] + horizontal * torch.sin(phase) * right
        position = position + vertical * (torch.cos(phase) - 1) * down
        forward = torch.nn.functional.normalize(target - position, dim=0)
        local_right = torch.nn.functional.normalize(torch.linalg.cross(down, forward), dim=0)
        local_down = torch.linalg.cross(forward, local_right)
        pose = torch.eye(4, device=poses.device)
        pose[:3, :3] = torch.stack([local_right, local_down, forward], dim=1)
        pose[:3, 3] = position
        result.append(
            {**cameras[nearest], "name": f"orbit_{len(result):04d}", "camtoworld": pose, "K": K}
        )
    return result


def orbit_framing(cameras: list[dict[str, Any]], points: torch.Tensor) -> dict[str, Any]:
    """Fit central baseline point support inside a fixed margin for the entire orbit."""
    bounds = []
    zooms = []
    quantiles = points.new_tensor([0.02, 0.98])
    for camera in cameras:
        pose = camera["camtoworld"]
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        local = local[local[:, 2] > 0.01]
        if not len(local):
            raise ValueError("No baseline support lies in front of the orbit camera.")
        offsets = local[:, :2] / local[:, 2:] * camera["K"].diagonal()[:2]
        limits = torch.quantile(offsets, quantiles, dim=0)
        radius = limits.abs().max(dim=0).values.clamp_min(1.0)
        size = points.new_tensor([camera["width"], camera["height"]])
        zooms.append(float((0.70 * size / (2 * radius)).min()))
        bounds.append(limits.tolist())
    return {
        "zoom": min(zooms),
        "quantiles": [0.02, 0.98],
        "frame_fill": 0.70,
        "projected_pixel_offset_bounds": bounds,
        "points": len(points),
    }


@torch.no_grad()
def export_model(
    model: dict[str, torch.Tensor], cameras: list[dict[str, Any]],
    orbit: list[dict[str, Any]], target: torch.Tensor, degree: int, gain: float,
    footprints: torch.Tensor, normalizer: float, output: Path,
) -> dict[str, Any]:
    output.mkdir()
    linear_views: dict[str, np.ndarray] = {}
    sensor_equivalent_views: dict[str, np.ndarray] = {}
    observed_views: dict[str, np.ndarray] = {}
    radiance_scale = gain / normalizer
    total = torch.zeros_like(target)
    for index, camera in enumerate(cameras):
        batch, _ = render_batch(model, [camera], degree)
        linear = batch[0]
        if not torch.isfinite(linear).all():
            raise FloatingPointError(f"Nonfinite render for {camera['name']}")
        total += sensor_contribution(batch, footprints[index:index + 1], normalizer)
        sensor_equivalent = linear * radiance_scale
        observed = sensor_equivalent * footprints[index, ..., None]
        linear_views[camera["name"]] = linear.cpu().numpy()
        sensor_equivalent_views[camera["name"]] = sensor_equivalent.cpu().numpy()
        observed_views[camera["name"]] = observed.cpu().numpy()
        save_image(linear_to_srgb(sensor_equivalent), output / f"view_{Path(camera['name']).stem}.png")
        save_image(linear_to_srgb(observed), output / f"observed_view_{Path(camera['name']).stem}.png")
    np.savez_compressed(output / "views_linear.npz", **linear_views)
    np.savez_compressed(output / "views_sensor_equivalent_linear.npz", **sensor_equivalent_views)
    np.savez_compressed(output / "views_observed_linear.npz", **observed_views)
    prediction = linear_to_srgb(total * gain)
    residual = prediction - target
    mse = float(residual.square().mean())
    save_image(prediction, output / "sensor_prediction.png")
    save_image(residual.abs(), output / "sensor_absolute_residual.png")
    np.save(output / "sensor_residual.npy", residual.cpu().numpy())
    with cast(Any, imageio.get_writer(
        output / "orbit.mp4", fps=FPS, macro_block_size=1, quality=8,
    )) as writer:
        for index, camera in enumerate(orbit):
            batch, _ = render_batch(model, [camera], degree)
            encoded = linear_to_srgb(batch[0] * radiance_scale).clamp(0, 1)
            if not torch.isfinite(encoded).all():
                raise FloatingPointError(f"Nonfinite orbit frame {index}")
            pixels = (encoded.cpu().numpy() * 255).round().astype(np.uint8)
            writer.append_data(pixels)
            if index in (0, 60, 120, 180, 239):
                Image.fromarray(pixels).save(output / f"orbit_{index:04d}.png")
    frames, seconds = count_frames_and_secs(str(output / "orbit.mp4"))
    if frames != FRAMES:
        raise RuntimeError(f"Decoded {frames} frames instead of {FRAMES}")
    if not math.isclose(seconds, FRAMES / FPS, abs_tol=1 / FPS):
        raise RuntimeError(f"Decoded video duration is {seconds}, expected {FRAMES / FPS}")
    reader = read_frames(str(output / "orbit.mp4"))
    metadata = next(reader)
    reader.close()
    size = (orbit[0]["width"], orbit[0]["height"])
    if tuple(metadata["size"]) != size or not math.isclose(metadata["fps"], FPS, abs_tol=1e-6):
        raise RuntimeError(f"Unexpected video dimensions or frame rate: {metadata}")
    return {
        "points": len(model["means"]), "sensor_mse": mse,
        "sensor_nmse": mse / float(target.square().mean()),
        "sensor_psnr": -10 * math.log10(max(mse, 1e-30)),
        "view_count": len(linear_views), "decoded_frames": frames, "video_seconds": seconds,
        "video_size": list(size), "video_fps": metadata["fps"],
    }


@torch.no_grad()
def export_run(run: Path, output: Path) -> None:
    """Export only the completed fixed pipeline; cleanup and refitting run in training."""
    run = run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["completed_steps"] != 6500:
        raise ValueError("Export requires the completed 3000 + 3000 + 500 step pipeline.")
    if manifest["radiance_domain"] != "linear" or manifest["sensor_normalization"] != "max_calibrated_lens_overlap":
        raise ValueError("Export requires linear radiance and calibrated lens-overlap normalization.")
    surface_path = run / "stages/surface/point_cloud.ply"
    vertices = PlyData.read(surface_path)["vertex"].data
    keep = np.load(run / "cleanup/keep.npy", allow_pickle=False)
    if keep.dtype != np.bool_ or keep.shape != (len(vertices),) or not keep.any():
        raise ValueError("Cleanup indices must select a nonempty subset of the surface model.")
    selected = PlyData.read(run / "cleanup/selected.ply")["vertex"].data
    if selected.dtype != vertices.dtype or not np.array_equal(selected, vertices[keep]):
        raise ValueError("Cleanup initialization differs from the retained surface subset.")
    device = torch.device("cuda")
    model = load_model(run / "point_cloud.ply", device)
    if len(model["means"]) != int(keep.sum()):
        raise ValueError("The refit must preserve the retained model size.")
    calibration_path = run / "inputs/calibration.npz"
    cameras = load_cameras(calibration_path, device, manifest["downsample"])
    footprints, normalizer = load_footprints(calibration_path, cameras, device)
    if normalizer != manifest["sensor_normalizer"]:
        raise ValueError("Frozen footprint normalization differs from the training manifest.")
    target = measurement(run / "inputs/measurement.png", cameras[0], device)
    parent = load_model(surface_path, device)
    opacity = parent["opacities"].sigmoid()
    support = parent["means"][opacity >= opacity.median()]
    target_point = support.median(dim=0).values
    framing = orbit_framing(orbit_cameras(cameras, 8, target_point), support)
    orbit = orbit_cameras(cameras, FRAMES, target_point, framing["zoom"])
    for camera in orbit:
        camera["K"][0, 2] = camera["width"] / 2
        camera["K"][1, 2] = camera["height"] / 2
    output.mkdir(parents=True, exist_ok=False)
    source_output = output / "source/multiplexed"
    source_output.mkdir(parents=True)
    for name in ("__init__.py", "train.py", "render.py"):
        shutil.copy2(Path(__file__).with_name(name), source_output / name)
    save_image(target, output / "sensor_target.png")
    orbit_payload = [{key: value.tolist() if isinstance(value, torch.Tensor) else value
                      for key, value in camera.items()
                      if key in ("name", "width", "height", "K", "camtoworld")}
                     for camera in orbit]
    (output / "orbit_cameras.json").write_text(json.dumps(orbit_payload, indent=2) + "\n")
    report = {
        "status": "exporting", "parent_run": str(run), "parent_manifest": manifest,
        "reference_images_read": False, "object_masks_used": False,
        "sensor_normalizer": normalizer, "radiance_scale": manifest["gain"] / normalizer,
        "display": "Whole unmasked views: nominal sRGB of saved gain times learned radiance divided by maximum lens overlap; clipping for display only.",
        "view_archives": {
            "views_linear.npz": "Raw unmasked learned linear radiance L.",
            "views_sensor_equivalent_linear.npz": "Unmasked U = saved_gain * L / max_calibrated_lens_overlap; encoded for view PNGs and orbit frames.",
            "views_observed_linear.npz": "Calibrated lens footprint times U; summing and encoding once reproduces the sensor prediction.",
        },
        "orbit": {"target": target_point.tolist(), "framing": framing,
                  "source": "Surface model before cleanup; points at or above median opacity.",
                  "interpretation": "Model-derived display path, not a measured physical orbit."},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["final"] = export_model(
        model, cameras, orbit, target, 0, manifest["gain"], footprints, normalizer, output / "final",
    )
    if not math.isclose(report["final"]["sensor_mse"], manifest["final_sensor_mse"], rel_tol=0, abs_tol=1e-6):
        raise RuntimeError("Export does not reproduce the fitted sensor MSE.")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["final"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_run(args.run, args.output)


if __name__ == "__main__":
    main()
