"""Reconstruct one combined capture with the fixed bounded-surface pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove
from gsplat.utils import normalized_quat_to_rotmat
from scipy.spatial import ConvexHull, cKDTree  # ty: ignore[unresolved-import]
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


DOWNSAMPLE = 8
CAMERA_BATCH = 6
TV_WEIGHT = 0.1
STAGE_STEPS = {"volume": 3000, "surface": 3000, "refit": 500}


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.clamp_min(0.0031308).pow(1 / 2.4) - 0.055,
    )


def load_cameras(path: Path, device: Any, downsample: int = 8) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as calibration:
        width, height = calibration["sensor_size"].tolist()
        cameras: list[dict[str, Any]] = [
            {"name": str(name), "width": width, "height": height, "K": K, "camtoworld": pose}
            for name, K, pose in zip(
                calibration["names"], calibration["K"], calibration["camtoworld"], strict=True,
            )
        ]
    if not cameras or downsample < 1:
        raise ValueError("Need calibrated cameras and a positive downsample factor.")
    result = []
    for source in cameras:
        camera = dict(source)
        width, height = round(source["width"] / downsample), round(source["height"] / downsample)
        K = torch.tensor(source["K"], device=device, dtype=torch.float32)
        K[0] *= width / source["width"]
        K[1] *= height / source["height"]
        camera.update(
            source_width=source["width"],
            source_height=source["height"],
            width=width,
            height=height,
            K=K,
            camtoworld=torch.tensor(source["camtoworld"], device=device, dtype=torch.float32),
        )
        result.append(camera)
    if len({(c["width"], c["height"]) for c in result}) != 1:
        raise ValueError("All lenses must use the same sensor grid.")
    return result


def load_model(path: Path, device: Any) -> dict[str, torch.Tensor]:
    vertex = PlyData.read(path)["vertex"]
    names = vertex.data.dtype.names or ()
    rest = sorted(
        (n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.removeprefix("f_rest_"))
    )

    def columns(fields: list[str]) -> torch.Tensor:
        return torch.tensor(
            np.column_stack([vertex[n] for n in fields]), dtype=torch.float32, device=device
        )

    return {
        "means": columns(["x", "y", "z"]),
        "scales": columns([f"scale_{i}" for i in range(3)]),
        "quats": columns([f"rot_{i}" for i in range(4)]),
        "opacities": columns(["opacity"])[:, 0],
        "sh0": columns([f"f_dc_{i}" for i in range(3)])[:, None],
        "shN": (
            columns(rest).reshape(-1, 3, len(rest) // 3).transpose(1, 2).contiguous()
            if rest
            else torch.empty((len(vertex), 0, 3), device=device)
        ),
    }


def save_model(model: dict[str, torch.Tensor], path: Path) -> None:
    arrays = [
        model["means"],
        torch.zeros_like(model["means"]),
        model["sh0"][:, 0],
        model["shN"].transpose(1, 2).flatten(1),
        model["opacities"][:, None],
        model["scales"],
        model["quats"],
    ]
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(model["shN"].shape[1] * 3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    values = torch.cat(arrays, dim=1).detach().cpu().numpy()
    vertices = np.empty(len(values), dtype=[(n, "f4") for n in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def measurement(path: Path, camera: dict[str, Any], device: Any) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    expected = (camera["width"], camera["height"])
    if image.size != expected:
        raise ValueError(f"Measurement dimensions {image.size} do not match calibration {expected}.")
    return torch.tensor(np.asarray(image).copy(), device=device, dtype=torch.float32) / 255


def save_image(value: torch.Tensor, path: Path) -> None:
    Image.fromarray((value.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(
        path
    )


def _render_volume(
    model: dict[str, torch.Tensor] | torch.nn.ParameterDict,
    cameras: list[dict[str, Any]],
    degree: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render linear radiance with the full calibrated intrinsic matrices."""
    from gsplat.rendering import rasterization

    rgb, _alpha, info = rasterization(
        means=model["means"],
        quats=F.normalize(model["quats"], dim=-1),
        scales=model["scales"].exp(),
        opacities=model["opacities"].sigmoid(),
        colors=torch.cat((model["sh0"], model["shN"]), dim=1),
        viewmats=torch.linalg.inv(torch.stack([c["camtoworld"] for c in cameras])),
        Ks=torch.stack([c["K"] for c in cameras]),
        width=cameras[0]["width"], height=cameras[0]["height"],
        sh_degree=degree, packed=False, near_plane=0.01, far_plane=100.0,
        rasterize_mode="classic",
    )
    return rgb.clamp_min(0), info


@torch.no_grad()
def project_nonnegative_sh0(sh0: torch.Tensor) -> None:
    """Project SH0 to the exact float32 nonnegative-radiance boundary."""
    lower = -sh0.new_tensor(0.5) / sh0.new_tensor(0.28209479177387814)
    sh0.clamp_min_(lower)


def render_batch(
    model: dict[str, torch.Tensor] | torch.nn.ParameterDict,
    cameras: list[dict[str, Any]],
    degree: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render planar surfels and their surface statistics using full intrinsics."""
    from gsplat.rendering import rasterization_2dgs

    rgb_depth, alpha, normals, depth_normals, distortion, _median, info = rasterization_2dgs(
        means=model["means"],
        quats=F.normalize(model["quats"], dim=-1),
        scales=model["scales"].exp(),
        opacities=model["opacities"].sigmoid(),
        colors=torch.cat((model["sh0"], model["shN"]), dim=1),
        viewmats=torch.linalg.inv(torch.stack([c["camtoworld"] for c in cameras])),
        Ks=torch.stack([c["K"] for c in cameras]),
        width=cameras[0]["width"], height=cameras[0]["height"],
        sh_degree=degree, packed=False, near_plane=0.01, far_plane=100.0,
        render_mode="RGB+ED", distloss=True, depth_mode="expected",
    )
    # gsplat squeezes the depth-normal camera dimension when C=1.
    info.update(alpha=alpha, rendered_normals=normals,
                depth_normals=depth_normals.reshape_as(normals),
                expected_depth=rgb_depth[..., 3:4], distortion=distortion)
    return rgb_depth[..., :3].clamp_min(0), info


def surface_regularizers(info: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Native depth distortion and opacity-neutral surface-direction agreement."""
    direction = F.normalize(info["rendered_normals"], dim=-1, eps=1e-6)
    cosine = (direction * info["depth_normals"]).sum(dim=-1).clamp(-1, 1)
    normal = (info["alpha"].detach().squeeze(-1) * (1 - cosine)).mean()
    return info["distortion"].mean(), normal


def image_tv(images: torch.Tensor) -> torch.Tensor:
    return (images[:, 1:] - images[:, :-1]).abs().mean() + (
        images[:, :, 1:] - images[:, :, :-1]
    ).abs().mean()


def random_scene_cameras(
    cameras: list[dict[str, Any]], target: torch.Tensor, count: int = 3,
) -> list[dict[str, Any]]:
    """Sample the calibrated camera-plane spread and aim at the sparse centroid."""
    poses = torch.stack([c["camtoworld"] for c in cameras])
    origins = poses[:, :3, 3]
    center = origins.mean(0)
    nearest = int((origins - center).square().sum(-1).argmin())
    anchor = poses[nearest]
    right, down = anchor[:3, 0], anchor[:3, 1]
    offsets = (origins - center) @ torch.stack((right, down), dim=1)
    sampled = (2 * torch.rand(count, 2, device=target.device) - 1) * offsets.std(0)
    result = []
    for offset in sampled:
        position = center + offset[0] * right + offset[1] * down
        forward = F.normalize(target - position, dim=0)
        local_right = F.normalize(torch.linalg.cross(down, forward), dim=0)
        pose = torch.eye(4, device=target.device)
        pose[:3, :3] = torch.stack(
            (local_right, torch.linalg.cross(forward, local_right), forward), dim=1
        )
        pose[:3, 3] = position
        K = cameras[nearest]["K"].clone()
        K[0, 2], K[1, 2] = cameras[nearest]["width"] / 2, cameras[nearest]["height"] / 2
        result.append({**cameras[nearest], "camtoworld": pose, "K": K})
    return result


def load_footprints(
    path: Path, cameras: list[dict[str, Any]], device: Any,
) -> tuple[torch.Tensor, float]:
    """Decode 4x4 aperture sample counts and preserve calibrated overlap normalization."""
    names = [camera["name"] for camera in cameras]
    if not names or len(set(names)) != len(names):
        raise ValueError("Calibrated lens names must be nonempty and unique.")
    with np.load(path, allow_pickle=False) as archive:
        if archive["names"].tolist() != names:
            raise ValueError("Footprint names must match every calibrated lens exactly.")
        counts = archive["footprints"]
    if counts.ndim != 3 or counts.shape[0] != len(names) or counts.dtype != np.uint8:
        raise ValueError("Footprints must contain uint8 sample counts for each calibrated lens.")
    weights = torch.from_numpy(counts.astype(np.float32)) / 16
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Footprint coverage must be finite and between zero and one.")
    if (weights.sum(dim=(1, 2)) == 0).any():
        raise ValueError("Every calibrated lens needs nonempty sensor coverage.")
    source_height, source_width = weights.shape[1:]
    camera = cameras[0]
    if source_width * camera["source_height"] != source_height * camera["source_width"]:
        raise ValueError("Footprints and calibrated sensor must have the same aspect ratio.")
    height, width = camera["height"], camera["width"]
    if height > source_height or width > source_width:
        raise ValueError("Render resolution exceeds calibrated footprint resolution.")
    normalizer = float(weights.sum(0).max())
    if (height, width) != (source_height, source_width):
        weights = F.interpolate(weights[:, None], (height, width), mode="area")[:, 0]
    return weights.to(device=device), normalizer


def sensor_contribution(
    linear_views: torch.Tensor, footprints: torch.Tensor, normalizer: float,
) -> torch.Tensor:
    """Sum footprint-weighted linear views divided by maximum lens overlap."""
    return (linear_views * footprints[..., None]).sum(0) / normalizer


def build_support(points: np.ndarray) -> dict[str, np.ndarray]:
    """Build the padded hull of all density-supported calibration XYZ seeds."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) <= 16 or not np.isfinite(xyz).all():
        raise ValueError("Support needs at least 17 finite three-dimensional calibration points.")
    spacing = cKDTree(xyz).query(xyz, k=17, workers=4)[0][:, -1]
    labels = DBSCAN(eps=2 * float(np.median(spacing)), min_samples=17).fit_predict(xyz)
    indices = np.flatnonzero(labels >= 0)
    supported = xyz[indices]
    if len(supported) <= 16:
        raise ValueError("Too few density-supported calibration points for a hull.")
    spacing = cKDTree(supported).query(supported, k=17, workers=4)[0][:, -1]
    hull = ConvexHull(supported)
    return {
        "planes": hull.equations,
        "allowance": 2 * np.median(spacing[hull.simplices], axis=1),
        "center": supported.mean(axis=0),
        "supported_seed_indices": indices,
    }


def face_support(
    params: dict[str, torch.Tensor] | torch.nn.ParameterDict,
    planes: torch.Tensor,
    tangent_only: bool,
) -> torch.Tensor:
    """Maximum face coordinate over a complete three-sigma ellipsoid or tangent ellipse."""
    dimensions = 2 if tangent_only else 3
    rotation = normalized_quat_to_rotmat(F.normalize(params["quats"], dim=-1))
    local = torch.einsum("nki,fk->nfi", rotation, planes[:, :3])[:, :, :dimensions]
    radius = 3 * (local.square() * params["scales"][:, :dimensions].exp().square()[:, None]).sum(-1).sqrt()
    return params["means"] @ planes[:, :3].T + planes[:, 3] + radius


@torch.no_grad()
def project_hull_(
    params: dict[str, torch.Tensor] | torch.nn.ParameterDict,
    planes: torch.Tensor,
    allowance: torch.Tensor,
    center: torch.Tensor,
    tangent_only: bool,
) -> torch.Tensor:
    """Contract violating supports about the seed center, preserving Adam state and identity."""
    center_value = center @ planes[:, :3].T + planes[:, 3]
    direction_support = face_support(params, planes, tangent_only) - center_value
    factor = torch.where(
        direction_support > 0,
        (allowance - center_value) / direction_support,
        torch.ones_like(direction_support),
    ).amin(dim=1).clamp_max(1)
    affected = factor < 1
    # Numerical clearance only; this is the accepted eight-epsilon projection.
    factor[affected] *= 1 - 8 * torch.finfo(factor.dtype).eps
    selected = factor[affected, None]
    params["means"][affected] = center + selected * (params["means"][affected] - center)
    dimensions = 2 if tangent_only else 3
    params["scales"][affected, :dimensions] += selected.log()
    return affected.sum()


def convert_to_surface(
    model: dict[str, torch.Tensor] | torch.nn.ParameterDict,
) -> dict[str, torch.Tensor]:
    """Put the two largest covariance axes first without changing the full covariance."""
    scales = model["scales"].detach().cpu().numpy()
    quats = model["quats"].detach().cpu().numpy()[:, [1, 2, 3, 0]].astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rotation = Rotation.from_quat(quats).as_matrix()
    order = np.argsort(-scales, axis=1, kind="stable")
    reordered = np.take_along_axis(rotation, order[:, None, :], axis=2)
    reordered[np.linalg.det(reordered) < 0, :, 2] *= -1
    converted = {name: value.detach().clone() for name, value in model.items()}
    converted["scales"] = torch.as_tensor(
        np.take_along_axis(scales, order, axis=1),
        dtype=model["scales"].dtype, device=model["scales"].device,
    )
    converted["quats"] = torch.as_tensor(
        Rotation.from_matrix(reordered).as_quat()[:, [3, 0, 1, 2]],
        dtype=model["quats"].dtype, device=model["quats"].device,
    )
    return converted


def quarter_keep(
    model: dict[str, torch.Tensor] | torch.nn.ParameterDict,
    geometry: dict[str, np.ndarray],
) -> np.ndarray:
    """Retain complete tangent ellipses inside one quarter of the original hull padding."""
    planes, allowance = geometry["planes"], geometry["allowance"]
    means = model["means"].detach().cpu().numpy().astype(np.float64)
    scales = np.exp(model["scales"].detach().cpu().numpy()[:, :2].astype(np.float64))
    quats = model["quats"].detach().cpu().numpy()[:, [1, 2, 3, 0]].astype(np.float64)
    rotation = Rotation.from_quat(quats).as_matrix()
    maximum = np.full(len(means), -np.inf)
    for start in range(0, len(planes), 16):
        plane = planes[start:start + 16]
        signed = means @ plane[:, :3].T + plane[:, 3]
        local = np.einsum("nki,fk->nfi", rotation, plane[:, :3])[:, :, :2]
        radius = 3 * np.sqrt(np.sum(local ** 2 * scales[:, None] ** 2, axis=-1))
        maximum = np.maximum(maximum, (signed + radius - .25 * allowance[start:start + 16]).max(axis=1))
    keep = maximum <= 0
    if not keep.any():
        raise ValueError("The fixed quarter-padding cleanup removes every surface Gaussian.")
    return keep


@torch.no_grad()
def split_tangent(
    params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
    mask: torch.Tensor,
) -> None:
    """Replace each selected surfel with two children in its tangent plane."""
    selected = torch.where(mask)[0]
    remaining = torch.where(~mask)[0]
    radii = params["scales"][selected, :2].exp()
    rotation = normalized_quat_to_rotmat(F.normalize(params["quats"][selected], dim=-1))
    samples = torch.randn((2, len(selected), 2), device=radii.device, dtype=radii.dtype)
    offsets = torch.einsum("nij,bnj->bni", rotation[:, :, :2], samples * radii)

    def param_fn(name: str, value: torch.Tensor) -> torch.Tensor:
        if name == "means":
            children = (value[selected] + offsets).reshape(-1, 3)
        elif name == "scales":
            children = value[selected].clone()
            children[:, :2] = (radii / 1.6).log()
            children = children.repeat(2, 1)
        else:
            children = value[selected].repeat([2] + [1] * (value.ndim - 1))
        return torch.nn.Parameter(torch.cat((value[remaining], children)),
                                  requires_grad=value.requires_grad)

    def optimizer_fn(_key: str, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value[remaining], value.new_zeros((2 * len(selected), *value.shape[1:]))))

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            children = value[selected].repeat([2] + [1] * (value.ndim - 1))
            state[name] = torch.cat((value[remaining], children))


class SurfaceStrategy(DefaultStrategy):
    """Keep the existing schedule while growing and pruning by tangent radii."""

    def __init__(self) -> None:
        super().__init__(
            prune_opa=.006, grow_grad2d=3.5e-4,
            grow_scale3d=.012, grow_scale2d=.05,
            prune_scale3d=.22, prune_scale2d=.12,
            refine_start_iter=500, refine_every=100, refine_stop_iter=2400,
            refine_scale2d_stop_iter=2400, reset_every=100_000,
            key_for_gradient="gradient_2dgs",
        )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> tuple[int, int]:
        grads = state["grad2d"] / state["count"].clamp_min(1)
        high = grads > self.grow_grad2d
        small = params["scales"][:, :2].exp().max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
        to_duplicate = high & small
        to_split = high & ~small
        if step < self.refine_scale2d_stop_iter:
            to_split |= state["radii"] > self.grow_scale2d
        n_duplicate, n_split = int(to_duplicate.sum()), int(to_split.sum())
        if n_duplicate:
            duplicate(params, optimizers, state, to_duplicate)
        to_split = torch.cat((to_split, torch.zeros(n_duplicate, dtype=torch.bool, device=grads.device)))
        if n_split:
            split_tangent(params, optimizers, state, to_split)
        return n_duplicate, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> int:
        to_prune = params["opacities"].flatten().sigmoid() < self.prune_opa
        count = int(to_prune.sum())
        if count:
            remove(params, optimizers, state, to_prune)
        return count


def _fit_stage(
    stage: str,
    params: torch.nn.ParameterDict,
    cameras: list[dict[str, Any]],
    footprints: torch.Tensor,
    target: torch.Tensor,
    normalizer: float,
    calibration_points: torch.Tensor,
    geometry: dict[str, np.ndarray],
    output: Path,
    inherited_gain: float,
    completed_steps: int,
) -> dict[str, Any]:
    """Run the same sensor objective with the three fixed stage differences."""
    from fused_ssim import fused_ssim
    from static.gaussians import build_optimizers, initialize_default_strategy

    output.mkdir(parents=True)
    steps = STAGE_STEPS[stage]
    tangent_only = stage != "volume"
    fixed_topology = stage == "refit"
    renderer = render_batch if tangent_only else _render_volume
    gradient_key = "gradient_2dgs" if tangent_only else "means2d"
    center = calibration_points.median(dim=0).values
    extent = float(torch.quantile((calibration_points - center).norm(dim=-1), 0.95))
    planes = torch.as_tensor(geometry["planes"], device=target.device, dtype=torch.float32)
    allowance = torch.as_tensor(
        geometry["allowance"] * (0.25 if fixed_topology else 1),
        device=target.device, dtype=torch.float32,
    )
    hull_center = torch.as_tensor(geometry["center"], device=target.device, dtype=torch.float32)
    assert (planes[:, :3] @ hull_center + planes[:, 3] < 0).all()
    initial_count = len(params["means"])
    initial_third_scale = params["scales"][:, 2].detach().clone()
    if fixed_topology:
        params["shN"].requires_grad_(False)
    project_nonnegative_sh0(params["sh0"])
    optimizers = build_optimizers(params, scene_scale=extent)
    strategy, state = None, None
    if stage == "volume":
        strategy, state = initialize_default_strategy(
            extent, params, optimizers, refine_stop_iter=2400,
        )
    elif stage == "surface":
        strategy = SurfaceStrategy()
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=extent)
    else:
        del optimizers["shN"]
    log_gain = None if fixed_topology else torch.zeros(
        (), device=target.device, requires_grad=True,
    )
    gain_optimizer = None if log_gain is None else torch.optim.Adam([log_gain], lr=0.001)

    def current_gain() -> torch.Tensor | float:
        return inherited_gain if log_gain is None else log_gain.exp()

    manifest = {
        "status": "running", "stage": stage, "planned_steps": steps,
        "parent_completed_steps": completed_steps,
        "planned_total_optimization_steps": completed_steps + steps,
        "initial_points": initial_count, "scene_extent": extent,
        "source_snapshot": "../../source", "inputs": "../../inputs",
        "initialization": {
            "volume": "Neutral RGB 0.5 at calibration XYZ; original axes and random quaternions",
            "surface": "Largest-axis conversion of completed volume; preserve full covariance",
            "refit": "Exact quarter-hull subset of completed surface; no further conversion",
        }[stage],
        "primitive": "2D tangent Gaussian" if tangent_only else "3D Gaussian",
        "gain_optimization": not fixed_topology,
        "optimizer_state": "Fresh Adam states at the start of each stage",
        "trainable": list(optimizers),
        "densification": vars(strategy) if strategy is not None else "disabled",
        "density_gradient_camera_divisor": len(cameras) if strategy is not None else None,
        "support_allowance_multiplier": 0.25 if fixed_topology else 1,
        "support_axes": 2 if tangent_only else 3,
        "surface_regularization": {
            "distortion_weight": 0.01 if tangent_only else 0,
            "normal_weight": 0.05 if tangent_only else 0,
            "distortion_start_step": 1 if fixed_topology else 301,
            "normal_start_step": 1 if fixed_topology else 701,
        },
        "tv_weight": TV_WEIGHT, "seed": 0, "active_sh_degree": 0,
        "checkpoint_gain_representation": "gain is the actual linear scalar",
    }
    save_model(dict(params.items()), output / "initial.ply")
    manifest["initial_contracted_gaussians"] = int(project_hull_(
        params, planes, allowance, hull_center, tangent_only,
    ))
    save_model(dict(params.items()), output / "initial_projected.ply")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    batches = [(cameras[i:i + CAMERA_BATCH], footprints[i:i + CAMERA_BATCH])
               for i in range(0, len(cameras), CAMERA_BATCH)]
    history = []
    started = time.monotonic()
    for step in range(steps):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if gain_optimizer is not None:
            gain_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            total = torch.zeros_like(target)
            for batch, weights in batches:
                total += sensor_contribution(renderer(params, batch, 0)[0], weights, normalizer)
            if step == 0 and log_gain is not None:
                gain = (total * srgb_to_linear(target)).sum() / total.square().sum().clamp_min(1e-12)
                log_gain.copy_(gain.clamp_min(1e-8).log())
        total.requires_grad_(True)
        prediction = linear_to_srgb(total * current_gain())
        l1 = (prediction - target).abs().mean()
        dssim = 1 - fused_ssim(prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * dssim
        loss.backward()
        assert total.grad is not None
        means2d_grad, radii = [], []
        training_tv = training_distortion = training_normal = opaque_fraction = 0.0
        distortion_weight = 0.01 if tangent_only and (fixed_topology or step >= 300) else 0.0
        normal_weight = 0.05 if tangent_only and (fixed_topology or step >= 700) else 0.0
        for batch, weights in batches:
            rgb, info = renderer(params, batch, 0)
            if strategy is not None:
                info[gradient_key].retain_grad()
            objective = (sensor_contribution(rgb, weights, normalizer) * total.grad).sum()
            objective.backward(retain_graph=True)
            if strategy is not None:
                # Density sees only sensor gradients, with the C-camera multiplier canceled.
                means2d_grad.append(info[gradient_key].grad.detach().clone())
                radii.append(info["radii"].detach())
            tv = image_tv(linear_to_srgb(rgb * current_gain() / normalizer)) * len(batch) / len(cameras)
            training_tv += float(tv.detach())
            if tangent_only:
                distortion, normal = surface_regularizers(info)
                fraction = len(batch) / len(cameras)
                training_distortion += float(distortion.detach()) * fraction
                training_normal += float(normal.detach()) * fraction
                opaque_fraction += float((info["alpha"].detach() > 0.95).float().mean()) * fraction
                (TV_WEIGHT * tv + fraction * (
                    distortion_weight * distortion + normal_weight * normal)).backward()
            else:
                (TV_WEIGHT * tv).backward()
        novel, novel_info = renderer(params, random_scene_cameras(cameras, center), 0)
        novel_tv = image_tv(linear_to_srgb(novel * current_gain() / normalizer))
        if tangent_only:
            novel_distortion, novel_normal = surface_regularizers(novel_info)
            (TV_WEIGHT * novel_tv + distortion_weight * novel_distortion
             + normal_weight * novel_normal).backward()
        else:
            (TV_WEIGHT * novel_tv).backward()
        for name, optimizer in optimizers.items():
            if name == "means":
                optimizer.param_groups[0]["lr"] = 1.2e-4 * extent * 0.01 ** (step / steps)
            optimizer.step()
        project_nonnegative_sh0(params["sh0"])
        if gain_optimizer is not None:
            gain_optimizer.step()
        if strategy is not None:
            assert state is not None
            means2d = torch.empty_like(torch.cat(means2d_grad))
            means2d.grad = torch.cat(means2d_grad) / len(cameras)
            strategy.step_post_backward(params, optimizers, state, step, {
                "width": cameras[0]["width"], "height": cameras[0]["height"],
                "n_cameras": len(cameras), gradient_key: means2d,
                "radii": torch.cat(radii), "gaussian_ids": None,
            }, packed=False)
        contracted = project_hull_(params, planes, allowance, hull_center, tangent_only)
        if step % 100 == 0 or step + 1 == steps:
            gain = inherited_gain if log_gain is None else float(log_gain.detach().exp())
            row = {"stage": stage, "step": step + 1, "points": len(params["means"]),
                   "sensor_loss": float(loss.detach()), "training_tv": training_tv,
                   "novel_tv": float(novel_tv.detach()), "contracted_gaussians": int(contracted),
                   "sensor_mse": float((prediction.detach() - target).square().mean()),
                   "gain": gain, "elapsed_seconds": time.monotonic() - started}
            if tangent_only:
                row.update(training_distortion=training_distortion, training_normal=training_normal,
                           novel_distortion=float(novel_distortion.detach()),
                           novel_normal=float(novel_normal.detach()),
                           distortion_weight=distortion_weight, normal_weight=normal_weight,
                           training_opaque_pixel_fraction=opaque_fraction)
            history.append(row)
            print(json.dumps(row), flush=True)
        if (step + 1) % 500 == 0 or step + 1 == steps:
            if not all(torch.isfinite(value).all() for value in params.values()):
                raise FloatingPointError("Nonfinite Gaussian parameters")
            assert (params["sh0"] * 0.28209479177387814 + 0.5 >= 0).all()
            assert torch.count_nonzero(params["shN"]) == 0
            if fixed_topology:
                assert len(params["means"]) == initial_count
                assert torch.equal(params["scales"][:, 2], initial_third_scale)
            with torch.no_grad():
                maximum_excess = float((face_support(params, planes, tangent_only) - allowance).max())
                gain = float(current_gain())
            assert maximum_excess < 2e-6
            save_model(dict(params.items()), output / "point_cloud.ply")
            torch.save({"model": params.state_dict(), "step": step + 1,
                        "total_completed_optimization_steps": completed_steps + step + 1,
                        "gain": gain}, output / "checkpoint.pt")
            manifest.update(completed_steps=step + 1, points=len(params["means"]), gain=gain,
                            total_completed_optimization_steps=completed_steps + step + 1,
                            maximum_final_face_excess_world_units=maximum_excess)
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    with torch.no_grad():
        total = torch.zeros_like(target)
        for batch, weights in batches:
            total += sensor_contribution(renderer(params, batch, 0)[0], weights, normalizer)
        prediction = linear_to_srgb(total * current_gain())
        mse = float((prediction - target).square().mean())
        if not math.isfinite(mse):
            raise FloatingPointError("Nonfinite final sensor fit")
        save_image(prediction, output / "sensor_prediction.png")
    manifest.update(status="complete", final_sensor_mse=mse,
                    elapsed_seconds=time.monotonic() - started)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def train(args: argparse.Namespace) -> None:
    from static.gaussians import init_gaussians

    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    inputs = args.output / "inputs"
    inputs.mkdir(parents=True)
    for name in ("measurement.png", "calibration.npz"):
        shutil.copy2(args.data / name, inputs / name)
    source_root = Path(__file__).resolve().parents[1]
    for relative in (
        "multiplexed/__init__.py", "multiplexed/train.py", "multiplexed/render.py",
        "static/__init__.py", "static/gaussians.py",
        "pyproject.toml",
    ):
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    calibration_path = inputs / "calibration.npz"
    with np.load(calibration_path, allow_pickle=False) as calibration:
        points = calibration["points"]
        metadata = json.loads(str(calibration["metadata"]))
    geometry = build_support(points)
    np.savez_compressed(inputs / "support_geometry.npz", **geometry)
    device = torch.device("cuda")
    cameras = load_cameras(calibration_path, device, DOWNSAMPLE)
    footprints, normalizer = load_footprints(calibration_path, cameras, device)
    target = measurement(inputs / "measurement.png", cameras[0], device)
    calibration_points = torch.as_tensor(points, device=device, dtype=torch.float32)
    save_image(target, args.output / "measurement.png")
    torch.save(target.cpu(), inputs / "measurement_resized.pt")
    manifest: dict[str, Any] = {
        "status": "running", "host": socket.gethostname(), "job_id": os.getenv("SLURM_JOB_ID"),
        "method": "bounded projected-color volume, bounded surface, quarter-hull cleanup, fixed-gain refit",
        "initialization": metadata["initialization"],
        "footprint_calibration": metadata["footprint_calibration"],
        "measurement_preparation": metadata["measurement_preparation"],
        "photometric_target_count": 1, "calibration_rgb_loaded_by_trainer": False,
        "runtime_reference_images_loaded": False, "historical_reference_informed_development": True,
        "camera_optimization": False, "source_snapshot": "source", "inputs": "inputs",
        "stage_steps": STAGE_STEPS, "seed": 0, "downsample": DOWNSAMPLE,
        "camera_batch": CAMERA_BATCH, "tv_weight": TV_WEIGHT,
        "active_sh_degree": 0, "max_sh_degree": 0, "allocated_sh_degree": 3,
        "radiance_domain": "linear", "sensor_normalization": "max_calibrated_lens_overlap",
        "sensor_normalizer": normalizer, "sensor_response": "nominal sRGB JPEG approximation",
        "spatial_support": "Density-supported calibration hull; full 3-sigma support; quarter padding after surface stage",
        "support_assumption": "The padded sparse calibration hull contains the object; unsupported valid geometry may be constrained",
        "checkpoint_selection": "Fixed final iteration of each stage; no runtime reference selection",
        "stages": {},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inherited_gain, completed_steps = 1.0, 0
    for stage in STAGE_STEPS:
        # These were separate accepted jobs: preserve their independent RNG and optimizer starts.
        torch.manual_seed(0)
        if stage == "volume":
            params = init_gaussians(points, np.full(points.shape, 127.5, dtype=np.float32), device="cuda")
        elif stage == "surface":
            parent = load_model(args.output / "stages/volume/point_cloud.ply", device)
            params = torch.nn.ParameterDict({
                key: torch.nn.Parameter(value) for key, value in convert_to_surface(parent).items()
            })
        else:
            parent = load_model(args.output / "stages/surface/point_cloud.ply", device)
            keep = quarter_keep(parent, geometry)
            cleanup = args.output / "cleanup"
            cleanup.mkdir()
            np.save(cleanup / "keep.npy", keep)
            selected = {key: value[torch.as_tensor(keep, device=device)] for key, value in parent.items()}
            save_model(selected, cleanup / "selected.ply")
            original = PlyData.read(args.output / "stages/surface/point_cloud.ply")["vertex"].data
            retained = PlyData.read(cleanup / "selected.ply")["vertex"].data
            assert retained.dtype == original.dtype and retained.tobytes() == original[keep].tobytes()
            report = {
                "status": "complete", "before_points": len(keep), "kept_points": int(keep.sum()),
                "removed_points": int((~keep).sum()), "padding_factor": 0.25,
                "support_geometry": "../inputs/support_geometry.npz",
                "formula": "Retain iff every face satisfies n*mean+b+3*sqrt(sum over two tangent axes of ((R.T*n)*sigma)^2) <= 0.25*original_face_allowance",
                "retained_attributes_byte_exact": True, "gain_refit": False,
                "reference_images_read": False, "image_masks_used": False,
                "additional_opacity_selection": False,
            }
            (cleanup / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            manifest["cleanup"] = report
            params = torch.nn.ParameterDict({
                key: torch.nn.Parameter(value) for key, value in load_model(cleanup / "selected.ply", device).items()
            })
        stage_manifest = _fit_stage(
            stage, params, cameras, footprints, target, normalizer, calibration_points, geometry,
            args.output / "stages" / stage, inherited_gain, completed_steps,
        )
        completed_steps += STAGE_STEPS[stage]
        inherited_gain = stage_manifest["gain"]
        manifest["stages"][stage] = stage_manifest
        manifest.update(completed_steps=completed_steps, total_completed_optimization_steps=completed_steps)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("point_cloud.ply", "checkpoint.pt"):
        shutil.copy2(args.output / "stages/refit" / name, args.output / name)
    manifest.update(status="complete", points=len(params["means"]), gain=inherited_gain,
                    final_sensor_mse=stage_manifest["final_sensor_mse"],
                    allocated_sh_degree=math.isqrt(params["shN"].shape[1] + 1) - 1)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from multiplexed.render import export_run
    export_run(args.output, args.output / "export")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="train.py multiplexed", description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Prepared input folder containing measurement.png and calibration.npz.",
    )
    parser.add_argument("--output", type=Path, required=True)
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
