"""2D Gaussian surfels: conversion from 3D Gaussians, regularizers and in-plane densification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove
from gsplat.utils import normalized_quat_to_rotmat
from scipy.spatial.transform import Rotation
from torch import Tensor

from multiplexed.sensor import Params

# Split children shrink by this factor along the split axes, as in gsplat's split.
SPLIT_SHRINK = 1.6


def surface_regularizers(info: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """Depth distortion loss and alpha-weighted normal consistency loss of a 2DGS render."""
    direction = F.normalize(info["normals"], dim=-1, eps=1e-6)
    cosine = (direction * info["depth_normals"]).sum(dim=-1).clamp(-1, 1)
    normal = (info["alpha"].detach().squeeze(-1) * (1 - cosine)).mean()
    return info["distortion"].mean(), normal


def convert_to_surface(params: Params) -> dict[str, Tensor]:
    """Reorder each Gaussian's axes so the two largest come first, keeping its covariance."""
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()[:, [1, 2, 3, 0]].astype(np.float64)
    rotation = Rotation.from_quat(quats).as_matrix()
    order = np.argsort(-scales, axis=1, kind="stable")
    reordered = np.take_along_axis(rotation, order[:, None, :], axis=2)
    reordered[np.linalg.det(reordered) < 0, :, 2] *= -1
    converted = {name: value.detach().clone() for name, value in params.items()}
    converted["scales"] = torch.as_tensor(
        np.take_along_axis(scales, order, axis=1), device=params["scales"].device
    )
    converted["quats"] = torch.as_tensor(
        Rotation.from_matrix(reordered).as_quat()[:, [3, 0, 1, 2]],
        dtype=params["quats"].dtype,
        device=params["quats"].device,
    )
    return converted


@torch.no_grad()
def split_tangent(
    params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
    mask: Tensor,
) -> None:
    """Replace each selected surfel with two children sampled in its tangent plane."""
    selected = torch.where(mask)[0]
    remaining = torch.where(~mask)[0]
    radii = params["scales"][selected, :2].exp()
    rotation = normalized_quat_to_rotmat(F.normalize(params["quats"][selected], dim=-1))
    samples = torch.randn((2, len(selected), 2), device=radii.device, dtype=radii.dtype)
    offsets = torch.einsum("nij,bnj->bni", rotation[:, :, :2], samples * radii)

    def param_fn(name: str, value: Tensor) -> Tensor:
        if name == "means":
            children = (value[selected] + offsets).reshape(-1, 3)
        elif name == "scales":
            children = value[selected].clone()
            children[:, :2] = (radii / SPLIT_SHRINK).log()
            children = children.repeat(2, 1)
        else:
            children = value[selected].repeat([2] + [1] * (value.ndim - 1))
        return torch.nn.Parameter(
            torch.cat((value[remaining], children)), requires_grad=value.requires_grad
        )

    def optimizer_fn(_key: str, value: Tensor) -> Tensor:
        return torch.cat((value[remaining], value.new_zeros((2 * len(selected), *value.shape[1:]))))

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for name, value in state.items():
        if isinstance(value, Tensor):
            children = value[selected].repeat([2] + [1] * (value.ndim - 1))
            state[name] = torch.cat((value[remaining], children))


class SurfaceStrategy(DefaultStrategy):
    """DefaultStrategy that sizes surfels by their two tangent axes, splits them in-plane
    and prunes only by opacity."""

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
        size = params["scales"][:, :2].exp().max(dim=-1).values
        small = size <= self.grow_scale3d * state["scene_scale"]
        to_duplicate = high & small
        to_split = high & ~small
        if step < self.refine_scale2d_stop_iter:
            to_split |= state["radii"] > self.grow_scale2d
        n_duplicate, n_split = int(to_duplicate.sum()), int(to_split.sum())
        if n_duplicate:
            duplicate(params, optimizers, state, to_duplicate)
        # Duplicates are appended after the existing surfels and are never split.
        to_split = torch.cat((to_split, to_split.new_zeros(n_duplicate)))
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
