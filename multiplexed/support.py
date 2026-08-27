"""Padded convex hull of the calibration points that bounds every Gaussian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.utils import normalized_quat_to_rotmat
from scipy.spatial import ConvexHull, cKDTree  # ty: ignore[import-not-found]
from sklearn.cluster import DBSCAN
from torch import Tensor

from multiplexed.sensor import Params

# Point spacing is the distance to the 16th nearest neighbor (17 including the point itself).
# DBSCAN keeps points with that many neighbors within SPACING_MULTIPLE median spacings, and
# each hull face is padded by SPACING_MULTIPLE times the median spacing of its vertices.
NEIGHBORS = 17
SPACING_MULTIPLE = 2.0


@dataclass(frozen=True)
class Hull:
    # [F, 4] outward unit normals and offsets; a point x is inside when n.x + d <= 0.
    planes: Tensor
    # [F] distance a Gaussian's 3-sigma extent may extend past each face.
    allowance: Tensor
    center: Tensor


def _spacing(points: np.ndarray) -> np.ndarray:
    return cKDTree(points).query(points, k=NEIGHBORS, workers=4)[0][:, -1]


def build_support(points: Tensor) -> Hull:
    """Padded convex hull of the calibration points that DBSCAN keeps as dense."""
    xyz = points.cpu().numpy().astype(np.float64)
    eps = SPACING_MULTIPLE * float(np.median(_spacing(xyz)))
    dense = xyz[DBSCAN(eps=eps, min_samples=NEIGHBORS).fit_predict(xyz) >= 0]
    hull = ConvexHull(dense)
    allowance = SPACING_MULTIPLE * np.median(_spacing(dense)[hull.simplices], axis=1)
    return Hull(
        *(
            torch.as_tensor(value, dtype=torch.float32, device=points.device)
            for value in (hull.equations, allowance, dense.mean(axis=0))
        )
    )


def face_support(params: Params, planes: Tensor, tangent_only: bool) -> Tensor:
    """[N, F] signed distance from each face to the far edge of each Gaussian's 3-sigma extent.

    Surfels have no extent along their third axis, so `tangent_only` uses the first two.
    """
    dimensions = 2 if tangent_only else 3
    rotation = normalized_quat_to_rotmat(F.normalize(params["quats"], dim=-1))
    local = torch.einsum("nki,fk->nfi", rotation, planes[:, :3])[:, :, :dimensions]
    scales = params["scales"][:, :dimensions].exp()
    radius = 3 * (local.square() * scales.square()[:, None]).sum(-1).sqrt()
    return params["means"] @ planes[:, :3].T + planes[:, 3] + radius


def inside_hull(params: Params, hull: Hull, tangent_only: bool) -> Tensor:
    """[N] whether each Gaussian's 3-sigma extent stays within the padded hull."""
    return (face_support(params, hull.planes, tangent_only) <= hull.allowance).all(dim=1)


@torch.no_grad()
def project_hull_(params: Params, hull: Hull, tangent_only: bool) -> None:
    """Shrink Gaussians that cross the padded hull toward its center, in place."""
    center_value = hull.center @ hull.planes[:, :3].T + hull.planes[:, 3]
    direction_support = face_support(params, hull.planes, tangent_only) - center_value
    factor = (
        torch.where(
            direction_support > 0,
            (hull.allowance - center_value) / direction_support,
            torch.ones_like(direction_support),
        )
        .amin(dim=1)
        .clamp_max(1)
    )
    affected = factor < 1
    # Shrink a little extra so rounding cannot leave a Gaussian outside the hull.
    factor[affected] *= 1 - 8 * torch.finfo(factor.dtype).eps
    selected = factor[affected, None]
    params["means"][affected] = hull.center + selected * (params["means"][affected] - hull.center)
    dimensions = 2 if tangent_only else 3
    params["scales"][affected, :dimensions] += selected.log()
