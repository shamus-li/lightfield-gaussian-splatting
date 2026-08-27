from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap  # ty: ignore[unresolved-import]

from utils.camera import image_camtoworld


def _mean_up(camtoworlds: np.ndarray) -> np.ndarray:
    ups = camtoworlds[:, :3, 1]
    norms = np.linalg.norm(ups, axis=1, keepdims=True)
    ups = ups / norms
    mean = ups.mean(axis=0)
    return mean / np.linalg.norm(mean)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 0.9999:
        return np.eye(3, dtype=np.float64)
    if c < -0.9999:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(v, v)
    s = np.linalg.norm(v)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s**2))


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    transformed = np.einsum("nij,ki->nkj", camtoworlds.copy(), matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    return transformed


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def similarity_from_cameras(camtoworlds: np.ndarray) -> np.ndarray:
    t = camtoworlds[:, :3, 3]
    rotations = camtoworlds[:, :3, :3]
    ups = np.sum(rotations * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up_norm = float(np.linalg.norm(world_up))
    world_up /= world_up_norm

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        rotation_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        rotation_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotations = rotation_align @ rotations
    forwards = np.sum(rotations * np.array([0.0, 0.0, 1.0]), axis=-1)
    translated = (rotation_align @ t[..., None])[..., 0]
    nearest = translated + (forwards * -translated).sum(-1)[:, None] * forwards
    translate = -np.median(nearest, axis=0)

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = rotation_align
    camera_distances = np.linalg.norm(translated + translate, axis=-1)
    denom = float(np.median(camera_distances))
    transform[:3, :] *= 1.0 / denom
    return transform


def align_principal_axes(points: np.ndarray) -> np.ndarray:
    centroid = np.median(points, axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sort_indices = _eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def _load_alignment_inputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reconstruction = pycolmap.Reconstruction(str(path / "sparse"))
    images = sorted(reconstruction.images.values(), key=lambda image: str(image.name))
    camtoworlds = [image_camtoworld(image) for image in images]
    camtoworld_array = np.stack(camtoworlds, axis=0)
    transform1 = similarity_from_cameras(camtoworld_array)
    camtoworld_array = transform_cameras(transform1, camtoworld_array)

    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
    points = transform_points(transform1, points)
    transform2 = align_principal_axes(points)
    camtoworld_array = transform_cameras(transform2, camtoworld_array)
    transform = transform2 @ transform1
    return transform, camtoworld_array


def compute_alignment(
    train_dir: Path,
    subset_dir: Path,
) -> np.ndarray:
    base, train_camtoworlds = _load_alignment_inputs(train_dir)
    support, subset_camtoworlds = _load_alignment_inputs(subset_dir)
    align = base @ np.linalg.inv(support)

    train_up = _mean_up(train_camtoworlds)
    eval_up = _mean_up(transform_cameras(align, subset_camtoworlds))
    cosine = float(np.dot(train_up, eval_up))
    if cosine < 0.0:
        rot = _rotation_between(eval_up, train_up)
        rot4 = np.eye(4, dtype=np.float64)
        rot4[:3, :3] = rot
        align = rot4 @ align
        print(f"[align] Detected flipped up vector (cos={cosine:.3f}); applying correction.")
    return align @ support


def write_alignment(train_dir: Path, test_dir: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, compute_alignment(train_dir, test_dir))
    return output
