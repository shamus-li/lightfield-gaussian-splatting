from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pycolmap
import tyro

from static.covisible import write_covisible_masks
from utils.colmap import link_file, sparse_subset
from utils.io import read_image, read_yaml, write_config, write_json
from utils.metrics import compute_pair_metrics, summarize_metric_rows
from utils.runtime import REPO, run_in_env

ITERATION = 30_000
SUBMODULE = {"fsgs": "FSGS", "sparsegs": "SparseGS"}
TRAIN_FLAGS = {
    "fsgs": [
        "--n_views", "0",
        "--position_lr_max_steps", str(ITERATION),
        "--densify_until_iter", "700",
        "--densify_grad_threshold", "0.0005",
        "--prune_threshold", "0.005",
    ],
    "sparsegs": [
        "--no_load_depth",
        "--opacity_reset_interval", str(ITERATION),
        "--lambda_local_pearson", "0.0",
        "--lambda_pearson", "0.0",
        "--box_p", "128",
        "--p_corr", "0.5",
        "--prune_exp", "7.0",
        "--prune_perc", "0.98",
        "--densify_lag", "1000000",
        "--power_thresh", "-4.0",
        "--densify_period", "5000",
        "--step_ratio", "0.95",
        "--lambda_diffusion", "0.0",
        "--SDS_freq", "0.1",
        "--lambda_reg", "0.0",
        "--warp_reg_start_itr", "4999",
        "--beta", "5.0",
    ],
}  # fmt: skip
# SparseGS trains the three iPhone lenses together, anchored on the wide lens.
SPARSEGS_IPHONE_FLAGS = [
    "--train_all_cameras",
    "--anchor_camera_name", "iphone__wide.png",
    "--aux_camera_loss_weight", "0.02",
]  # fmt: skip
# The FSGS fork renders the eval scene as its training split.
RENDER_FLAGS = {"fsgs": ["--skip_test"], "sparsegs": ["--no_load_depth"]}
RENDER_DIR = {
    "fsgs": Path(f"train/ours_{ITERATION}/renders"),
    "sparsegs": Path(f"renders/ours_{ITERATION}/renders"),
}


@dataclass(frozen=True)
class Config:
    # Baseline method; --eval reads it from the run's cfg.yml.
    method: tyro.conf.Positional[Literal["fsgs", "sparsegs"]]
    # Scene directory containing static/shared/ from prepare-static.
    data: Path
    # Run directory for the baseline's outputs, renders/ and metrics.json.
    result_dir: Path
    # Input camera design; --eval reads it from the run's cfg.yml.
    camera_model: Literal["monocular", "iphone", "stereo", "lightfield"] = "monocular"
    # Render and score the existing run in result-dir instead of training.
    eval: bool = False


def coordinate_scale(reconstruction: pycolmap.Reconstruction, names: set[str]) -> float:
    """Scale that lifts a median point depth below 0.5 to one, clear of the 0.2 near plane."""
    xyz = np.array([point.xyz for point in reconstruction.points3D.values()])
    xyz = xyz[:: int(np.ceil(len(xyz) / 100_000))]
    depths = []
    for image in reconstruction.images.values():
        if image.name in names:
            world_to_camera = image.cam_from_world()
            depth = xyz @ world_to_camera.rotation.matrix()[2] + world_to_camera.translation[2]
            depths.append(depth[depth > 0.0])
    median_depth = float(np.median(np.concatenate(depths)))
    return 1.0 if median_depth >= 0.5 else 1.0 / median_depth


def write_fsgs_bounds(reconstruction: pycolmap.Reconstruction, path: Path) -> None:
    """Write LLFF poses_bounds.npy for the registered images, sorted by name."""
    xyz = np.array([point.xyz for point in reconstruction.points3D.values()])
    rows = []
    for image in sorted(reconstruction.images.values(), key=lambda image: image.name):
        camera = reconstruction.cameras[image.camera_id]
        world_to_camera = np.eye(4)
        world_to_camera[:3] = image.cam_from_world().matrix()
        depths = xyz @ world_to_camera[2, :3] + world_to_camera[2, 3]
        depths = depths[depths > 1e-4]
        near = max(0.01, np.percentile(depths, 0.5))
        far = max(near + 1e-3, np.percentile(depths, 99.5))
        hwf = [[camera.height], [camera.width], [camera.params[0]]]
        pose = np.concatenate([np.linalg.inv(world_to_camera)[:3], hwf], axis=1)
        rows.append(np.concatenate([pose.reshape(-1), [near, far]]))
    np.save(path, np.stack(rows).astype(np.float32))


def write_dataset(
    source: pycolmap.Reconstruction,
    image_dir: Path,
    names: list[str],
    destination: Path,
    *,
    scale: float,
    method: str,
) -> None:
    """Write a 3DGS-style dataset of `names` with every point of `source`, scaled by `scale`."""
    for name in names:
        link_file(image_dir / name, destination / "images" / name)
    reconstruction = sparse_subset(source, names, all_points=True)
    reconstruction.transform(pycolmap.Sim3d(scale=scale))
    sparse = destination / "sparse/0"
    sparse.mkdir(parents=True)
    reconstruction.write(str(sparse))
    reconstruction.export_PLY(str(sparse / "points3D.ply"))
    if method == "fsgs":
        link_file(sparse / "points3D.ply", destination / "0_views/dense/fused.ply")
        write_fsgs_bounds(reconstruction, destination / "poses_bounds.npy")


def train(cfg: Config, shared: Path) -> None:
    source = pycolmap.Reconstruction(str(shared / "sparse"))
    train_names = sorted(
        path.name for path in (shared / "subsets" / cfg.camera_model / "images").iterdir()
    )
    test_names = sorted(path.name for path in (shared / "subsets/iphone_eval/images").iterdir())
    scale = coordinate_scale(source, set(train_names))
    for split, names in (("train", train_names), ("eval", test_names)):
        write_dataset(
            source,
            shared / "images",
            names,
            cfg.result_dir / "data" / split,
            scale=scale,
            method=cfg.method,
        )
    flags = TRAIN_FLAGS[cfg.method]
    if cfg.method == "sparsegs" and cfg.camera_model == "iphone":
        flags = flags + SPARSEGS_IPHONE_FLAGS
    root = REPO / "submodules" / SUBMODULE[cfg.method]
    run_in_env(
        cfg.method,
        root / "train.py",
        [
            "--source_path", cfg.result_dir / "data/train",
            "--model_path", cfg.result_dir,
            "--images", "images",
            "--resolution", "1",
            "--iterations", ITERATION,
            "--test_iterations", ITERATION,
            "--save_iterations", ITERATION,
            # Past the last iteration, so no training checkpoint is written.
            "--checkpoint_iterations", ITERATION + 1,
            *flags,
            "--quiet",
        ],
        cwd=root,
    )  # fmt: skip


def evaluate(cfg: Config, shared: Path) -> dict[str, float]:
    """Render the eval images with the trained baseline, move them to renders/ and score them."""
    root = REPO / "submodules" / SUBMODULE[cfg.method]
    eval_dir = cfg.result_dir / "data/eval"
    render_dir = cfg.result_dir / "renders"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    run_in_env(
        cfg.method,
        root / "render.py",
        [
            "--source_path", eval_dir,
            "--model_path", cfg.result_dir,
            "--images", "images",
            "--resolution", "1",
            *RENDER_FLAGS[cfg.method],
            "--iteration", ITERATION,
            "--quiet",
        ],
        cwd=root,
    )  # fmt: skip
    rendered = cfg.result_dir / RENDER_DIR[cfg.method]
    rendered.rename(cfg.result_dir / "renders_tmp")
    shutil.rmtree(cfg.result_dir / RENDER_DIR[cfg.method].parts[0])
    (cfg.result_dir / "renders_tmp").rename(render_dir)

    mask_dir = cfg.result_dir / "covisible"
    write_covisible_masks(
        shared / "subsets/iphone_eval", shared / "subsets" / cfg.camera_model, mask_dir
    )
    rows = []
    for gt_path in sorted((eval_dir / "images").iterdir()):
        name = gt_path.with_suffix(".png").name
        rows.append(
            compute_pair_metrics(
                read_image(render_dir / name, "RGB").astype(np.float32) / 255.0,
                read_image(gt_path, "RGB").astype(np.float32) / 255.0,
                mask=read_image(mask_dir / name, "L") > 127,
            )
        )
    shutil.rmtree(mask_dir)
    return summarize_metric_rows(rows)


def main(argv: Sequence[str]) -> None:
    cfg = tyro.cli(
        Config,
        args=argv,
        prog="train.py baseline",
        description="Train and evaluate FSGS or SparseGS on a real static scene.",
    )
    cfg = replace(cfg, data=cfg.data.resolve(), result_dir=cfg.result_dir.resolve())
    if cfg.eval:
        saved = read_yaml(cfg.result_dir / "cfg.yml")
        cfg = replace(cfg, method=saved["method"], camera_model=saved["camera_model"])
    shared = cfg.data / "static/shared"
    if not cfg.eval:
        cfg.result_dir.mkdir(parents=True)
        write_config(cfg.result_dir / "cfg.yml", cfg)
        train(cfg, shared)
    metrics = evaluate(cfg, shared)
    write_json(cfg.result_dir / "metrics.json", metrics)
    print(metrics)
