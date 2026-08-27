from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from static import baseline_data
from static.covisible import write_covisible_masks
from utils.metrics import (
    compute_pair_metrics,
    summarize_metric_rows,
)
from utils.cli import parse_args
from utils.io import read_image, write_json
from utils.runtime import torch_env


def finish_baseline_run(
    method: str,
    output_dir: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "submodules" / {"fsgs": "FSGS", "sparsegs": "SparseGS"}[method]
    iteration = 30_000
    render_command = [
        str(repo / ".envs" / method / "bin/python"),
        str(root / "render.py"),
        "--source_path",
        str(output_dir / "data/eval"),
        "--model_path",
        str(output_dir),
        "--images",
        "images",
    ]
    if method == "fsgs":
        render_command.extend(("--skip_test", "--resolution", "1"))
    else:
        render_command.extend(("--no_load_depth", "--resolution", "1"))
    render_command.extend(("--iteration", str(iteration), "--quiet"))
    print(shlex.join(render_command))
    subprocess.run(render_command, cwd=root, env=torch_env(), check=True)
    renders_dir = (
        output_dir / "train" / f"ours_{iteration}" / "renders"
        if method == "fsgs"
        else output_dir / "renders" / f"ours_{iteration}" / "renders"
    )
    rows: list[dict[str, float]] = []
    for ground_truth in sorted((output_dir / "data/eval/images").iterdir()):
        name = ground_truth.with_suffix(".png").name
        prediction = read_image(renders_dir / name, "RGB").astype(np.float32) / 255.0
        target = read_image(ground_truth, "RGB").astype(np.float32) / 255.0
        mask = read_image(output_dir / "covisible/1x/val" / name, "L") > 127
        rows.append(compute_pair_metrics(prediction, target, mask=mask, lpips_net="vgg"))
    metrics = summarize_metric_rows(rows)
    write_json(output_dir / "metrics.json", metrics)


def train_baseline(
    method: str,
    data_root: Path,
    camera_model: str,
    *,
    output_dir: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "submodules" / {"fsgs": "FSGS", "sparsegs": "SparseGS"}[method]

    data_dir = data_root.expanduser().resolve() / "static" / "shared"
    output_dir.mkdir()
    train_list = data_dir / "splits" / f"{camera_model}.txt"
    eval_camera = "iphone" if camera_model == "monocular" else camera_model
    eval_list = data_dir / "splits" / f"{eval_camera}_eval.txt"
    coordinate_scale = baseline_data.baseline_coordinate_scale(data_dir, train_list)
    write_covisible_masks(
        data_dir / "subsets" / f"{eval_camera}_eval",
        data_dir / "subsets" / camera_model,
        output_dir / "covisible",
        support_test_every=1,
    )
    baseline_data.prepare_baseline_dataset(
        method=method,
        source_path=data_dir,
        split_path=train_list,
        destination=output_dir / "data/train",
        coordinate_scale=coordinate_scale,
        training=True,
    )
    baseline_data.prepare_baseline_dataset(
        method=method,
        source_path=data_dir,
        split_path=eval_list,
        destination=output_dir / "data/eval",
        coordinate_scale=coordinate_scale,
        training=False,
    )
    iteration = 30_000
    command = [
        str(repo / ".envs" / method / "bin/python"),
        str(root / "train.py"),
        "--source_path",
        str(output_dir / "data/train"),
        "--model_path",
        str(output_dir),
        "--images",
        "images",
        "--resolution",
        "1",
        "--iterations",
        str(iteration),
        "--test_iterations",
        str(iteration),
        "--save_iterations",
        str(iteration),
        "--checkpoint_iterations",
        str(iteration + 1),
    ]
    if method == "fsgs":
        command.extend(
            (
                "--n_views",
                "0",
                "--position_lr_max_steps",
                str(iteration),
                "--densify_until_iter",
                "700",
                "--densify_grad_threshold",
                "0.0005",
                "--prune_threshold",
                "0.005",
            )
        )
    else:
        command.extend(
            (
                "--no_load_depth",
                "--opacity_reset_interval",
                str(iteration),
                "--lambda_local_pearson",
                "0.0",
                "--lambda_pearson",
                "0.0",
                "--box_p",
                "128",
                "--p_corr",
                "0.5",
                "--prune_exp",
                "7.0",
                "--prune_perc",
                "0.98",
                "--densify_lag",
                "1000000",
                "--power_thresh",
                "-4.0",
                "--densify_period",
                "5000",
                "--step_ratio",
                "0.95",
                "--lambda_diffusion",
                "0.0",
                "--SDS_freq",
                "0.1",
                "--lambda_reg",
                "0.0",
                "--warp_reg_start_itr",
                "4999",
                "--beta",
                "5.0",
            )
        )
        if camera_model == "iphone":
            command.extend(
                (
                    "--train_all_cameras",
                    "--anchor_camera_name",
                    "iphone__wide.png",
                    "--aux_camera_loss_weight",
                    "0.02",
                )
            )
    command.append("--quiet")
    print(shlex.join(command))
    subprocess.run(command, cwd=root, env=torch_env(), check=True)
    finish_baseline_run(method, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py baseline",
        description="Train and evaluate FSGS or SparseGS on a static scene.",
    )
    parser.add_argument(
        "method",
        choices=("fsgs", "sparsegs"),
        help="Baseline to train or evaluate.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory containing static/shared/; required for training.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Run outputs and metrics.",
    )
    parser.add_argument(
        "--camera-model",
        choices=("monocular", "iphone", "stereo", "lightfield"),
        default="monocular",
        help="Input camera design (default: monocular).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Recompute metrics from renders recorded by an existing baseline run.",
    )
    args = parse_args(parser, argv)
    output_dir = args.result_dir.expanduser().resolve()
    if args.eval:
        finish_baseline_run(args.method, output_dir)
        return
    train_baseline(
        args.method,
        args.data,
        args.camera_model,
        output_dir=output_dir,
    )
