from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

from casual.options import (
    add_capture_arguments,
    camera_match,
)
from casual.prepare import prepare_scene
from utils.checkpoints import (
    select_model,
    validation_steps,
)
from utils.metrics import (
    compute_pair_metrics,
    summarize_metric_rows,
)
from static.alignment import write_alignment
from static.covisible import write_covisible_masks
from utils.cli import parse_args
from utils.io import read_image, read_json, read_lines, write_json
from utils.runtime import torch_env


def difix_env() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    environment = torch_env(repo / "submodules/Difix3D", repo)
    environment["HF_HOME"] = str(repo / "models/huggingface")
    return environment


def train_difix(
    data_root: Path,
    result_dir: Path,
    *,
    modality: str,
    method: str,
) -> None:
    match = camera_match(modality, method)
    repo = Path(__file__).resolve().parents[1]
    difix_root = repo / "submodules/Difix3D"
    trainer = difix_root / "examples" / "gsplat" / "simple_trainer_difix3d.py"
    data_dir = data_root.expanduser().resolve() / "subsets" / "train"
    result_dir = result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    steps = validation_steps(30_000)
    command = [
        str(repo / ".envs/difix3d/bin/python"),
        str(trainer),
        "default",
        "--disable_viewer",
        "--data_factor",
        "1",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--pose_opt",
        "--antialiased",
        "--max_steps",
        "30000",
        "--test_every",
        "8",
        "--strategy.reset_every",
        "100000",
        "--strategy.pause_refine_after_reset",
        "0",
        "--strategy.prune_scale3d",
        "0.22",
        "--strategy.prune_scale2d",
        "0.12",
        "--strategy.prune_opa",
        "0.006",
        "--strategy.grow_grad2d",
        "0.00035",
        "--strategy.grow_scale3d",
        "0.012",
        "--strategy.refine_stop_iter",
        "26000",
        "--strategy.refine_scale2d_stop_iter",
        "26000",
        "--scale_reg",
        "0.0005",
        "--lpips_net",
        "alex",
        "--eval_steps",
        *(str(step) for step in steps),
        "--save_steps",
        *(str(step) for step in steps),
    ]
    if match:
        command.extend(("--match_string", match))
    config = {
        "method": method,
        "modality": modality,
    }
    write_json(result_dir / "difix.json", config)
    subprocess.run(
        command,
        cwd=difix_root,
        env=difix_env(),
        check=True,
    )


def evaluate_difix(
    data_root: Path,
    out_dir: Path,
    *,
    ckpt_path: Path,
    covisible_dir: Path,
) -> None:
    data_root = data_root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    eval_dir = data_root / "subsets" / "test"
    eval_list = data_root / "splits" / "eval.txt"
    alignment_path = out_dir / "alignments" / "test_to_train.npy"
    covisible_dir = covisible_dir.expanduser().resolve()
    evaluation_dir = out_dir / "evaluation"
    repo = Path(__file__).resolve().parents[1]
    difix_root = repo / "submodules/Difix3D"
    trainer_script = difix_root / "examples" / "gsplat" / "simple_trainer_difix3d.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / "renders")
    command = [
        str(repo / ".envs/difix3d/bin/python"),
        str(trainer_script),
        "default",
        "--disable_viewer",
        "--data_factor",
        "1",
        "--data_dir",
        str(eval_dir),
        "--result_dir",
        str(evaluation_dir),
        "--ckpt",
        str(ckpt_path),
        "--test_every",
        "1",
        "--max_steps",
        "30000",
        "--eval_only",
        "--lpips_net",
        "alex",
        "--dataset_transform_path",
        str(alignment_path),
        "--pose_opt",
        "--antialiased",
        "--camera-model",
        "pinhole",
    ]
    print("Executing:", shlex.join(command))
    subprocess.run(
        command,
        env=difix_env(),
        check=True,
    )
    render_dir = next((evaluation_dir / "renders" / "val").iterdir())
    eval_names = sorted(read_lines(eval_list))
    rows: list[dict[str, float]] = []
    for name in eval_names:
        image_name = Path(name).with_suffix(".png").name
        pred = read_image(render_dir / "Pred" / image_name, "RGB").astype(np.float32) / 255.0
        gt = read_image(render_dir / "GT" / image_name, "RGB").astype(np.float32) / 255.0
        mask = read_image(covisible_dir / Path(name).with_suffix(".png"), "L")
        rows.append(compute_pair_metrics(pred, gt, mask=mask, lpips_net="alex"))
    metrics = summarize_metric_rows(rows)
    write_json(out_dir / "metrics.json", metrics)
    (render_dir / "Pred").rename(out_dir / "renders")
    shutil.rmtree(evaluation_dir)
    print(f"Wrote metrics: {out_dir / 'metrics.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py difix",
        description="Prepare, train, and evaluate Difix3D+ on a casual video capture.",
    )
    add_capture_arguments(parser)
    args = parse_args(parser, argv)
    if args.eval:
        config = read_json(args.result_dir.expanduser().resolve() / "difix.json")
        args.modality = config["modality"]
        args.method = config["method"]
    result_dir = args.result_dir.expanduser().resolve()
    if args.eval:
        prepared = result_dir / "prepared"
        checkpoint = result_dir / "model.pt"
    else:
        prepared = prepare_scene(
            args.data,
            result_dir / "prepared",
            modality=args.modality,
            initializer=args.initializer,
        )
        train_difix(
            prepared,
            result_dir,
            modality=args.modality,
            method=args.method,
        )
        checkpoint = select_model(result_dir)
        shutil.rmtree(result_dir / "ply")
    train_dir = prepared / "subsets" / "train"
    test_dir = prepared / "subsets" / "test"
    match = camera_match(args.modality, args.method)
    write_alignment(
        train_dir,
        test_dir,
        result_dir / "alignments" / "test_to_train.npy",
    )
    covisible_dir = write_covisible_masks(
        test_dir,
        train_dir,
        result_dir / "covisible" / "test",
        support_test_every=8,
        support_match=match or None,
    )
    evaluate_difix(
        prepared,
        result_dir,
        ckpt_path=checkpoint,
        covisible_dir=covisible_dir,
    )
    shutil.rmtree(result_dir / "alignments")
    shutil.rmtree(result_dir / "covisible")
