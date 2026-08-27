from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dynamic.eval import evaluate_dynamic_renders, load_iteration
from dynamic.megasam import prepare_model as prepare_megasam_model
from dynamic.model import generate_config, run
from dynamic.prepare import camera_frames, prepare_dynamic_dataset, prepare_vggt_model
from utils.cli import parse_args
from utils.io import write_json
from utils.vggt import run_vggt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py dynamic",
        description="Train, render, or evaluate the dynamic model from video captures.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene containing iphone/ and stereo/ captures.",
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--modality", choices=("iphone", "stereo"), default="iphone")
    parser.add_argument("--method", choices=("multiview", "monocular"), default="multiview")
    parser.add_argument("--initializer", choices=("vggt", "megasam"), default="megasam")
    parser.add_argument("--eval", action="store_true")
    args = parse_args(parser, argv)

    result_dir = args.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = result_dir / "prepared"
    model = result_dir / "model"
    modality = args.modality

    if args.eval:
        modality = (
            "iphone"
            if any("wide" in name.lower() for name in camera_frames(dataset_dir))
            else "stereo"
        )
        shutil.rmtree(model / "test")
    else:
        scene_dir = args.data.expanduser().resolve()
        videos = sorted(
            path for path in (scene_dir / modality).iterdir() if path.suffix.lower() == ".mov"
        )
        prepare_dynamic_dataset(
            videos=videos,
            output_dir=dataset_dir,
            modality=modality,
        )
        run_vggt(
            dataset_dir / "tmp_vggt",
            "--conf_thres_value",
            "3.0",
            "--stage",
            "both",
            "--use_ba",
            "--vis_thresh",
            "0.05",
            "--min_inlier_per_frame",
            "16",
        )
        dataset_name = f"{args.data.name}_{modality}"
        if args.initializer == "megasam":
            prepare_megasam_model(
                dataset_dir,
                modality=modality,
                dataset_name=dataset_name,
            )
        else:
            prepare_vggt_model(dataset_dir)
        shutil.rmtree(dataset_dir / "tmp_vggt")

        cameras = camera_frames(dataset_dir)
        config = generate_config(
            dataset_name=dataset_name,
            camera_count=len(cameras),
            frame_count=len(next(iter(cameras.values()))),
        )
        if args.method == "monocular":
            iteration = 28_000
            config += (
                "\nOptimizationParams['iterations'] = 28000\n"
                "OptimizationParams['coarse_iterations'] = 6000\n"
            )
        else:
            iteration = int(
                next(
                    line.removeprefix("    iterations=").removesuffix(",")
                    for line in config.splitlines()
                    if line.startswith("    iterations=")
                )
            )
        config += (
            f"test_iterations = [{iteration}]\n"
            f"save_iterations = [{iteration}]\n"
            "checkpoint_iterations = []\n"
        )
        source_config = dataset_dir / "config.py"
        source_config.write_text(config)
        model.mkdir()
        train_args = [
            "--source_path",
            str(dataset_dir),
            "--model_path",
            str(model),
            "--expname",
            f"{args.data.name}_{modality}_{args.method}",
            "--match_string",
            "wide" if modality == "iphone" else "right",
            "--skip_post_eval",
            "--configs",
            str(source_config),
        ]
        if args.method == "monocular":
            train_args.append("--filter_training")
        run("train.py", train_args)
        source_config.unlink()

    iteration = load_iteration(model)
    run(
        "render.py",
        [
            "--model_path",
            str(model),
            "--source_path",
            str(dataset_dir),
            "--iteration",
            str(iteration),
            "--skip_train",
            "--skip_video",
        ],
    )
    metrics = evaluate_dynamic_renders(
        model,
        dataset_dir=dataset_dir,
        match_string="wide" if modality == "iphone" else "right",
        iteration=iteration,
    )
    write_json(result_dir / "metrics.json", metrics)
