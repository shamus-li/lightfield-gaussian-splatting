from __future__ import annotations

import argparse
import shutil
from typing import Sequence

from casual.options import add_capture_arguments, camera_match
from casual.prepare import prepare_scene
from utils.checkpoints import select_model
from static.alignment import write_alignment
from static.covisible import write_covisible_masks
from static.train import StaticConfig, run
from utils.cli import parse_args
from utils.io import read_yaml


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py casual",
        description="Prepare, train, and evaluate a Gaussian model on a casual video capture.",
    )
    add_capture_arguments(parser)
    args = parse_args(parser, argv)
    if args.eval:
        config = read_yaml(args.result_dir.expanduser().resolve() / "cfg.yml")
        args.modality = config["camera_model"]
        args.method = "monocular" if config["match"] else "multiview"
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
    train_dir = prepared / "subsets" / "train"
    test_dir = prepared / "subsets" / "test"
    match = camera_match(args.modality, args.method)

    if not args.eval:
        train_config = StaticConfig(
            data_dir=train_dir,
            result_dir=result_dir,
            camera_model=args.modality,
            match=match,
            max_steps=30_000,
            refine_stop_iter=26_000,
            test_every=8,
            lpips_net="alex",
        )
        run(train_config)
        checkpoint = select_model(result_dir)
    alignment = write_alignment(
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
    eval_config = StaticConfig(
        data_dir=test_dir,
        result_dir=result_dir,
        camera_model=args.modality,
        checkpoint=checkpoint,
        alignment=alignment,
        covisible_dir=covisible_dir,
        max_steps=30_000,
        test_every=1,
        lpips_net="alex",
    )
    run(eval_config)
    shutil.rmtree(result_dir / "alignments")
    shutil.rmtree(result_dir / "covisible")
