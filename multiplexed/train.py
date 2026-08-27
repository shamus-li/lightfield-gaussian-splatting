from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from multiplexed.prepare import prepare
from utils.cli import parse_args
from utils.runtime import torch_env


def run_model(mode: str, output_dir: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    hardware_dir = repo / "submodules/hardware-gaussians"
    subprocess.run(
        [
            str(repo / ".venv/bin/python"),
            str(hardware_dir / "hardware.py"),
            mode,
            "--data",
            str(output_dir),
        ],
        cwd=hardware_dir,
        env=torch_env(),
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py multiplexed",
        description="Prepare, train, and evaluate the multiplexed capture.",
    )
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded multiplexed dataset.",
    )
    parser.add_argument("--result-dir", type=Path, help="Prepared data and run outputs.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the completed run in RESULT_DIR instead of training.",
    )
    args = parse_args(parser, argv)
    result_dir = args.result_dir.expanduser().resolve()
    if args.eval:
        run_model("eval", result_dir)
        return
    prepare(args.data, result_dir)
    (result_dir / "run").mkdir(parents=True)
    run_model("train", result_dir)
    run_model("eval", result_dir)
