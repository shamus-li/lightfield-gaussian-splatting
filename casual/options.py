from __future__ import annotations

import argparse
from pathlib import Path


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        metavar="DATA",
        type=Path,
        help="Downloaded scene directory containing <modality>-train/ and <modality>-eval/.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Directory for prepared data, checkpoints, renders, and metrics.",
    )
    parser.add_argument(
        "--modality",
        choices=("iphone", "stereo"),
        default="iphone",
        help="Capture type (default: iphone).",
    )
    parser.add_argument(
        "--method",
        choices=("monocular", "multiview"),
        default="multiview",
        help=(
            "Use one camera stream or every synchronized stream "
            "(training default: multiview; loaded from the run during evaluation)."
        ),
    )
    parser.add_argument(
        "--initializer",
        choices=("colmap", "vggt"),
        default="colmap",
        help=(
            "Camera-pose initializer "
            "(training default: colmap; loaded from the run during evaluation)."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the completed run in RESULT_DIR instead of training.",
    )


def camera_match(modality: str, method: str) -> str:
    if method == "multiview":
        return ""
    return "wide" if modality == "iphone" else "right"
