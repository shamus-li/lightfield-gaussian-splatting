from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from utils.colmap import link_file
from utils.runtime import REPO, run_in_env

MEGASAM = REPO / "submodules/mega-sam"
MODELS = REPO / "models"
PYTHON_PATHS = (
    MEGASAM / "UniDepth",
    MEGASAM / "base",
    MEGASAM / "base/droid_slam",
    MEGASAM / "base/thirdparty/lietorch",
    MEGASAM / "cvd_opt/core",
)


def run_megasam(frames_dir: Path, workdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depths [F, h, w], camera-to-world poses [F, 4, 4] and intrinsics [3, 3] of a video's frames.

    MegaSaM's scripts write relative to their working directory, which is deleted afterwards.
    """
    scene = frames_dir.name
    steps = [
        (
            "Depth-Anything/run_videos.py",
            [
                "--encoder",
                "vitl",
                "--load-from",
                MODELS / "mega-sam/depth-anything/checkpoints/depth_anything_vitl14.pth",
                "--localhub",
                "--img-path",
                frames_dir,
                "--outdir",
                workdir / "Depth-Anything" / scene,
            ],
        ),
        (
            "UniDepth/scripts/demo_mega-sam.py",
            ["--img-path", frames_dir, "--scene-name", scene, "--outdir", workdir / "UniDepth"],
        ),
        (
            "camera_tracking_scripts/test_demo.py",
            [
                "--datapath",
                frames_dir,
                "--weights",
                MODELS / "mega-sam/megasam_final.pth",
                "--scene_name",
                scene,
                "--mono_depth_path",
                workdir / "Depth-Anything",
                "--metric_depth_path",
                workdir / "UniDepth",
                "--disable_vis",
            ],
        ),
        (
            "cvd_opt/preprocess_flow.py",
            [
                "--datapath",
                frames_dir,
                "--model",
                MODELS / "dycheck/raft-things.pth",
                "--scene_name",
                scene,
                "--mixed_precision",
            ],
        ),
        ("cvd_opt/cvd_opt.py", ["--scene_name", scene, "--output_dir", workdir / "outputs_cvd"]),
    ]
    # Depth-Anything's --localhub loads DINOv2 from torchhub/ in the working directory.
    link_file(REPO / "submodules/dinov2", workdir / "torchhub/facebookresearch_dinov2_main")
    for script, args in steps:
        run_in_env(
            "video",
            MEGASAM / script,
            args,
            cwd=workdir,
            python_paths=PYTHON_PATHS,
            env_extra={"HF_HOME": str(MODELS / "huggingface")},
        )
    with np.load(workdir / "outputs_cvd" / f"{scene}_sgd_cvd_hr.npz") as result:
        depths = result["depths"].astype(np.float32)
        camtoworlds = result["cam_c2w"].astype(np.float64)
        K = result["intrinsic"].astype(np.float64)
    shutil.rmtree(workdir)
    return depths, camtoworlds, K
