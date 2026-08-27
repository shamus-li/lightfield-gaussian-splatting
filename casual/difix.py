from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from casual.run import HOLDOUT_EVERY, LPIPS_NET, MAX_STEPS, REFINE_STOP_ITER, run
from static.data import ColmapDataset
from static.train import SCALE_REG
from utils.gaussians import STRATEGY
from utils.io import read_image, read_json
from utils.metrics import compute_pair_metrics, summarize_metric_rows
from utils.runtime import REPO, run_in_env

DIFIX = REPO / "submodules/Difix3D"
TRAINER = DIFIX / "examples/gsplat/simple_trainer_difix3d.py"
# Checkpoints scored on the held-out training frames; the best one becomes model.pt.
CHECKPOINT_STEPS = (3_000, 7_000, MAX_STEPS)


def run_difix(args: Sequence[str | Path | float]) -> None:
    run_in_env(
        "video",
        TRAINER,
        ["default", "--disable_viewer", "--data_factor", "1", "--pose_opt", "--antialiased", *args],
        cwd=DIFIX,
        python_paths=[DIFIX, REPO],
        env_extra={"HF_HOME": str(REPO / "models/huggingface")},
    )


def train_difix(result_dir: Path, train_dir: Path, match: str) -> None:
    """Train Difix3D+ and keep the checkpoint with the best held-out PSNR as model.pt."""
    # The Difix trainer writes its own cfg.yml, so it runs in a subdirectory.
    output = result_dir / "difix"
    strategy_flags = [
        arg for key, value in STRATEGY.items() for arg in (f"--strategy.{key}", value)
    ]
    run_difix(
        [
            "--data_dir", train_dir,
            "--result_dir", output,
            "--max_steps", MAX_STEPS,
            "--test_every", HOLDOUT_EVERY,
            *strategy_flags,
            "--strategy.refine_stop_iter", REFINE_STOP_ITER,
            "--strategy.refine_scale2d_stop_iter", REFINE_STOP_ITER,
            "--scale_reg", SCALE_REG,
            "--lpips_net", LPIPS_NET,
            "--eval_steps", *CHECKPOINT_STEPS,
            "--save_steps", *CHECKPOINT_STEPS,
            *(["--match_string", match] if match else []),
        ]
    )  # fmt: skip
    # Stats and checkpoints are named by the zero-based step at which they were written.
    psnr = {
        path.stem.removeprefix("val_step"): read_json(path)["psnr"]
        for path in (output / "stats").glob("val_step*.json")
    }
    step = max(psnr, key=psnr.__getitem__)
    print(f"Selected checkpoint {step} with held-out PSNR {psnr[step]:.3f}")
    (output / "ckpts" / f"ckpt_{int(step)}_rank0.pt").rename(result_dir / "model.pt")
    shutil.rmtree(output)


def evaluate_difix(
    result_dir: Path, train_dir: Path, test_dir: Path, mask_dir: Path
) -> dict[str, float]:
    """Render the test images with Difix3D's evaluator, move them to renders/ and score them."""
    output = result_dir / "difix_eval"
    output.mkdir()
    # Difix normalizes the test cameras by themselves; map them back to the raw frame and then
    # into the model's train-normalized frame.
    test_to_train = ColmapDataset(train_dir).normalize() @ np.linalg.inv(
        ColmapDataset(test_dir).normalize()
    )
    np.save(output / "test_to_train.npy", test_to_train)
    run_difix(
        [
            "--data_dir", test_dir,
            "--result_dir", output,
            "--ckpt", result_dir / "model.pt",
            # 0 selects every image; 1 would split them by name tokens that ours lack.
            "--test_every", "0",
            "--eval_only",
            "--lpips_net", LPIPS_NET,
            "--dataset_transform_path", output / "test_to_train.npy",
        ]
    )  # fmt: skip
    step_dir = next((output / "renders/val").iterdir())
    render_dir = result_dir / "renders"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir()
    names = ColmapDataset(test_dir).image_names
    rows = []
    # Difix writes the i-th test image, in name order, as Pred/ and GT/ <i:04d>.png.
    for index, name in enumerate(names):
        pred_path = step_dir / "Pred" / f"{index:04d}.png"
        pred = read_image(pred_path, "RGB").astype(np.float32) / 255.0
        gt = read_image(step_dir / "GT" / f"{index:04d}.png", "RGB").astype(np.float32) / 255.0
        # gsplat crops undistorted SIMPLE_RADIAL images to OpenCV's ROI, one pixel short of
        # the full size.
        mask = read_image(mask_dir / Path(name).with_suffix(".png"), "L")[
            : gt.shape[0], : gt.shape[1]
        ]
        rows.append(compute_pair_metrics(pred, gt, mask=mask > 127, lpips_net=LPIPS_NET))
        pred_path.rename(render_dir / Path(name).with_suffix(".png"))
    shutil.rmtree(output)
    return summarize_metric_rows(rows)


def main(argv: Sequence[str]) -> None:
    run(
        argv,
        prog="train.py difix",
        description="Prepare, train and evaluate Difix3D+ on a casual video capture.",
        train=train_difix,
        evaluate=evaluate_difix,
    )
