from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import torch
from torchvision.utils import save_image

from utils.camera import CameraView
from utils.io import write_json
from utils.metrics import get_lpips_model, psnr, ssim_window
from utils.render import RenderConfig, render_item
from synthetic.config import TrainConfig


def evaluate_synthetic(
    *,
    gaussians: torch.nn.ParameterDict,
    cameras: Sequence[CameraView],
    cfg: TrainConfig,
    run_dir: Path,
) -> None:
    split = "adjacent_test_camera" if cfg.num_exposures == 1 else "full_test_camera"
    render_config = RenderConfig(
        sh_degree=cfg.sh_degree,
        antialiased=False,
        packed=True,
    )
    device = gaussians["means"].device
    render_dir = run_dir / "renders" / split
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)

    lpips = get_lpips_model("vgg", device, spatial=False)
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            rendered, _ = render_item(
                gaussians,
                camera,
                cfg=render_config,
            )
            prediction = rendered.permute(2, 0, 1).clamp(0.0, 1.0)
            target = camera.image.to(device=device, dtype=torch.float32).permute(2, 0, 1)
            save_image(prediction.cpu(), render_dir / f"{index:04d}.png")
            prediction_batch = prediction.unsqueeze(0)
            target_batch = target.unsqueeze(0)
            values = {
                "psnr": float(psnr(prediction_batch, target_batch).item()),
                "ssim": float(ssim_window(prediction_batch, target_batch).item()),
                "lpips": float(
                    lpips(
                        prediction_batch * 2.0 - 1.0,
                        target_batch * 2.0 - 1.0,
                    )
                    .mean()
                    .item()
                ),
            }
            for name in totals:
                totals[name] += values[name]

    mean = {name: value / len(cameras) for name, value in totals.items()}
    print(
        f"Evaluating {split.replace('_', ' ')}: "
        f"PSNR {mean['psnr']:.4f} "
        f"SSIM {mean['ssim']:.4f} LPIPS {mean['lpips']:.4f}"
    )
    write_json(run_dir / "metrics.json", mean)
