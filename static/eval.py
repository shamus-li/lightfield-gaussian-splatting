from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from static.data import ColmapDataset
from utils import metrics
from utils.gaussians import rasterize
from utils.io import read_image, write_image


@torch.no_grad()
def evaluate(
    model_path: Path,
    train_dir: Path,
    test_dir: Path,
    mask_dir: Path,
    *,
    lpips_net: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Render every image in test_dir with the model and score it inside its covisible mask."""
    params = torch.load(model_path, map_location="cuda", weights_only=True)
    testset = ColmapDataset(test_dir)
    # Test and train share one COLMAP world frame, so the train normalization maps the test
    # cameras into the model's frame.
    testset.apply_transform(ColmapDataset(train_dir).normalize())
    scores: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    renders: dict[str, np.ndarray] = {}
    for view in testset:
        name = testset.image_names[view.image_id]
        rendered = rasterize(
            params,
            view.camtoworld[None].cuda(),
            view.K[None].cuda(),
            view.width,
            view.height,
            sh_degree=3,
            packed=False,
            rasterize_mode="antialiased",
        )[0][0].clamp(0.0, 1.0)
        pred = rendered.permute(2, 0, 1)[None]
        gt = view.image.cuda().permute(2, 0, 1)[None]
        mask = read_image(mask_dir / Path(name).with_suffix(".png"), "L") > 127
        mask = torch.from_numpy(mask).cuda()[None, None]
        scores["psnr"].append(metrics.psnr(pred, gt, mask).item())
        scores["ssim"].append(metrics.ssim(pred, gt, mask).item())
        scores["lpips"].append(metrics.lpips(pred, gt, mask, net=lpips_net).item())
        renders[name] = (rendered.cpu().numpy() * 255.0).astype(np.uint8)
    return {key: float(np.mean(values)) for key, values in scores.items()}, renders


def save_renders(render_dir: Path, renders: dict[str, np.ndarray]) -> None:
    """Write each render to render_dir/<image name>.png."""
    render_dir.mkdir(exist_ok=True)
    for name, image in renders.items():
        write_image(render_dir / Path(name).with_suffix(".png"), image)
