from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import utils.metrics as metric_lib
import utils.render as render_mod
from static.data import ColmapDataset
from utils.io import read_image, write_image, write_json

if TYPE_CHECKING:
    from static.train import StaticConfig


@torch.no_grad()
def evaluate_static_model(
    config: StaticConfig,
    valset: ColmapDataset,
    gaussians: torch.nn.ParameterDict,
    render_cfg: render_mod.RenderConfig,
    *,
    step: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    result_dir = config.result_dir.expanduser().resolve()
    render_dir = result_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    covisible_dir = config.covisible_dir.expanduser().resolve() if config.covisible_dir else None
    values: dict[str, list[torch.Tensor]] = {"psnr": [], "ssim": [], "lpips": []}

    for row in range(len(valset)):
        item = valset[row]
        target = item.image.to(device=device, dtype=torch.float32)
        rendered = render_mod.render_item(
            gaussians,
            item,
            cfg=render_cfg,
        )[0].clamp(0.0, 1.0)
        prediction = rendered.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)

        if covisible_dir is None:
            values["psnr"].append(metric_lib.psnr(prediction, target))
            values["ssim"].append(metric_lib.ssim_dycheck(prediction, target))
            values["lpips"].append(metric_lib.lpips(prediction, target, net=config.lpips_net))
        else:
            mask = read_image(
                covisible_dir / Path(valset.image_name(item.image_id)).with_suffix(".png"),
                "L",
            )
            mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None, None].to(
                device=device
            )
            values["psnr"].append(metric_lib.mpsnr(prediction, target, mask_tensor))
            values["ssim"].append(metric_lib.mssim(prediction, target, mask_tensor))
            values["lpips"].append(
                metric_lib.mlpips(prediction, target, mask_tensor, net=config.lpips_net)
            )

        if config.checkpoint is not None:
            image = (rendered.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            write_image(render_dir / f"{row:04d}.png", image)

    metrics = {name: float(torch.stack(items).mean()) for name, items in values.items()}
    if config.checkpoint is None:
        path = result_dir / "stats" / f"val_step{step:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, metrics)
    return metrics
