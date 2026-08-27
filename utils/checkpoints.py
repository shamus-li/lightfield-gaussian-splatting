from __future__ import annotations

import shutil
from pathlib import Path

from utils.io import read_json


def validation_steps(max_steps: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((min(3_000, max_steps), min(7_000, max_steps), max_steps)))


def select_model(
    result_dir: Path,
) -> Path:
    result_dir = result_dir.expanduser().resolve()
    stats_dir = result_dir / "stats"
    scores = []
    for path in stats_dir.glob("val_step*.json"):
        step = int(path.stem.removeprefix("val_step"))
        stats = read_json(path)
        scores.append(
            (
                float(stats["psnr"]),
                float(stats["ssim"]),
                -float(stats["lpips"]),
                -step,
                step,
            )
        )
    selected = max(scores)
    step = selected[-1]
    model = result_dir / "model.pt"
    (result_dir / "ckpts" / f"ckpt_{step}_rank0.pt").replace(model)
    shutil.rmtree(result_dir / "ckpts")
    shutil.rmtree(stats_dir)
    return model
