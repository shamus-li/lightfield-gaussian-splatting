from __future__ import annotations

import runpy
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from utils.runtime import torch_env


def generate_config(*, dataset_name: str, camera_count: int, frame_count: int) -> str:
    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1] / "submodules/4DGaussians/utils/config_templates.py"
        )
    )
    return namespace["generate_multiview_config"](
        camera_count,
        frame_count,
        dataset_name=dataset_name,
    )


def run(name: str, argv: Sequence[str]) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "submodules/4DGaussians"
    script = root / name
    command = [str(repo / ".envs/dynamic/bin/python"), str(script), *map(str, argv)]
    print(f"[dynamic:model:{script.stem}] {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=root,
        env=torch_env(root),
        check=True,
    )
