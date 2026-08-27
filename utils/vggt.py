from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from utils.runtime import torch_env


def run_vggt(dataset_dir: Path, *options: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    root = dataset_dir.expanduser().resolve()
    command = [
        str(repo / ".envs/vggt/bin/python"),
        str(Path(__file__).resolve()),
        str(repo / "submodules/vggt/demo_colmap.py"),
        str(repo / "submodules/dinov2"),
        "--scene_dir",
        str(root),
        *options,
    ]
    print("$", " ".join(command))
    subprocess.run(command, cwd=root, env=torch_env(), check=True)


if __name__ == "__main__":
    script = Path(sys.argv[1]).expanduser().resolve()
    dinov2 = Path(sys.argv[2]).expanduser().resolve()
    original_load = torch.hub.load

    def load_local_dinov2(
        repo_or_dir: str,
        model: str,
        *model_args: object,
        **model_kwargs: Any,
    ) -> Any:
        if repo_or_dir == "facebookresearch/dinov2":
            model_kwargs["source"] = "local"
            repo_or_dir = str(dinov2)
        return original_load(repo_or_dir, model, *model_args, **model_kwargs)

    setattr(torch.hub, "load", load_local_dinov2)
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")
