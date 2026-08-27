from __future__ import annotations

import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from utils.runtime import REPO, torch_env


def run_vggt(dataset_dir: Path, *options: str) -> None:
    """Run VGGT's COLMAP export on dataset_dir/images in a subprocess."""
    root = dataset_dir.resolve()
    command = [
        sys.executable,
        "-m",
        "utils.vggt",
        str(REPO / "submodules/vggt/demo_colmap.py"),
        str(REPO / "submodules/dinov2"),
        "--scene_dir",
        str(root),
        *options,
    ]
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, env=torch_env(REPO), check=True)


if __name__ == "__main__":
    script = Path(sys.argv[1])
    dinov2 = Path(sys.argv[2])
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

    torch.hub.load = load_local_dinov2  # ty: ignore[invalid-assignment]
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")
