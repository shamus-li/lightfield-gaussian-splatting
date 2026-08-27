from __future__ import annotations

import os
import random
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_env(*python_paths: Path) -> dict[str, str]:
    """Environment for a torch subprocess in another venv, with optional PYTHONPATH entries."""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TORCH_HOME"] = str(REPO / "models")
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(str(path.resolve()) for path in python_paths)
    return environment


def run_in_env(
    env: str,
    script: Path,
    args: Sequence[str | Path | float],
    *,
    cwd: Path,
    python_paths: Sequence[Path] = (),
    env_extra: Mapping[str, str] | None = None,
) -> None:
    """Run `script` with the Python of .envs/<env>, printing the command first."""
    command = [str(REPO / ".envs" / env / "bin/python"), str(script), *map(str, args)]
    environment = torch_env(*python_paths)
    environment.update(env_extra or {})
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)
