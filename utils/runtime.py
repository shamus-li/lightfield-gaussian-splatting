from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_env(*python_paths: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TORCH_HOME"] = str(Path(__file__).resolve().parents[1] / "models")
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(
            str(path.expanduser().resolve()) for path in python_paths
        )
    return environment
