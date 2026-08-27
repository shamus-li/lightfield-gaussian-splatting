from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def read_image(path: Path, mode: str | None = None) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert(mode) if mode else image)


def write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_config(path: Path, config: Any) -> None:
    """Write a dataclass config as YAML with paths stored as strings."""
    write_yaml(
        path,
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    )
