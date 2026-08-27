from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from utils.io import read_yaml


def parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    args = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config, _ = config_parser.parse_known_args(args)
    parser.add_argument("--config", type=Path, help="YAML file containing command defaults.")
    if config.config:
        parser.set_defaults(**read_yaml(config.config))
    return parser.parse_args(args)
