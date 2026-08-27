from __future__ import annotations

import argparse
import importlib
import sys

# Command name -> (module, function); modules are imported only when their command runs.
COMMANDS = {
    "baseline": ("static.baseline", "main"),
    "casual": ("casual.run", "main"),
    "depth": ("synthetic.depth", "main"),
    "difix": ("casual.difix", "main"),
    "dynamic": ("dynamic.train", "main"),
    "multiplexed": ("multiplexed.train", "main"),
    "multiplexed-render": ("multiplexed.render", "main"),
    "prepare-depth": ("synthetic.depth", "prepare_main"),
    "prepare-static": ("static.prepare", "main"),
    "static": ("static.train", "main"),
    "synthetic": ("synthetic.train", "main"),
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Train and evaluate light-field Gaussian splatting models."
    )
    parser.add_argument("command", choices=COMMANDS)
    command = parser.parse_args(args[:1]).command
    module, function = COMMANDS[command]
    getattr(importlib.import_module(module), function)(args[1:])


if __name__ == "__main__":
    main()
