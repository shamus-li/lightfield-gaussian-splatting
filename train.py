from __future__ import annotations

import argparse
import sys

from casual.difix import main as difix_main
from casual.train import main as casual_main
from dynamic.train import main as dynamic_main
from multiplexed.train import main as multiplexed_main
from static.baseline import main as baseline_main
from static.prepare import main as prepare_static_main
from static.train import main as static_main
from synthetic.depth import main as depth_main
from synthetic.depth import prepare_main as prepare_depth_main
from synthetic.train import main as synthetic_main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "baseline": baseline_main,
        "casual": casual_main,
        "depth": depth_main,
        "difix": difix_main,
        "dynamic": dynamic_main,
        "multiplexed": multiplexed_main,
        "prepare-depth": prepare_depth_main,
        "prepare-static": prepare_static_main,
        "static": static_main,
        "synthetic": synthetic_main,
    }
    parser = argparse.ArgumentParser(
        description="Train and evaluate light-field Gaussian splatting models."
    )
    parser.add_argument("command", choices=commands)
    command = parser.parse_args(args[:1]).command
    commands[command](args[1:])


if __name__ == "__main__":
    main()
