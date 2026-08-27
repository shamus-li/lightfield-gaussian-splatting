"""4DGaussians training and rendering in the dynamic environment."""

from __future__ import annotations

import runpy
from collections.abc import Sequence
from pathlib import Path

from utils.runtime import REPO, run_in_env

FOURDGS = REPO / "submodules/4DGaussians"
# Monocular training runs longer, with a longer coarse (static) stage, than the template's.
MONOCULAR_ITERATIONS = 28_000
MONOCULAR_COARSE_ITERATIONS = 6_000


def write_config(
    path: Path, *, dataset_name: str, camera_count: int, frame_count: int, monocular: bool
) -> None:
    """Write 4DGaussians' tuned multi-view config, testing and saving only the final iteration.

    4DGaussians lets config-file values override its command-line arguments, so every override
    goes into this file.
    """
    template = runpy.run_path(str(FOURDGS / "utils/config_templates.py"))
    source = template["generate_multiview_config"](
        camera_count, frame_count, dataset_name=dataset_name
    )
    # The template returns the config as Python source.
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102
    config = {key: value for key, value in namespace.items() if not key.startswith("__")}
    optimization = config["OptimizationParams"]
    if monocular:
        optimization.update(
            iterations=MONOCULAR_ITERATIONS, coarse_iterations=MONOCULAR_COARSE_ITERATIONS
        )
    final = [optimization["iterations"]]
    config.update(test_iterations=final, save_iterations=final, checkpoint_iterations=[])
    path.write_text("".join(f"{key} = {value!r}\n" for key, value in config.items()))


def train(source: Path, model_dir: Path, *, match: str, monocular: bool) -> None:
    """Train on source with the config at model_dir/config.py.

    `match` names the camera 4DGaussians renders for testing; monocular also trains only on it.
    """
    _run(
        "train.py",
        [
            "--source_path",
            source,
            "--model_path",
            model_dir,
            "--configs",
            model_dir / "config.py",
            "--match_string",
            match,
            "--skip_post_eval",
            *(["--filter_training"] if monocular else []),
        ],
    )


def render_test(source: Path, model_dir: Path, iteration: int) -> Path:
    """Render the test views and return the folder of renders."""
    _run(
        "render.py",
        [
            "--model_path",
            model_dir,
            "--source_path",
            source,
            "--iteration",
            iteration,
            "--skip_train",
            "--skip_video",
        ],
    )
    return model_dir / "test" / f"ours_{iteration}" / "renders"


def latest_iteration(model_dir: Path) -> int:
    return max(
        int(path.name.removeprefix("iteration_"))
        for path in (model_dir / "point_cloud").glob("iteration_*")
    )


def _run(script: str, args: Sequence[str | Path | float]) -> None:
    run_in_env("video", FOURDGS / script, args, cwd=FOURDGS, python_paths=(FOURDGS,))
