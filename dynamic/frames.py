from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio

from utils.io import write_image

# Every Nth video frame is kept.
FRAME_STEP = {"iphone": 3, "stereo": 2}


def extract_frames(videos: Sequence[Path], output_dir: Path, *, step: int) -> dict[str, list[Path]]:
    """Write every `step`-th frame of each video to output_dir/<stem>_<NNN>/frame_<NNNNN>.png."""
    for index, video in enumerate(videos, start=1):
        camera_dir = output_dir / f"{video.stem}_{index:03d}"
        camera_dir.mkdir(parents=True)
        frames = itertools.islice(iio.imiter(video), 0, None, step)
        for number, frame in enumerate(frames, start=1):
            write_image(camera_dir / f"frame_{number:05d}.png", frame)
    return camera_frames(output_dir)


def camera_frames(dataset_dir: Path) -> dict[str, list[Path]]:
    """Sorted frame paths of each camera folder in dataset_dir, by camera name."""
    return {
        directory.name: sorted(directory.glob("frame_*.png"))
        for directory in sorted(dataset_dir.iterdir())
        if (directory / "frame_00001.png").is_file()
    }
