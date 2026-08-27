from __future__ import annotations

import os
import shutil
from pathlib import Path

import pycolmap

from utils.colmap import extract_frames, vggt_sparse, write_subsets

FPS = {"iphone": 10, "stereo": 20}


def run_colmap(root: Path) -> None:
    """Reconstruct root/images into root/sparse with COLMAP SIMPLE_RADIAL cameras."""
    pycolmap.set_random_seed(0)
    database = root / "database.db"
    num_threads = len(os.sched_getaffinity(0))
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = num_threads
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    matching.max_num_matches = 32768
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = num_threads
    mapping.mapper.num_threads = num_threads
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(root / "images"),
        reader_options=reader,
        extraction_options=extraction,
    )
    pycolmap.match_exhaustive(database_path=str(database), matching_options=matching)
    models = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(root / "images"),
        output_path=str(root / "colmap"),
        options=mapping,
    )
    # The mapper can also keep small models from failed initial pairs.
    index = max(models, key=lambda index: models[index].num_reg_images())
    (root / "colmap" / str(index)).rename(root / "sparse")
    shutil.rmtree(root / "colmap")
    database.unlink()


def extract_capture(capture_dir: Path, images: Path, *, label: str, fps: float) -> list[str]:
    names = []
    for video in sorted(path for path in capture_dir.iterdir() if path.suffix.lower() == ".mov"):
        frames = extract_frames(video, images, fps=fps, prefix=f"{label}__{video.stem}__")
        names += [frame.name for frame in frames]
    return names


def prepare_scene(data: Path, root: Path, *, camera_model: str, initializer: str) -> None:
    """Extract the train and eval videos into root/images, pose them and split them into subsets."""
    images = root / "images"
    fps = FPS[camera_model]
    train_names = extract_capture(data / f"{camera_model}-train", images, label="train", fps=fps)
    test_names = extract_capture(data / f"{camera_model}-eval", images, label="eval", fps=fps)
    if initializer == "colmap":
        run_colmap(root)
    else:
        vggt_sparse(root)
    write_subsets(root, {"train": train_names, "test": test_names})
    print(f"Prepared casual capture at {root}")
