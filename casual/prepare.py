from __future__ import annotations

import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pycolmap  # ty: ignore[unresolved-import]

from static.prepare_data import materialize_subset_dirs
from utils.io import write_lines
from utils.vggt import run_vggt


def run_colmap(
    image_dir: Path,
    out_root: Path,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    pycolmap.set_random_seed(0)
    db_path = out_root / "database.db"
    sparse_path = out_root / "sparse"
    num_threads = max(1, len(os.sched_getaffinity(0)))
    extraction_options = pycolmap.SiftExtractionOptions()
    extraction_options.num_threads = num_threads

    matching_options = pycolmap.SiftMatchingOptions()
    matching_options.num_threads = num_threads
    matching_options.max_num_matches = 32768

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.num_threads = num_threads
    mapping_options.mapper.num_threads = num_threads
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_dir),
        camera_model="SIMPLE_RADIAL",
        sift_options=extraction_options,
    )
    pycolmap.match_exhaustive(database_path=str(db_path), sift_options=matching_options)
    pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(sparse_path),
        options=mapping_options,
    )
    db_path.unlink()
    return sparse_path / "0"


def capture_sources(
    capture_root: Path,
    extraction_root: Path,
    *,
    fps: float,
) -> list[Path]:
    extracted_frames: list[Path] = []
    for video_path in sorted(
        path for path in capture_root.iterdir() if path.suffix.lower() == ".mov"
    ):
        output_dir = extraction_root / video_path.stem
        output_dir.mkdir(parents=True)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps:g}",
                "-pix_fmt",
                "rgb24",
                str(output_dir / "frame_%06d.png"),
            ],
            check=True,
        )
        extracted_frames.extend(sorted(output_dir.glob("frame_*.png")))
    return extracted_frames


def link_images(
    *,
    label: str,
    sources: list[Path],
    output_root: Path,
) -> list[str]:
    names: list[str] = []
    for source in sources:
        name = f"{label}__{source.parent.name}__{source.name}"
        os.link(source, output_root / "images" / name)
        names.append(name)
    return names


def prepare_scene(
    data_root: Path,
    output_root: Path,
    *,
    modality: str,
    initializer: str,
) -> Path:
    data_root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    fps = 10.0 if modality == "iphone" else 20.0

    output_root.mkdir(parents=True)
    train_sources = capture_sources(
        data_root / f"{modality}-train",
        output_root / "extracted_frames",
        fps=fps,
    )
    eval_sources = capture_sources(
        data_root / f"{modality}-eval",
        output_root / "extracted_frames/eval",
        fps=fps,
    )
    (output_root / "images").mkdir()
    train_names = link_images(
        label="train",
        sources=train_sources,
        output_root=output_root,
    )
    eval_names = link_images(
        label="eval",
        sources=eval_sources,
        output_root=output_root,
    )
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True)
    for label, names in (("train", train_names), ("eval", eval_names)):
        write_lines(split_dir / f"{label}.txt", names)

    if initializer == "colmap":
        sparse_model = run_colmap(
            image_dir=output_root / "images",
            out_root=output_root / "colmap",
        )
        (output_root / "sparse").symlink_to(
            os.path.relpath(sparse_model, start=output_root),
            target_is_directory=True,
        )
    else:
        run_vggt(output_root, "--use_ba")
        (output_root / "sparse/points.ply").unlink()

    materialize_subset_dirs(
        output_dir=output_root,
        subsets={"train": train_names, "test": eval_names},
        combined_sparse=output_root / "sparse",
    )

    print(f"Prepared casual capture at {output_root}")
    return output_root
