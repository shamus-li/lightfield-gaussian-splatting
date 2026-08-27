from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import torch
from addict import Dict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from static.data import select_images
from utils.io import read_image, write_image
from utils.runtime import REPO

# Longer image side fed to RAFT.
FLOW_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 4
RAFT_ITERS = 20
# A pixel is occluded in a support view when its forward-backward flow error exceeds
# FB_RELATIVE * (|forward|^2 + |backward|^2) + FB_ABSOLUTE.
FB_RELATIVE = 0.01
FB_ABSOLUTE = 0.5


def load_dycheck_raft() -> tuple[Any, nn.Module]:
    raft_dir = REPO / "submodules/dycheck/dycheck/processors/raft/_impl"
    package_name = "_dycheck_raft"
    spec = importlib.util.spec_from_file_location(
        package_name,
        raft_dir / "__init__.py",
        submodule_search_locations=[str(raft_dir)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    raft_module = importlib.import_module(f"{package_name}.raft")
    utils_module = importlib.import_module(f"{package_name}.utils.utils")
    args = Dict(small=False, mixed_precision=False, dropout=0, alternate_corr=False)
    model = nn.DataParallel(raft_module.RAFT(args))
    model.load_state_dict(torch.load(REPO / "models/dycheck/raft-things.pth", map_location="cpu"))
    return utils_module.InputPadder, model.module


def min_visible_views(support_count: int) -> int:
    """Number of support views that must see a pixel for it to count as covisible."""
    if support_count <= 4:
        return support_count
    return min(support_count, max(3, math.ceil(0.05 * support_count)))


def downscale(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = FLOW_SIZE / max(height, width)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def fit_to(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize `image` to fit `target_hw` and pad the rest with black."""
    target_h, target_w = target_hw
    if image.shape[:2] == target_hw:
        return image
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        target_h - new_h - top,
        left,
        target_w - new_w - left,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


class ImagePairs(Dataset[tuple[np.ndarray, np.ndarray, int]]):
    """Every (test image, support image) pair, downscaled to a common flow resolution."""

    def __init__(
        self,
        test_paths: Sequence[Path],
        support_paths: Sequence[Path],
        flow_hw: tuple[int, int],
    ) -> None:
        self.test_paths = test_paths
        self.support_paths = support_paths
        self.flow_hw = flow_hw

    def __len__(self) -> int:
        return len(self.test_paths) * len(self.support_paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        test_index, support_index = divmod(index, len(self.support_paths))
        test = read_image(self.test_paths[test_index], "RGB")
        support = read_image(self.support_paths[support_index], "RGB")
        return (
            fit_to(downscale(test), self.flow_hw),
            fit_to(downscale(support), self.flow_hw),
            test_index,
        )


def image_paths(data_dir: Path, *, holdout_every: int = 0, match: str = "") -> list[Path]:
    reconstruction = pycolmap.Reconstruction(str(data_dir / "sparse"))
    names = sorted(image.name for image in reconstruction.images.values())
    indices = select_images(names, holdout_every=holdout_every, match=match)
    return [data_dir / "images" / names[index] for index in indices]


def write_covisible_masks(
    test_dir: Path,
    support_dir: Path,
    out_dir: Path,
    *,
    holdout_every: int = 0,
    match: str = "",
) -> None:
    """Write out_dir/<test image>.png marking pixels that enough support images also see."""
    test_paths = image_paths(test_dir)
    support_paths = image_paths(support_dir, holdout_every=holdout_every, match=match)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    InputPadder, model = load_dycheck_raft()
    model = model.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    first = read_image(test_paths[0], "RGB")
    height, width = first.shape[:2]
    flow_height, flow_width = downscale(first).shape[:2]
    flow_hw = (flow_height, flow_width)
    loader = DataLoader(
        ImagePairs(test_paths, support_paths, flow_hw),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    threshold = min_visible_views(len(support_paths))
    print(
        f"Covisibility: {len(support_paths)} support images, keeping pixels seen by "
        f">= {threshold}; flow at {flow_hw}, masks at {(height, width)}.",
        flush=True,
    )

    occlusions: list[list[np.ndarray]] = [[] for _ in test_paths]
    with torch.backends.cudnn.flags(enabled=False), torch.inference_mode():
        for test_batch, support_batch, test_indices in tqdm(loader, desc="Covisible masks"):
            test_t = test_batch.permute(0, 3, 1, 2).float().to(device, non_blocking=True)
            support_t = support_batch.permute(0, 3, 1, 2).float().to(device, non_blocking=True)
            padder = InputPadder(test_t.shape)
            test_t, support_t = padder.pad(test_t, support_t)
            flow_fw = model(test_t, support_t, iters=RAFT_ITERS, test_mode=True)[1]
            flow_bw = model(support_t, test_t, iters=RAFT_ITERS, test_mode=True)[1]
            flow_fw = padder.unpad(flow_fw).permute(0, 2, 3, 1).cpu().numpy()
            flow_bw = padder.unpad(flow_bw).permute(0, 2, 3, 1).cpu().numpy()

            for fw, bw, test_index in zip(flow_fw, flow_bw, test_indices.tolist()):
                x, y = np.meshgrid(
                    np.arange(fw.shape[1], dtype=fw.dtype),
                    np.arange(fw.shape[0], dtype=fw.dtype),
                    indexing="xy",
                )
                bw_warped = cv2.remap(bw, x + fw[..., 0], y + fw[..., 1], cv2.INTER_LINEAR)
                fb_error = np.sum((fw + bw_warped) ** 2, axis=-1)
                fb_scale = np.sum(fw**2 + bw_warped**2, axis=-1)
                occlusions[test_index].append(fb_error > FB_RELATIVE * fb_scale + FB_ABSOLUTE)
                if len(occlusions[test_index]) < len(support_paths):
                    continue
                visible = len(support_paths) - np.sum(occlusions[test_index], axis=0)
                mask = (visible >= threshold).astype(np.uint8) * 255
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                write_image(out_dir / Path(test_paths[test_index].name).with_suffix(".png"), mask)
                occlusions[test_index] = []
