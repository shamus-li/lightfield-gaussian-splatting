from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

from addict import Dict
import cv2
import numpy as np
import pycolmap  # ty: ignore[unresolved-import]
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from static.data import select_dataset_indices
from utils.io import read_image, write_image


def load_dycheck_raft() -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[1] / "submodules/dycheck"
    raft_dir = root / "dycheck" / "processors" / "raft" / "_impl"
    init_path = raft_dir / "__init__.py"
    package_name = "_dycheck_raft"
    spec: Any = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(raft_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    raft_module = importlib.import_module(f"{package_name}.raft")
    utils_module = importlib.import_module(f"{package_name}.utils.utils")

    args = Dict(
        small=False,
        mixed_precision=False,
        dropout=0,
        alternate_corr=False,
    )
    model = nn.DataParallel(raft_module.RAFT(args))
    model.load_state_dict(
        torch.load(
            Path(__file__).resolve().parents[1] / "models/dycheck/raft-things.pth",
            map_location="cpu",
        )
    )
    return utils_module.InputPadder, model.module


def load_selected_image_paths(
    data_dir: Path,
    *,
    split: str,
    test_every: int,
    match_string: str | None = None,
) -> list[Path]:
    root = data_dir.expanduser().resolve()
    reconstruction = pycolmap.Reconstruction(str(root / "sparse"))
    images = sorted(reconstruction.images.values(), key=lambda image: str(image.name))
    image_names = [str(image.name) for image in images]
    indices = select_dataset_indices(
        image_names,
        split=split,
        test_every=test_every,
        match_string=match_string,
    )
    return [root / "images" / image_names[index] for index in indices]


def maybe_downscale_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    longer = max(h, w)
    if longer <= 512:
        return image

    scale = 512 / float(longer)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_and_pad_image(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image

    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


class ColmapRAFTDenseDataset(Dataset[tuple[np.ndarray, np.ndarray, int, int]]):
    def __init__(
        self,
        base_paths: Sequence[Path],
        support_paths: Sequence[Path],
        target_hw: tuple[int, int],
    ) -> None:
        self.base_paths = list(base_paths)
        self.support_paths = list(support_paths)
        self.target_hw = target_hw

    def __len__(self) -> int:
        return len(self.base_paths) * len(self.support_paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        support_count = len(self.support_paths)
        base_index, support_index = divmod(index, support_count)
        base = resize_and_pad_image(
            maybe_downscale_image(
                read_image(self.base_paths[base_index], "RGB"),
            ),
            self.target_hw,
        )
        support = resize_and_pad_image(
            maybe_downscale_image(
                read_image(self.support_paths[support_index], "RGB"),
            ),
            self.target_hw,
        )
        return base, support, base_index, support_index


def compute_covisible_for_base(
    base_paths: Sequence[Path],
    support_paths: Sequence[Path],
    *,
    out_dir: Path,
) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    InputPadder, model = load_dycheck_raft()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_preview = read_image(base_paths[0], "RGB")
    output_hw = base_preview.shape[:2]
    model_hw = maybe_downscale_image(base_preview).shape[:2]

    dataset = ColmapRAFTDenseDataset(base_paths, support_paths, model_hw)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = model.to(device).eval()

    support_count = len(support_paths)
    threshold = (
        support_count
        if support_count <= 4
        else min(support_count, max(3, int(np.ceil(0.05 * support_count))))
    )
    print(
        f"[covisible] Support frames: {support_count}, keeping pixels visible in >= {threshold} frames.",
        flush=True,
    )
    print(
        f"[covisible] Model HW: {model_hw}, output HW: {output_hw}",
        flush=True,
    )

    occ_cache: list[list[np.ndarray]] = [[] for _ in base_paths]
    with torch.backends.cudnn.flags(enabled=False), torch.inference_mode():
        for batch in tqdm(dataloader, desc="Computing covisible"):
            img0_batch, img1_batch, bi_batch, _ = batch
            img0_t = (
                img0_batch.permute(0, 3, 1, 2).contiguous().float().to(device, non_blocking=True)
            )
            img1_t = (
                img1_batch.permute(0, 3, 1, 2).contiguous().float().to(device, non_blocking=True)
            )

            padder = InputPadder(img0_t.shape)
            img0_t, img1_t = padder.pad(img0_t, img1_t)
            flow_fw = model(img0_t, img1_t, iters=20, test_mode=True)[1]
            flow_bw = model(img1_t, img0_t, iters=20, test_mode=True)[1]
            flow_fw = padder.unpad(flow_fw).permute(0, 2, 3, 1).cpu().numpy()
            flow_bw = padder.unpad(flow_bw).permute(0, 2, 3, 1).cpu().numpy()

            for item_index, base_slot in enumerate(bi_batch.tolist()):
                fw = flow_fw[item_index]
                bw = flow_bw[item_index]
                x, y = np.meshgrid(
                    np.arange(fw.shape[1], dtype=fw.dtype),
                    np.arange(fw.shape[0], dtype=fw.dtype),
                    indexing="xy",
                )
                warp = np.stack([x, y], axis=-1) + fw
                bw_res = cv2.remap(bw, warp[..., 0], warp[..., 1], cv2.INTER_LINEAR)
                fb_sq_diff = np.sum((fw + bw_res) ** 2, axis=-1, keepdims=True)
                fb_sum_sq = np.sum(fw**2 + bw_res**2, axis=-1, keepdims=True)
                occ_cache[int(base_slot)].append(
                    (fb_sq_diff > 0.01 * fb_sum_sq + 0.5).astype(np.float32)
                )

                if len(occ_cache[int(base_slot)]) != support_count:
                    continue
                visible_counts = (1.0 - np.stack(occ_cache[int(base_slot)], axis=0)).sum(axis=0)
                covisible = (visible_counts >= threshold).astype(np.float32)
                rel_name = base_paths[int(base_slot)].name
                out_path = out_dir / Path(rel_name).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                mask = (covisible[..., 0] * 255).astype(np.uint8)
                if mask.shape != output_hw:
                    mask = cv2.resize(
                        mask, (output_hw[1], output_hw[0]), interpolation=cv2.INTER_NEAREST
                    )
                write_image(out_path, mask)
                occ_cache[int(base_slot)] = []


def write_covisible_masks(
    base_dir: Path,
    support_dir: Path,
    output_root: Path,
    *,
    support_test_every: int,
    support_match: str | None = None,
) -> Path:
    base_dir = base_dir.expanduser().resolve()
    support_dir = support_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    base_paths = load_selected_image_paths(base_dir, split="val", test_every=1)
    support_paths = load_selected_image_paths(
        support_dir,
        split="train",
        test_every=support_test_every,
        match_string=support_match,
    )
    out_dir = output_root / "1x" / "val"
    compute_covisible_for_base(
        base_paths,
        support_paths,
        out_dir=out_dir,
    )
    return out_dir
