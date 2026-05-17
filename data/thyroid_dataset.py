from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

THYROID_ROOT = Path(
    os.environ.get(
        "THYROID_ROOT",
        "Thyroid_US/Thyroid Dataset",
    )
)

def _resolve_ddti_stage1(root: Path) -> Path:
    candidates = [
        root / "DDTI" / "2_preprocessed_data" / "stage1",
        root / "DDTI dataset" / "DDTI" / "2_preprocessed_data" / "stage1",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find DDTI stage1 under any of: {[str(c) for c in candidates]}"
    )


DDTI_STAGE1_DIR = _resolve_ddti_stage1(THYROID_ROOT)
TN3K_DIR = THYROID_ROOT / "tn3k"
# BUSI lives one level above THYROID_ROOT in the on-disk layout that the
# repo uses (sibling of "Thyroid Dataset"). Allow an explicit override too.
BUSI_DIR = Path(
    os.environ.get(
        "BUSI_ROOT",
        str(THYROID_ROOT.parent / "Dataset_BUSI_with_GT"),
    )
)

Mode = Literal["pretrain", "finetune"]
Split = Literal["train", "val", "test", "all"]

MaskStrategy = Literal["block", "random_pixel", "grid", "random_patch", "mixed"]


@dataclass
class MaskingConfig:

    patch_size: int = 40
    ratio: float = 0.2
    max_iters: int = 10_000
    strategy: MaskStrategy = "block"
    # "mixed" uses both patch_size and patch_size_b (split 50/50 of `ratio`).
    patch_size_b: int = 10


def _sample_block_mask(h: int, w: int, p: int, ratio: float, max_iters: int,
                       rng: np.random.Generator) -> np.ndarray:
    eh, ew = h + p, w + p
    mask_ext = np.zeros((eh, ew), dtype=np.uint8)
    target_pixels = int(ratio * h * w)
    masked_pixels = 0
    for _ in range(max_iters):
        if masked_pixels >= target_pixels:
            break
        y = int(rng.integers(0, eh))
        x = int(rng.integers(0, ew))
        mask_ext[y : y + p, x : x + p] = 1
        masked_pixels = int(mask_ext[p:, p:].sum())
    return mask_ext[p:, p:].astype(np.uint8)


def _sample_random_pixel_mask(h: int, w: int, ratio: float,
                              rng: np.random.Generator) -> np.ndarray:
    return (rng.random((h, w)) < ratio).astype(np.uint8)


def _sample_grid_mask(h: int, w: int, p: int, ratio: float) -> np.ndarray:
    step = max(p + 1, int(round(p / max(ratio, 1e-6) ** 0.5)))
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            mask[y : y + p, x : x + p] = 1
    return mask


def _sample_random_patch_mask(h: int, w: int, p: int, ratio: float,
                              rng: np.random.Generator) -> np.ndarray:
    n_y = max(1, h // p)
    n_x = max(1, w // p)
    n_cells = n_y * n_x
    n_mask = max(1, int(round(ratio * n_cells)))
    idx = rng.choice(n_cells, size=n_mask, replace=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    for c in idx:
        cy, cx = c // n_x, c % n_x
        mask[cy * p : (cy + 1) * p, cx * p : (cx + 1) * p] = 1
    return mask


def sample_block_mask(
    h: int,
    w: int,
    cfg: MaskingConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if cfg.strategy == "block":
        return _sample_block_mask(h, w, cfg.patch_size, cfg.ratio, cfg.max_iters, rng)
    if cfg.strategy == "random_pixel":
        return _sample_random_pixel_mask(h, w, cfg.ratio, rng)
    if cfg.strategy == "grid":
        return _sample_grid_mask(h, w, cfg.patch_size, cfg.ratio)
    if cfg.strategy == "random_patch":
        return _sample_random_patch_mask(h, w, cfg.patch_size, cfg.ratio, rng)
    if cfg.strategy == "mixed":
        # Each sub-strategy is asked for half of the target ratio.
        a = _sample_block_mask(h, w, cfg.patch_size, cfg.ratio / 2,
                               cfg.max_iters, rng)
        b = _sample_block_mask(h, w, cfg.patch_size_b, cfg.ratio / 2,
                               cfg.max_iters, rng)
        return np.maximum(a, b).astype(np.uint8)
    raise ValueError(f"Unknown masking strategy: {cfg.strategy!r}")

def _random_scale_crop(
    img: np.ndarray,
    seg: np.ndarray | None,
    size: int,
    rng: np.random.Generator,
    min_scale: float = 0.67,
    max_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = img.shape[:2]
    scale = float(rng.uniform(min_scale, max_scale))
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    img_c = img[top : top + crop_h, left : left + crop_w]
    seg_c = None if seg is None else seg[top : top + crop_h, left : left + crop_w]
    img_r = np.array(
        Image.fromarray(img_c).resize((size, size), Image.BILINEAR)
    )
    seg_r = (
        None
        if seg_c is None
        else np.array(Image.fromarray(seg_c).resize((size, size), Image.NEAREST))
    )
    return img_r, seg_r


def _random_flip(
    img: np.ndarray,
    seg: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        seg = None if seg is None else np.ascontiguousarray(seg[:, ::-1])
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
        seg = None if seg is None else np.ascontiguousarray(seg[::-1, :])
    return img, seg

def _load_gray(path: Path, size: int) -> np.ndarray:
    # Load image as uint8 grayscale, resized to (size, size)
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.array(im, dtype=np.uint8)


def _load_mask(path: Path, size: int) -> np.ndarray:
    # Load binary segmentation mask, resized to (size, size), values in {0,1}
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.NEAREST)
        arr = np.array(im, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _load_mask_union(paths: Sequence[Path], size: int) -> np.ndarray:
    # OR-combine several binary masks into one, all resized to (size, size)
    if not paths:
        raise ValueError("at least one mask path required")
    union = np.zeros((size, size), dtype=np.uint8)
    for p in paths:
        union = np.maximum(union, _load_mask(p, size))
    return union

@dataclass
class Sample:
    image_path: Path
    mask_path: Path | None
    sample_id: str
    extra_mask_paths: list[Path] = field(default_factory=list)


def _ddti_index(split: Split, seed: int = 0) -> list[Sample]:
    img_dir = DDTI_STAGE1_DIR / "p_image"
    msk_dir = DDTI_STAGE1_DIR / "p_mask"
    items = sorted(
        img_dir.glob("*.PNG"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not items:
        items = sorted(img_dir.glob("*.png"))
    samples = [
        Sample(image_path=p, mask_path=msk_dir / p.name, sample_id=p.stem)
        for p in items
    ]
    n_test = 128
    if split == "all":
        return samples
    if split == "test":
        return samples[-n_test:]
    if split in ("train", "val"):
        return samples[:-n_test]
    raise ValueError(f"Unknown DDTI split: {split}")


def _busi_index(split: Split, classes: tuple[str, ...] = ("benign", "malignant"),
                seed: int = 0, val_frac: float = 0.0, test_frac: float = 0.2) -> list[Sample]:
    samples: list[Sample] = []
    for cls in classes:
        cls_dir = BUSI_DIR / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"BUSI class dir missing: {cls_dir}")
        imgs = sorted(p for p in cls_dir.glob("*.png") if "_mask" not in p.stem)
        for img in imgs:
            base = img.stem
            primary = cls_dir / f"{base}_mask.png"
            extras = sorted(cls_dir.glob(f"{base}_mask_*.png"))
            if not primary.exists():
                continue
            samples.append(Sample(
                image_path=img,
                mask_path=primary,
                sample_id=f"{cls}:{base}",
                extra_mask_paths=extras,
            ))
    # Deterministic shuffle, then split.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    samples = [samples[i] for i in idx]
    n = len(samples)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    n_train = n - n_test - n_val
    if split == "test":
        return samples[n_train + n_val :]
    if split == "val":
        return samples[n_train : n_train + n_val]
    if split == "train":
        return samples[:n_train]
    if split == "all":
        # Exclude test partition so pretraining never sees held-out images.
        return samples[: n_train + n_val]
    raise ValueError(f"Unknown BUSI split: {split}")


def _tn3k_index(split: Split, fold: int = 0) -> list[Sample]:
    # Index TN3k samples using the official fold JSONs
    if split == "test":
        img_dir = TN3K_DIR / "test-image"
        msk_dir = TN3K_DIR / "test-mask"
        items = sorted(img_dir.glob("*.jpg"))
        return [
            Sample(image_path=p, mask_path=msk_dir / p.name, sample_id=p.stem)
            for p in items
        ]

    img_dir = TN3K_DIR / "trainval-image"
    msk_dir = TN3K_DIR / "trainval-mask"
    all_items = sorted(img_dir.glob("*.jpg"))

    if split == "all":
        return [
            Sample(image_path=p, mask_path=msk_dir / p.name, sample_id=p.stem)
            for p in all_items
        ]

    fold_path = TN3K_DIR / f"tn3k-trainval-fold{fold}.json"
    with open(fold_path) as fh:
        fold_idx = json.load(fh)
    key = "train" if split == "train" else "val"
    selected = set(fold_idx[key])
    return [
        Sample(image_path=p, mask_path=msk_dir / p.name, sample_id=p.stem)
        for i, p in enumerate(all_items)
        if i in selected
    ]


class ThyroidUSDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        mode: Mode,
        image_size: int = 256,
        augment: bool = True,
        masking: MaskingConfig | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in ("pretrain", "finetune"):
            raise ValueError(f"mode must be 'pretrain' or 'finetune', got {mode!r}")
        self.samples = list(samples)
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.masking = masking or MaskingConfig()
        self.seed = seed
        if mode == "finetune":
            missing = [s.sample_id for s in self.samples if s.mask_path is None]
            if missing:
                raise ValueError(
                    f"finetune mode requires segmentation masks; "
                    f"{len(missing)} samples without mask, first few: {missing[:3]}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, idx: int) -> np.random.Generator:
        base = torch.initial_seed() % (2**31)
        return np.random.default_rng(self.seed + idx + base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        rng = self._rng(idx)

        img = _load_gray(sample.image_path, self.image_size)
        if sample.mask_path is None:
            seg = None
        elif sample.extra_mask_paths:
            seg = _load_mask_union(
                [sample.mask_path, *sample.extra_mask_paths], self.image_size
            )
        else:
            seg = _load_mask(sample.mask_path, self.image_size)

        if self.augment:
            img, seg = _random_scale_crop(img, seg, self.image_size, rng)
            img, seg = _random_flip(img, seg, rng)

        img_f = img.astype(np.float32) / 255.0  # [0, 1] per paper
        img_t = torch.from_numpy(img_f).unsqueeze(0)  # (1, H, W)

        if self.mode == "pretrain":
            mask = sample_block_mask(
                self.image_size, self.image_size, self.masking, rng
            )
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
            input_t = img_t * (1.0 - mask_t)  # zero-out masked region
            return {
                "input": input_t,
                "target": img_t,
                "loss_mask": mask_t,
                "sample_id": sample.sample_id,
            }

        # finetune
        seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)
        return {
            "image": img_t,
            "seg": seg_t,
            "sample_id": sample.sample_id,
        }

def build_dataset(
    name: Literal["ddti", "tn3k", "busi"],
    split: Split = "train",
    mode: Mode = "pretrain",
    image_size: int = 256,
    augment: bool | None = None,
    masking: MaskingConfig | None = None,
    n_labeled: int | None = None,
    fold: int = 0,
    seed: int = 0,
) -> ThyroidUSDataset:
    if augment is None:
        augment = split in ("train", "all")

    if name == "ddti":
        samples = _ddti_index(split)
    elif name == "tn3k":
        samples = _tn3k_index(split, fold=fold)
    elif name == "busi":
        samples = _busi_index(split, seed=seed)
    else:
        raise ValueError(f"Unknown dataset name: {name!r}")

    if n_labeled is not None:
        if mode != "finetune":
            raise ValueError("n_labeled only applies to finetune mode")
        if n_labeled > len(samples):
            raise ValueError(
                f"n_labeled={n_labeled} exceeds split size {len(samples)}"
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=n_labeled, replace=False)
        samples = [samples[i] for i in sorted(idx.tolist())]

    return ThyroidUSDataset(
        samples=samples,
        mode=mode,
        image_size=image_size,
        augment=augment,
        masking=masking,
        seed=seed,
    )


def build_dataloader(
    name: Literal["ddti", "tn3k", "busi"],
    split: Split = "train",
    mode: Mode = "pretrain",
    batch_size: int = 10,
    num_workers: int = 4,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    image_size: int = 256,
    augment: bool | None = None,
    masking: MaskingConfig | None = None,
    n_labeled: int | None = None,
    fold: int = 0,
    seed: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = build_dataset(
        name=name,
        split=split,
        mode=mode,
        image_size=image_size,
        augment=augment,
        masking=masking,
        n_labeled=n_labeled,
        fold=fold,
        seed=seed,
    )
    if shuffle is None:
        shuffle = split in ("train", "all")
    if drop_last is None:
        drop_last = shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "MaskingConfig",
    "MaskStrategy",
    "Sample",
    "ThyroidUSDataset",
    "build_dataset",
    "build_dataloader",
    "sample_block_mask",
    "DDTI_STAGE1_DIR",
    "TN3K_DIR",
    "BUSI_DIR",
]
