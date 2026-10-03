"""
spectra/data/trans10k_dataset.py
Dataset loader for Trans10K, GSD, and ClearPose.
Handles multi-frame temporal pairs required by OFCV.
"""
import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.augmentation import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Material class mapping
# ---------------------------------------------------------------------------

MATERIAL_CLASSES = {
    "background": 0,
    "glass": 1,
    "water": 2,
    "plastic": 3,
    "specular_metal": 4,
}

MATERIAL_NAMES = {v: k for k, v in MATERIAL_CLASSES.items()}


# ---------------------------------------------------------------------------
# Trans10K Dataset
# ---------------------------------------------------------------------------

class Trans10KDataset(Dataset):
    """
    Trans10K transparent object segmentation dataset.
    https://xieenze.github.io/projects/TransLAB/TransLAB.html

    Folder structure expected:
        root/
            train/
                images/   *.jpg
                masks/    *.png   (binary: 255=transparent, 0=background)
            validation/
            test/
    """

    MATERIAL_MAP = {
        # Trans10K provides 11 categories; we collapse to 4
        "bottle":        1,  # glass
        "cup":           1,
        "glass_door":    1,
        "window":        1,
        "bowl":          1,
        "box":           3,  # plastic
        "container":     3,
        "screen":        4,  # specular metal / screen
        "storage":       3,
        "aquarium":      2,  # water
        "other":         1,
    }

    def __init__(
        self,
        root: str,
        split: str = "train",             # train | validation | test
        image_size: Tuple[int, int] = (512, 512),
        temporal_stride: int = 1,
        label_fraction: float = 1.0,      # semi-supervised fraction
        transform=None,
    ):
        super().__init__()
        self.root = Path(root) / "Trans10K" / split
        self.split = split
        self.image_size = image_size
        self.temporal_stride = temporal_stride
        self.transform = transform if transform is not None else (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )

        self.samples = []

        image_paths = sorted(
            glob.glob(str(self.root / "images" / "*.jpg")) +
            glob.glob(str(self.root / "images" / "*.png"))
        )

        for img_path in image_paths:
            stem = Path(img_path).stem

            # Trans10K masks are named like: 123_mask.png
            mask_path = self.root / "masks" / f"{stem}_mask.png"

            if mask_path.exists():
                self.samples.append((img_path, str(mask_path)))

        assert len(self.samples) > 0, "No valid image-mask pairs found"

        # Semi-supervised: only use a fraction of labels
        if label_fraction < 1.0 and split == "train":
            n = max(1, int(len(self.samples) * label_fraction))
            indices = random.sample(range(len(self.samples)), n)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot load mask: {path}")
        # Binarise: Trans10K masks use value 76 for transparent regions
        # Any non-zero pixel is foreground
        return (mask > 0).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        image_path, mask_path = self.samples[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        if image.shape[:2] != mask.shape[:2]:
            # Try rotation first (common EXIF rotation issue)
            if image.shape[:2] == mask.shape[:2][::-1]:
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            else:
                # Hard fallback: resize mask to match image
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        pair_idx = min(idx + self.temporal_stride, len(self.samples) - 1)
        image_t1 = self._load_image(self.samples[pair_idx][0])

        # Trans10K filenames are numeric — no keyword-based material info
        # Use 1 (transparent) for all foreground objects
        mat_class = 1

        if self.transform is not None:
            # Apply same spatial augmentations to both frames and mask
            augmented = self.transform(
                image=image,
                mask=mask,
                image_t1=image_t1,
            )
            image    = augmented["image"]
            mask     = augmented["mask"]
            image_t1 = augmented["image_t1"]

        return {
            "image":        image,                             # (3, H, W) float32 [0,1]
            "image_t1":     image_t1,                         # (3, H, W) float32 [0,1]
            "mask":         mask.long(),                       # (H, W) int64 {0,1}
            "material":     torch.tensor(mat_class, dtype=torch.long),
            "image_path":   image_path,
        }


# ---------------------------------------------------------------------------
# GSD Dataset
# ---------------------------------------------------------------------------

class GSDDataset(Dataset):
    """
    Glass Surface Detection dataset.
    https://github.com/Mhaiyang/CVPR2021_PDNet
    Folder structure:
        root/GSD/train/image/*.jpg  + mask/*.png
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        transform=None,
    ):
        super().__init__()
        self.root = Path(root) / "GSD" / split
        self.image_size = image_size
        self.transform = transform if transform is not None else (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )
        self.image_paths = sorted(glob.glob(str(self.root / "image" / "*.jpg")))
        self.mask_paths  = sorted(glob.glob(str(self.root / "mask"  / "*.png")))
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask  = (cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        dummy_t1 = image.copy()  # GSD is single-frame; reuse same image

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask, image_t1=dummy_t1)
            image    = aug["image"]
            mask     = aug["mask"]
            dummy_t1 = aug["image_t1"]

        return {
            "image":    image,
            "image_t1": dummy_t1,
            "mask":     mask.long(),
            "material": torch.tensor(1, dtype=torch.long),  # all glass
            "image_path": self.image_paths[idx],
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

try:
    from data.clearpose_dataset import ClearPoseDataset
except Exception:  # ClearPose is optional
    ClearPoseDataset = None

DATASETS = {
    "trans10k":  Trans10KDataset,
    "gsd":       GSDDataset,
}
if ClearPoseDataset is not None:
    DATASETS["clearpose"] = ClearPoseDataset

# HuggingFace-backed datasets (lazy — only registered if available)
try:
    from data.hf_datasets import register_hf_datasets
    register_hf_datasets(DATASETS)
except Exception:
    pass


def build_eval_dataset(name: str, root: str, split: str, image_size, **kwargs):
    """Build a single eval-only dataset by name. Used by cross-dataset eval."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    cls_or_fn = DATASETS[name]
    if name == "trans10k":
        return cls_or_fn(root=root, split=split, image_size=image_size, label_fraction=1.0, **kwargs)
    # HF adapter factories accept root=None
    return cls_or_fn(root=root, split=split, image_size=image_size, **kwargs)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from config."""
    DatasetCls = DATASETS[cfg.data.dataset]

    train_ds = DatasetCls(
        root=cfg.data.root,
        split="train",
        image_size=tuple(cfg.data.image_size),
        label_fraction=cfg.train.label_fraction,
    )
    val_ds = DatasetCls(
        root=cfg.data.root,
        split="validation",
        image_size=tuple(cfg.data.image_size),
        label_fraction=1.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
