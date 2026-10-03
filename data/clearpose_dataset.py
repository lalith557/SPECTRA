"""
spectra/data/clearpose_dataset.py
ClearPose dataset loader (Chen et al., ECCV 2022).
Used for zero-shot cross-dataset generalization tests.

ClearPose contains transparent object videos with pose & mask annotations.
We treat it as a flat segmentation dataset by sampling consecutive frame pairs.

Expected folder structure (after download + extract):
    root/ClearPose/
        set1/scene1/
            000000-color.png
            000000-label.png    # instance segmentation
            000001-color.png
            ...
        set1/scene2/
            ...
        set2/...

The loader walks every scene directory, pairs consecutive frames, and uses the
binary union of the instance labels as the transparent-object mask.
"""
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import get_val_transforms


_FRAME_RE = re.compile(r"(\d+)-color\.(png|jpg)$", re.IGNORECASE)


class ClearPoseDataset(Dataset):
    """
    Loader for the ClearPose benchmark (https://github.com/opipari/ClearPose).

    Args:
        root:           top-level dataset root containing ClearPose/
        image_size:     (H, W)
        temporal_stride: frames between (image, image_t1)
        split:          'all' | 'set1' | 'set2' | 'set3' (subdirectory name)
        transform:      DualFrameTransform; defaults to val transforms
        max_samples:    cap dataset size (for fast smoke tests)
    """

    def __init__(
        self,
        root: str,
        split: str = "all",
        image_size: Tuple[int, int] = (448, 448),
        temporal_stride: int = 1,
        transform=None,
        max_samples: int = -1,
    ):
        super().__init__()
        self.root = Path(root) / "ClearPose"
        self.split = split
        self.image_size = image_size
        self.temporal_stride = temporal_stride
        self.transform = transform or get_val_transforms(image_size)

        # Walk scenes
        if split == "all":
            scene_dirs = sorted([p for p in self.root.rglob("scene*") if p.is_dir()])
        else:
            scene_dirs = sorted([p for p in (self.root / split).rglob("scene*") if p.is_dir()])

        self.samples: List[Tuple[str, str, str]] = []   # (image_t0, image_t1, label_t0)

        for scene in scene_dirs:
            color_paths = sorted(scene.glob("*-color.png")) + sorted(scene.glob("*-color.jpg"))
            for i, color_t0 in enumerate(color_paths):
                # next-frame index, clamped
                j = min(i + temporal_stride, len(color_paths) - 1)
                color_t1 = color_paths[j]

                stem = _FRAME_RE.match(color_t0.name)
                if stem is None:
                    continue
                frame_id = stem.group(1)
                label_t0 = scene / f"{frame_id}-label.png"
                if not label_t0.exists():
                    # fall back to alt suffix
                    label_t0 = scene / f"{frame_id}-mask.png"
                if not label_t0.exists():
                    continue
                self.samples.append((str(color_t0), str(color_t1), str(label_t0)))

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            # Soft-fail with a clear error so eval scripts can skip
            self._missing = True
        else:
            self._missing = False

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _binarise_label(lbl: np.ndarray) -> np.ndarray:
        # ClearPose labels are uint16 instance maps; any non-zero pixel is a
        # transparent object instance.
        return (lbl > 0).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        if self._missing:
            raise FileNotFoundError(
                f"ClearPose root '{self.root}' contains no scenes. "
                "Download from https://github.com/opipari/ClearPose and extract."
            )

        p_t0, p_t1, p_lbl = self.samples[idx]
        img_t0 = cv2.cvtColor(cv2.imread(p_t0, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_t1 = cv2.cvtColor(cv2.imread(p_t1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        lbl = cv2.imread(p_lbl, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            raise FileNotFoundError(f"Cannot read label: {p_lbl}")
        mask = self._binarise_label(lbl)

        if img_t0.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img_t0.shape[1], img_t0.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img_t0, mask=mask, image_t1=img_t1)
        return {
            "image":      aug["image"],
            "image_t1":   aug["image_t1"],
            "mask":       aug["mask"].long(),
            "material":   torch.tensor(1, dtype=torch.long),
            "image_path": p_t0,
        }
