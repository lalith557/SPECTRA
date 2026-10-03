"""
spectra/data/video_dataset.py

Video dataset loader for temporal evaluation of transparent-object segmentation.

Supports two formats:
  - directory of frame folders with optional per-frame masks
  - raw .mp4 / .avi videos (mask-less, used for qualitative results)

The loader yields windows of T consecutive frames so we can measure
frame-to-frame temporal consistency (critical for video downstream tasks).

Expected directory layout for annotated videos:
    root/transparent_videos/
        clip01/
            frames/000001.jpg ... 000300.jpg
            masks/000001.png ... 000300.png   (optional)
        clip02/
            ...
"""
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import get_val_transforms


class TransparentVideoDataset(Dataset):
    """
    Returns windows of T consecutive frames + masks (if available).

    Each sample dict:
        frames:  (T, 3, H, W) float tensors normalised
        masks:   (T, H, W) long {0, 1} or None if unannotated
        clip:    str — clip identifier
        idx_start: starting frame index in clip
    """

    def __init__(
        self,
        root: str,
        clips: Optional[List[str]] = None,
        window_size: int = 4,
        stride: int = 2,                # window stride within a clip
        image_size: Tuple[int, int] = (448, 448),
        transform=None,
        require_masks: bool = False,
    ):
        super().__init__()
        self.root = Path(root) / "transparent_videos"
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.transform = transform or get_val_transforms(image_size)
        self.require_masks = require_masks

        all_clips = sorted([p for p in self.root.iterdir() if p.is_dir()]) \
            if self.root.exists() else []
        if clips is not None:
            all_clips = [p for p in all_clips if p.name in clips]

        self.samples: List[Tuple[Path, int, List[Path], Optional[List[Path]]]] = []
        for clip in all_clips:
            frame_dir = clip / "frames"
            mask_dir  = clip / "masks"
            frames    = sorted(glob.glob(str(frame_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(frame_dir / "*.png")))
            if len(frames) < window_size:
                continue
            masks = None
            if mask_dir.exists():
                masks = sorted(glob.glob(str(mask_dir / "*.png")))
                if len(masks) != len(frames):
                    masks = None  # fall back to unannotated
            if require_masks and masks is None:
                continue

            frame_paths = [Path(p) for p in frames]
            mask_paths  = [Path(p) for p in masks] if masks else None

            for start in range(0, len(frame_paths) - window_size + 1, stride):
                self.samples.append((clip, start, frame_paths, mask_paths))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        clip, start, frame_paths, mask_paths = self.samples[idx]
        T = self.window_size

        frames, masks = [], []
        H, W = self.image_size

        # Apply transform per-pair, sharing spatial randomness across the window
        first_img = cv2.cvtColor(cv2.imread(str(frame_paths[start])),
                                 cv2.COLOR_BGR2RGB)
        if mask_paths is not None:
            first_mask = (cv2.imread(str(mask_paths[start]),
                                     cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        else:
            first_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)

        for t in range(T):
            i = start + t
            img = cv2.cvtColor(cv2.imread(str(frame_paths[i])), cv2.COLOR_BGR2RGB)
            if mask_paths is not None:
                msk = (cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            else:
                msk = np.zeros(img.shape[:2], dtype=np.uint8)

            aug = self.transform(image=img, mask=msk, image_t1=img)
            frames.append(aug["image"])
            masks.append(aug["mask"].long())

        frames_t = torch.stack(frames, dim=0)        # (T, 3, H, W)
        masks_t  = torch.stack(masks, dim=0)         # (T, H, W)

        return {
            "frames":     frames_t,
            "masks":      masks_t,
            "clip":       clip.name,
            "idx_start":  start,
        }
