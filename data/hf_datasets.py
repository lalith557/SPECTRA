"""
spectra/data/hf_datasets.py

HuggingFace-hosted dataset loaders. Three datasets supported:

    gsd_s     - garrying/GSD-S       (single-image glass segmentation, RGB)
    rgbd_gsd  - garrying/RGBD-GSD    (RGB + depth glass segmentation)
    vgsd_d    - garrying/VGSD-D      (video glass surface dataset, with depth)

Each dataset is wrapped to match SPECTRA's standard sample contract:
    {
        "image":      (3, H, W) float tensor (normalised)
        "image_t1":   (3, H, W) float tensor  (next frame, or copy for static)
        "mask":       (H, W) long {0, 1}
        "material":   long scalar (1 = glass)
        "image_path": str (HF index used as id)
        "depth":      (1, H, W) float tensor, only for rgbd_gsd / vgsd_d
    }

These wrap the HF dataset interface — first call downloads & caches.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Lazy importer — keep `datasets` optional in the inference image
# ---------------------------------------------------------------------------

def _hf_load(repo: str, split: Optional[str] = None):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required: pip install datasets"
        ) from e
    return load_dataset(repo, split=split) if split else load_dataset(repo)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_np(pil_img) -> np.ndarray:
    if pil_img is None:
        return None
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _binarise(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1), np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# GSD-S (single-image)
# ---------------------------------------------------------------------------

class HFGsdSDataset(Dataset):
    """
    Loader for `garrying/GSD-S` from HuggingFace.

    Sample keys provided by the HF dataset:
        image_id, image (PIL), mask (PIL binary), seg (PIL semantic),
        seg_colored (PIL viz)

    image_t1 is set to the same image (no video frames in GSD-S).
    """

    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (448, 448),
        transform=None,
        max_samples: int = -1,
    ):
        super().__init__()
        self.ds = _hf_load("garrying/GSD-S", split=split)
        self.image_size = image_size
        self.transform = transform or (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )
        self._n = len(self.ds) if max_samples < 0 else min(max_samples, len(self.ds))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict:
        rec = self.ds[idx]
        img  = _pil_to_np(rec["image"])
        mask = _binarise(_pil_to_np(rec["mask"]))
        if img.shape[:2] != mask.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        aug = self.transform(image=img, mask=mask, image_t1=img.copy())
        return {
            "image":      aug["image"],
            "image_t1":   aug["image_t1"],
            "mask":       aug["mask"].long(),
            "material":   torch.tensor(1, dtype=torch.long),
            "image_path": str(rec.get("image_id", idx)),
        }


# ---------------------------------------------------------------------------
# RGBD-GSD (RGB + depth)
# ---------------------------------------------------------------------------

class HFRgbdGsdDataset(Dataset):
    """
    Loader for `garrying/RGBD-GSD`. Sample keys:
        image_id, image (PIL), mask (PIL binary), depth (PIL 16-bit or float)
    """

    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (448, 448),
        transform=None,
        use_depth: bool = True,
        max_samples: int = -1,
    ):
        super().__init__()
        self.ds = _hf_load("garrying/RGBD-GSD", split=split)
        self.image_size = image_size
        self.use_depth = use_depth
        self.transform = transform or (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )
        self._n = len(self.ds) if max_samples < 0 else min(max_samples, len(self.ds))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict:
        rec = self.ds[idx]
        img  = _pil_to_np(rec["image"])
        mask = _binarise(_pil_to_np(rec["mask"]))
        if img.shape[:2] != mask.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img, mask=mask, image_t1=img.copy())

        sample = {
            "image":      aug["image"],
            "image_t1":   aug["image_t1"],
            "mask":       aug["mask"].long(),
            "material":   torch.tensor(1, dtype=torch.long),
            "image_path": str(rec.get("image_id", idx)),
        }
        if self.use_depth and "depth" in rec:
            depth = _pil_to_np(rec["depth"]).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth / max(depth.max(), 1.0)
            # Resize depth to match image
            import cv2
            depth = cv2.resize(depth, (sample["image"].shape[-1],
                                       sample["image"].shape[-2]),
                               interpolation=cv2.INTER_LINEAR)
            sample["depth"] = torch.from_numpy(depth).unsqueeze(0).float()
        return sample


# ---------------------------------------------------------------------------
# VGSD-D (Video Glass Surface Dataset with Depth)
# ---------------------------------------------------------------------------

class HFVgsdDataset(Dataset):
    """
    Loader for `garrying/VGSD-D` — video glass surface dataset.

    The HF dataset is expected to expose temporal frames with fields like:
        clip_id, frame_index, image (PIL), image_next (PIL),
        mask (PIL), depth (PIL, optional)

    If `image_next` is not present we look up the next frame within the
    same `clip_id` ourselves.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (448, 448),
        transform=None,
        max_samples: int = -1,
    ):
        super().__init__()
        try:
            self.ds = _hf_load("garrying/VGSD-D", split=split)
        except Exception as e:
            # Dataset may not exist publicly under that exact name; surface a
            # helpful message but don't crash module import.
            self.ds = None
            self._error = str(e)
            return
        self._error = None
        self.image_size = image_size
        self.transform = transform or (
            get_train_transforms(image_size) if split == "train"
            else get_val_transforms(image_size)
        )

        # Pre-index by clip so we can find next-frames if image_next is missing.
        self._clip_index = {}
        for i, rec in enumerate(self.ds):
            cid = rec.get("clip_id") or rec.get("clip") or "default"
            self._clip_index.setdefault(cid, []).append(i)

        self._n = len(self.ds) if max_samples < 0 else min(max_samples, len(self.ds))

    def __len__(self) -> int:
        return 0 if self.ds is None else self._n

    def _next_frame(self, rec, idx: int):
        cid = rec.get("clip_id") or rec.get("clip") or "default"
        order = self._clip_index[cid]
        pos = order.index(idx)
        return self.ds[order[min(pos + 1, len(order) - 1)]]

    def __getitem__(self, idx: int) -> Dict:
        if self.ds is None:
            raise RuntimeError(f"VGSD-D not loaded: {self._error}")
        rec = self.ds[idx]
        img = _pil_to_np(rec["image"])
        if "image_next" in rec and rec["image_next"] is not None:
            img_t1 = _pil_to_np(rec["image_next"])
        else:
            img_t1 = _pil_to_np(self._next_frame(rec, idx)["image"])
        mask = _binarise(_pil_to_np(rec["mask"])) if "mask" in rec else None

        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        if img.shape[:2] != mask.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img, mask=mask, image_t1=img_t1)
        sample = {
            "image":      aug["image"],
            "image_t1":   aug["image_t1"],
            "mask":       aug["mask"].long(),
            "material":   torch.tensor(1, dtype=torch.long),
            "image_path": f"{rec.get('clip_id', 'clip')}/{rec.get('frame_index', idx)}",
        }
        if "depth" in rec and rec["depth"] is not None:
            depth = _pil_to_np(rec["depth"]).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth / max(depth.max(), 1.0)
            import cv2
            depth = cv2.resize(depth, (sample["image"].shape[-1],
                                       sample["image"].shape[-2]),
                               interpolation=cv2.INTER_LINEAR)
            sample["depth"] = torch.from_numpy(depth).unsqueeze(0).float()
        return sample


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------

def register_hf_datasets(registry: dict) -> dict:
    """
    Add HF-backed datasets to an existing DATASETS-style registry. We use a
    lightweight adapter so signatures stay (root, split, image_size).
    """

    def gsd_s(root=None, split="train", image_size=(448, 448), **kw):
        return HFGsdSDataset(split=split, image_size=tuple(image_size), **kw)

    def rgbd_gsd(root=None, split="train", image_size=(448, 448), **kw):
        return HFRgbdGsdDataset(split=split, image_size=tuple(image_size), **kw)

    def vgsd_d(root=None, split="train", image_size=(448, 448), **kw):
        return HFVgsdDataset(split=split, image_size=tuple(image_size), **kw)

    registry.update({
        "gsd_s":    gsd_s,
        "rgbd_gsd": rgbd_gsd,
        "vgsd_d":   vgsd_d,
    })
    return registry
