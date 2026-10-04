"""
eval/local_cross_dataset_eval.py

Real cross-dataset / video evaluation against the LOCAL on-disk datasets that
the original cross_dataset_eval.py missed because of path-layout mismatches:

  - ClearPose   : ./clearpose_dataset/{set8,set9}/scene*/{id}-color.png + {id}-label.png
  - VSGD (video): ./VSGD_dataset/test/<clip>/JPEGImages/*.jpg + SegmentationClassPNG/*.png

For VSGD we additionally run a controlled A/B:
  - video : image_t1 = genuine next frame  (real two-frame parallax -> real OFCV)
  - static: image_t1 = image              (duplicate frame -> degenerate OFCV)
This isolates whether a true second frame changes anything (the core question).

Usage:
  venv/Scripts/python.exe eval/local_cross_dataset_eval.py --max 1200
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_config, load_checkpoint, to_device          # noqa: E402
from models.spectra_model import SPECTRA                            # noqa: E402
from eval.metrics import TransparentObjectMetrics                   # noqa: E402
from data.augmentation import get_val_transforms                   # noqa: E402

_CP_RE = re.compile(r"(\d+)-color\.png$", re.IGNORECASE)


def _even_subsample(items, max_n):
    if max_n <= 0 or len(items) <= max_n:
        return items
    step = len(items) / max_n
    return [items[int(i * step)] for i in range(max_n)]


def clearpose_samples(root="clearpose_dataset", max_n=-1):
    root = ROOT / root
    samples = []  # (img_t0, img_t1, label)
    for scene in sorted(p for p in root.rglob("scene*") if p.is_dir()):
        colors = sorted(scene.glob("*-color.png"))
        for i, c0 in enumerate(colors):
            c1 = colors[min(i + 1, len(colors) - 1)]
            m = _CP_RE.search(c0.name)
            if not m:
                continue
            lbl = scene / f"{m.group(1)}-label.png"
            if lbl.exists():
                samples.append((str(c0), str(c1), str(lbl)))
    return _even_subsample(samples, max_n)


def vsgd_samples(root="VSGD_dataset/test", max_n=-1):
    root = ROOT / root
    samples = []  # (img_t0, img_t1, mask_t0)
    for clip in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(glob.glob(str(clip / "JPEGImages" / "*.jpg")))
        for i, f0 in enumerate(frames):
            f1 = frames[min(i + 1, len(frames) - 1)]
            stem = Path(f0).stem
            mask = clip / "SegmentationClassPNG" / f"{stem}.png"
            if mask.exists():
                samples.append((f0, f1, str(mask)))
    return _even_subsample(samples, max_n)


class PairDataset(Dataset):
    """Generic (img_t0, img_t1, mask) loader with val transforms.
    static=True duplicates the first frame as the 'next' frame."""

    def __init__(self, samples, image_size=(448, 448), static=False):
        self.samples = samples
        self.static = static
        self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p0, p1, pm = self.samples[idx]
        img0 = cv2.cvtColor(cv2.imread(p0, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img1 = img0 if self.static else cv2.cvtColor(cv2.imread(p1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(pm, cv2.IMREAD_GRAYSCALE)
        mask = (lbl > 0).astype(np.uint8)
        if img0.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
        aug = self.transform(image=img0, mask=mask, image_t1=img1)
        return {"image": aug["image"], "image_t1": aug["image_t1"], "mask": aug["mask"].long()}


@torch.no_grad()
def evaluate(model, ds, device, bs=4, image_size=(448, 448)):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    metrics = TransparentObjectMetrics()
    model.eval()
    H, W = image_size
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["image"], batch["image_t1"], return_intermediates=False)
        prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear", align_corners=False)
        metrics.update(prob, batch["mask"])
    res = metrics.compute()
    res["n_images"] = len(ds)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    ap.add_argument("--max", type=int, default=1200, help="max samples per dataset (-1=all)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="outputs/cross_dataset/local_results.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = cfg.data.image_size
    print(f"[device] {device}")

    model = SPECTRA(cfg, use_gnn=False).to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, device=str(device))
    print(f"[model ] loaded {args.checkpoint}")

    rows = []

    cp = clearpose_samples(max_n=args.max)
    print(f"[data  ] ClearPose pairs: {len(cp)}")
    if cp:
        r = evaluate(model, PairDataset(cp, (H, W)), device, args.bs, (H, W))
        r.update(dataset="clearpose", mode="video", split="set8+set9")
        rows.append(r)
        print("  clearpose(video) ", r)

    vs = vsgd_samples(max_n=args.max)
    print(f"[data  ] VSGD pairs: {len(vs)}")
    if vs:
        rv = evaluate(model, PairDataset(vs, (H, W), static=False), device, args.bs, (H, W))
        rv.update(dataset="vsgd", mode="video", split="test")
        rows.append(rv)
        print("  vsgd(video)  ", rv)
        rs = evaluate(model, PairDataset(vs, (H, W), static=True), device, args.bs, (H, W))
        rs.update(dataset="vsgd", mode="static", split="test")
        rows.append(rs)
        print("  vsgd(static) ", rs)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": args.checkpoint, "max_per_dataset": args.max, "rows": rows},
              open(outp, "w"), indent=2)
    print(f"[done  ] wrote {outp}")


if __name__ == "__main__":
    main()
