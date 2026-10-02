"""
train/run_vgsd_ablation.py

VGSD NATIVE ablation (#4): train the 5 variants on REAL video (true inter-frame
parallax) and test whether OFCV/BRF help when the flow signal is genuinely
present -- the question Trans10K (static frames) cannot answer.

Data: VGSD-D (Liu et al., AAAI 2024). If a proper train split exists at
--train-root it is used; otherwise the 105 local test clips are split
CLIP-DISJOINT 80/20 (dev mode, clearly labelled non-standard).

Results -> EXPERIMENTS/10_vgsd_native/<variant>/result.json (+ summary.json).

Usage:
  venv/Scripts/python.exe train/run_vgsd_ablation.py --epochs 12
  venv/Scripts/python.exe train/run_vgsd_ablation.py --train-root path/to/VGSD/train
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_config, to_device                            # noqa: E402
from models.spectra_model import SPECTRA                            # noqa: E402
from train.losses import SPECTRALoss                                # noqa: E402
from eval.metrics import TransparentObjectMetrics                   # noqa: E402
from data.augmentation import get_train_transforms, get_val_transforms  # noqa: E402

VARIANTS = [
    {"name": "neither",   "use_ofcv": False, "use_brf": False, "use_gnn": False},
    {"name": "ofcv_only", "use_ofcv": True,  "use_brf": False, "use_gnn": False},
    {"name": "brf_only",  "use_ofcv": False, "use_brf": True,  "use_gnn": False},
    {"name": "ofcv_brf",  "use_ofcv": True,  "use_brf": True,  "use_gnn": False},
]
OUT = ROOT / "EXPERIMENTS" / "10_vgsd_native"


def clip_pairs(clip_dir):
    frames = sorted(glob.glob(str(clip_dir / "JPEGImages" / "*.jpg")))
    pairs = []
    for i, f0 in enumerate(frames):
        f1 = frames[min(i + 1, len(frames) - 1)]
        m = clip_dir / "SegmentationClassPNG" / f"{Path(f0).stem}.png"
        if m.exists():
            pairs.append((f0, f1, str(m)))
    return pairs


def build_splits(train_root, dev_root, val_frac=0.2):
    """Return (train_pairs, val_pairs), clip-disjoint."""
    if train_root and Path(train_root).exists():
        tr_clips = sorted(p for p in Path(train_root).iterdir() if p.is_dir())
        va_root = Path(train_root).parent / "test"
        va_clips = sorted(p for p in va_root.iterdir() if p.is_dir()) if va_root.exists() else tr_clips[-5:]
        mode = "official-train-split"
    else:
        clips = sorted(p for p in Path(dev_root).iterdir() if p.is_dir())
        n_val = max(1, int(len(clips) * val_frac))
        va_clips, tr_clips = clips[:n_val], clips[n_val:]   # disjoint clips
        mode = f"DEV clip-disjoint split of {len(clips)} test clips ({len(tr_clips)} train / {len(va_clips)} val)"
    tr = [p for c in tr_clips for p in clip_pairs(c)]
    va = [p for c in va_clips for p in clip_pairs(c)]
    return tr, va, mode


class VGSDPairs(Dataset):
    def __init__(self, pairs, image_size, train):
        self.pairs = pairs
        self.tf = get_train_transforms(image_size) if train else get_val_transforms(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p0, p1, pm = self.pairs[i]
        i0 = cv2.cvtColor(cv2.imread(p0), cv2.COLOR_BGR2RGB)
        i1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(pm, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        if i0.shape[:2] != m.shape[:2]:
            m = cv2.resize(m, (i0.shape[1], i0.shape[0]), interpolation=cv2.INTER_NEAREST)
        a = self.tf(image=i0, mask=m, image_t1=i1)
        return {"image": a["image"], "image_t1": a["image_t1"],
                "mask": a["mask"].long(), "material": torch.tensor(1)}


@torch.no_grad()
def evaluate(model, loader, device, hw):
    model.eval(); met = TransparentObjectMetrics(); H, W = hw
    for b in loader:
        b = to_device(b, device)
        out = model(b["image"], b["image_t1"], return_intermediates=False)
        prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear", align_corners=False)
        met.update(prob, b["mask"])
    return met.compute()


def train_variant(v, tr_ds, va_ds, cfg, device, epochs, bs):
    H, W = cfg.data.image_size
    model = SPECTRA(cfg, use_gnn=v["use_gnn"], use_ofcv=v["use_ofcv"], use_brf=v["use_brf"]).to(device)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = AdamW([{"params": backbone, "lr": 1e-5}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
    crit = SPECTRALoss()
    scaler = torch.amp.GradScaler("cuda")
    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)
    steps = max(1, len(tl)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-7)
    per_epoch, best = [], -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tl:
            b = to_device(b, device)
            with torch.amp.autocast("cuda"):
                out = model(b["image"], b["image_t1"], return_intermediates=True)
                loss = crit(predictions=out, targets={"mask": b["mask"], "material": b["material"]})["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        m = evaluate(model, vl, device, (H, W))
        per_epoch.append({"epoch": ep, **m})
        best = max(best, m["iou"])
        print(f"   [{v['name']}] E{ep}/{epochs} val IoU={m['iou']:.4f}", flush=True)
    return {"variant": v["name"], "best_iou": best, "per_epoch": per_epoch,
            "config": {k: v[k] for k in ("use_ofcv", "use_brf", "use_gnn")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train-root", default="")
    ap.add_argument("--dev-root", default=str(ROOT / "VSGD_dataset" / "test"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2)  # 2 fits 8 GB (OFCV variants add RAFT); raise if you have more VRAM
    ap.add_argument("--image-size", type=int, default=0,
                    help="override cfg.data.image_size; must be divisible by 14 (DINOv2 patch) AND 8 (RAFT stride) -> multiples of 56, e.g. 336")
    args = ap.parse_args()

    if not Path(args.dev_root).exists() and not (args.train_root and Path(args.train_root).exists()):
        print("ERROR: no VGSD data found.\n"
              "Download VGSD-D (Liu et al., AAAI 2024): https://github.com/fawnliu/VGSD\n"
              "and pass --train-root, or place clips under VSGD_dataset/test/.")
        sys.exit(2)

    cfg = load_config(args.config)
    if args.image_size > 0:
        if args.image_size % 56 != 0:
            print(f"ERROR: --image-size {args.image_size} must be a multiple of 56 (LCM of 14 and 8).")
            sys.exit(2)
        cfg.data.image_size = [args.image_size, args.image_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, mode = build_splits(args.train_root, args.dev_root)
    print(f"[VGSD ablation] mode={mode}  train_pairs={len(tr)}  val_pairs={len(va)}  image_size={cfg.data.image_size}")
    OUT.mkdir(parents=True, exist_ok=True)
    tr_ds, va_ds = VGSDPairs(tr, cfg.data.image_size, True), VGSDPairs(va, cfg.data.image_size, False)

    summary = {"mode": mode, "n_train_pairs": len(tr), "n_val_pairs": len(va), "epochs": args.epochs, "variants": []}
    for v in VARIANTS:
        t0 = time.time()
        rec = train_variant(v, tr_ds, va_ds, cfg, device, args.epochs, args.bs)
        rec["train_min"] = round((time.time() - t0) / 60, 1)
        (OUT / v["name"]).mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(OUT / v["name"] / "result.json", "w"), indent=2)
        summary["variants"].append({"variant": v["name"], "best_iou": rec["best_iou"]})
        torch.cuda.empty_cache()
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"[done] {OUT/'summary.json'}")
    ious = [s["best_iou"] for s in summary["variants"]]
    print(f"  spread = {max(ious)-min(ious):.4f} across variants "
          f"(neither vs physics: does OFCV help on REAL video?)")


if __name__ == "__main__":
    main()
