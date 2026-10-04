"""
spectra/eval/failure_taxonomy.py

Quantitative, automatically-labelled failure-mode taxonomy:

    reflection      — high specular highlight inside GT mask
    motion_blur     — high RAFT residual everywhere (flow degraded)
    thin_glass      — GT mask < 2% of image and high aspect ratio
    strong_occlusion— GT mask broken into many small components

For each bucket we report:
  - sample count
  - mean IoU within bucket (model performance)
  - mean IoU in the complement bucket (baseline)
  - gap (transparency-physics-specific weakness)

Outputs:
    results/failure_taxonomy/taxonomy.json
    results/failure_taxonomy/taxonomy.md
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_config, get_logger, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset
from eval.metrics import compute_iou

logger = get_logger("spectra.failure_taxonomy")


# ---------------------------------------------------------------------------
# Bucket detectors (work on numpy arrays)
# ---------------------------------------------------------------------------

def detect_reflection(img_rgb: np.ndarray, gt: np.ndarray) -> bool:
    """High specular highlight inside transparent region."""
    if gt.sum() < 100:
        return False
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    inside = gray[gt > 0]
    # >5% of GT pixels are near-saturated → likely strong specular highlight
    return float((inside > 0.92).mean()) > 0.05


def detect_motion_blur(img_rgb: np.ndarray, residual: np.ndarray) -> bool:
    """Flow residual is uniformly high — model can't isolate object."""
    if residual is None:
        return False
    return float(residual.mean()) > 0.55 and float(residual.std()) < 0.10


def detect_thin_glass(gt: np.ndarray) -> bool:
    coverage = float(gt.mean())
    if coverage > 0.04 or coverage == 0:
        return False
    # Use moments to estimate aspect ratio of the mask
    ys, xs = np.where(gt > 0)
    if len(xs) < 20:
        return False
    h = ys.max() - ys.min(); w = xs.max() - xs.min()
    aspect = max(h, w) / max(1, min(h, w))
    return aspect > 4.0


def detect_strong_occlusion(gt: np.ndarray) -> bool:
    if gt.sum() < 100:
        return False
    n_components, _ = cv2.connectedComponents((gt > 0).astype(np.uint8))
    return n_components > 4


BUCKETS = {
    "reflection":        detect_reflection,
    "motion_blur":       detect_motion_blur,
    "thin_glass":        detect_thin_glass,
    "strong_occlusion":  detect_strong_occlusion,
}


# ---------------------------------------------------------------------------
# Per-image record builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    p.add_argument("--dataset",    default="trans10k")
    p.add_argument("--split",      default="test")
    p.add_argument("--output",     default="results/failure_taxonomy/taxonomy.json")
    p.add_argument("--max-images", type=int, default=-1)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = cfg.data.image_size

    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    load_checkpoint(args.checkpoint, model, device=str(device))
    ds = build_eval_dataset(name=args.dataset, root=cfg.data.root, split=args.split,
                            image_size=(H, W))
    if args.max_images > 0 and hasattr(ds, "samples"):
        ds.samples = ds.samples[:args.max_images]
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    records: List[Dict] = []
    mean_rgb = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std_rgb  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for batch in tqdm(loader, desc="taxonomy"):
        batch = to_device(batch, device)
        out = model(batch["image"], batch["image_t1"], return_intermediates=True)
        prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().cpu().numpy()
        pred_bin = (prob > 0.5).astype(np.uint8)

        gt = batch["mask"].squeeze().cpu().numpy().astype(np.uint8)
        iou = compute_iou(torch.from_numpy(pred_bin), torch.from_numpy(gt))

        # Get unnormalised RGB
        img = batch["image"].squeeze().cpu().numpy()
        img_rgb = (img * std_rgb + mean_rgb) * 255
        img_rgb = img_rgb.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

        residual = out.get("residual_map")
        residual_np = residual.squeeze().cpu().numpy() if residual is not None else None

        labels = []
        if detect_reflection(img_rgb, gt):        labels.append("reflection")
        if detect_motion_blur(img_rgb, residual_np): labels.append("motion_blur")
        if detect_thin_glass(gt):                 labels.append("thin_glass")
        if detect_strong_occlusion(gt):           labels.append("strong_occlusion")
        if not labels:
            labels = ["clean"]

        records.append({
            "image_path": batch["image_path"][0],
            "iou": float(iou),
            "labels": labels,
        })

    # Aggregate
    summary: Dict[str, Dict] = {}
    iou_all = np.array([r["iou"] for r in records])
    for bucket in list(BUCKETS) + ["clean"]:
        in_b  = np.array([r["iou"] for r in records if bucket in r["labels"]])
        out_b = np.array([r["iou"] for r in records if bucket not in r["labels"]])
        summary[bucket] = {
            "n_in":        int(in_b.size),
            "n_out":       int(out_b.size),
            "iou_in":      float(in_b.mean())  if in_b.size  else 0.0,
            "iou_out":     float(out_b.mean()) if out_b.size else 0.0,
            "iou_gap":     float((in_b.mean() if in_b.size else 0.0)
                                 - (out_b.mean() if out_b.size else 0.0)),
            "share_of_total": float(in_b.size / max(len(records), 1)),
        }

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n": len(records), "iou_overall": float(iou_all.mean()),
            "summary": summary, "per_image": records,
        }, f, indent=2)
    md = ["# SPECTRA failure-mode taxonomy",
          "",
          f"- Total images: **{len(records)}**",
          f"- Overall mean IoU: **{iou_all.mean():.4f}**",
          "",
          "| bucket | n_in | share | IoU(in) | IoU(out) | gap |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, v in summary.items():
        md.append(f"| {k} | {v['n_in']} | {v['share_of_total']:.2%} | "
                  f"{v['iou_in']:.4f} | {v['iou_out']:.4f} | {v['iou_gap']:+.4f} |")
    with open(out.with_suffix(".md"), "w") as f:
        f.write("\n".join(md))
    logger.info(f"wrote: {out}")


if __name__ == "__main__":
    main()
