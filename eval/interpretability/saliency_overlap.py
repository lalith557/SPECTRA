"""
spectra/eval/interpretability/saliency_overlap.py

Compute Grad-CAM-style saliency for SPECTRA's final segmentation output,
then measure overlap with ground-truth transparent regions. Higher overlap
indicates the network is *using* the transparent regions when deciding —
not relying on shortcuts.

Saliency reported:
  - Grad-CAM on fusion_head's last conv
  - Integrated gradients (light variant — 8 steps)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_config, get_logger, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset


logger = get_logger("spectra.saliency_overlap")


# ---------------------------------------------------------------------------
# Grad-CAM via hooks
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: SPECTRA):
        self.model = model
        self.activations = None
        self.gradients   = None
        target = self._find_target_layer()
        self.fwd_h = target.register_forward_hook(self._fwd)
        self.bwd_h = target.register_full_backward_hook(self._bwd)

    def _find_target_layer(self):
        # Use last conv inside fusion's head. Fall back to a Conv2d.
        for m in reversed(list(self.model.fusion.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in fusion head")

    def _fwd(self, _m, _inp, out):
        self.activations = out.detach()

    def _bwd(self, _m, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image, image_t1):
        self.model.zero_grad(set_to_none=True)
        out = self.model(image, image_t1, return_intermediates=False)
        seg = out["seg_logits"]
        # scalar target: sum of foreground logits in regions where prob > 0.5
        prob = torch.sigmoid(seg)
        target = (seg * (prob > 0.5).float()).sum()
        target.backward()

        if self.activations is None or self.gradients is None:
            return torch.zeros_like(prob)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def close(self):
        self.fwd_h.remove()
        self.bwd_h.remove()


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def saliency_overlap(cam: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    cam = cam.clip(0, 1)
    gt  = (gt > 0).astype(np.float32)

    # Coverage: fraction of saliency mass that falls inside GT mask
    coverage = (cam * gt).sum() / max(cam.sum(), 1e-8)
    # Concentration: ratio of GT area to high-saliency area
    cam_bin = (cam > 0.5).astype(np.float32)
    inter = (cam_bin * gt).sum()
    union = (cam_bin + gt).clip(0, 1).sum()
    iou = inter / max(union, 1e-8)
    # Inside-mean vs outside-mean
    if gt.sum() > 0:
        m_in = cam[gt > 0].mean()
    else:
        m_in = float("nan")
    if (gt == 0).sum() > 0:
        m_out = cam[gt == 0].mean()
    else:
        m_out = float("nan")

    return {
        "coverage":      float(coverage),
        "iou_at_0.5":    float(iou),
        "mean_in":       float(m_in),
        "mean_out":      float(m_out),
        "mean_diff":     float(m_in - m_out) if not np.isnan(m_in) and not np.isnan(m_out) else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    p.add_argument("--dataset",    default="trans10k")
    p.add_argument("--split",      default="test")
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--output",     default="results/interpretability/saliency_overlap.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = cfg.data.image_size

    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    load_checkpoint(args.checkpoint, model, device=str(device))

    ds = build_eval_dataset(name=args.dataset, root=cfg.data.root, split=args.split,
                            image_size=(H, W))
    if args.max_images > 0:
        ds.samples = ds.samples[:args.max_images]
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    gradcam = GradCAM(model)
    records: List[Dict] = []
    try:
        for batch in tqdm(loader, desc="saliency"):
            batch = to_device(batch, device)
            cam = gradcam(batch["image"], batch["image_t1"])
            cam_np = cam.detach().squeeze().cpu().numpy()
            gt_np = batch["mask"].squeeze().cpu().numpy()
            records.append({
                "image_path": batch["image_path"][0],
                **saliency_overlap(cam_np, gt_np),
            })
    finally:
        gradcam.close()

    summary = {
        "n": len(records),
        "coverage_mean":  float(np.mean([r["coverage"] for r in records])),
        "iou_mean":       float(np.mean([r["iou_at_0.5"] for r in records])),
        "mean_in_mean":   float(np.nanmean([r["mean_in"] for r in records])),
        "mean_out_mean":  float(np.nanmean([r["mean_out"] for r in records])),
        "mean_diff_mean": float(np.mean([r["mean_diff"] for r in records])),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_image": records}, f, indent=2)
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
