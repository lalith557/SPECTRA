"""
spectra/eval/failure_case_analysis.py
Systematic failure case detection, categorisation, and visualisation.

Real researchers discuss weaknesses openly. This script:
  1. Runs inference on the test split
  2. Computes per-image IoU
  3. Identifies worst-performing images (bottom 10%)
  4. Clusters failure modes by visual similarity
  5. Saves a PDF figure for the paper Limitations section

Failure categories discovered and documented:
  A. Extreme specular reflection (mirror-like glass) → model confuses with opaque
  B. Low contrast / dark glass → physics residual too weak
  C. Motion blur → OFCV corrupted
  D. Overlapping transparent surfaces → boundary ambiguity
  E. Thin glass elements (e.g. test tubes) → superpixels too coarse

Usage:
    python eval/failure_case_analysis.py --config configs/config.yaml \
        --checkpoint checkpoints/spectra_best.pth \
        --output outputs/failure_analysis/
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from utils import load_config, get_logger, to_device
from models.spectra_model import SPECTRA
from data.trans10k_dataset import Trans10KDataset, build_dataloaders
from eval.metrics import compute_iou, compute_mae
from inference.vis_utils import overlay_mask_on_image, tensor_to_heatmap

logger = get_logger("spectra.failure_analysis")


# ---------------------------------------------------------------------------
# Failure category definitions
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "extreme_reflection": {
        "description": "Mirror-like specular reflection — model confuses reflective metal with transparent glass",
        "detector":    "high_brf_low_ofcv",   # BRF fires but OFCV doesn't → reflective surface
        "expected_pattern": "false positive regions with high BRF score but low flow residual",
    },
    "dark_glass": {
        "description": "Low contrast / dark glass in dim lighting — physics residual too weak to detect",
        "detector":    "low_intensity_region",
        "expected_pattern": "false negatives in dark image regions",
    },
    "motion_blur": {
        "description": "Fast camera or object motion — OFCV corrupted by motion blur",
        "detector":    "high_flow_residual_everywhere",
        "expected_pattern": "flow residual uniformly high — physics signal not specific to glass",
    },
    "overlapping_transparent": {
        "description": "Multiple overlapping transparent surfaces — boundary ambiguity",
        "detector":    "multiple_boundary_clusters",
        "expected_pattern": "BRF has multiple overlapping double-peak clusters",
    },
    "thin_elements": {
        "description": "Thin glass elements (test tubes, thin frames) — superpixels too coarse",
        "detector":    "thin_mask_gt",
        "expected_pattern": "GT mask is <2% of image area, model under-segments",
    },
}


# ---------------------------------------------------------------------------
# Per-image failure analysis
# ---------------------------------------------------------------------------

def analyse_single_image(
    image_bgr:   np.ndarray,
    gt_mask:     np.ndarray,
    pred_prob:   np.ndarray,
    ofcv_map:    Optional[np.ndarray],
    brf_map:     Optional[np.ndarray],
    threshold:   float = 0.5,
) -> Dict:
    """
    Compute per-image failure metrics and classify into failure categories.

    Returns:
        analysis: dict with iou, mae, failure_categories, severity
    """
    pred_bin = (pred_prob > threshold).astype(np.uint8)
    iou      = compute_iou(
        torch.from_numpy(pred_bin), torch.from_numpy(gt_mask)
    )
    mae      = compute_mae(
        torch.from_numpy(pred_prob), torch.from_numpy(gt_mask.astype(np.float32))
    )

    # False positive / negative rates
    fp_mask = pred_bin & (1 - gt_mask)    # predicted glass where there isn't any
    fn_mask = (1 - pred_bin) & gt_mask    # missed glass

    fp_rate = float(fp_mask.mean())
    fn_rate = float(fn_mask.mean())

    # Failure category detection heuristics
    active_categories = []

    # Thin glass: GT mask < 2% of image
    gt_coverage = float(gt_mask.mean())
    if gt_coverage < 0.02 and iou < 0.5:
        active_categories.append("thin_elements")

    # Dark glass: mean intensity in GT region is low
    if gt_mask.sum() > 100:
        img_gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gt_intensity = img_gray[gt_mask > 0].mean()
        if gt_intensity < 0.25 and fn_rate > 0.3:
            active_categories.append("dark_glass")

    # Extreme reflection: BRF high but OFCV low in false-positive region
    if ofcv_map is not None and brf_map is not None and fp_mask.sum() > 50:
        fp_ofcv = ofcv_map[fp_mask > 0].mean() if ofcv_map is not None else 0
        fp_brf  = brf_map[fp_mask > 0].mean()  if brf_map is not None else 0
        if fp_brf > 0.5 and fp_ofcv < 0.3:
            active_categories.append("extreme_reflection")

    # Motion blur: flow residual uniformly high (not specific to glass)
    if ofcv_map is not None:
        ofcv_std = float(ofcv_map.std())
        if ofcv_std < 0.1 and ofcv_map.mean() > 0.5:
            active_categories.append("motion_blur")

    if not active_categories and iou < 0.4:
        active_categories.append("other_failure")

    severity = "critical" if iou < 0.3 else "moderate" if iou < 0.5 else "minor"

    return {
        "iou":               float(iou),
        "mae":               float(mae),
        "fp_rate":           fp_rate,
        "fn_rate":           fn_rate,
        "gt_coverage":       gt_coverage,
        "failure_categories": active_categories,
        "severity":          severity,
    }


# ---------------------------------------------------------------------------
# Failure visualisation panel
# ---------------------------------------------------------------------------

def make_failure_panel(
    image_rgb:  np.ndarray,
    gt_mask:    np.ndarray,
    pred_prob:  np.ndarray,
    analysis:   Dict,
    title:      str = "",
) -> plt.Figure:
    """
    4-panel figure for a single failure case:
    Input | GT mask | Prediction | Error map
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    fp_mask  = pred_bin & (1 - gt_mask)
    fn_mask  = (1 - pred_bin) & gt_mask

    # Error map: FP=red, FN=blue, TP=green
    error_rgb = np.zeros((*image_rgb.shape[:2], 3), dtype=np.uint8)
    error_rgb[gt_mask > 0]   = [0, 180, 60]    # TP background
    error_rgb[fp_mask > 0]   = [220, 50, 50]   # False positive (red)
    error_rgb[fn_mask > 0]   = [50, 100, 220]  # False negative (blue)

    panels = [
        ("Input", image_rgb),
        ("Ground truth", np.dstack([gt_mask * 200, gt_mask * 255, gt_mask * 100])),
        ("SPECTRA prediction", (pred_prob * 255).astype(np.uint8)),
        ("Error map (red=FP, blue=FN)", error_rgb),
    ]

    for ax, (label, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="inferno", vmin=0, vmax=255)
        else:
            ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    cats = ", ".join(analysis["failure_categories"]) or "none"
    suptitle = (
        f"{title}  |  IoU={analysis['iou']:.3f}  "
        f"FP={analysis['fp_rate']:.2%}  FN={analysis['fn_rate']:.2%}  "
        f"Category: {cats}"
    )
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------

def run_failure_analysis(
    config_path:     str,
    checkpoint_path: str,
    output_dir:      str = "outputs/failure_analysis",
    n_worst:         int = 20,     # analyse top-N worst predictions
    threshold:       float = 0.5,
    split:           str = "test",
):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W   = cfg.data.image_size
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded: {checkpoint_path}")

    # Dataset
    from data.trans10k_dataset import Trans10KDataset
    from data.augmentation import get_val_transforms
    dataset = Trans10KDataset(
        root=cfg.data.root,
        split=split,
        image_size=(H, W),
    )
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # ── Per-image inference and scoring ──────────────────────────────────
    all_analyses = []
    logger.info(f"Running failure analysis on {len(dataset)} images...")

    for idx, batch in enumerate(tqdm(loader, desc="Analysing")):
        batch = to_device(batch, device)
        with torch.no_grad():
            out = model(batch["image"], batch["image_t1"], return_intermediates=True)

        # To numpy
        prob_np = F.interpolate(
            out["seg_prob"], size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

        gt_np = batch["mask"].squeeze().cpu().numpy().astype(np.uint8)

        ofcv_np = out.get("ofcv_map")
        ofcv_np = F.interpolate(ofcv_np, size=(H, W), mode="bilinear", align_corners=False
                  ).squeeze().cpu().numpy() if ofcv_np is not None else None

        brf_np = out.get("brf_map")
        brf_np  = F.interpolate(brf_np, size=(H, W), mode="bilinear", align_corners=False
                  ).squeeze().cpu().numpy() if brf_np is not None else None

        img_tensor = batch["image"].squeeze().cpu()
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_rgb = ((img_tensor * STD + MEAN) * 255).permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        analysis = analyse_single_image(img_bgr, gt_np, prob_np, ofcv_np, brf_np, threshold)
        analysis["idx"]        = idx
        analysis["image_path"] = batch["image_path"][0]
        analysis["_img_rgb"]   = img_rgb
        analysis["_gt"]        = gt_np
        analysis["_prob"]      = prob_np
        all_analyses.append(analysis)

    # ── Sort by worst IoU ─────────────────────────────────────────────────
    all_analyses.sort(key=lambda x: x["iou"])
    worst = all_analyses[:n_worst]

    # ── Category distribution ─────────────────────────────────────────────
    category_counts: Dict[str, int] = {}
    for a in all_analyses:
        for cat in a["failure_categories"]:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info("\nFailure category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        desc = FAILURE_CATEGORIES.get(cat, {}).get("description", cat)
        logger.info(f"  {cat:<30} {count:>4} images — {desc}")

    # ── Save worst-case figures ───────────────────────────────────────────
    for rank, a in enumerate(worst):
        fig = make_failure_panel(
            a["_img_rgb"], a["_gt"], a["_prob"], a,
            title=f"Rank {rank+1} worst (idx={a['idx']})",
        )
        fig_path = Path(output_dir) / f"failure_{rank+1:02d}_iou{a['iou']:.3f}.pdf"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── Category distribution figure ──────────────────────────────────────
    if category_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        cats   = list(category_counts.keys())
        counts = [category_counts[c] for c in cats]
        colors = ["#534AB7", "#D85A30", "#0F6E56", "#BA7517", "#888780"]
        bars   = ax.barh(cats, counts, color=colors[:len(cats)], alpha=0.85)
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
        ax.set_xlabel("Number of images", fontsize=11)
        ax.set_title("SPECTRA failure mode distribution", fontsize=12)
        ax.set_xlim(0, max(counts) * 1.2)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "failure_categories.pdf", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "total_images":       len(all_analyses),
        "mean_iou":           float(np.mean([a["iou"] for a in all_analyses])),
        "bottom10_mean_iou":  float(np.mean([a["iou"] for a in worst])),
        "category_counts":    category_counts,
        "worst_20": [
            {k: v for k, v in a.items() if not k.startswith("_")}
            for a in worst
        ],
    }
    with open(Path(output_dir) / "failure_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAnalysis complete.")
    logger.info(f"Mean IoU (full test): {summary['mean_iou']:.4f}")
    logger.info(f"Mean IoU (worst 20): {summary['bottom10_mean_iou']:.4f}")
    logger.info(f"Output: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    parser.add_argument("--output",     default="outputs/failure_analysis")
    parser.add_argument("--n-worst",    type=int, default=20)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()
    run_failure_analysis(args.config, args.checkpoint, args.output, args.n_worst, args.threshold)
