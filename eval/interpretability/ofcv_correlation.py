"""
spectra/eval/interpretability/ofcv_correlation.py

Measures how well the OFCV activation map correlates with ground-truth
transparent regions. This is the central interpretability claim of the paper:
the OFCV detector responds *specifically* to transparency physics.

Reported per-image:
  - Pearson r between OFCV(x,y) and GT(x,y)
  - AUROC: how well OFCV alone (no learned head!) discriminates transparent pixels
  - Mean OFCV inside vs outside GT mask
  - Statistical significance via paired permutation test

Outputs:
    results/interpretability/ofcv_correlation.json
    results/interpretability/ofcv_correlation.md
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure we can import sibling SPECTRA modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_config, get_logger, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset

logger = get_logger("spectra.ofcv_correlation")


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / max(denom, 1e-8))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUROC. O(n log n)."""
    scores = scores.ravel(); labels = labels.ravel()
    order = np.argsort(scores)
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, labels.size + 1, dtype=np.float64)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def paired_permutation_p(in_mean: float, out_mean: float,
                          in_vals: np.ndarray, out_vals: np.ndarray,
                          n_perm: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    combined = np.concatenate([in_vals, out_vals])
    n_in = in_vals.size
    obs = in_mean - out_mean
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        a = combined[:n_in].mean()
        b = combined[n_in:].mean()
        if abs(a - b) >= abs(obs):
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    p.add_argument("--dataset",    default="trans10k")
    p.add_argument("--split",      default="test")
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--output",     default="results/interpretability/ofcv_correlation.json")
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
    per_image = []

    for batch in tqdm(loader, desc="ofcv-corr"):
        batch = to_device(batch, device)
        out = model(batch["image"], batch["image_t1"], return_intermediates=True)
        ofcv = out.get("ofcv_map")
        if ofcv is None:
            continue
        ofcv = F.interpolate(ofcv, size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().cpu().numpy()
        gt = batch["mask"].squeeze().cpu().numpy().astype(np.float32)

        r = pearson_r(ofcv, gt)
        a = auroc(ofcv, gt.astype(np.int64))
        m_in  = float(ofcv[gt > 0].mean()) if gt.sum() > 0 else float("nan")
        m_out = float(ofcv[gt == 0].mean()) if (gt == 0).sum() > 0 else float("nan")
        p_perm = paired_permutation_p(
            m_in, m_out, ofcv[gt > 0], ofcv[gt == 0]
        ) if gt.sum() > 0 and (gt == 0).sum() > 0 else 1.0

        per_image.append({
            "image_path": batch["image_path"][0],
            "pearson_r":  r, "auroc": a,
            "mean_in":    m_in, "mean_out": m_out,
            "perm_p":     p_perm,
        })

    summary = {
        "n":                len(per_image),
        "pearson_r_mean":   float(np.nanmean([r["pearson_r"] for r in per_image])),
        "pearson_r_median": float(np.nanmedian([r["pearson_r"] for r in per_image])),
        "auroc_mean":       float(np.nanmean([r["auroc"] for r in per_image])),
        "mean_in_mean":     float(np.nanmean([r["mean_in"] for r in per_image])),
        "mean_out_mean":    float(np.nanmean([r["mean_out"] for r in per_image])),
        "frac_perm_sig":    float(np.mean([r["perm_p"] < 0.05 for r in per_image])),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    md = [
        "# OFCV — ground-truth correlation",
        "",
        f"- N images: **{summary['n']}**",
        f"- Mean Pearson r: **{summary['pearson_r_mean']:.4f}** "
        f"(median {summary['pearson_r_median']:.4f})",
        f"- Mean OFCV-alone AUROC: **{summary['auroc_mean']:.4f}**",
        f"- Mean OFCV inside  GT mask: {summary['mean_in_mean']:.4f}",
        f"- Mean OFCV outside GT mask: {summary['mean_out_mean']:.4f}",
        f"- Fraction images with p < 0.05 (permutation): "
        f"**{summary['frac_perm_sig']:.2%}**",
    ]
    with open(out.with_suffix(".md"), "w") as f:
        f.write("\n".join(md))
    logger.info(f"wrote: {out}")


if __name__ == "__main__":
    main()
