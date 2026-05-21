"""
spectra/scripts/build_portfolio_visuals.py
Produce presentation-quality figures for the portfolio / paper:

  fig01_training_story.png  - val IoU + OFCV variance vs epoch (dual axis)
  fig02_robustness_bars.png - bar chart of IoU @ severity 5 per corruption
  fig03_ablation.png        - ablation bars at E1 vs E5 for full/no_ofcv/no_brf
  fig04_before_after.pdf    - 4 samples x (input | E1 pred | E10 pred | OFCV E10)

Reads metrics from results/causal_model/* JSONs and the training log.
Re-runs inference for the before/after figure since we need both E1 + E10 outputs.
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders


# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Fig 1 — training story (IoU + OFCV variance vs epoch)
# ---------------------------------------------------------------------------

def parse_train_log(log_path):
    """Pull (epoch, ofcv_var) per-step and (epoch, val_iou) per validation."""
    ofcv_by_epoch = {}
    val_iou_by_epoch = {}
    step_re = re.compile(r"\[E(\d+)\|S(\d+)\] .* ofcv_var=([\d.eE+-]+)")
    val_re  = re.compile(r"\[Val E(\d+)\] IoU=([\d.]+)")
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = step_re.search(line)
        if m:
            ep = int(m.group(1))
            v  = float(m.group(3))
            ofcv_by_epoch.setdefault(ep, []).append(v)
        m = val_re.search(line)
        if m:
            val_iou_by_epoch[int(m.group(1))] = float(m.group(2))
    return ofcv_by_epoch, val_iou_by_epoch


def fig_training_story(log_path, out_path):
    ofcv, val_iou = parse_train_log(log_path)
    epochs = sorted(ofcv.keys())
    ofcv_mean = [np.mean(ofcv[e]) for e in epochs]

    val_epochs = sorted(val_iou.keys())
    val_vals   = [val_iou[e] for e in val_epochs]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color_ofcv = "#534AB7"
    color_iou  = "#0F6E56"

    l1 = ax1.plot(epochs, ofcv_mean, "o-", color=color_ofcv,
                  linewidth=2, markersize=7, label="OFCV variance (mean per epoch)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("OFCV variance", color=color_ofcv)
    ax1.tick_params(axis="y", labelcolor=color_ofcv)
    ax1.set_ylim(0, max(ofcv_mean) * 1.15)
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    l2 = ax2.plot(val_epochs, val_vals, "s-", color=color_iou,
                  linewidth=2, markersize=8, label="Val IoU")
    ax2.set_ylabel("Validation IoU", color=color_iou)
    ax2.tick_params(axis="y", labelcolor=color_iou)
    ax2.set_ylim(0.80, 0.95)
    ax2.spines["right"].set_visible(True)

    for e, v in zip(val_epochs, val_vals):
        ax2.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=10, color=color_iou)

    plt.title("Training trajectory: OFCV variance grows while IoU climbs",
              fontweight="bold")
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 2 — robustness bar chart
# ---------------------------------------------------------------------------

def fig_robustness_bars(json_path, out_path):
    data = json.loads(Path(json_path).read_text())
    clean_iou = data["clean"]["iou"]
    corruptions = [k for k in data.keys() if k != "clean"]
    sev5_iou = [data[c]["5"]["iou"] for c in corruptions]
    drops    = [clean_iou - i for i in sev5_iou]

    # Sort by severity-5 IoU descending (best-handled first)
    order = sorted(range(len(corruptions)), key=lambda i: -sev5_iou[i])
    corruptions = [corruptions[i] for i in order]
    sev5_iou    = [sev5_iou[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#0F6E56" if i > 0.85 else ("#BA7517" if i > 0.78 else "#D85A30")
              for i in sev5_iou]
    bars = ax.bar([c.replace("_", " ") for c in corruptions], sev5_iou,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(clean_iou, color="#888780", linestyle="--", linewidth=1.5,
               label=f"Clean baseline ({clean_iou:.3f})")
    for b, v in zip(bars, sev5_iou):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel("IoU @ severity 5")
    ax.set_title("Hard-case robustness — IoU at maximum corruption severity",
                 fontweight="bold")
    ax.legend(loc="lower left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 3 — ablation (E1 vs E5 for full/no_ofcv/no_brf)
# ---------------------------------------------------------------------------

def fig_ablation(json_path, out_path):
    data = json.loads(Path(json_path).read_text())
    variants = data["variants"]
    names = [v["name"] for v in variants]
    e1 = [v["epochs"][0]["iou"] for v in variants]
    e5 = [v["epochs"][1]["iou"] for v in variants]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(x - w / 2, e1, w, color="#BA7517", edgecolor="black",
                linewidth=0.5, label="Epoch 1")
    b5 = ax.bar(x + w / 2, e5, w, color="#0F6E56", edgecolor="black",
                linewidth=0.5, label="Epoch 5")

    for bars in (b1, b5):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names])
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Val IoU")
    ax.set_title("Ablation: OFCV/BRF buy convergence speed, not final accuracy",
                 fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 4 — before/after E1 vs E10 (re-runs inference)
# ---------------------------------------------------------------------------

def denormalize(t):
    img = t.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def heatmap(arr, cmap=cv2.COLORMAP_VIRIDIS):
    arr = arr.squeeze()
    lo, hi = arr.min(), arr.max()
    arr = ((arr - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(arr, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(img, prob, thresh=0.5):
    mask = (prob > thresh).astype(np.uint8)
    color = np.zeros_like(img)
    color[..., 0] = 255
    return (img * 0.55 + np.where(mask[..., None].astype(bool), color * 0.45, 0)).astype(np.uint8)


@torch.no_grad()
def fig_before_after(cfg_path, ckpt_e1, ckpt_e10, n_samples, out_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = build_dataloaders(cfg)

    model = SPECTRA(cfg, use_gnn=False).to(device).eval()

    def load_state(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)

    samples = []
    # First pass: collect images + GT
    for batch in val_loader:
        for i in range(batch["image"].shape[0]):
            samples.append({
                "image": batch["image"][i:i + 1].to(device),
                "image_t1": batch["image_t1"][i:i + 1].to(device),
                "mask": batch["mask"][i].cpu().numpy(),
            })
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    # E1 outputs
    load_state(ckpt_e1)
    e1_seg, e1_ofcv = [], []
    for s in samples:
        out = model(s["image"], s["image_t1"], return_intermediates=True)
        e1_seg.append(out["seg_prob"][0].cpu().numpy())
        e1_ofcv.append(out["ofcv_map"][0].cpu().numpy())

    # E10 outputs
    load_state(ckpt_e10)
    e10_seg, e10_ofcv = [], []
    for s in samples:
        out = model(s["image"], s["image_t1"], return_intermediates=True)
        e10_seg.append(out["seg_prob"][0].cpu().numpy())
        e10_ofcv.append(out["ofcv_map"][0].cpu().numpy())

    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3.0 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    col_titles = ["Input", "Ground truth", "E1 prediction",
                  "E10 prediction", "E10 OFCV map"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontweight="bold", fontsize=12)

    for r, s in enumerate(samples):
        img_vis = denormalize(s["image"][0])
        gt = (s["mask"] * 255).astype(np.uint8)
        e1_overlay = overlay(img_vis, e1_seg[r].squeeze())
        e10_overlay = overlay(img_vis, e10_seg[r].squeeze())
        ofcv_vis = heatmap(e10_ofcv[r], cv2.COLORMAP_VIRIDIS)

        axes[r, 0].imshow(img_vis)
        axes[r, 1].imshow(gt, cmap="bone")
        axes[r, 2].imshow(e1_overlay)
        axes[r, 3].imshow(e10_overlay)
        axes[r, 4].imshow(ofcv_vis)
        for c in range(5):
            axes[r, c].axis("off")

    plt.suptitle("Before / after: same images at epoch 1 vs epoch 10",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("results/portfolio")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_training_story(
        "results/causal_model/logs/train_final10.log",
        out_dir / "fig01_training_story.png",
    )

    fig_robustness_bars(
        "results/causal_model/final/robustness/robustness_results.json",
        out_dir / "fig02_robustness_bars.png",
    )

    fig_ablation(
        "results/causal_model/ablations/ablation_table.json",
        out_dir / "fig03_ablation.png",
    )

    fig_before_after(
        cfg_path="configs/config.yaml",
        ckpt_e1="checkpoints/checkpoint_epoch001.pth",
        ckpt_e10="checkpoints/spectra_best.pth",
        n_samples=4,
        out_path=out_dir / "fig04_before_after.pdf",
    )


if __name__ == "__main__":
    main()
