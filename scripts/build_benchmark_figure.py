"""
Generate fig05_benchmark_comparison.png: side-by-side IoU bars + scatter
of IoU vs train minutes across SPECTRA and the 4 baselines.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def main():
    data = json.loads(Path("benchmarks/comparison_table.json").read_text())
    models = data["models"]

    names = [m["name"].split("(")[0].strip() for m in models]
    short_names = [
        "SAM\n(zero-shot)",
        "SegFormer-B0",
        "U-Net\n(ResNet-34)",
        "DeepLabV3+\n(ResNet-50)",
        "SPECTRA\n(ours)",
    ]
    ious = [m["val_iou"] for m in models]
    train_mins = [m["train_minutes"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ── Panel A: IoU bars ──────────────────────────────────────────────
    colors = ["#888780", "#BA7517", "#0F6E56", "#D85A30", "#534AB7"]
    bars = axes[0].bar(short_names, ious, color=colors,
                       edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, ious):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold" if "SPECTRA" in b.get_label() else "normal")
    # Highlight SPECTRA bar
    bars[-1].set_edgecolor("#534AB7")
    bars[-1].set_linewidth(2.5)
    axes[0].set_ylabel("Trans10K val IoU")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("(a) Validation IoU — matched 10-epoch training, same splits",
                      fontweight="bold")
    axes[0].axhline(0.5, color="#888780", linestyle=":", linewidth=0.8, alpha=0.5)

    # ── Panel B: IoU vs train time scatter ─────────────────────────────
    # Drop SAM (train_mins=0) from this view since it skews the axes
    plot_models = models[1:]   # skip SAM
    plot_x = [m["train_minutes"] for m in plot_models]
    plot_y = [m["val_iou"] for m in plot_models]
    plot_names = short_names[1:]
    plot_colors = colors[1:]
    plot_sizes  = [m["params_M"] * 8 for m in plot_models]

    for x, y, n, c, s in zip(plot_x, plot_y, plot_names, plot_colors, plot_sizes):
        is_spectra = "SPECTRA" in n
        axes[1].scatter(x, y, s=s, c=c, edgecolor="black",
                        linewidth=2.5 if is_spectra else 0.7, alpha=0.85,
                        zorder=3)
        offset_y = 0.005 if not is_spectra else 0.013
        axes[1].annotate(n.replace("\n", " "),
                         (x, y), textcoords="offset points",
                         xytext=(0, 14 if is_spectra else 10),
                         ha="center", fontsize=9,
                         fontweight="bold" if is_spectra else "normal")
    axes[1].set_xlabel("Training time (min, RTX 4070)")
    axes[1].set_ylabel("Trans10K val IoU")
    axes[1].set_xlim(30, max(plot_x) * 1.12)
    axes[1].set_ylim(0.84, 0.94)
    axes[1].set_title("(b) Accuracy vs compute — bubble size ∝ params",
                      fontweight="bold")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Trans10K matched-condition comparison (val 1000 imgs, 10 epochs)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    out = Path("results/portfolio/fig05_benchmark_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
