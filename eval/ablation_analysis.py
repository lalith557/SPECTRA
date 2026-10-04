"""
spectra/eval/ablation_analysis.py

Generate the paper-ready ablation analysis figure + table:
  - convergence curves (val IoU vs epoch, per variant)
  - final-epoch comparison bar chart
  - markdown / LaTeX tables

Input:
    results/full_ablations/summary.json

Outputs:
    results/full_ablations/convergence.pdf
    results/full_ablations/final_iou.pdf
    results/full_ablations/table.md
    results/full_ablations/table.tex
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np


COLOURS = {
    "neither":   "#888780",
    "ofcv_only": "#534AB7",
    "brf_only":  "#0F6E56",
    "ofcv_brf":  "#D85A30",
    "full":      "#BA7517",
}


def plot_convergence(results, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        epochs = [e["epoch"] for e in r["per_epoch"]]
        ious   = [e["iou"]   for e in r["per_epoch"]]
        ax.plot(epochs, ious, marker="o", markersize=3, linewidth=2,
                color=COLOURS.get(r["variant"], None),
                label=f"{r['variant']} (best {r['best_iou']:.3f})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val IoU", fontsize=12)
    ax.set_title("SPECTRA full-length ablation: convergence curves", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_bar(results, output_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = [r["variant"] for r in results]
    finals = [r["final_iou"] for r in results]
    bests  = [r["best_iou"]  for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, finals, w, label="Final epoch IoU", color="#534AB7")
    ax.bar(x + w/2, bests,  w, label="Best IoU",        color="#D85A30")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("IoU")
    ax.set_title("Final vs best IoU across ablation variants")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def to_latex(results) -> str:
    lines = [
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Variant & Epochs & Best IoU & Best @ Ep. & Final IoU & $\Delta$ \\",
        r"\midrule",
    ]
    base = next((r for r in results if r["variant"] == "neither"), None)
    base_iou = base["best_iou"] if base else None
    for r in results:
        delta = (r["best_iou"] - base_iou) if base_iou is not None else 0.0
        lines.append(
            f"{r['variant']} & {r['epochs']} & {r['best_iou']:.3f} & "
            f"{r['best_epoch']} & {r['final_iou']:.3f} & "
            f"{delta:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/full_ablations/summary.json")
    p.add_argument("--output-dir", default="results/full_ablations")
    args = p.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_convergence(results, out / "convergence.pdf")
    plot_final_bar(results,   out / "final_iou.pdf")

    base = next((r for r in results if r["variant"] == "neither"), None)
    md = ["# Full-length ablation results\n"]
    md.append("| variant | epochs | best_iou | best_epoch | final_iou | Δ vs neither |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        d = (r["best_iou"] - base["best_iou"]) if base else 0.0
        md.append(f"| {r['variant']} | {r['epochs']} | {r['best_iou']:.4f} | "
                  f"{r['best_epoch']} | {r['final_iou']:.4f} | {d:+.4f} |")
    with open(out / "table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(out / "table.tex", "w", encoding="utf-8") as f:
        f.write(to_latex(results))
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
