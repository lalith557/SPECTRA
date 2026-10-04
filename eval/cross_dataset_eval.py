"""
spectra/eval/cross_dataset_eval.py
Zero-shot cross-dataset evaluation: train on Trans10K, evaluate on GSD / ClearPose
(or any registered dataset) WITHOUT any finetuning.

Purpose:
    Reviewers must believe the model learned transparency *physics*, not Trans10K
    pixel statistics. This script measures generalisation gap directly.

Usage:
    python eval/cross_dataset_eval.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/spectra_best.pth \
        --datasets trans10k:test gsd:test clearpose:all \
        --output outputs/cross_dataset/

Outputs:
    outputs/cross_dataset/results.json
    outputs/cross_dataset/results.md           (paper-ready table)
    outputs/cross_dataset/results.tex          (LaTeX table)

Each row reports {dataset, split, iou, f_measure, mae, ber, n_images}.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, get_logger, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from eval.metrics import TransparentObjectMetrics
from data.trans10k_dataset import build_eval_dataset

logger = get_logger("spectra.cross_dataset")


# ---------------------------------------------------------------------------
# One eval pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 2,
    image_size: Tuple[int, int] = (448, 448),
) -> Dict[str, float]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    metrics = TransparentObjectMetrics()
    model.eval()

    H, W = image_size
    for batch in tqdm(loader, desc="eval"):
        batch = to_device(batch, device)
        out   = model(batch["image"], batch["image_t1"], return_intermediates=False)
        prob  = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear", align_corners=False)
        metrics.update(prob, batch["mask"])

    results = metrics.compute()
    results["n_images"] = len(dataset)
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def to_markdown(rows: List[Dict]) -> str:
    cols = ["dataset", "split", "n_images", "iou", "f_measure", "mae", "ber"]
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            f"{r.get(c, '-'):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "-"))
            for c in cols
        ) + " |")
    return "\n".join([head, sep, *body])


def to_latex(rows: List[Dict]) -> str:
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Dataset & Split & N & IoU $\uparrow$ & F$_\beta$ $\uparrow$ & MAE $\downarrow$ & BER $\downarrow$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['dataset']} & {r['split']} & {r['n_images']} & "
            f"{r['iou']:.3f} & {r['f_measure']:.3f} & "
            f"{r['mae']:.3f} & {r['ber']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cross_dataset(
    config_path: str,
    checkpoint_path: str,
    dataset_specs: List[str],     # e.g. ["trans10k:test", "gsd:test"]
    output_dir: str,
    batch_size: int = 4,
):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = cfg.data.image_size
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load Trans10K-trained model (no finetuning) ─────────────────────────
    model = SPECTRA(cfg, use_gnn=False).to(device)
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, optimizer=None, device=str(device))
        logger.info(f"Loaded: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint missing — evaluating randomly-initialised model: {checkpoint_path}")

    # ── Evaluate on each dataset ────────────────────────────────────────────
    rows: List[Dict] = []
    for spec in dataset_specs:
        if ":" in spec:
            name, split = spec.split(":", 1)
        else:
            name, split = spec, "test"

        logger.info(f"--- Evaluating on {name} [{split}] ---")
        try:
            ds = build_eval_dataset(
                name=name, root=cfg.data.root, split=split,
                image_size=tuple(cfg.data.image_size),
            )
        except Exception as e:
            logger.error(f"Skipping {name}:{split} — {e}")
            rows.append({"dataset": name, "split": split, "n_images": 0,
                         "iou": float("nan"), "f_measure": float("nan"),
                         "mae": float("nan"), "ber": float("nan"),
                         "error": str(e)})
            continue

        if len(ds) == 0:
            logger.warning(f"Dataset {name}:{split} is empty — skipping")
            rows.append({"dataset": name, "split": split, "n_images": 0,
                         "iou": float("nan"), "f_measure": float("nan"),
                         "mae": float("nan"), "ber": float("nan")})
            continue

        res = evaluate_on_dataset(
            model, ds, device,
            batch_size=batch_size,
            num_workers=cfg.data.num_workers,
            image_size=(H, W),
        )
        res["dataset"] = name
        res["split"] = split
        rows.append(res)
        logger.info(
            f"{name}:{split} — IoU={res['iou']:.4f} F={res['f_measure']:.4f} "
            f"MAE={res['mae']:.4f} BER={res['ber']:.4f} (n={res['n_images']})"
        )

    # ── Save results ────────────────────────────────────────────────────────
    out = Path(output_dir)
    with open(out / "results.json", "w") as f:
        json.dump({"checkpoint": checkpoint_path, "rows": rows}, f, indent=2)
    with open(out / "results.md", "w") as f:
        f.write("# Cross-dataset zero-shot generalisation\n\n")
        f.write(f"Checkpoint: `{checkpoint_path}`\n\n")
        f.write(to_markdown(rows))
        f.write("\n")
    with open(out / "results.tex", "w") as f:
        f.write(to_latex(rows))

    logger.info(f"Results written to {output_dir}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    p.add_argument("--datasets",   nargs="+",
                   default=["trans10k:test", "gsd:test", "clearpose:all"],
                   help="dataset:split pairs to evaluate")
    p.add_argument("--output",     default="outputs/cross_dataset")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    run_cross_dataset(
        args.config, args.checkpoint, args.datasets, args.output, args.batch_size
    )
