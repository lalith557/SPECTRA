"""
spectra/eval/aggregate_seeds.py

Read per-seed checkpoints (results from train/run_multi_seed.py), re-evaluate
each on the official Trans10K test split, and report mean ± std per metric.

This is the canonical "we ran 3-5 seeds and report mean ± std" plot+table
that top venues expect.

Outputs:
    results/multi_seed/<variant>/aggregated.json
    results/multi_seed/<variant>/aggregated.md
"""
import argparse
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from utils import load_config, get_logger, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset
from eval.metrics import TransparentObjectMetrics

logger = get_logger("spectra.aggregate")


@torch.no_grad()
def evaluate_checkpoint(cfg, ckpt_path: Path, device, dataset_name: str,
                        split: str = "test") -> Dict:
    model = SPECTRA(cfg, use_gnn=False).to(device)
    load_checkpoint(str(ckpt_path), model, optimizer=None, device=str(device))

    ds = build_eval_dataset(
        name=dataset_name, root=cfg.data.root, split=split,
        image_size=tuple(cfg.data.image_size),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    metrics = TransparentObjectMetrics()
    H, W = cfg.data.image_size
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["image"], batch["image_t1"])
        prob = F.interpolate(out["seg_prob"], size=(H, W),
                             mode="bilinear", align_corners=False)
        metrics.update(prob, batch["mask"])
    return metrics.compute()


def aggregate(per_seed_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
    if not per_seed_metrics:
        return {}
    keys = ["iou", "f_measure", "mae", "ber", "precision", "recall"]
    out = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if k in m]
        if not vals:
            continue
        mean = stats.mean(vals)
        std  = stats.stdev(vals) if len(vals) > 1 else 0.0
        out[k] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals),
                  "n": len(vals)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--variant", default="full")
    p.add_argument("--variant-root", default="results/multi_seed")
    p.add_argument("--dataset", default="trans10k")
    p.add_argument("--split",   default="test")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.variant_root) / args.variant
    seed_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        logger.error(f"No seed directories under {root}")
        return

    per_seed = []
    for sd in seed_dirs:
        ckpt = sd / "spectra_best.pth"
        if not ckpt.exists():
            logger.warning(f"missing checkpoint: {ckpt}")
            continue
        seed = int(sd.name.split("_")[1])
        logger.info(f"evaluating seed {seed}...")
        m = evaluate_checkpoint(cfg, ckpt, device, args.dataset, args.split)
        m["seed"] = seed
        per_seed.append(m)
        logger.info(f"  IoU={m['iou']:.4f}  F={m['f_measure']:.4f}")

    agg = aggregate(per_seed)
    output = {
        "variant": args.variant, "dataset": args.dataset, "split": args.split,
        "per_seed": per_seed, "aggregate": agg,
    }
    with open(root / "aggregated.json", "w") as f:
        json.dump(output, f, indent=2)

    md = [f"# Multi-seed aggregate ({args.variant}, {args.dataset}/{args.split})\n"]
    md.append("| metric | mean | std | min | max | n |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for k, v in agg.items():
        md.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | "
                  f"{v['min']:.4f} | {v['max']:.4f} | {v['n']} |")
    md.append("\n## Per-seed values\n")
    md.append("| seed | iou | f_measure | mae | ber |")
    md.append("| --- | --- | --- | --- | --- |")
    for m in per_seed:
        md.append(f"| {m['seed']} | {m['iou']:.4f} | {m['f_measure']:.4f} | "
                  f"{m['mae']:.4f} | {m['ber']:.4f} |")
    with open(root / "aggregated.md", "w") as f:
        f.write("\n".join(md))

    logger.info(f"Wrote {root / 'aggregated.json'}")


if __name__ == "__main__":
    main()
