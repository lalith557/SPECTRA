"""
spectra/train/run_multi_seed.py

Run the same training config under N random seeds and record per-seed metrics.
Used together with eval/aggregate_seeds.py and eval/statistical_significance.py.

Usage:
    python train/run_multi_seed.py --seeds 42 7 13 1234 2025 --variant full
    python train/run_multi_seed.py --seeds 42 7 13 --variant brf_only
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils import load_config, get_logger, set_seed
from train.train_baseline import train

logger = get_logger("spectra.multi_seed")


VARIANTS = {
    "full":      {"use_ofcv": True,  "use_brf": True,  "use_gnn": False},
    "ofcv_only": {"use_ofcv": True,  "use_brf": False, "use_gnn": False},
    "brf_only":  {"use_ofcv": False, "use_brf": True,  "use_gnn": False},
    "neither":   {"use_ofcv": False, "use_brf": False, "use_gnn": False},
    "full_gnn":  {"use_ofcv": True,  "use_brf": True,  "use_gnn": True},
}


def _load_summary(path: Path):
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _upsert(summary, entry):
    for i, e in enumerate(summary):
        if e.get("seed") == entry["seed"] and e.get("variant") == entry["variant"]:
            summary[i] = entry
            return summary
    summary.append(entry)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--seeds",  nargs="+", type=int,
                   default=[42, 7, 13, 1234, 2025])
    p.add_argument("--variant", choices=list(VARIANTS), default="full")
    p.add_argument("--output", default="results/multi_seed")
    p.add_argument("--force", action="store_true",
                   help="Retrain seeds even if spectra_best.pth already exists.")
    args = p.parse_args()

    flags = VARIANTS[args.variant]
    out_root = Path(args.output) / args.variant
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "summary.json"
    summary = _load_summary(summary_path)

    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        best_path = seed_dir / "spectra_best.pth"

        # Skip if already trained (unless --force). Preserves prior seed results.
        if best_path.exists() and not args.force:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_iou = ckpt.get("best_iou")
            entry = {
                "seed": seed, "variant": args.variant,
                "best_iou": best_iou, "duration_s": None, "ok": True,
                "skipped": True,
            }
            summary = _upsert(summary, entry)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"seed {seed}: SKIP (exists) best IoU {best_iou}")
            del ckpt
            gc.collect()
            continue

        cfg = load_config(args.config)
        cfg.seed = seed
        cfg.logging.save_dir = str(seed_dir)
        cfg.experiment = f"{args.variant}_seed{seed}"
        set_seed(seed)

        t0 = time.time()
        try:
            train(
                cfg_override=cfg,
                experiment_name=cfg.experiment,
                use_ofcv=flags["use_ofcv"],
                use_brf=flags["use_brf"],
                use_gnn=flags["use_gnn"],
            )
            ok = True
        except Exception as e:
            logger.error(f"seed {seed} failed: {e}")
            ok = False
        duration = time.time() - t0

        # Best metric is stamped on the best checkpoint
        best_iou = None
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_iou = ckpt.get("best_iou")
            del ckpt

        entry = {
            "seed": seed, "variant": args.variant,
            "best_iou": best_iou, "duration_s": duration, "ok": ok,
        }
        summary = _upsert(summary, entry)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"seed {seed}: best IoU {best_iou}")

        # Free GPU memory before the next seed so torch.manual_seed()
        # does not OOM on a fragmented allocator.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    logger.info("done — see results/multi_seed/<variant>/summary.json")


if __name__ == "__main__":
    main()
