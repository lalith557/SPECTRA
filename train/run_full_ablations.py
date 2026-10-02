"""
spectra/train/run_full_ablations.py

Full-length ablation suite: trains every variant for 30-50 epochs so we can
prove the novel modules matter at convergence (not just for early epochs).

Variants:
    neither    - no physics modules (DINOv2 + FusionHead only) — baseline
    ofcv_only  - C1 only
    brf_only   - C2 only
    ofcv_brf   - C1 + C2
    full       - C1 + C2 + C3 (MBP-GNN)

For each variant we record per-epoch val metrics so we can also plot the
convergence curves and demonstrate whether the novel modules confer
final-epoch gains, not just early-epoch acceleration.

Usage:
    python train/run_full_ablations.py --epochs 50 --output results/full_ablations
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils import load_config, get_logger, set_seed
from train.train_baseline import (
    build_scheduler, train_one_epoch, validate,
)
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders
from train.losses import SPECTRALoss
from eval.metrics import TransparentObjectMetrics

logger = get_logger("spectra.full_ablations")


VARIANTS = [
    {"name": "neither",   "use_ofcv": False, "use_brf": False, "use_gnn": False,
     "desc": "DINOv2 backbone + FusionHead only (no physics)"},
    {"name": "ofcv_only", "use_ofcv": True,  "use_brf": False, "use_gnn": False,
     "desc": "C1: OFCV only"},
    {"name": "brf_only",  "use_ofcv": False, "use_brf": True,  "use_gnn": False,
     "desc": "C2: BRF only"},
    {"name": "ofcv_brf",  "use_ofcv": True,  "use_brf": True,  "use_gnn": False,
     "desc": "C1 + C2"},
    {"name": "full",      "use_ofcv": True,  "use_brf": True,  "use_gnn": True,
     "desc": "C1 + C2 + C3 (MBP-GNN)"},
]


def _save_partial_result(output_dir: Path, variant: Dict, cfg, per_epoch,
                         best_iou, best_epoch, t0, status: str) -> Dict:
    """Write result.json after every epoch so a crash doesn't lose progress."""
    rec = {
        "variant":    variant["name"],
        "desc":       variant["desc"],
        "epochs":     cfg.train.epochs,
        "best_iou":   best_iou,
        "best_epoch": best_epoch,
        "final_iou":  per_epoch[-1]["iou"] if per_epoch else None,
        "per_epoch":  per_epoch,
        "duration_s": time.time() - t0,
        "status":     status,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def _is_transient_cuda_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "cudnn", "cublas", "stream_mismatch", "cuda error",
        "out of memory", "device-side assert",
    ))


def train_variant(cfg, variant: Dict, output_dir: Path,
                  max_cuda_retries: int = 3) -> Dict:
    from torch.optim import AdamW
    from torch.amp import GradScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    model = SPECTRA(cfg, use_gnn=variant["use_gnn"],
                    use_ofcv=variant["use_ofcv"],
                    use_brf=variant["use_brf"]).to(device)

    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("backbone") and p.requires_grad]
    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.train.lr_backbone},
        {"params": other_params,    "lr": cfg.train.lr},
    ], weight_decay=cfg.train.weight_decay)
    criterion = SPECTRALoss(
        lambda_seg=cfg.loss.lambda_seg, lambda_mat=cfg.loss.lambda_mat,
        lambda_bnd=cfg.loss.lambda_bnd,
        lambda_consist=cfg.loss.get("lambda_consist", 0.1),
        lambda_refl=cfg.loss.get("lambda_refl", 0.1),
        aux_var_weight=cfg.ofcv.get("aux_var_weight", 0.1),
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler("cuda")

    per_epoch = []
    step = 0
    best_iou = 0.0
    best_epoch = 0
    t0 = time.time()

    _save_partial_result(output_dir, variant, cfg, per_epoch,
                         best_iou, best_epoch, t0, status="running")

    for epoch in range(1, cfg.train.epochs + 1):
        # Robust training step: retry once or twice on transient cuDNN/CUBLAS
        # errors. These can happen after long runs when CUDA streams desync.
        last_err = None
        for attempt in range(max_cuda_retries + 1):
            try:
                _, step = train_one_epoch(
                    model, train_loader, optimizer, scheduler, criterion,
                    scaler, device, epoch, cfg, step,
                )
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                if not _is_transient_cuda_error(e):
                    raise
                logger.warning(
                    f"[{variant['name']}] transient CUDA error on epoch "
                    f"{epoch} (attempt {attempt + 1}/{max_cuda_retries + 1}): {e}"
                )
                # Recover GPU state
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2 + 2 * attempt)
        if last_err is not None:
            # Persist whatever we have so future runs can resume
            _save_partial_result(output_dir, variant, cfg, per_epoch,
                                 best_iou, best_epoch, t0,
                                 status=f"failed_at_epoch_{epoch}")
            raise last_err

        # Validate every epoch for proper curves
        try:
            r = validate(model, val_loader, criterion, device, epoch)
        except RuntimeError as e:
            if _is_transient_cuda_error(e):
                logger.warning(f"[{variant['name']}] val cuda error: {e}. "
                               "Synchronising and retrying once.")
                torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
                r = validate(model, val_loader, criterion, device, epoch)
            else:
                raise

        per_epoch.append({"epoch": epoch, **r})

        if r["iou"] > best_iou:
            best_iou = r["iou"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_iou": best_iou,
                "variant": variant["name"],
            }, output_dir / "best.pth")

        # Incremental save — every epoch
        _save_partial_result(output_dir, variant, cfg, per_epoch,
                             best_iou, best_epoch, t0, status="running")

        torch.cuda.empty_cache()
        gc.collect()

    final = _save_partial_result(output_dir, variant, cfg, per_epoch,
                                 best_iou, best_epoch, t0,
                                 status="completed")
    return final


def _load_existing(out_root: Path, variant_name: str):
    p = out_root / variant_name / "result.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--output", default="results/full_ablations")
    p.add_argument("--variants", nargs="+",
                   default=[v["name"] for v in VARIANTS],
                   help="Subset of variants to run.")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip variants whose result.json has status='completed'.")
    args = p.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [v for v in VARIANTS if v["name"] in args.variants]
    for v in selected:
        existing = _load_existing(out_root, v["name"])
        if args.skip_completed and existing and existing.get("status") == "completed":
            logger.info(f"--- Skip {v['name']} (already completed, best_iou="
                        f"{existing.get('best_iou')}) ---")
            continue

        logger.info(f"--- Variant: {v['name']} ({v['desc']}) ---")
        v_dir = out_root / v["name"]
        v_dir.mkdir(exist_ok=True)

        cfg = load_config(args.config)
        cfg.train.epochs = args.epochs
        cfg.logging.save_dir = str(v_dir)
        cfg.experiment = f"full_ablation_{v['name']}"

        train_variant(cfg, v, v_dir)   # incremental save inside

    # Merge ALL variants that have a result.json on disk — even ones we
    # didn't train this run — so summary.json is always complete.
    merged: list[Dict] = []
    for v in VARIANTS:
        rec = _load_existing(out_root, v["name"])
        if rec is not None:
            merged.append(rec)

    with open(out_root / "summary.json", "w") as f:
        json.dump(merged, f, indent=2)

    md = ["# Full-length ablation summary\n",
          "| variant | desc | epochs | best_iou | best_epoch | final_iou | status |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in merged:
        final = r.get("final_iou")
        md.append(
            f"| {r['variant']} | {r['desc']} | {r['epochs']} | "
            f"{r['best_iou']:.4f} | {r['best_epoch']} | "
            f"{final if final is None else f'{final:.4f}'} | "
            f"{r.get('status', '-')} |"
        )
    with open(out_root / "summary.md", "w") as f:
        f.write("\n".join(md))
    logger.info(f"summary: {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
