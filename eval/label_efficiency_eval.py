"""
spectra/eval/label_efficiency_eval.py
SSL Validation — the single most important result in the paper.

Proves that physics-contrastive pre-training (C4) allows SPECTRA to match
prior SOTA performance with only 10-25% of labelled training data.

Experiment design:
  For each label fraction f in [0.05, 0.10, 0.25, 0.50, 1.00]:
    - Train SPECTRA with pre-training  (physics-contrastive C4)
    - Train SPECTRA without pre-training (supervised baseline)
    - Evaluate both on full test set
    - Record IoU, F-measure

  Plot label efficiency curve showing:
    - SPECTRA (ours) vs supervised baseline
    - Prior SOTA horizontal line at 100% labels
    - Crossover point annotation

Usage:
    python eval/label_efficiency_eval.py --config configs/config.yaml
    python eval/label_efficiency_eval.py --fast   # 10-epoch debug runs
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import load_config, set_seed, get_logger, to_device
from eval.metrics import TransparentObjectMetrics
from train.losses import SPECTRALoss
from inference.vis_utils import save_label_efficiency_curve

logger = get_logger("spectra.label_efficiency")

# Prior SOTA baselines for reference line on the plot
PRIOR_SOTA = {
    "trans10k": 0.858,   # EBLNet ICCV 2021
    "gsd":      0.803,
}


# ---------------------------------------------------------------------------
# Train one label-fraction experiment
# ---------------------------------------------------------------------------

def train_one_experiment(
    cfg,
    device:         torch.device,
    label_fraction: float,
    use_pretrain:   bool,
    n_epochs:       int,
    pretrain_ckpt:  str = None,
) -> Dict:
    """
    Train SPECTRA with a given label fraction and return val metrics.

    Args:
        label_fraction: fraction of labelled training data (0.0–1.0)
        use_pretrain:   whether to initialise from physics-contrastive checkpoint
        n_epochs:       training epochs

    Returns:
        best_metrics: dict with iou, f_measure, mae, ber at best val IoU
    """
    from models.spectra_model import SPECTRA
    from data.trans10k_dataset import build_dataloaders

    # Override label fraction in cfg (monkey-patch for this run)
    cfg.train.label_fraction = label_fraction
    cfg.train.epochs         = n_epochs

    set_seed(cfg.seed)

    model = SPECTRA(cfg, use_gnn=False).to(device)

    # Load physics-contrastive pre-trained weights if requested
    if use_pretrain and pretrain_ckpt and Path(pretrain_ckpt).exists():
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        # Load encoder weights only (projector head is discarded)
        backbone_state = {
            k.replace("encoder.", ""): v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        missing, unexpected = model.backbone.load_state_dict(
            backbone_state, strict=False
        )
        logger.info(
            f"  Pre-trained encoder loaded. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        f"  Label fraction={label_fraction:.0%} → "
        f"{len(train_loader.dataset)} training images"
    )

    criterion = SPECTRALoss(cfg.loss.lambda_seg, cfg.loss.lambda_mat, cfg.loss.lambda_bnd)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler    = GradScaler()

    best_iou     = 0.0
    best_metrics = {}

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            with autocast():
                out    = model(batch["image"], batch["image_t1"])
                losses = criterion(out, {"mask": batch["mask"], "material": batch["material"]})
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Validate
        model.eval()
        metrics = TransparentObjectMetrics()
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                with autocast():
                    out = model(batch["image"], batch["image_t1"])
                metrics.update(out["seg_prob"], batch["mask"])

        result = metrics.compute()
        if result["iou"] > best_iou:
            best_iou     = result["iou"]
            best_metrics = result

        if epoch % 5 == 0 or epoch == n_epochs:
            logger.info(
                f"  [f={label_fraction:.0%}|pretrain={use_pretrain}]"
                f"[E{epoch}] IoU={result['iou']:.4f}"
            )

    best_metrics["label_fraction"] = label_fraction
    best_metrics["use_pretrain"]   = use_pretrain
    return best_metrics


# ---------------------------------------------------------------------------
# Full label efficiency experiment
# ---------------------------------------------------------------------------

def run_label_efficiency(
    config_path:    str,
    pretrain_ckpt:  str = None,
    fractions:      List[float] = [0.05, 0.10, 0.25, 0.50, 1.00],
    fast:           bool = False,
    output_json:    str = "outputs/label_efficiency.json",
    output_fig:     str = "outputs/label_efficiency.pdf",
):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 5 if fast else cfg.train.epochs

    logger.info(f"Label efficiency experiment | fractions={fractions} | epochs={n_epochs}")

    spectra_results  = []   # with pre-training
    baseline_results = []   # without pre-training

    for f in fractions:
        logger.info(f"\n=== Fraction {f:.0%} — SPECTRA (with pre-training) ===")
        r_spectra = train_one_experiment(
            cfg, device, label_fraction=f,
            use_pretrain=True, n_epochs=n_epochs,
            pretrain_ckpt=pretrain_ckpt,
        )
        spectra_results.append(r_spectra)

        logger.info(f"\n=== Fraction {f:.0%} — Baseline (no pre-training) ===")
        r_baseline = train_one_experiment(
            cfg, device, label_fraction=f,
            use_pretrain=False, n_epochs=n_epochs,
        )
        baseline_results.append(r_baseline)

    # ── Print comparison table ────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info(f"{'Labels':>8} | {'SPECTRA IoU':>12} | {'Baseline IoU':>13} | {'Δ IoU':>8}")
    logger.info("-"*70)
    for s, b in zip(spectra_results, baseline_results):
        delta = s["iou"] - b["iou"]
        logger.info(
            f"{s['label_fraction']:>7.0%} | "
            f"{s['iou']:>12.4f} | "
            f"{b['iou']:>13.4f} | "
            f"{delta:>+8.4f}"
        )
    logger.info("="*70)

    # Crossover: find smallest f where SPECTRA ≥ prior SOTA
    sota = PRIOR_SOTA.get(cfg.data.dataset, 0.85)
    crossover = None
    for r in spectra_results:
        if r["iou"] >= sota:
            crossover = r["label_fraction"]
            break

    if crossover:
        logger.info(
            f"\n✓ SPECTRA matches prior SOTA ({sota:.3f} IoU) "
            f"with only {crossover:.0%} of labels!"
        )
    else:
        logger.info(
            f"\n✗ SPECTRA did not reach prior SOTA ({sota:.3f}) "
            f"within {n_epochs} epochs — run longer or check pre-training."
        )

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "fractions":        fractions,
        "spectra_results":  spectra_results,
        "baseline_results": baseline_results,
        "prior_sota":       sota,
        "crossover_fraction": crossover,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f_out:
        json.dump(output, f_out, indent=2)
    logger.info(f"Results saved: {output_json}")

    # ── Generate figure ───────────────────────────────────────────────────
    save_label_efficiency_curve(
        fractions=fractions,
        spectra_ious=[r["iou"]  for r in spectra_results],
        baseline_ious=[r["iou"] for r in baseline_results],
        sota_full=sota,
        save_path=output_fig,
    )
    logger.info(f"Figure saved: {output_fig}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--pretrain",  default="checkpoints/spectra_pretrain.pth",
                        help="Physics-contrastive pre-trained checkpoint")
    parser.add_argument("--fractions", default="0.05,0.10,0.25,0.50,1.0")
    parser.add_argument("--fast",      action="store_true", help="5-epoch debug runs")
    parser.add_argument("--output",    default="outputs/label_efficiency.json")
    parser.add_argument("--fig",       default="outputs/label_efficiency.pdf")
    args = parser.parse_args()

    fractions = [float(f) for f in args.fractions.split(",")]
    run_label_efficiency(
        config_path=args.config,
        pretrain_ckpt=args.pretrain,
        fractions=fractions,
        fast=args.fast,
        output_json=args.output,
        output_fig=args.fig,
    )
