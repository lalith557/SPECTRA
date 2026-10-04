"""
train/run_full_ablations_resumable.py

RESUMABLE wrapper around the Trans10K full-length ablation (P2), for running
publication experiments across interrupted Kaggle sessions.

It changes NOTHING scientific. It reuses, verbatim:
  * data.trans10k_dataset.build_dataloaders
  * models.spectra_model.SPECTRA
  * train.losses.SPECTRALoss
  * train.train_baseline.build_scheduler / train_one_epoch / validate
  * train.run_full_ablations.VARIANTS
The optimizer/criterion/scheduler/scaler construction below is copied verbatim
from train.run_full_ablations.train_variant so the two runners are identical in
every scientific respect. The ONLY additions are: full-state checkpointing
after each completed epoch, exact --resume, and a --stop-after-epoch staging
switch (which never mutates the requested total epoch count / scheduler length).

Staged usage (P2: ofcv_only, 30 epochs total, split 15 + 15):

  # session 1
  python train/run_full_ablations_resumable.py --epochs 30 --variants ofcv_only \
      --output EXPERIMENTS/17_corrected_trans10k_ablation_2026-10-02 --stop-after-epoch 15
  # session 2 (new machine, same code + checkpoint)
  python train/run_full_ablations_resumable.py --epochs 30 --variants ofcv_only \
      --output EXPERIMENTS/17_corrected_trans10k_ablation_2026-10-02 \
      --resume EXPERIMENTS/17_corrected_trans10k_ablation_2026-10-02/ofcv_only/checkpoint_latest.pth
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils import load_config, get_logger, set_seed
from train.train_baseline import build_scheduler, train_one_epoch, validate
from train.run_full_ablations import VARIANTS
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders
from train.losses import SPECTRALoss
import train.checkpoint_utils as cu

logger = get_logger("spectra.full_ablations.resumable")


# ---------------------------------------------------------------------------
# Scientific object construction — VERBATIM from run_full_ablations.train_variant
# ---------------------------------------------------------------------------

def _build_training_objects(cfg, variant: Dict, device) -> Dict:
    from torch.optim import AdamW
    from torch.amp import GradScaler

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
    return {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "scaler": scaler, "criterion": criterion,
        "train_loader": train_loader, "val_loader": val_loader,
        "steps_per_epoch": len(train_loader),
    }


def _is_transient_cuda_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "cudnn", "cublas", "stream_mismatch", "cuda error",
        "out of memory", "device-side assert",
    ))


# ---------------------------------------------------------------------------
# result.json writer (status-aware)
# ---------------------------------------------------------------------------

def _write_result(output_dir: Path, variant: Dict, total_epochs: int,
                  per_epoch, best_iou, best_epoch, elapsed_s, status: str) -> Dict:
    rec = {
        "variant": variant["name"],
        "desc": variant.get("desc", ""),
        "epochs": total_epochs,                         # requested TOTAL (not the stage)
        "epochs_completed": per_epoch[-1]["epoch"] if per_epoch else 0,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "final_iou": per_epoch[-1]["iou"] if per_epoch else None,
        "per_epoch": per_epoch,
        "duration_s": elapsed_s,
        "status": status,   # running | staged_partial | completed | failed_at_epoch_N
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result.json", "w") as f:
        json.dump(rec, f, indent=2)
    return rec


# ---------------------------------------------------------------------------
# The resumable epoch driver. Dependency-injected build/train/validate so the
# smoke test exercises this EXACT control flow with lightweight stand-ins,
# while production uses the real SPECTRA functions above.
# ---------------------------------------------------------------------------

def run_variant_resumable(
    cfg, variant: Dict, output_dir: Path, total_epochs: int, *,
    resume_path: Optional[str] = None,
    stop_after_epoch: Optional[int] = None,
    build_objs: Callable = _build_training_objects,
    train_epoch_fn: Callable = train_one_epoch,
    validate_fn: Callable = validate,
    max_cuda_retries: int = 3,
    log=logger,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (1) Deterministic flags + base seed FIRST. On resume this is overwritten
    #     by the restored RNG state (step 11) so it does not destroy progression.
    set_seed(cfg.seed)

    # (2) Build all objects (identical to a fresh run).
    objs = build_objs(cfg, variant, device)
    model, optimizer = objs["model"], objs["optimizer"]
    scheduler, scaler, criterion = objs["scheduler"], objs["scaler"], objs["criterion"]
    train_loader, val_loader = objs["train_loader"], objs["val_loader"]
    steps_per_epoch = objs["steps_per_epoch"]
    batch_size = cfg.train.batch_size

    current_prov = cu.make_provenance(
        variant=variant, dataset=cfg.data.dataset, image_size=cfg.data.image_size,
        batch_size=batch_size, total_epochs=total_epochs, steps_per_epoch=steps_per_epoch,
        seed=cfg.seed, lr=cfg.train.lr, lr_backbone=cfg.train.lr_backbone,
        weight_decay=cfg.train.weight_decay, warmup_epochs=cfg.train.warmup_epochs,
    )

    per_epoch = []
    step = 0
    best_iou = 0.0
    best_epoch = 0
    start_epoch = 1
    elapsed_base = 0.0

    latest_path = output_dir / "checkpoint_latest.pth"

    if resume_path:
        ckpt = cu.load_checkpoint_file(resume_path, device=device)
        cu.verify_resume_compat(ckpt, current_prov, total_epochs)  # FAILS LOUDLY
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["global_step"]
        best_iou = ckpt["best_iou"]
        best_epoch = ckpt["best_epoch"]
        per_epoch = [e for e in ckpt["per_epoch"] if e["epoch"] <= ckpt["epoch"]]  # dedup guard
        start_epoch = ckpt["epoch"] + 1
        elapsed_base = ckpt.get("elapsed_s", 0.0)
        cu.restore_rng_states(ckpt.get("rng"), log)                 # (11) RNG LAST
        log.info(f"[{variant['name']}] RESUME {resume_path}: next_epoch={start_epoch} "
                 f"step={step} best_iou={best_iou:.4f}@E{best_epoch} "
                 f"epochs_done={len(per_epoch)}")
        if start_epoch > total_epochs:
            log.info(f"[{variant['name']}] checkpoint already complete; nothing to do.")
            return _write_result(output_dir, variant, total_epochs, per_epoch,
                                 best_iou, best_epoch, elapsed_base, status="completed")
    else:
        # Fresh run must not silently clobber an existing (partial) run.
        if latest_path.exists():
            raise RuntimeError(
                f"{latest_path} already exists but --resume was not given. "
                "Refusing to overwrite an existing run. Pass --resume to continue "
                "it, or choose a fresh --output directory.")

    t0 = time.time()

    def _elapsed() -> float:
        return elapsed_base + (time.time() - t0)

    def _full_state(epoch: int, status: str) -> Dict:
        return cu.build_full_state(
            epoch=epoch, total_epochs=total_epochs, global_step=step,
            best_iou=best_iou, best_epoch=best_epoch, per_epoch=per_epoch,
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            provenance=current_prov, status=status, elapsed_s=_elapsed())

    # staged stop point (0 / None / >=total => run to total)
    stop_epoch = total_epochs
    if stop_after_epoch is not None and stop_after_epoch > 0:
        stop_epoch = min(stop_after_epoch, total_epochs)
        if stop_epoch < start_epoch:
            raise RuntimeError(
                f"--stop-after-epoch {stop_after_epoch} is before the resume "
                f"point (next epoch {start_epoch}); nothing to do.")

    _write_result(output_dir, variant, total_epochs, per_epoch,
                  best_iou, best_epoch, _elapsed(), status="running")

    for epoch in range(start_epoch, total_epochs + 1):
        # transient-CUDA retry — same policy as run_full_ablations.train_variant
        last_err = None
        for attempt in range(max_cuda_retries + 1):
            try:
                _, step = train_epoch_fn(model, train_loader, optimizer, scheduler,
                                         criterion, scaler, device, epoch, cfg, step)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                if not _is_transient_cuda_error(e):
                    raise
                log.warning(f"[{variant['name']}] transient CUDA error epoch {epoch} "
                            f"(attempt {attempt + 1}/{max_cuda_retries + 1}): {e}")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache(); gc.collect()
                time.sleep(2 + 2 * attempt)
        if last_err is not None:
            _write_result(output_dir, variant, total_epochs, per_epoch, best_iou,
                          best_epoch, _elapsed(), status=f"failed_at_epoch_{epoch}")
            raise last_err

        try:
            r = validate_fn(model, val_loader, criterion, device, epoch)
        except RuntimeError as e:
            if _is_transient_cuda_error(e):
                log.warning(f"[{variant['name']}] val cuda error: {e}; retrying once.")
                torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
                r = validate_fn(model, val_loader, criterion, device, epoch)
            else:
                raise

        per_epoch.append({"epoch": epoch, **r})
        if r["iou"] > best_iou:
            best_iou = r["iou"]
            best_epoch = epoch
            cu.atomic_save(_full_state(epoch, status="best"), output_dir / "best.pth")

        # LATEST is written only AFTER the epoch fully completed (train+val).
        is_final = (epoch == total_epochs)
        cu.atomic_save(_full_state(epoch, status="completed" if is_final else "staged_partial"),
                       latest_path)
        _write_result(output_dir, variant, total_epochs, per_epoch, best_iou,
                      best_epoch, _elapsed(), status="completed" if is_final else "running")

        torch.cuda.empty_cache(); gc.collect()

        if epoch >= stop_epoch and not is_final:
            rec = _write_result(output_dir, variant, total_epochs, per_epoch, best_iou,
                                best_epoch, _elapsed(), status="staged_partial")
            log.info(f"[{variant['name']}] STAGED STOP after epoch {epoch} "
                     f"(<{total_epochs}). Resume with:\n    --resume {latest_path}")
            return rec

    return _write_result(output_dir, variant, total_epochs, per_epoch, best_iou,
                         best_epoch, _elapsed(), status="completed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Resumable Trans10K ablation runner (P2).")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--epochs", type=int, required=True,
                   help="TOTAL epochs for the experiment (defines scheduler length). "
                        "Never shortened by --stop-after-epoch.")
    p.add_argument("--output", required=True)
    p.add_argument("--variants", nargs="+", default=["ofcv_only", "ofcv_brf"],
                   help="Variant names from run_full_ablations.VARIANTS.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint_latest.pth to continue. Requires exactly one --variants.")
    p.add_argument("--stop-after-epoch", type=int, default=None,
                   help="Stop cleanly after this epoch (staged), saving a resumable checkpoint. "
                        "Does NOT change --epochs.")
    args = p.parse_args()

    names = {v["name"] for v in VARIANTS}
    for n in args.variants:
        if n not in names:
            p.error(f"unknown variant {n!r}; choose from {sorted(names)}")
    if args.resume and len(args.variants) != 1:
        p.error("--resume requires exactly one --variants (checkpoints are per-variant).")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [v for v in VARIANTS if v["name"] in args.variants]

    for v in selected:
        log_hdr = f"--- Variant: {v['name']} ({v.get('desc','')}) [total {args.epochs} ep] ---"
        logger.info(log_hdr)
        v_dir = out_root / v["name"]
        cfg = load_config(args.config)
        cfg.train.epochs = args.epochs        # TOTAL — scheduler built on this
        cfg.logging.save_dir = str(v_dir)
        cfg.experiment = f"full_ablation_{v['name']}_resumable"
        rec = run_variant_resumable(
            cfg, v, v_dir, total_epochs=args.epochs,
            resume_path=args.resume, stop_after_epoch=args.stop_after_epoch,
        )
        logger.info(f"[{v['name']}] status={rec['status']} "
                    f"best_iou={rec['best_iou']:.4f}@E{rec['best_epoch']} "
                    f"epochs_completed={rec['epochs_completed']}/{args.epochs}")

    # Merge any result.json present → summary (reflects whatever stage we're at).
    merged = []
    for v in VARIANTS:
        rp = out_root / v["name"] / "result.json"
        if rp.exists():
            try:
                merged.append(json.load(open(rp)))
            except Exception:
                pass
    json.dump(merged, open(out_root / "summary.json", "w"), indent=2)
    logger.info(f"summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
