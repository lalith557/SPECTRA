"""
spectra/train/train_baseline.py
Main training loop for SPECTRA.
Handles: warmup LR schedule, mixed precision, gradient clipping,
wandb logging, checkpoint saving, validation.
"""
import os
import sys
import math
import time
import argparse
import gc
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch import amp
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from utils import load_config, set_seed, get_logger, save_checkpoint, to_device
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders
from train.losses import SPECTRALoss
from eval.metrics import TransparentObjectMetrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = get_logger("spectra.train")


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
    total_steps  = cfg.train.epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    if cosine_steps <= 0:
        # Warmup spans the entire training run; no cosine phase. Avoid T_max=0.
        return warmup
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-7,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    criterion:  nn.Module,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    cfg,
    step:       int,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = to_device(batch, device)

        with amp.autocast("cuda"):
            outputs = model(
                image=batch["image"],
                image_t1=batch["image_t1"],
                return_intermediates=True,
            )
            losses = criterion(
                predictions=outputs,
                targets={"mask": batch["mask"], "material": batch["material"]},
            )
            loss = losses["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1
        step       += 1

        if step % cfg.logging.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            
            # Diagnostics
            ofcv_var = 0.0
            ofcv_grad_norm = 0.0
            if "ofcv_map" in outputs and outputs["ofcv_map"] is not None:
                ofcv_var = outputs["ofcv_map"].var().item()
                # Compute gradient norm for OFCV parameters
                ofcv_grads = [p.grad for n, p in model.named_parameters() if "ofcv" in n and p.grad is not None]
                if ofcv_grads:
                    ofcv_grad_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in ofcv_grads])).item()

            logger.info(
                f"[E{epoch}|S{step}] loss={loss.item():.4f} "
                f"seg={losses['seg'].item():.4f} "
                f"bnd={losses['bnd'].item():.4f} "
                f"cns={losses.get('consist', 0.0):.4f} "
                f"ref={losses.get('refl', 0.0):.4f} "
                f"var={losses.get('var', 0.0):.4f} "
                f"ofcv_var={ofcv_var:.2e} "
                f"lr={lr:.2e}"
            )
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "train/loss_total": loss.item(),
                    "train/loss_seg":   losses["seg"].item(),
                    "train/loss_bnd":   losses["bnd"].item(),
                    "train/loss_consist": losses.get("consist", 0.0),
                    "train/loss_refl":  losses.get("refl", 0.0),
                    "train/loss_var":   losses.get("var", 0.0),
                    "train/ofcv_map_var": ofcv_var,
                    "train/ofcv_grad_norm": ofcv_grad_norm,
                    "train/lr":         lr,
                    "step":             step,
                })

    return total_loss / max(n_batches, 1), step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:  torch.device,
    epoch:   int,
) -> dict:
    model.eval()
    metrics  = TransparentObjectMetrics()
    val_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = to_device(batch, device)

        with amp.autocast("cuda"):
            outputs = model(
                image=batch["image"],
                image_t1=batch["image_t1"],
                return_intermediates=False,
            )
            losses = criterion(
                predictions=outputs,
                targets={"mask": batch["mask"], "material": batch["material"]},
            )

        metrics.update(outputs["seg_prob"], batch["mask"])
        val_loss  += losses["total"].item()
        n_batches += 1

    results = metrics.compute()
    results["val_loss"] = val_loss / max(n_batches, 1)

    logger.info(
        f"[Val E{epoch}] IoU={results['iou']:.4f} "
        f"F={results['f_measure']:.4f} "
        f"MAE={results['mae']:.4f} "
        f"BER={results['ber']:.4f} "
        f"loss={results['val_loss']:.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg_path: str = None, cfg_override=None, experiment_name=None, use_ofcv=True, use_brf=True, use_gnn=False, resume_path=None):
    if cfg_override is not None:
        cfg = cfg_override
    else:
        cfg = load_config(cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # W&B init
    if WANDB_AVAILABLE and cfg.logging.get("wandb_project"):
        run_name = experiment_name if experiment_name else cfg.experiment
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            name=run_name,
            config=cfg.__dict__,
        )

    # Dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model = SPECTRA(cfg, use_gnn=use_gnn, use_ofcv=use_ofcv, use_brf=use_brf).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer: different LRs for backbone vs rest
    backbone_params = list(model.backbone.parameters())
    other_params    = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone") and p.requires_grad]

    optimizer = AdamW([
        {"params": backbone_params, "lr": cfg.train.lr_backbone},
        {"params": other_params,    "lr": cfg.train.lr},
    ], weight_decay=cfg.train.weight_decay)

    # Loss
    criterion = SPECTRALoss(
        lambda_seg=cfg.loss.lambda_seg,
        lambda_mat=cfg.loss.lambda_mat,
        lambda_bnd=cfg.loss.lambda_bnd,
        lambda_consist=cfg.loss.get("lambda_consist", 0.1),
        lambda_refl=cfg.loss.get("lambda_refl", 0.1),
        aux_var_weight=cfg.ofcv.get("aux_var_weight", 0.1),
    )

    # Scheduler + scaler
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler("cuda")

    best_iou = 0.0
    step     = 0
    start_epoch = 1

    if resume_path:
        from utils import load_checkpoint
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", 0.0)
        # Approximate step count
        step = (start_epoch - 1) * len(train_loader)
        logger.info(f"Resumed from {resume_path} (Starting Epoch {start_epoch})")

    patience = cfg.train.get("patience", 10)
    patience_counter = 0

    for epoch in range(start_epoch, cfg.train.epochs + 1):

        avg_loss, step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch, cfg, step,
        )

        # Validate at epoch 1 and every 5 epochs
        if epoch == 1 or epoch % 5 == 0:
            val_results = validate(model, val_loader, criterion, device, epoch)

            # Checkpoint
            is_best = val_results["iou"] > best_iou
            if is_best:
                best_iou = val_results["iou"]
                patience_counter = 0
            else:
                patience_counter += 1
            
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "val/iou":       val_results["iou"],
                    "val/f_measure": val_results["f_measure"],
                    "val/mae":       val_results["mae"],
                    "val/ber":       val_results["ber"],
                    "val/loss":      val_results["val_loss"],
                    "epoch":         epoch,
                })
        else:
            is_best = False

        if is_best or True:  # SAVE EVERY EPOCH during causal validation phase
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "cfg": cfg.__dict__,
                },
                save_dir=cfg.logging.save_dir,
                filename=f"checkpoint_epoch{epoch:03d}.pth",
                is_best=is_best,
            )

        logger.info(f"Best IoU so far: {best_iou:.4f} (Patience: {patience_counter}/{patience})")
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training complete.")
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    from typing import Tuple  # deferred import for type hint in function
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()
    train(args.config, resume_path=args.resume)
