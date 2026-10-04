"""
tests/test_resume_smoke.py

CPU-only smoke test for the full-state checkpoint/resume machinery
(train/checkpoint_utils.py + train/run_full_ablations_resumable.run_variant_resumable).

It forces CUDA off and swaps the heavy SPECTRA model / real dataloaders / real
train_one_epoch+validate for tiny CPU stand-ins, so it tests the EXACT resume
control flow (epoch continuation, no duplicate epochs, optimizer/scheduler/scaler/
step/best/RNG restoration, atomic writes, status transitions, config-mismatch
guard) WITHOUT any GPU work, datasets, or thermal load. The scientific functions
are untouched; production runs use them unchanged.

Run:  python tests/test_resume_smoke.py     (exits non-zero on failure)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU BEFORE torch import ("" is not honored on all builds)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import sys
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

import train.checkpoint_utils as cu
from train.run_full_ablations_resumable import run_variant_resumable

assert not torch.cuda.is_available(), "smoke test must run on CPU (set CUDA_VISIBLE_DEVICES='')"

DIM = 4
STEPS_PER_EPOCH = 3
X = torch.arange(DIM, dtype=torch.float32).reshape(1, DIM)  # fixed input (not RNG-derived)
VARIANT = {"name": "ofcv_only", "use_ofcv": True, "use_brf": False, "use_gnn": False, "desc": "test"}

EXECUTED = []  # epochs for which the (injected) train step actually ran


def make_cfg(total_epochs, batch_size=4):
    return SimpleNamespace(
        seed=42,
        data=SimpleNamespace(dataset="trans10k", image_size=[448, 448]),
        train=SimpleNamespace(batch_size=batch_size, lr=1e-4, lr_backbone=1e-5,
                              weight_decay=1e-4, warmup_epochs=1, epochs=total_epochs),
    )


def build_objs(cfg, variant, device):
    """Tiny deterministic stand-in. Called AFTER set_seed(cfg.seed) in the driver,
    so model init is reproducible across fresh runs."""
    model = nn.Linear(DIM, 1, bias=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    total_steps = cfg.train.epochs * STEPS_PER_EPOCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # disabled -> CPU safe
    loader = list(range(STEPS_PER_EPOCH))
    return {"model": model, "optimizer": opt, "scheduler": sched, "scaler": scaler,
            "criterion": None, "train_loader": loader, "val_loader": loader,
            "steps_per_epoch": STEPS_PER_EPOCH}


def train_epoch_fn(model, loader, optimizer, scheduler, criterion, scaler, device, epoch, cfg, step):
    """Mimics train_one_epoch's signature. Consumes torch CPU RNG each batch so
    the optimisation trajectory genuinely depends on RNG state (makes RNG restore
    observable)."""
    model.train()
    last = 0.0
    for _ in loader:
        optimizer.zero_grad()
        scale = 1.0 + 0.01 * torch.randn(()).item()      # <- consumes torch RNG
        out = model(X.to(device))
        loss = out.sum() * scale
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        last = float(loss.item())
    EXECUTED.append(epoch)
    return last, step


def validate_fn(model, loader, criterion, device, epoch):
    # Highest IoU at epoch 1 so best_epoch stays 1 while 'latest' advances.
    return {"iou": 1.0 - 0.1 * epoch, "f_measure": 0.0, "mae": 0.0, "ber": 0.0, "val_loss": 0.0}


def _run(out_dir, total_epochs, resume=None, stop_after=None, batch_size=4):
    cfg = make_cfg(total_epochs, batch_size)
    return run_variant_resumable(
        cfg, VARIANT, Path(out_dir), total_epochs=total_epochs,
        resume_path=resume, stop_after_epoch=stop_after,
        build_objs=build_objs, train_epoch_fn=train_epoch_fn, validate_fn=validate_fn)


def _final_weights(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    return sd["weight"].clone()


CHECKS = []
def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="spectra_resume_smoke_"))
    try:
        # ---- unit: RNG capture/restore round-trip ----
        st = cu.capture_rng_states()
        a_t = torch.randn(3); a_n = __import__("numpy").random.rand(3); a_p = __import__("random").random()
        cu.restore_rng_states(st)
        b_t = torch.randn(3); b_n = __import__("numpy").random.rand(3); b_p = __import__("random").random()
        check("RNG restore reproduces torch draws", torch.allclose(a_t, b_t))
        check("RNG restore reproduces numpy draws", (a_n == b_n).all())
        check("RNG restore reproduces python draws", a_p == b_p)

        # ---- unit: atomic_save leaves no tmp + reloads ----
        ap = tmp / "atomic.pth"
        cu.atomic_save({"format": cu.CKPT_FORMAT, "x": 1}, ap)
        check("atomic_save wrote file", ap.exists())
        check("atomic_save left no .tmp", not (tmp / "atomic.pth.tmp").exists())

        # ---- STAGE 1: epochs 1-2 of 4 (staged stop) ----
        EXECUTED.clear()
        d = tmp / "staged" / VARIANT["name"]
        rec1 = _run(d, total_epochs=4, stop_after=2)
        check("stage1 status == staged_partial", rec1["status"] == "staged_partial")
        check("stage1 ran epochs [1,2]", EXECUTED == [1, 2])
        check("stage1 per_epoch == [1,2]", [e["epoch"] for e in rec1["per_epoch"]] == [1, 2])
        check("stage1 checkpoint_latest.pth exists", (d / "checkpoint_latest.pth").exists())
        check("stage1 best.pth exists", (d / "best.pth").exists())
        check("stage1 best_epoch == 1", rec1["best_epoch"] == 1)
        ck = cu.load_checkpoint_file(d / "checkpoint_latest.pth")
        check("stage1 ckpt.epoch == 2", ck["epoch"] == 2)
        check("stage1 ckpt.next_epoch == 3", ck["next_epoch"] == 3)
        check("stage1 global_step == 2*spe", ck["global_step"] == 2 * STEPS_PER_EPOCH)

        # ---- STAGE 2: resume -> epochs 3-4 only ----
        EXECUTED.clear()
        rec2 = _run(d, total_epochs=4, resume=str(d / "checkpoint_latest.pth"))
        check("stage2 status == completed", rec2["status"] == "completed")
        check("stage2 ran ONLY epochs [3,4]", EXECUTED == [3, 4])
        check("stage2 per_epoch == [1,2,3,4] (no dups)",
              [e["epoch"] for e in rec2["per_epoch"]] == [1, 2, 3, 4])
        check("stage2 best_epoch still 1", rec2["best_epoch"] == 1)
        check("stage2 best_iou preserved (~0.9)", abs(rec2["best_iou"] - 0.9) < 1e-9)
        ck2 = cu.load_checkpoint_file(d / "checkpoint_latest.pth")
        check("stage2 global_step == 4*spe (continued)", ck2["global_step"] == 4 * STEPS_PER_EPOCH)

        # ---- DECISIVE: uninterrupted 4-epoch run == staged(2)+resume(2) final weights ----
        EXECUTED.clear()
        du = tmp / "uninterrupted" / VARIANT["name"]
        _run(du, total_epochs=4)
        w_uninterrupted = _final_weights(du / "checkpoint_latest.pth")
        w_staged = _final_weights(d / "checkpoint_latest.pth")
        check("resumed final weights == uninterrupted final weights (exact continuation)",
              torch.allclose(w_uninterrupted, w_staged, atol=0, rtol=0))

        # ---- GUARD: config mismatch fails loudly (batch_size) ----
        raised = ""
        try:
            _run(d, total_epochs=4, resume=str(d / "checkpoint_latest.pth"), batch_size=8)
        except RuntimeError as e:
            raised = str(e)
        check("batch_size mismatch raises", "batch_size mismatch" in raised)

        # ---- GUARD: total_epochs mismatch fails loudly ----
        raised = ""
        try:
            _run(tmp / "staged" / VARIANT["name"], total_epochs=5,
                 resume=str(d / "checkpoint_latest.pth"))
        except RuntimeError as e:
            raised = str(e)
        check("total_epochs mismatch raises", "total_epochs mismatch" in raised)

        # ---- GUARD: fresh run refuses to clobber existing checkpoint ----
        raised = ""
        try:
            _run(d, total_epochs=4)   # no --resume into a dir that already has checkpoint_latest
        except RuntimeError as e:
            raised = str(e)
        check("fresh run over existing checkpoint refuses", "Refusing to overwrite" in raised)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok in CHECKS if not ok)
    print(f"\n{'='*60}\n{len(CHECKS)-n_fail}/{len(CHECKS)} checks passed"
          f"{'' if n_fail == 0 else f'  ({n_fail} FAILED)'}\n{'='*60}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
