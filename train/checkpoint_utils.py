"""
train/checkpoint_utils.py

Full-state checkpoint/resume INFRASTRUCTURE for SPECTRA ablation training.

This module contains NO scientific definitions — no model, optimizer, loss,
scheduler, augmentation, dataset, or evaluation logic. It only:
  * snapshots / restores RNG state (python, numpy, torch-CPU, torch-CUDA),
  * atomically writes a checkpoint (tmp file -> os.replace),
  * assembles a full-state checkpoint dict from objects the caller passes in,
  * verifies a checkpoint belongs to the EXACT experiment configuration
    before a resume is allowed (fails loudly on any mismatch).

It is deliberately generic so the two resumable runners
(train/run_full_ablations_resumable.py  — P2 / Trans10K,
 scripts/run_vgsd_corrected_resumable.py — P3 / VGSD)
share one tested implementation of the full-state machinery.

Nothing here changes the training computation.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

CKPT_FORMAT = "spectra_full_ckpt_v1"

# Provenance keys compared on resume. A runner may pass its own subset
# (VGSD, for instance, has no seed / warmup / per-group LR exposed).
DEFAULT_CHECK_KEYS = [
    "variant_name", "variant_flags", "dataset", "image_size", "batch_size",
    "seed", "lr", "lr_backbone", "weight_decay", "warmup_epochs",
    "steps_per_epoch",   # guards scheduler T_max against a different dataset size
]


# ---------------------------------------------------------------------------
# RNG snapshot / restore
# ---------------------------------------------------------------------------

def capture_rng_states() -> Dict[str, Any]:
    """Snapshot every RNG that can influence training. Non-mutating."""
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()  # list[ByteTensor]
        except Exception:
            cuda_states = None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "n_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def restore_rng_states(rng: Optional[Dict[str, Any]], logger=None) -> None:
    """Restore RNG state saved by capture_rng_states(). Best-effort + loud."""
    if rng is None:
        _warn(logger, "no RNG state in checkpoint; continuing without RNG restore")
        return
    try:
        random.setstate(rng["python"])
    except Exception as e:
        _warn(logger, f"could not restore Python RNG: {e}")
    try:
        np.random.set_state(rng["numpy"])
    except Exception as e:
        _warn(logger, f"could not restore NumPy RNG: {e}")
    try:
        cpu = rng["torch_cpu"]
        if isinstance(cpu, torch.Tensor):
            torch.set_rng_state(cpu.cpu().to(torch.uint8))
    except Exception as e:
        _warn(logger, f"could not restore torch CPU RNG: {e}")

    cuda_states = rng.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            avail = torch.cuda.device_count()
            states = [s.cpu().to(torch.uint8) for s in cuda_states]
            if len(states) == avail:
                torch.cuda.set_rng_state_all(states)
            else:
                for i in range(min(avail, len(states))):
                    torch.cuda.set_rng_state(states[i], device=i)
                _warn(logger, f"CUDA device count changed (ckpt={len(states)}, now={avail}); "
                              f"restored first {min(avail, len(states))} device RNG state(s)")
        except Exception as e:
            _warn(logger, f"could not restore CUDA RNG: {e}")


def _warn(logger, msg: str) -> None:
    if logger is not None:
        logger.warning(f"[checkpoint_utils] {msg}")
    else:
        print(f"[checkpoint_utils][WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def atomic_save(state: Dict[str, Any], path) -> None:
    """Write to a .tmp sibling then os.replace() so a crash mid-write never
    corrupts the previous valid checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass  # fsync is a durability nicety, not required for correctness
    os.replace(tmp, path)  # atomic on same filesystem; replaces existing


def load_checkpoint_file(path, device="cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != CKPT_FORMAT:
        raise RuntimeError(
            f"{path} is not a SPECTRA full-state checkpoint "
            f"(format={ckpt.get('format')!r}, expected {CKPT_FORMAT!r}). "
            "Refusing to resume from a weights-only / legacy checkpoint."
        )
    return ckpt


# ---------------------------------------------------------------------------
# Provenance + full-state dict assembly
# ---------------------------------------------------------------------------

def make_provenance(*, variant: Dict[str, Any], dataset: str, image_size,
                    batch_size: int, total_epochs: int, steps_per_epoch: int,
                    seed=None, lr=None, lr_backbone=None, weight_decay=None,
                    warmup_epochs=None, extra: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    prov = {
        "variant_name": variant["name"],
        "variant_flags": {k: bool(variant.get(k, False))
                          for k in ("use_ofcv", "use_brf", "use_gnn")},
        "dataset": dataset,
        "image_size": _as_list(image_size),
        "batch_size": int(batch_size),
        "total_epochs": int(total_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "seed": seed,
        "lr": lr,
        "lr_backbone": lr_backbone,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
    }
    if extra:
        prov.update(extra)
    return prov


def build_full_state(*, epoch: int, total_epochs: int, global_step: int,
                     best_iou: float, best_epoch: int, per_epoch: List[Dict],
                     model, optimizer, scheduler, scaler,
                     provenance: Dict[str, Any], status: str,
                     elapsed_s: float) -> Dict[str, Any]:
    """Assemble the complete resumable checkpoint. RNG is snapshotted HERE,
    i.e. at the save point (end of a completed epoch) — which is exactly the
    state the next epoch's DataLoader iterator would see in an uninterrupted
    run, so resume reproduces that epoch's shuffle + augmentation seeding."""
    return {
        "format": CKPT_FORMAT,
        "epoch": int(epoch),                 # last FULLY completed epoch
        "next_epoch": int(epoch) + 1,        # unambiguous resume point
        "total_epochs": int(total_epochs),
        "global_step": int(global_step),
        "best_iou": float(best_iou),
        "best_epoch": int(best_epoch),
        "per_epoch": list(per_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng": capture_rng_states(),
        "provenance": provenance,
        "status": status,
        "elapsed_s": float(elapsed_s),
        "torch_version": torch.__version__,
    }


# ---------------------------------------------------------------------------
# Resume compatibility verification — FAIL LOUDLY
# ---------------------------------------------------------------------------

def verify_resume_compat(ckpt: Dict[str, Any], current_prov: Dict[str, Any],
                         requested_total_epochs: int,
                         check_keys: Optional[List[str]] = None) -> None:
    """Raise RuntimeError if the checkpoint does not belong to the exact
    experiment configuration being requested now."""
    check_keys = check_keys if check_keys is not None else DEFAULT_CHECK_KEYS
    errs: List[str] = []
    cprov = ckpt.get("provenance", {}) or {}

    done = ckpt.get("epoch")
    if done is None:
        errs.append("checkpoint has no 'epoch' field")
    elif done >= requested_total_epochs:
        errs.append(f"checkpoint epoch={done} is not < requested total "
                    f"epochs={requested_total_epochs}: nothing left to train")

    # The requested total MUST equal the checkpoint's total, otherwise the
    # per-batch scheduler length (T_max) differs and the LR trajectory would
    # silently change — exactly the protocol mixing we must prevent.
    if cprov.get("total_epochs") != requested_total_epochs:
        errs.append(f"total_epochs mismatch: checkpoint={cprov.get('total_epochs')}, "
                    f"requested={requested_total_epochs} (scheduler T_max would differ)")

    for k in check_keys:
        want = _norm(current_prov.get(k))
        have = _norm(cprov.get(k))
        if want != have:
            errs.append(f"{k} mismatch: checkpoint={cprov.get(k)!r}, current={current_prov.get(k)!r}")

    if errs:
        raise RuntimeError(
            "Checkpoint configuration mismatch — refusing to resume "
            "(no silent protocol mixing):\n  - " + "\n  - ".join(errs))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_list(x):
    if isinstance(x, (list, tuple)):
        return [int(v) if isinstance(v, (int, np.integer)) else v for v in x]
    return x


def _norm(v):
    """Normalise for equality: lists/tuples -> tuple, floats compared with a
    small tolerance to absorb yaml/json round-tripping (1e-4 vs 0.0001)."""
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _norm(val)) for k, val in v.items()))
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v
