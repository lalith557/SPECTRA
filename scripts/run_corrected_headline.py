"""
scripts/run_corrected_headline.py  — AUTHORIZED Priority-1 ONLY.

Regenerate the SPECTRA headline checkpoint under corrected RAFT normalization.
Uses train/train_baseline.py::train() VERBATIM (exact original headline protocol).
ONLY scientific change vs the legacy run = RAFT [0,255]->[-1,1] (working-tree fix).

Config defaults already encode the required protocol; this driver only:
  * points cfg.logging.save_dir at a NEW dir (never overwrites checkpoints/spectra_best.pth)
  * sets experiment name, seed 42, use_ofcv=True, use_brf=True, use_gnn=False
and asserts the protocol before launching (fail-loud -> STOP on any mismatch).

Writes -> EXPERIMENTS/13_corrected_headline_2026-10-02/checkpoints/{checkpoint_epochNNN,spectra_best}.pth
          EXPERIMENTS/13_corrected_headline_2026-10-02/meta.json
"""
import os, sys, json, time, subprocess
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils import load_config
from train.train_baseline import train

D = ROOT / "EXPERIMENTS" / "13_corrected_headline_2026-10-02"
CKPT = D / "checkpoints"


def main():
    cfg = load_config("configs/config.yaml")

    # --- protocol assertions (STOP if any mismatch; do not auto-fix) ---
    assert list(cfg.data.image_size) == [448, 448], f"image_size != 448: {cfg.data.image_size}"
    assert cfg.train.epochs == 10, f"epochs != 10: {cfg.train.epochs}"
    assert cfg.train.batch_size == 4, f"batch_size != 4: {cfg.train.batch_size}"
    assert abs(cfg.train.lr - 1e-4) < 1e-12, f"lr != 1e-4: {cfg.train.lr}"
    assert abs(cfg.train.lr_backbone - 1e-5) < 1e-12, f"lr_backbone != 1e-5: {cfg.train.lr_backbone}"
    assert abs(cfg.train.weight_decay - 1e-4) < 1e-12, f"wd != 1e-4: {cfg.train.weight_decay}"
    assert cfg.train.warmup_epochs == 1, f"warmup != 1: {cfg.train.warmup_epochs}"
    assert cfg.model.backbone == "dinov2_vits14", f"backbone: {cfg.model.backbone}"
    assert cfg.data.dataset == "trans10k", f"dataset: {cfg.data.dataset}"
    # RAFT correction present, old range absent
    rw = (ROOT / "flow" / "raft_wrapper.py").read_text(encoding="utf-8")
    assert "x * 2.0 - 1.0" in rw, "RAFT [-1,1] correction NOT present"
    assert "(x * 255.0).clamp(0, 255)" not in rw, "OLD [0,255] RAFT line still present"

    CKPT.mkdir(parents=True, exist_ok=True)
    cfg.logging.save_dir = str(CKPT)
    cfg.experiment = "spectra_full_corrected_raft"
    cfg.seed = 42
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        head = "unknown"

    print(f"[headline] CORRECTED-RAFT SPECTRA  ofcv=T brf=T gnn=F  seed=42  ep=10  448  bs=4", flush=True)
    print(f"[headline] save_dir={CKPT}  git={head}  RAFT=[-1,1] corrected", flush=True)
    print(f"[headline] legacy reference: spectra_best.pth best_iou=0.9217 (broken-RAFT, NOT overwritten)", flush=True)

    t0 = time.time()
    train(cfg_override=cfg, experiment_name=cfg.experiment,
          use_ofcv=True, use_brf=True, use_gnn=False)   # EXACT original training code
    dur_min = round((time.time() - t0) / 60, 1)

    best_path = CKPT / "spectra_best.pth"
    best_iou = best_epoch = None
    if best_path.exists():
        bc = torch.load(best_path, map_location="cpu", weights_only=False)
        best_iou = bc.get("best_iou"); best_epoch = bc.get("epoch")
    ckpts = sorted(str(p.name) for p in CKPT.glob("checkpoint_epoch*.pth"))
    meta = {
        "experiment": "spectra_full_corrected_raft (Priority-1 headline regeneration)",
        "classification": "D — corrected RAFT (train+eval)",
        "model": {"use_ofcv": True, "use_brf": True, "use_gnn": False, "backbone": cfg.model.backbone},
        "dataset": "trans10k", "train_split": "train", "val_split": "validation",
        "image_size": [448, 448], "batch_size": 4, "epochs": 10, "seed": 42,
        "optimizer": "AdamW bb=1e-5 rest=1e-4 wd=1e-4",
        "scheduler": "LinearLR warmup(1ep,0.01->1.0) -> CosineAnnealingLR eta_min=1e-7",
        "amp": True, "grad_clip": 1.0, "loss": "SPECTRALoss defaults",
        "checkpoint_selection": "best validation IoU (val at E1/E5/E10 per original protocol)",
        "raft_normalization": "[-1,1] CORRECTED",
        "only_scientific_change_vs_legacy": "RAFT normalization [0,255]->[-1,1]",
        "git_head": head, "git_state_file": "../GIT_STATE.txt",
        "best_iou": best_iou, "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "epoch_checkpoints": ckpts,
        "runtime_min": dur_min,
        "legacy_reference_only": {"spectra_best.pth": 0.9217, "note": "broken-RAFT; provenance/context only, not final"},
    }
    json.dump(meta, open(D / "meta.json", "w"), indent=2)
    print(f"[headline] DONE best_iou={best_iou} (E{best_epoch}) runtime={dur_min}min -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
