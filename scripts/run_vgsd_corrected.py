"""
scripts/run_vgsd_corrected.py  — AUTHORIZED corrected-RAFT VGSD runs.

Re-runs ONLY ofcv_only and ofcv_brf at 336 / 12 ep with the verified RAFT [-1,1]
normalization fix active (flow/raft_wrapper.py). Reuses train_variant from
train/run_vgsd_ablation.py VERBATIM -> identical optimizer/scheduler/loss/batch/
augmentation/eval as the existing neither & brf_only runs (which are reused).

Protocol note: matches the existing VGSD ablation's initialization protocol,
which did NOT set an explicit seed (run_vgsd_ablation.py applies no manual seed);
we keep that so all four variants share one protocol. Seed state is recorded as
"unset (matches existing VGSD ablation)". No hyperparameters changed.

Writes -> EXPERIMENTS/18_corrected_vgsd_ablation_2026-10-02/vgsd_4way/<variant>/result.json (+ model.pth, meta.json)
Does NOT touch EXPERIMENTS/10_vgsd_native.
"""
import os, sys, json, time, subprocess
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils import load_config
from train.run_vgsd_ablation import build_splits, VGSDPairs, train_variant, VARIANTS

IMG = 336; EPOCHS = 12; BS = 2
# Fresh, clearly-named output dir (NOT the interrupted 11_... pre-audit dir).
OUTDIR = ROOT / "EXPERIMENTS" / "18_corrected_vgsd_ablation_2026-10-02" / "vgsd_4way"
RUN = ["ofcv_only", "ofcv_brf"]   # only the RAFT-dependent variants; reuse neither/brf_only from 10_vgsd_native


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml"); cfg.data.image_size = [IMG, IMG]
    H, W = cfg.data.image_size
    tr, va, mode = build_splits("", str(ROOT / "VSGD_dataset" / "test"))
    tr_ds = VGSDPairs(tr, cfg.data.image_size, True)
    va_ds = VGSDPairs(va, cfg.data.image_size, False)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        head = "unknown"
    print(f"[vgsd-corrected] mode={mode} train={len(tr)} val={len(va)} img={IMG} ep={EPOCHS} bs={BS}", flush=True)
    print(f"[vgsd-corrected] RAFT fix active; running {RUN}; seed UNSET (matches existing protocol)", flush=True)

    variant_map = {v["name"]: v for v in VARIANTS}
    for name in RUN:
        v = variant_map[name]
        vdir = OUTDIR / name; vdir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rec = train_variant(v, tr_ds, va_ds, cfg, device, EPOCHS, BS)   # EXACT ablation logic
        rec["train_min"] = round((time.time() - t0) / 60, 1)
        gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else None
        meta = {
            "experiment": "vgsd_corrected_raft",
            "variant": name,
            "raft_normalization": "[-1,1] CORRECTED (flow/raft_wrapper.py)",
            "git_head": head,
            "git_note": "RAFT fix uncommitted in working tree; see ../GIT_STATE.txt",
            "dataset": "VGSD-D (Liu et al. 2024)", "split": mode,
            "n_train_pairs": len(tr), "n_val_pairs": len(va),
            "image_size": [IMG, IMG], "batch_size": BS, "epochs": EPOCHS,
            "optimizer": "AdamW bb=1e-5 rest=1e-4 wd=1e-4", "scheduler": "cosine->1e-7",
            "loss": "SPECTRALoss() defaults", "amp": True, "grad_clip": 1.0,
            "seed": "unset (matches existing VGSD ablation protocol)",
            "init": "DINOv2 pretrained (torch.hub) + random heads; from-scratch, no checkpoint loaded",
            "train_min": rec["train_min"], "gpu_max_mem_mb": gpu_mb,
            "best_iou": rec["best_iou"],
            "final_iou": rec["per_epoch"][-1]["iou"],
            "best_epoch": max(rec["per_epoch"], key=lambda e: e["iou"])["epoch"],
            "mean_iou": round(sum(e["iou"] for e in rec["per_epoch"]) / len(rec["per_epoch"]), 4),
        }
        json.dump(rec, open(vdir / "result.json", "w"), indent=2)
        json.dump(meta, open(vdir / "meta.json", "w"), indent=2)
        print(f"[vgsd-corrected] {name}: best={rec['best_iou']:.4f} (E{meta['best_epoch']}) "
              f"final={meta['final_iou']:.4f} mean={meta['mean_iou']:.4f} "
              f"time={rec['train_min']}min gpu={gpu_mb}MB -> {vdir/'result.json'}", flush=True)
        torch.cuda.empty_cache()
    print("[vgsd-corrected] DONE both variants.", flush=True)


if __name__ == "__main__":
    main()
