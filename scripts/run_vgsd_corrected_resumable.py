"""
scripts/run_vgsd_corrected_resumable.py

RESUMABLE version of scripts/run_vgsd_corrected.py (P3): corrected-RAFT VGSD
ablation for ofcv_only + ofcv_brf at 336 / 12 ep, able to stop after N epochs
and resume in a fresh Kaggle session with full optimizer / scheduler / AMP /
step / best / RNG state restored.

Scientific reuse (unchanged):
  * train.run_vgsd_ablation.build_splits / VGSDPairs / evaluate / VARIANTS
  * models.spectra_model.SPECTRA, train.losses.SPECTRALoss
The per-variant object construction and the per-batch update below are mirrored
VERBATIM from train.run_vgsd_ablation.train_variant (lines ~110-138). They MUST
stay byte-identical to that function so the four VGSD variants share one
protocol. The ONLY additions are full-state checkpoint / resume / stop-after.

Matches run_vgsd_corrected.py exactly: same OUTDIR, RUN set, IMG/EPOCHS/BS, and
the "seed UNSET (matches existing VGSD ablation protocol)" convention.

Staged usage (per variant; VGSD checkpoints are per-variant):
  python scripts/run_vgsd_corrected_resumable.py --variant ofcv_only --stop-after-epoch 6
  python scripts/run_vgsd_corrected_resumable.py --variant ofcv_only \
      --resume EXPERIMENTS/18_corrected_vgsd_ablation_2026-10-02/vgsd_4way/ofcv_only/checkpoint_latest.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity with evaluate import path)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_config, get_logger, to_device  # noqa: E402
from models.spectra_model import SPECTRA              # noqa: E402
from train.losses import SPECTRALoss                  # noqa: E402
from train.run_vgsd_ablation import build_splits, VGSDPairs, evaluate, VARIANTS  # noqa: E402
import train.checkpoint_utils as cu                   # noqa: E402

logger = get_logger("spectra.vgsd.resumable")

IMG = 336
EPOCHS = 12
BS = 2
OUTDIR = ROOT / "EXPERIMENTS" / "18_corrected_vgsd_ablation_2026-10-02" / "vgsd_4way"
RUN = ["ofcv_only", "ofcv_brf"]  # RAFT-dependent variants; reuse neither/brf_only from 10_vgsd_native

VGSD_CHECK_KEYS = ["variant_name", "variant_flags", "dataset", "image_size", "batch_size", "steps_per_epoch"]


def _build_vgsd_objects(v, cfg, device, epochs, bs, n_train_pairs):
    """Mirror of run_vgsd_ablation.train_variant object construction (lines ~111-121).
    KEEP IN SYNC with that function — any change there must be reflected here."""
    from torch.optim import AdamW
    model = SPECTRA(cfg, use_gnn=v["use_gnn"], use_ofcv=v["use_ofcv"], use_brf=v["use_brf"]).to(device)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = AdamW([{"params": backbone, "lr": 1e-5}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
    crit = SPECTRALoss()
    scaler = torch.amp.GradScaler("cuda")
    # steps/T_max use the TOTAL epochs (never the stage count) so the cosine
    # schedule is identical whether or not the run is interrupted.
    steps = max(1, n_train_pairs // bs) * epochs  # len(DataLoader with drop_last) == n//bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-7)
    return model, opt, crit, scaler, sched


def _vgsd_train_one_epoch(model, tl, opt, sched, scaler, crit, device, step):
    """Mirror of the inner per-batch loop in run_vgsd_ablation.train_variant
    (lines ~125-132). KEEP IN SYNC — identical op order / clip / step."""
    model.train()
    for b in tl:
        b = to_device(b, device)
        with torch.amp.autocast("cuda"):
            out = model(b["image"], b["image_t1"], return_intermediates=True)
            loss = crit(predictions=out, targets={"mask": b["mask"], "material": b["material"]})["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
    return step


def run_variant_resumable(v, cfg, device, tr_ds, va_ds, n_train_pairs, vdir: Path,
                          total_epochs=EPOCHS, bs=BS, resume_path=None, stop_after_epoch=None):
    from torch.utils.data import DataLoader
    vdir = Path(vdir); vdir.mkdir(parents=True, exist_ok=True)
    H, W = cfg.data.image_size
    latest_path = vdir / "checkpoint_latest.pth"

    model, opt, crit, scaler, sched = _build_vgsd_objects(v, cfg, device, total_epochs, bs, n_train_pairs)
    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)

    current_prov = cu.make_provenance(
        variant=v, dataset="VGSD-D", image_size=cfg.data.image_size, batch_size=bs,
        total_epochs=total_epochs, steps_per_epoch=max(1, n_train_pairs // bs),
        seed="unset (matches existing VGSD ablation protocol)",
        extra={"image_px": IMG})

    per_epoch, step, best, best_epoch, start_epoch, elapsed_base = [], 0, -1.0, 0, 1, 0.0

    if resume_path:
        ckpt = cu.load_checkpoint_file(resume_path, device=device)
        cu.verify_resume_compat(ckpt, current_prov, total_epochs, check_keys=VGSD_CHECK_KEYS)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["global_step"]; best = ckpt["best_iou"]; best_epoch = ckpt["best_epoch"]
        per_epoch = [e for e in ckpt["per_epoch"] if e["epoch"] <= ckpt["epoch"]]
        start_epoch = ckpt["epoch"] + 1
        elapsed_base = ckpt.get("elapsed_s", 0.0)
        cu.restore_rng_states(ckpt.get("rng"), logger)  # RNG restored LAST
        logger.info(f"[{v['name']}] RESUME {resume_path}: next_epoch={start_epoch} step={step} "
                    f"best={best:.4f}@E{best_epoch}")
        if start_epoch > total_epochs:
            logger.info(f"[{v['name']}] already complete."); return _write(vdir, v, total_epochs, per_epoch, best, best_epoch, elapsed_base, "completed")
    elif latest_path.exists():
        raise RuntimeError(f"{latest_path} exists but --resume not given; refusing to overwrite. "
                           "Pass --resume or use a fresh dir.")

    stop_epoch = total_epochs
    if stop_after_epoch and stop_after_epoch > 0:
        stop_epoch = min(stop_after_epoch, total_epochs)
        if stop_epoch < start_epoch:
            raise RuntimeError(f"--stop-after-epoch {stop_after_epoch} precedes resume point {start_epoch}.")

    t0 = time.time()
    def _elapsed(): return elapsed_base + (time.time() - t0)
    def _full(epoch, status):
        return cu.build_full_state(epoch=epoch, total_epochs=total_epochs, global_step=step,
                                   best_iou=best, best_epoch=best_epoch, per_epoch=per_epoch,
                                   model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                                   provenance=current_prov, status=status, elapsed_s=_elapsed())

    for ep in range(start_epoch, total_epochs + 1):
        step = _vgsd_train_one_epoch(model, tl, opt, sched, scaler, crit, device, step)
        m = evaluate(model, vl, device, (H, W))
        per_epoch.append({"epoch": ep, **m})
        if m["iou"] > best:
            best = m["iou"]; best_epoch = ep
            cu.atomic_save(_full(ep, "best"), vdir / "best.pth")
        is_final = (ep == total_epochs)
        cu.atomic_save(_full(ep, "completed" if is_final else "staged_partial"), latest_path)
        _write(vdir, v, total_epochs, per_epoch, best, best_epoch, _elapsed(),
               "completed" if is_final else "running")
        print(f"   [{v['name']}] E{ep}/{total_epochs} val IoU={m['iou']:.4f}", flush=True)
        torch.cuda.empty_cache()
        if ep >= stop_epoch and not is_final:
            logger.info(f"[{v['name']}] STAGED STOP after epoch {ep} (<{total_epochs}). "
                        f"Resume with --resume {latest_path}")
            return _write(vdir, v, total_epochs, per_epoch, best, best_epoch, _elapsed(), "staged_partial")

    return _write(vdir, v, total_epochs, per_epoch, best, best_epoch, _elapsed(), "completed")


def _write(vdir: Path, v, total_epochs, per_epoch, best, best_epoch, elapsed_s, status) -> dict:
    rec = {"variant": v["name"], "best_iou": best, "best_epoch": best_epoch,
           "epochs": total_epochs, "epochs_completed": per_epoch[-1]["epoch"] if per_epoch else 0,
           "per_epoch": per_epoch, "duration_s": elapsed_s, "status": status,
           "config": {k: v[k] for k in ("use_ofcv", "use_brf", "use_gnn")}}
    json.dump(rec, open(vdir / "result.json", "w"), indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Resumable corrected-RAFT VGSD ablation (P3).")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dev-root", default=str(ROOT / "VSGD_dataset" / "test"))
    ap.add_argument("--train-root", default="")
    ap.add_argument("--variant", choices=RUN, default=None,
                    help="Run a single variant (required with --resume / --stop-after-epoch).")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    args = ap.parse_args()

    if (args.resume or args.stop_after_epoch) and not args.variant:
        ap.error("--resume / --stop-after-epoch require --variant (checkpoints are per-variant).")

    cfg = load_config(args.config)
    cfg.data.image_size = [IMG, IMG]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, mode = build_splits(args.train_root, args.dev_root)
    tr_ds, va_ds = VGSDPairs(tr, cfg.data.image_size, True), VGSDPairs(va, cfg.data.image_size, False)
    logger.info(f"[vgsd-resumable] mode={mode} train_pairs={len(tr)} val_pairs={len(va)} "
                f"img={IMG} ep={EPOCHS} bs={BS} seed=UNSET")

    variant_map = {v["name"]: v for v in VARIANTS}
    run_names = [args.variant] if args.variant else RUN
    for name in run_names:
        v = variant_map[name]
        rec = run_variant_resumable(v, cfg, device, tr_ds, va_ds, len(tr), OUTDIR / name,
                                    total_epochs=EPOCHS, bs=BS,
                                    resume_path=args.resume, stop_after_epoch=args.stop_after_epoch)
        logger.info(f"[{name}] status={rec['status']} best={rec['best_iou']:.4f}@E{rec['best_epoch']} "
                    f"epochs_completed={rec['epochs_completed']}/{EPOCHS}")


if __name__ == "__main__":
    main()
