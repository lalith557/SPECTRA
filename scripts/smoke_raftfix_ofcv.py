"""
scripts/smoke_raftfix_ofcv.py  —  6-epoch OFCV smoke test AFTER the RAFT [-1,1] fix.

Mirrors train/run_vgsd_ablation.py::train_variant EXACTLY (same optimizer groups,
LR, weight decay, cosine schedule, AMP, grad-clip, SPECTRALoss defaults, same
clip-disjoint split, same 336 res, bs 2). The ONLY differences vs the ablation:
  * variant fixed to ofcv_only, epochs=6 (smoke),
  * seed FIXED to 42 and recorded (ablation left it unset),
  * saves a checkpoint and OFCV/flow statistics for interpretation.
No hyperparameter tuning. No model/loss/eval edits.

Outputs -> results/smoke_raftfix/ofcv_only_seed42.{json,pth}
"""
import os, sys, json, time, random
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils import load_config, to_device
from models.spectra_model import SPECTRA
from train.losses import SPECTRALoss
from train.run_vgsd_ablation import build_splits, VGSDPairs, evaluate

SEED = 42
EPOCHS = 6
IMG = 336
BS = 2
OUT = ROOT / "results" / "smoke_raftfix"; OUT.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@torch.no_grad()
def ofcv_flow_stats(model, va_ds, device, hw, n=15):
    """gate=sigmoid(ofcv_map) stats + real-vs-dup deltas + flow magnitudes."""
    model.eval(); H, W = hw
    g_mean = []; g_std = []; g_min = []; g_max = []
    d_ofcv = []; d_pred = []; fmag_r = []; fmag_d = []
    idxs = list(range(0, min(len(va_ds), n * 3), 3))[:n]
    for i in idxs:
        b = va_ds[i]
        img = b["image"].unsqueeze(0).to(device); img1 = b["image_t1"].unsqueeze(0).to(device)
        out_r = model(img, img1, return_intermediates=True)
        out_d = model(img, img, return_intermediates=True)
        ofcv_r = out_r["ofcv_map"]; ofcv_d = out_d["ofcv_map"]
        gate = torch.sigmoid(ofcv_r)
        g_mean.append(float(gate.mean())); g_std.append(float(gate.std()))
        g_min.append(float(gate.min())); g_max.append(float(gate.max()))
        d_ofcv.append(float((ofcv_r - ofcv_d).abs().mean()))
        d_pred.append(float((out_r["seg_prob"] - out_d["seg_prob"]).abs().mean()))
        # flow magnitude real vs dup (confirms RAFT fix took effect)
        ff_r, _ = model.flow_model(img, img1); ff_d, _ = model.flow_model(img, img)
        fmag_r.append(float((ff_r ** 2).sum(1).sqrt().mean()))
        fmag_d.append(float((ff_d ** 2).sum(1).sqrt().mean()))
    mean = lambda a: float(np.mean(a))
    return dict(n=len(idxs), gate_mean=mean(g_mean), gate_std=mean(g_std),
                gate_min=mean(g_min), gate_max=mean(g_max),
                d_ofcv_real_vs_dup=mean(d_ofcv), d_pred_real_vs_dup=mean(d_pred),
                flow_mag_real=mean(fmag_r), flow_mag_dup=mean(fmag_d))


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml"); cfg.data.image_size = [IMG, IMG]
    H, W = cfg.data.image_size
    tr, va, mode = build_splits("", str(ROOT / "VSGD_dataset" / "test"))
    tr_ds = VGSDPairs(tr, cfg.data.image_size, True)
    va_ds = VGSDPairs(va, cfg.data.image_size, False)
    print(f"[smoke] mode={mode} train={len(tr)} val={len(va)} img={IMG} seed={SEED} epochs={EPOCHS}", flush=True)

    # ofcv_only variant (mirrors VARIANTS entry) --------------------------------
    model = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=False).to(device)

    # ---- verify RAFT fix BEFORE training: dup-frame flow must be ~0 ----
    with torch.no_grad():
        b0 = va_ds[0]; im = b0["image"].unsqueeze(0).to(device); im1 = b0["image_t1"].unsqueeze(0).to(device)
        ff_r, _ = model.flow_model(im, im1); ff_d, _ = model.flow_model(im, im)
        pre = dict(flow_mag_real=float((ff_r**2).sum(1).sqrt().mean()),
                   flow_mag_dup=float((ff_d**2).sum(1).sqrt().mean()))
    print(f"[smoke] RAFT-fix check (fresh model): flow real={pre['flow_mag_real']:.3f}px "
          f"dup={pre['flow_mag_dup']:.3f}px  (dup should be ~0; was ~184 before fix)", flush=True)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = AdamW([{"params": backbone, "lr": 1e-5}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
    crit = SPECTRALoss()
    scaler = torch.amp.GradScaler("cuda")
    tl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va_ds, batch_size=BS, shuffle=False, num_workers=0)
    steps = max(1, len(tl)) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-7)

    per_epoch = []; best = -1.0; t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tl:
            b = to_device(b, device)
            with torch.amp.autocast("cuda"):
                out = model(b["image"], b["image_t1"], return_intermediates=True)
                loss = crit(predictions=out, targets={"mask": b["mask"], "material": b["material"]})["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        m = evaluate(model, vl, device, (H, W))
        per_epoch.append({"epoch": ep, **m}); best = max(best, m["iou"])
        print(f"   [ofcv_only+raftfix] E{ep}/{EPOCHS} val IoU={m['iou']:.4f}", flush=True)

    train_min = round((time.time() - t0) / 60, 1)
    gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else None
    ckpt = OUT / "ofcv_only_seed42.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": EPOCHS,
                "best_iou": best, "variant": "ofcv_only", "seed": SEED,
                "img": IMG, "raft_fix": "[-1,1] normalization"}, ckpt)
    ofcv_stats = ofcv_flow_stats(model, va_ds, device, (H, W))

    rec = {
        "experiment": "smoke_raftfix_ofcv_only",
        "evidence_class": "A_implementation_debug + B_controlled (single-seed smoke)",
        "git_head": "5ca9f0f (RAFT fix uncommitted in working tree: flow/raft_wrapper.py)",
        "config": {"variant": "ofcv_only", "use_ofcv": True, "use_brf": False, "use_gnn": False},
        "seed": SEED, "split": mode, "image_size": [IMG, IMG], "batch_size": BS,
        "epochs": EPOCHS, "n_train_pairs": len(tr), "n_val_pairs": len(va),
        "optimizer": "AdamW bb=1e-5 rest=1e-4 wd=1e-4, cosine->1e-7, AMP, clip=1.0",
        "loss": "SPECTRALoss() defaults (seg+bnd+consist=0.1+refl=0.1+var=0.1)",
        "raft_fix_precheck": pre,
        "per_epoch": per_epoch,
        "best_iou": best, "final_iou": per_epoch[-1]["iou"],
        "train_min": train_min, "gpu_max_mem_mb": gpu_mb,
        "checkpoint": str(ckpt),
        "ofcv_flow_stats_posttrain": ofcv_stats,
    }
    json.dump(rec, open(OUT / "ofcv_only_seed42.json", "w"), indent=2)
    print("\n[smoke done] ->", OUT / "ofcv_only_seed42.json")
    print(f"  best={best:.4f} final={per_epoch[-1]['iou']:.4f} time={train_min}min gpu={gpu_mb}MB")
    print(f"  post-train flow real={ofcv_stats['flow_mag_real']:.3f} dup={ofcv_stats['flow_mag_dup']:.3f}")
    print(f"  gate mean={ofcv_stats['gate_mean']:.3f} std={ofcv_stats['gate_std']:.4f} "
          f"[{ofcv_stats['gate_min']:.3f},{ofcv_stats['gate_max']:.3f}]")
    print(f"  dOFCV(real vs dup)={ofcv_stats['d_ofcv_real_vs_dup']:.4f} "
          f"dPred={ofcv_stats['d_pred_real_vs_dup']:.6f}")


if __name__ == "__main__":
    main()
