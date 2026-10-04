"""
eval/diag_ofcv_temporal.py  — PHASE 3 + PHASE 4 diagnostics.

Question (Phase 3): does OFCV actually respond to TEMPORAL information?
  CASE A: real pair   I_t  -> I_{t+1}
  CASE B: duplicated  I_t  -> I_t     (no motion; residual/flow must collapse)
If OFCV(A) ~= OFCV(B), the learned head ignores its physics input.

Question (Phase 4): is the OFCV gate (as USED in fusion: sigmoid(ofcv_map))
  collapsed / saturated / uniform, and does it localize GT transparent regions?

Also measures whether the handcrafted RESIDUAL localizes GT regions at all
(the premise of the whole OFCV mechanism), and tests the RAFT [0,255] path (F3)
against the correct torchvision [-1,1] transform.

Read-only w.r.t. the model. Writes results/diagnostics/*.json and *.png.
NO training. NO architectural change.
"""
import os, sys, glob, json, math, random
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config, to_device                       # noqa
from models.spectra_model import SPECTRA                       # noqa
from flow.raft_wrapper import (RAFTWrapper, compute_warp_residual,  # noqa
                               compute_flow_consistency)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)
OUT = ROOT / "results" / "diagnostics"
FIG = ROOT / "results" / "figures"
DEV_ROOT = ROOT / "VSGD_dataset" / "test"


def load_frame(path, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0), img  # (1,3,H,W), uint8-ish rgb


def load_mask(path, size):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def clip_pairs(clip_dir):
    frames = sorted(glob.glob(str(clip_dir / "JPEGImages" / "*.jpg")))
    pairs = []
    for i, f0 in enumerate(frames[:-1]):
        f1 = frames[i + 1]
        m = clip_dir / "SegmentationClassPNG" / f"{Path(f0).stem}.png"
        if m.exists():
            pairs.append((f0, f1, str(m)))
    return pairs


def stat(t):
    t = t.detach().float().flatten()
    return dict(mean=float(t.mean()), std=float(t.std()),
                min=float(t.min()), max=float(t.max()),
                p50=float(t.median()), p95=float(torch.quantile(t, 0.95)))


def frac(t, op, thr):
    t = t.detach().float().flatten()
    return float((op(t, thr)).float().mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml")
    size = int(cfg.data.image_size[0])   # 448 (checkpoint native)
    print(f"[diag] device={device} size={size}")

    # ---- build model + load trained checkpoint (Trans10K, ViT-S, ofcv+brf) ----
    model = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=True).to(device).eval()
    ckpt_path = ROOT / "checkpoints" / "spectra_best.pth"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # user's own trusted ckpt
    sd = sd.get("model_state_dict") or sd.get("model") or sd.get("state_dict") or sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diag] loaded {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")

    raft = model.flow_model  # reuse the frozen RAFT already inside the model

    # ---- sample pairs across clips (one mid-motion pair per clip) ----
    clips = sorted(p for p in DEV_ROOT.iterdir() if p.is_dir())
    random.seed(0)
    sample_clips = clips[:40]
    agg = {k: [] for k in [
        "flow_mag_real", "flow_mag_dup",
        "resid_mean_real", "resid_mean_dup", "d_resid",
        "resid_in_gt", "resid_out_gt", "resid_localization_ratio",
        "ofcv_mean_real", "ofcv_mean_dup", "d_ofcv",
        "gate_mean", "gate_std", "gate_min", "gate_max",
        "gate_gt0.5", "gate_lt0.1", "gate_gt0.9",
        "gate_in_gt", "gate_out_gt",
        "d_pred_realdup", "seg_iou_real",
    ]}
    per_sample = []
    vis_saved = 0

    for ci, clip in enumerate(sample_clips):
        pairs = clip_pairs(clip)
        if len(pairs) < 2:
            continue
        p0, p1, pm = pairs[len(pairs) // 2]  # middle pair (motion likely)
        x0, rgb0 = load_frame(p0, size); x1, rgb1 = load_frame(p1, size)
        gt = load_mask(pm, size)
        if gt is None:
            continue
        x0, x1, gt = x0.to(device), x1.to(device), gt.to(device)

        with torch.no_grad():
            # ---- handcrafted substrate: real vs duplicated ----
            ff_r, fb_r = raft(x0, x1)          # current [0,255] path (F3)
            ff_d, fb_d = raft(x0, x0)
            res_r = compute_warp_residual(x0, x1, ff_r)   # (1,1,H,W)
            res_d = compute_warp_residual(x0, x0, ff_d)
            mag_r = (ff_r ** 2).sum(1, keepdim=True).sqrt()
            mag_d = (ff_d ** 2).sum(1, keepdim=True).sqrt()

            gtm = gt.bool()
            res_in  = res_r[gtm].mean() if gtm.any() else torch.tensor(float('nan'))
            res_out = res_r[~gtm].mean() if (~gtm).any() else torch.tensor(float('nan'))

            # ---- learned OFCV: real vs duplicated ----
            out_r = model(x0, x1, return_intermediates=True)
            out_d = model(x0, x0, return_intermediates=True)
            ofcv_r = out_r["ofcv_map"]   # (1,1,H,W) in [0,1], = ofcv_full
            ofcv_d = out_d["ofcv_map"]
            gate = torch.sigmoid(ofcv_r)  # exactly as fusion uses it
            pred_r = out_r["seg_prob"]; pred_d = out_d["seg_prob"]

            # gate localization vs GT (at full res)
            gate_in  = gate[gtm].mean() if gtm.any() else torch.tensor(float('nan'))
            gate_out = gate[~gtm].mean() if (~gtm).any() else torch.tensor(float('nan'))

            inter = ((pred_r > 0.5) & gtm).sum().float()
            union = (((pred_r > 0.5) | gtm).sum().float()).clamp(min=1)
            iou_r = float(inter / union)

        rec = dict(
            clip=clip.name,
            flow_mag_real=float(mag_r.mean()), flow_mag_dup=float(mag_d.mean()),
            resid_mean_real=float(res_r.mean()), resid_mean_dup=float(res_d.mean()),
            d_resid=float((res_r - res_d).abs().mean()),
            resid_in_gt=float(res_in), resid_out_gt=float(res_out),
            resid_localization_ratio=float(res_in / (res_out + 1e-8)),
            ofcv_mean_real=float(ofcv_r.mean()), ofcv_mean_dup=float(ofcv_d.mean()),
            d_ofcv=float((ofcv_r - ofcv_d).abs().mean()),
            gate_mean=float(gate.mean()), gate_std=float(gate.std()),
            gate_min=float(gate.min()), gate_max=float(gate.max()),
            **{"gate_gt0.5": frac(gate, torch.gt, 0.5),
               "gate_lt0.1": frac(gate, torch.lt, 0.1),
               "gate_gt0.9": frac(gate, torch.gt, 0.9)},
            gate_in_gt=float(gate_in), gate_out_gt=float(gate_out),
            d_pred_realdup=float((pred_r - pred_d).abs().mean()),
            seg_iou_real=iou_r,
        )
        per_sample.append(rec)
        for k in agg:
            v = rec.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                agg[k].append(v)

        # ---- save a few visualizations ----
        if vis_saved < 5:
            save_vis(FIG / f"phase3_{clip.name}.png", rgb0, rgb1,
                     mag_r[0, 0].cpu().numpy(), res_r[0, 0].cpu().numpy(),
                     ofcv_r[0, 0].cpu().numpy(), gate[0, 0].cpu().numpy(),
                     gt[0, 0].cpu().numpy(), pred_r[0, 0].cpu().numpy(),
                     res_d[0, 0].cpu().numpy(), ofcv_d[0, 0].cpu().numpy())
            vis_saved += 1

        del out_r, out_d, ff_r, fb_r, ff_d, fb_d
        torch.cuda.empty_cache()
        print(f"  [{ci+1}/{len(sample_clips)}] {clip.name}: "
              f"flow_real={rec['flow_mag_real']:.3f} flow_dup={rec['flow_mag_dup']:.3f} "
              f"dOFCV={rec['d_ofcv']:.4f} gate_mean={rec['gate_mean']:.3f} "
              f"resid_ratio={rec['resid_localization_ratio']:.3f}", flush=True)

    summary = {"n": len(per_sample), "size": size,
               "git_note": "5ca9f0f", "checkpoint": "spectra_best.pth (Trans10K, ViT-S)"}
    for k, v in agg.items():
        if v:
            summary[k] = dict(mean=float(np.mean(v)), std=float(np.std(v)),
                              min=float(np.min(v)), max=float(np.max(v)))
    json.dump({"summary": summary, "per_sample": per_sample},
              open(OUT / "phase3_4_ofcv_temporal.json", "w"), indent=2)
    print("\n[done] ->", OUT / "phase3_4_ofcv_temporal.json")
    # concise console readout of the decisive numbers
    def m(k): return summary.get(k, {}).get("mean", float("nan"))
    print(f"  flow_mag real={m('flow_mag_real'):.3f}  dup={m('flow_mag_dup'):.3f}")
    print(f"  residual localization (in_GT/out_GT) ratio = {m('resid_localization_ratio'):.3f}")
    print(f"  dOFCV(real vs dup) = {m('d_ofcv'):.4f}   dPred(real vs dup) = {m('d_pred_realdup'):.4f}")
    print(f"  gate mean={m('gate_mean'):.3f} std={m('gate_std'):.4f} "
          f"min={m('gate_min'):.3f} max={m('gate_max'):.3f}")
    print(f"  gate in_GT={m('gate_in_gt'):.4f}  out_GT={m('gate_out_gt'):.4f}")


def _norm01(a):
    a = a.astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def save_vis(path, rgb0, rgb1, flowmag, resid, ofcv, gate, gt, pred, resid_d, ofcv_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(18, 7.5))
    panels_top = [("frame t", rgb0), ("frame t+1", rgb1),
                  ("RAFT |flow| (real)", _norm01(flowmag)),
                  ("residual (real)", _norm01(resid)), ("GT mask", gt)]
    panels_bot = [("OFCV (real)", ofcv), ("gate=σ(ofcv)", gate),
                  ("prediction", pred),
                  ("residual (DUP t,t)", _norm01(resid_d)),
                  ("OFCV (DUP t,t)", ofcv_d)]
    for a, (t, im) in zip(ax[0], panels_top):
        a.imshow(im, cmap=None if im.ndim == 3 else "magma", vmin=0, vmax=1 if im.ndim==2 else None)
        a.set_title(t, fontsize=10); a.axis("off")
    for a, (t, im) in zip(ax[1], panels_bot):
        a.imshow(im, cmap="magma", vmin=0, vmax=1); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
