"""
eval/diag_gate_intervention.py  — inference-only OFCV gate-intervention control.

Uses the corrected-RAFT trained checkpoint (results/smoke_raftfix/ofcv_only_seed42.pth).
Replicates SPECTRA.forward by calling the trained submodules directly and substitutes
the patch-resolution OFCV map fed to the fusion head with 4 variants:
  REAL           : genuine OFCV(correct RAFT flow)
  SHUFFLED       : spatial permutation of REAL within each image (exact value dist preserved)
  CONSTANT       : per-image mean of REAL (spatial info removed, magnitude kept)
  RANDOM-MATCHED : N(mean,std) matched to REAL per image, clamped to [0,1]
No retraining. No model-code edits. No hyperparameter changes.

Question: does REAL differ from SHUFFLED/CONSTANT/RANDOM-MATCHED?
Reports IoU/F/MAE/BER per condition, pred-diff-from-REAL (whole-image, in-GT, boundary),
gate stats, per-image IoU. Saves 10 visualisations.
Outputs -> results/diagnostics/gate_intervention.json ; results/figures/gate_int_*.png
"""
import os, sys, json, random
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils import load_config, to_device
from models.spectra_model import SPECTRA
from train.run_vgsd_ablation import build_splits, VGSDPairs
from eval.metrics import TransparentObjectMetrics

SEED = 42
IMG = 336
OUT = ROOT / "results" / "diagnostics"; FIG = ROOT / "results" / "figures"
CONDS = ["REAL", "SHUFFLED", "CONSTANT", "RANDOM_MATCHED"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_variants(ofcv, gen):
    """ofcv: (B,1,h,w) in [0,1]. Return dict of the 4 intervened maps."""
    B, _, h, w = ofcv.shape
    out = {"REAL": ofcv}
    # SHUFFLED: per-image spatial permutation (exact value multiset preserved)
    flat = ofcv.view(B, 1, h * w)
    perm = torch.stack([torch.randperm(h * w, generator=gen, device=ofcv.device) for _ in range(B)])
    shuf = torch.gather(flat, 2, perm.unsqueeze(1)).view(B, 1, h, w)
    out["SHUFFLED"] = shuf
    # CONSTANT: per-image mean
    out["CONSTANT"] = ofcv.mean(dim=[2, 3], keepdim=True).expand_as(ofcv).contiguous()
    # RANDOM_MATCHED: N(mean,std) per image, clamped to [0,1]
    mean = ofcv.mean(dim=[2, 3], keepdim=True); std = ofcv.std(dim=[2, 3], keepdim=True)
    rnd = (torch.randn(ofcv.shape, generator=gen, device=ofcv.device) * std + mean).clamp(0, 1)
    out["RANDOM_MATCHED"] = rnd
    return out


def boundary_band(mask_np, width):
    k = np.ones((width, width), np.uint8)
    return ((cv2.dilate(mask_np, k) - cv2.erode(mask_np, k)) > 0)


def main():
    set_seed(SEED)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu"); gen.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml"); cfg.data.image_size = [IMG, IMG]; H, W = IMG, IMG

    model = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=False).to(device).eval()
    ck = torch.load(ROOT / "results" / "smoke_raftfix" / "ofcv_only_seed42.pth",
                    map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[gate-int] loaded corrected ckpt: missing={len(miss)} unexpected={len(unexp)}", flush=True)

    _, va, mode = build_splits("", str(ROOT / "VSGD_dataset" / "test"))
    va_ds = VGSDPairs(va, cfg.data.image_size, False)
    vl = DataLoader(va_ds, batch_size=2, shuffle=False, num_workers=0)
    print(f"[gate-int] val pairs={len(va)} split={mode}", flush=True)

    mets = {c: TransparentObjectMetrics() for c in CONDS}
    fidelity = TransparentObjectMetrics()  # model.forward REAL, to cross-check replication
    # pred-diff accumulators (vs REAL): whole / in-GT / boundary
    diff_acc = {c: {"whole": [0.0, 0], "gt": [0.0, 0], "bnd": [0.0, 0]} for c in CONDS if c != "REAL"}
    gate_acc = {c: [] for c in CONDS}
    per_image = {c: [] for c in CONDS}
    real_pred_ref_mean = [0.0, 0]
    vis = 0; idx = 0

    for b in vl:
        b = to_device(b, device)
        img, img1, mask = b["image"], b["image_t1"], b["mask"]  # mask (B,H,W)
        Bn = img.shape[0]
        with torch.no_grad():
            bo = model.backbone(img); patch = bo["patch_tokens"]; fpn = bo["features"]
            flow_fwd, residual, consistency = model._run_flow(img, img1)
            ofcv_real, _ = model.ofcv(patch_tokens=patch, residual_map=residual,
                                      consistency_map=consistency)  # (B,1,h,w)
            brf0 = torch.zeros(Bn, 1, H, W, device=device)  # use_brf=False path
            variants = make_variants(ofcv_real, gen)
            preds = {}
            for c in CONDS:
                fo = model.fusion(patch_tokens=patch, fpn_features=fpn,
                                  ofcv_map=variants[c], brf_map=brf0)
                p = fo["seg_prob"]
                if p.shape[-2:] != (H, W):
                    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
                preds[c] = p
                mets[c].update(p, mask)
                gate = torch.sigmoid(variants[c])
                gate_acc[c].append((float(gate.mean()), float(gate.std()),
                                    float(gate.min()), float(gate.max())))
            # fidelity: full model.forward REAL
            fo_full = model(img, img1, return_intermediates=False)
            fidelity.update(fo_full["seg_prob"], mask)

        # per-image metrics + diffs
        for i in range(Bn):
            gt = mask[i].unsqueeze(0).unsqueeze(0).float()
            gtb = mask[i].bool()
            bnp = mask[i].detach().cpu().numpy().astype(np.uint8)
            band = torch.from_numpy(boundary_band(bnp, max(3, IMG // 90))).to(device)
            pr = preds["REAL"][i:i+1]
            real_pred_ref_mean[0] += float(pr.mean()); real_pred_ref_mean[1] += 1
            for c in CONDS:
                inter = (((preds[c][i] > 0.5).squeeze(0)) & gtb).sum().float()
                union = ((((preds[c][i] > 0.5).squeeze(0)) | gtb).sum().float()).clamp(min=1)
                per_image[c].append(float(inter / union))
                if c == "REAL":
                    continue
                d = (preds[c][i:i+1] - pr).abs()
                diff_acc[c]["whole"][0] += float(d.mean()); diff_acc[c]["whole"][1] += 1
                if gtb.any():
                    diff_acc[c]["gt"][0] += float(d.squeeze()[gtb].mean()); diff_acc[c]["gt"][1] += 1
                if band.any():
                    diff_acc[c]["bnd"][0] += float(d.squeeze()[band].mean()); diff_acc[c]["bnd"][1] += 1

            # visualisations (10 samples)
            if vis < 10:
                save_vis(FIG / f"gate_int_{idx+i:03d}.png", img[i], variants, preds, mask[i], H, W)
                vis += 1
        idx += Bn
        if idx % 100 < Bn:
            print(f"  processed {idx}/{len(va)}", flush=True)

    real_mean_pred = real_pred_ref_mean[0] / max(real_pred_ref_mean[1], 1)
    summary = {"n_val": len(va), "checkpoint": "smoke_raftfix/ofcv_only_seed42.pth",
               "split": mode, "image_size": [IMG, IMG], "seed": SEED,
               "fidelity_modelforward_iou": fidelity.compute()["iou"]}
    for c in CONDS:
        m = mets[c].compute()
        ga = np.array(gate_acc[c])
        summary[c] = {"iou": round(m["iou"], 4), "f_measure": round(m["f_measure"], 4),
                      "mae": round(m["mae"], 4), "ber": round(m["ber"], 4),
                      "per_image_iou_mean": round(float(np.mean(per_image[c])), 4),
                      "per_image_iou_std": round(float(np.std(per_image[c])), 4),
                      "gate_mean": round(float(ga[:, 0].mean()), 4),
                      "gate_std": round(float(ga[:, 1].mean()), 4),
                      "gate_min": round(float(ga[:, 2].mean()), 4),
                      "gate_max": round(float(ga[:, 3].mean()), 4)}
        if c != "REAL":
            w = diff_acc[c]["whole"]; g = diff_acc[c]["gt"]; bd = diff_acc[c]["bnd"]
            wa = w[0]/max(w[1],1); summary[c]["pred_absdiff_from_REAL_whole"] = round(wa, 6)
            summary[c]["pred_absdiff_from_REAL_inGT"] = round(g[0]/max(g[1],1), 6)
            summary[c]["pred_absdiff_from_REAL_boundary"] = round(bd[0]/max(bd[1],1), 6)
            summary[c]["pred_reldiff_from_REAL_pct"] = round(100.0 * wa / max(real_mean_pred, 1e-8), 4)
    json.dump({"summary": summary, "per_image_iou": per_image},
              open(OUT / "gate_intervention.json", "w"), indent=2)

    print("\n[done] ->", OUT / "gate_intervention.json")
    print(f"  fidelity check: model.forward REAL IoU={summary['fidelity_modelforward_iou']:.4f} "
          f"vs replicated REAL IoU={summary['REAL']['iou']:.4f}")
    for c in CONDS:
        s = summary[c]
        extra = "" if c == "REAL" else (f"  |Δpred|whole={s['pred_absdiff_from_REAL_whole']:.5f} "
                                        f"({s['pred_reldiff_from_REAL_pct']:.3f}%) "
                                        f"inGT={s['pred_absdiff_from_REAL_inGT']:.5f} "
                                        f"bnd={s['pred_absdiff_from_REAL_boundary']:.5f}")
        print(f"  {c:14s} IoU={s['iou']:.4f} F={s['f_measure']:.4f} MAE={s['mae']:.4f} "
              f"BER={s['ber']:.4f} gate[{s['gate_min']:.3f},{s['gate_max']:.3f}]{extra}")


def _n01(a):
    a = a.astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def save_vis(path, img, variants, preds, mask, H, W):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mean = np.array([0.485,0.456,0.406],np.float32); std = np.array([0.229,0.224,0.225],np.float32)
    rgb = (img.detach().cpu().permute(1,2,0).numpy()*std+mean).clip(0,1)
    up = lambda m: F.interpolate(m, size=(H,W), mode="nearest")[0,0].detach().cpu().numpy()
    cols = [("RGB", rgb, None),
            ("REAL OFCV", up(variants["REAL"]), "magma"),
            ("SHUFFLED OFCV", up(variants["SHUFFLED"]), "magma"),
            ("CONSTANT OFCV", up(variants["CONSTANT"]), "magma"),
            ("REAL pred", preds["REAL"][0,0].detach().cpu().numpy(), "magma"),
            ("SHUFFLED pred", preds["SHUFFLED"][0,0].detach().cpu().numpy(), "magma"),
            ("CONSTANT pred", preds["CONSTANT"][0,0].detach().cpu().numpy(), "magma"),
            ("GT", mask.detach().cpu().numpy(), "gray")]
    fig, ax = plt.subplots(1, 8, figsize=(26, 3.6))
    for a,(t,im,cm) in zip(ax, cols):
        a.imshow(im if cm is None else im, cmap=cm, vmin=0, vmax=1 if cm else None)
        a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)


if __name__ == "__main__":
    main()
