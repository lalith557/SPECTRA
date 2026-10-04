"""
eval/diag_brf_intervention.py  — inference-only BRF gate-intervention control.

Analog of the OFCV control, for BRF. Uses spectra_best.pth (Trans10K, ofcv+brf,
the ONLY trained-BRF checkpoint, 448-res). Holds the OFCV map FIXED (at its REAL
value) and substitutes only the BRF map fed to the fusion head with 4 variants:
  REAL           : genuine brf_refined
  SHUFFLED       : patch-res spatial permutation of REAL (value dist preserved)
  CONSTANT       : per-image mean of REAL (spatial info removed)
  RANDOM-MATCHED : N(mean,std) matched to REAL at patch-res, clamped to [0,1]
BRF enters fusion downsampled to the patch grid, so variants are built at patch
resolution (32x32 for 448) and nearest-upsampled; REAL is passed at full res.
Runs on BOTH VGSD val and Trans10K test. No retraining, no model-code edits.

Outputs -> results/diagnostics/brf_intervention.json ; results/figures/brf_int_*.png
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
from data.trans10k_dataset import Trans10KDataset
from eval.metrics import TransparentObjectMetrics

SEED = 42
IMG = 448   # spectra_best.pth native
OUT = ROOT / "results" / "diagnostics"; FIG = ROOT / "results" / "figures"
CONDS = ["REAL", "SHUFFLED", "CONSTANT", "RANDOM_MATCHED"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def brf_variants(brf_ref, h, w, gen):
    """brf_ref: (B,1,H,W) in [0,1]. Build variants at patch res (h,w), nearest-up to full."""
    B, _, H, W = brf_ref.shape
    b_patch = F.interpolate(brf_ref, size=(h, w), mode="bilinear", align_corners=False)  # (B,1,h,w)
    out = {"REAL": brf_ref}  # pass real at full res (faithful; fusion downsamples it itself)
    flat = b_patch.view(B, 1, h * w)
    perm = torch.stack([torch.randperm(h * w, generator=gen, device=brf_ref.device) for _ in range(B)])
    shuf = torch.gather(flat, 2, perm.unsqueeze(1)).view(B, 1, h, w)
    out["SHUFFLED"] = F.interpolate(shuf, size=(H, W), mode="nearest")
    mean = b_patch.mean(dim=[2, 3], keepdim=True)
    out["CONSTANT"] = mean.expand(B, 1, H, W).contiguous()
    std = b_patch.std(dim=[2, 3], keepdim=True)
    rnd = (torch.randn(b_patch.shape, generator=gen, device=brf_ref.device) * std + mean).clamp(0, 1)
    out["RANDOM_MATCHED"] = F.interpolate(rnd, size=(H, W), mode="nearest")
    return out, b_patch


def boundary_band(mask_np, width):
    k = np.ones((width, width), np.uint8)
    return ((cv2.dilate(mask_np, k) - cv2.erode(mask_np, k)) > 0)


def run_dataset(name, ds, model, device, gen, H, W, hpw, n_vis=8):
    vl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    h, w = hpw
    mets = {c: TransparentObjectMetrics() for c in CONDS}
    fidelity = TransparentObjectMetrics()
    diff_acc = {c: {"whole": [0.0, 0], "gt": [0.0, 0], "bnd": [0.0, 0]} for c in CONDS if c != "REAL"}
    bstat = {c: [] for c in CONDS}
    per_image = {c: [] for c in CONDS}
    real_pred_mean = [0.0, 0]; vis = 0; idx = 0
    for b in vl:
        b = to_device(b, device)
        img, img1, mask = b["image"], b["image_t1"], b["mask"]
        Bn = img.shape[0]
        with torch.no_grad():
            bo = model.backbone(img); patch = bo["patch_tokens"]; fpn = bo["features"]
            flow_fwd, residual, consistency = model._run_flow(img, img1)
            ofcv_map, _ = model.ofcv(patch_tokens=patch, residual_map=residual,
                                     consistency_map=consistency)  # held FIXED across BRF conds
            brf_raw, brf_ref = model.brf(img)
            variants, b_patch_real = brf_variants(brf_ref, h, w, gen)
            preds = {}
            for c in CONDS:
                fo = model.fusion(patch_tokens=patch, fpn_features=fpn,
                                  ofcv_map=ofcv_map, brf_map=variants[c])
                p = fo["seg_prob"]
                if p.shape[-2:] != (H, W):
                    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
                preds[c] = p; mets[c].update(p, mask)
                bp = F.interpolate(variants[c], size=(h, w), mode="bilinear", align_corners=False)
                bstat[c].append((float(bp.mean()), float(bp.std()), float(bp.min()), float(bp.max())))
            fo_full = model(img, img1, return_intermediates=False)
            fidelity.update(fo_full["seg_prob"], mask)
        for i in range(Bn):
            gtb = mask[i].bool()
            bnp = mask[i].detach().cpu().numpy().astype(np.uint8)
            band = torch.from_numpy(boundary_band(bnp, max(3, IMG // 90))).to(device)
            pr = preds["REAL"][i:i+1]
            real_pred_mean[0] += float(pr.mean()); real_pred_mean[1] += 1
            for c in CONDS:
                inter = (((preds[c][i] > 0.5).squeeze(0)) & gtb).sum().float()
                union = ((((preds[c][i] > 0.5).squeeze(0)) | gtb).sum().float()).clamp(min=1)
                per_image[c].append(float(inter / union))
                if c == "REAL":
                    continue
                d = (preds[c][i:i+1] - pr).abs()
                diff_acc[c]["whole"][0] += float(d.mean()); diff_acc[c]["whole"][1] += 1
                if gtb.any(): diff_acc[c]["gt"][0] += float(d.squeeze()[gtb].mean()); diff_acc[c]["gt"][1] += 1
                if band.any(): diff_acc[c]["bnd"][0] += float(d.squeeze()[band].mean()); diff_acc[c]["bnd"][1] += 1
            if vis < n_vis:
                save_vis(FIG / f"brf_int_{name}_{idx+i:03d}.png", img[i], variants, preds, mask[i], H, W)
                vis += 1
        idx += Bn
        if idx % 500 < Bn:
            print(f"  [{name}] {idx}/{len(ds)}", flush=True)
    rpm = real_pred_mean[0] / max(real_pred_mean[1], 1)
    res = {"n": len(ds), "fidelity_iou": round(fidelity.compute()["iou"], 4)}
    for c in CONDS:
        m = mets[c].compute(); ga = np.array(bstat[c])
        res[c] = {"iou": round(m["iou"], 4), "f_measure": round(m["f_measure"], 4),
                  "mae": round(m["mae"], 4), "ber": round(m["ber"], 4),
                  "per_image_iou_mean": round(float(np.mean(per_image[c])), 4),
                  "brf_patch_mean": round(float(ga[:, 0].mean()), 4),
                  "brf_patch_std": round(float(ga[:, 1].mean()), 4),
                  "brf_patch_min": round(float(ga[:, 2].mean()), 4),
                  "brf_patch_max": round(float(ga[:, 3].mean()), 4)}
        if c != "REAL":
            w_ = diff_acc[c]["whole"]; g_ = diff_acc[c]["gt"]; bd = diff_acc[c]["bnd"]
            wa = w_[0]/max(w_[1],1)
            res[c]["pred_absdiff_whole"] = round(wa, 6)
            res[c]["pred_absdiff_inGT"] = round(g_[0]/max(g_[1],1), 6)
            res[c]["pred_absdiff_boundary"] = round(bd[0]/max(bd[1],1), 6)
            res[c]["pred_reldiff_pct"] = round(100.0 * wa / max(rpm, 1e-8), 4)
    return res, per_image


def _n01(a):
    a = a.astype(np.float32); lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def save_vis(path, img, variants, preds, mask, H, W):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mean = np.array([0.485,0.456,0.406],np.float32); std = np.array([0.229,0.224,0.225],np.float32)
    rgb = (img.detach().cpu().permute(1,2,0).numpy()*std+mean).clip(0,1)
    up = lambda m: F.interpolate(m, size=(H,W), mode="nearest")[0,0].detach().cpu().numpy()
    cols = [("RGB", rgb, None),
            ("REAL BRF", up(variants["REAL"]), "magma"),
            ("SHUFFLED BRF", up(variants["SHUFFLED"]), "magma"),
            ("CONSTANT BRF", up(variants["CONSTANT"]), "magma"),
            ("REAL pred", preds["REAL"][0,0].detach().cpu().numpy(), "magma"),
            ("SHUFFLED pred", preds["SHUFFLED"][0,0].detach().cpu().numpy(), "magma"),
            ("CONSTANT pred", preds["CONSTANT"][0,0].detach().cpu().numpy(), "magma"),
            ("GT", mask.detach().cpu().numpy(), "gray")]
    fig, ax = plt.subplots(1, 8, figsize=(26, 3.6))
    for a,(t,im,cm) in zip(ax, cols):
        a.imshow(im, cmap=cm, vmin=0, vmax=1 if cm else None); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=85); plt.close(fig)


def main():
    set_seed(SEED)
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu"); gen.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml"); cfg.data.image_size = [IMG, IMG]; H = W = IMG
    model = SPECTRA(cfg, use_gnn=False, use_ofcv=True, use_brf=True).to(device).eval()
    ck = torch.load(ROOT / "checkpoints" / "spectra_best.pth", map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[brf-int] loaded spectra_best.pth: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    h = w = IMG // 14  # patch grid (32)

    datasets = {}
    _, va, _ = build_splits("", str(ROOT / "VSGD_dataset" / "test"))
    datasets["VGSD"] = VGSDPairs(va, [IMG, IMG], False)
    datasets["Trans10K"] = Trans10KDataset(root="./datasets", split="test", image_size=(IMG, IMG))

    out = {"checkpoint": "spectra_best.pth (Trans10K, ofcv+brf)", "image_size": [IMG, IMG],
           "seed": SEED, "note": "OFCV held fixed; only BRF map varied. BRF trained on Trans10K."}
    per_img_all = {}
    for name, ds in datasets.items():
        print(f"[brf-int] === {name} (n={len(ds)}) ===", flush=True)
        res, per_img = run_dataset(name, ds, model, device, gen, H, W, (h, w))
        out[name] = res; per_img_all[name] = per_img
        print(f"  [{name}] fidelity IoU={res['fidelity_iou']} vs REAL IoU={res['REAL']['iou']}", flush=True)
        for c in CONDS:
            s = res[c]; extra = "" if c == "REAL" else (f" |dpred|whole={s['pred_absdiff_whole']:.5f}"
                       f"({s['pred_reldiff_pct']:.3f}%) bnd={s['pred_absdiff_boundary']:.5f}")
            print(f"    {c:14s} IoU={s['iou']:.4f} F={s['f_measure']:.4f} MAE={s['mae']:.4f} "
                  f"BER={s['ber']:.4f} brf_patch[{s['brf_patch_min']:.3f},{s['brf_patch_max']:.3f}]{extra}", flush=True)
    json.dump({"summary": out, "per_image_iou": per_img_all},
              open(OUT / "brf_intervention.json", "w"), indent=2)
    print("\n[done] ->", OUT / "brf_intervention.json")


if __name__ == "__main__":
    main()
