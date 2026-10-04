"""
eval/diag_brf_boundary.py  — PHASE 5 BRF boundary validation.

Does BRF carry boundary information, or is it mostly noise?
For representative VGSD frames:
  - compute brf_raw (handcrafted double-peak) and brf_refined (learned, from ckpt)
  - extract GT boundary band via morphological gradient (dilate - erode)
  - compare BRF inside boundary band vs outside: mean/std/median/percentiles
  - separation statistic: AUROC (BRF value predicting boundary membership) + Cohen's d
Visualize: input | GT boundary | BRF map | BRF x boundary.
Read-only. No training.
"""
import os, sys, glob, json
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import load_config                                   # noqa
from modules.brf import BoundaryResonanceField                 # noqa

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)
OUT = ROOT / "results" / "diagnostics"; FIG = ROOT / "results" / "figures"
DEV_ROOT = ROOT / "VSGD_dataset" / "test"


def load_frame(path, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0), img


def boundary_band(mask, width=5):
    k = np.ones((width, width), np.uint8)
    d = cv2.dilate(mask, k); e = cv2.erode(mask, k)
    return ((d - e) > 0).astype(np.uint8)


def auroc(scores, labels):
    # rank-based AUROC; labels in {0,1}
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; n_pos = pos.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float("nan")
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((a.mean() - b.mean()) / (sp + 1e-8))


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/config.yaml"); size = int(cfg.data.image_size[0])
    brf = BoundaryResonanceField(n_orientations=cfg.brf.n_orientations,
                                 n_scales=cfg.brf.n_scales,
                                 peak_window=cfg.brf.peak_window).to(device).eval()
    # load trained refine weights from checkpoint (brf.refine.*)
    sd = torch.load(ROOT / "checkpoints" / "spectra_best.pth", map_location="cpu", weights_only=False)  # user's own trusted ckpt
    sd = sd.get("model_state_dict") or sd.get("model") or sd.get("state_dict") or sd
    brf_sd = {k[len("brf."):]: v for k, v in sd.items() if k.startswith("brf.")}
    miss, unexp = brf.load_state_dict(brf_sd, strict=False)
    print(f"[brf] loaded refine: matched={len(brf_sd)} missing={len(miss)} unexpected={len(unexp)}")

    clips = sorted(p for p in DEV_ROOT.iterdir() if p.is_dir())[:40]
    raw_in, raw_out, ref_in, ref_out = [], [], [], []
    per_sample = []; vis = 0
    for ci, clip in enumerate(clips):
        frames = sorted(glob.glob(str(clip / "JPEGImages" / "*.jpg")))
        if not frames: continue
        f0 = frames[len(frames)//2]
        mp = clip / "SegmentationClassPNG" / f"{Path(f0).stem}.png"
        if not mp.exists(): continue
        x, rgb = load_frame(f0, size); x = x.to(device)
        m = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (size, size),
                       interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        band = boundary_band(m, width=max(3, size // 90))
        if band.sum() < 50: continue
        with torch.no_grad():
            braw, bref = brf(x)
        braw = braw[0, 0].cpu().numpy(); bref = bref[0, 0].cpu().numpy()
        bmask = band.astype(bool)
        raw_in.append(braw[bmask]); raw_out.append(braw[~bmask])
        ref_in.append(bref[bmask]); ref_out.append(bref[~bmask])
        # per-image separation
        lab = band.flatten()
        per_sample.append(dict(clip=clip.name,
                               raw_auroc=auroc(braw.flatten(), lab),
                               ref_auroc=auroc(bref.flatten(), lab),
                               raw_in=float(braw[bmask].mean()), raw_out=float(braw[~bmask].mean()),
                               ref_in=float(bref[bmask].mean()), ref_out=float(bref[~bmask].mean())))
        if vis < 5:
            save_vis(FIG / f"phase5_brf_{clip.name}.png", rgb, band, braw, bref)
            vis += 1
        print(f"  [{ci+1}/{len(clips)}] {clip.name}: raw_AUROC={per_sample[-1]['raw_auroc']:.3f} "
              f"ref_AUROC={per_sample[-1]['ref_auroc']:.3f}", flush=True)

    def cat(x): return np.concatenate(x) if x else np.array([0.0])
    RI, RO, FI, FO = cat(raw_in), cat(raw_out), cat(ref_in), cat(ref_out)
    # global AUROC on a subsample (memory)
    def sub(a, n=200000):
        return a if len(a) <= n else a[np.random.RandomState(0).choice(len(a), n, replace=False)]
    raw_scores = np.concatenate([sub(RI), sub(RO)]); raw_lab = np.concatenate([np.ones(len(sub(RI))), np.zeros(len(sub(RO)))])
    ref_scores = np.concatenate([sub(FI), sub(FO)]); ref_lab = np.concatenate([np.ones(len(sub(FI))), np.zeros(len(sub(FO)))])
    summary = dict(
        n=len(per_sample), size=size, checkpoint="spectra_best.pth",
        brf_raw=dict(in_mean=float(RI.mean()), out_mean=float(RO.mean()),
                     in_median=float(np.median(RI)), out_median=float(np.median(RO)),
                     cohen_d=cohen_d(RI, RO), global_auroc=auroc(raw_scores, raw_lab),
                     per_img_auroc_mean=float(np.nanmean([s["raw_auroc"] for s in per_sample]))),
        brf_refined=dict(in_mean=float(FI.mean()), out_mean=float(FO.mean()),
                         in_median=float(np.median(FI)), out_median=float(np.median(FO)),
                         cohen_d=cohen_d(FI, FO), global_auroc=auroc(ref_scores, ref_lab),
                         per_img_auroc_mean=float(np.nanmean([s["ref_auroc"] for s in per_sample]))),
    )
    json.dump({"summary": summary, "per_sample": per_sample},
              open(OUT / "phase5_brf_boundary.json", "w"), indent=2)
    print("\n[done] ->", OUT / "phase5_brf_boundary.json")
    print(f"  brf_raw : AUROC={summary['brf_raw']['global_auroc']:.3f} "
          f"d={summary['brf_raw']['cohen_d']:.3f} in={summary['brf_raw']['in_mean']:.3f} out={summary['brf_raw']['out_mean']:.3f}")
    print(f"  brf_ref : AUROC={summary['brf_refined']['global_auroc']:.3f} "
          f"d={summary['brf_refined']['cohen_d']:.3f} in={summary['brf_refined']['in_mean']:.3f} out={summary['brf_refined']['out_mean']:.3f}")
    print("  (AUROC 0.5 = BRF no better than chance at marking boundaries)")


def save_vis(path, rgb, band, braw, bref):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(rgb); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(band, cmap="gray"); ax[1].set_title("GT boundary band"); ax[1].axis("off")
    ax[2].imshow(braw, cmap="magma"); ax[2].set_title("BRF raw"); ax[2].axis("off")
    ax[3].imshow(bref, cmap="magma"); ax[3].set_title("BRF refined"); ax[3].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
