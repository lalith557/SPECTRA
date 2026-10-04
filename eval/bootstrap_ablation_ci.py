"""
eval/bootstrap_ablation_ci.py

Paired bootstrap confidence intervals for the central ablation claim, reusing the
already-trained 30-epoch variant checkpoints (NO retraining).

For each physics-equipped variant we test, on the SAME validation images, the
per-image IoU difference vs the no-physics "neither" baseline:
  - observed mean difference
  - 95% bootstrap CI (10,000 resamples)
  - two-sided bootstrap p-value
  - TOST equivalence verdict against a +/-0.01 IoU margin

Writes per-image JSONs + ci_summary.json into the consolidated results folder.

Usage:
  venv/Scripts/python.exe eval/bootstrap_ablation_ci.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_config, to_device, load_checkpoint           # noqa: E402
from models.spectra_model import SPECTRA                            # noqa: E402
from data.trans10k_dataset import build_eval_dataset                # noqa: E402
from eval.metrics import compute_iou                                # noqa: E402

# Non-GNN variants isolate the OFCV/BRF physics effect cleanly.
VARIANTS = [
    {"name": "neither",   "use_ofcv": False, "use_brf": False, "use_gnn": False},
    {"name": "ofcv_only", "use_ofcv": True,  "use_brf": False, "use_gnn": False},
    {"name": "brf_only",  "use_ofcv": False, "use_brf": True,  "use_gnn": False},
    {"name": "ofcv_brf",  "use_ofcv": True,  "use_brf": True,  "use_gnn": False},
]
CKPT = "results/full_ablations/{name}/best.pth"
OUT = ROOT / "EXPERIMENTS" / "05_bootstrap_ci"
SPLIT = "validation"
MARGIN = 0.01          # TOST equivalence margin on IoU
N_BOOT = 10000
SEED = 42


@torch.no_grad()
def per_image_iou(variant, cfg, device):
    model = SPECTRA(cfg, use_gnn=variant["use_gnn"],
                    use_ofcv=variant["use_ofcv"],
                    use_brf=variant["use_brf"]).to(device).eval()
    load_checkpoint(CKPT.format(name=variant["name"]), model, device=str(device))
    ds = build_eval_dataset(name="trans10k", root=cfg.data.root, split=SPLIT,
                            image_size=tuple(cfg.data.image_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    H, W = cfg.data.image_size
    ious = []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["image"], batch["image_t1"])
        prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().clamp(0, 1)
        gt = batch["mask"].squeeze().float()
        ious.append(compute_iou((prob > 0.5).long(), gt.long()))
    del model
    torch.cuda.empty_cache()
    return np.array(ious, dtype=np.float64)


def paired_bootstrap(diff, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(diff)
    means = np.empty(n)
    for i in range(n):
        means[i] = diff[rng.integers(0, m, m)].mean()
    obs = float(diff.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p-value (does the resample distribution cross 0?)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    # TOST: equivalent if the 95% CI lies entirely within (-margin, +margin)
    equiv = bool(lo > -MARGIN and hi < MARGIN)
    return obs, float(lo), float(hi), float(min(p, 1.0)), equiv


def main():
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}  split={SPLIT}")

    iou = {}
    for v in VARIANTS:
        print(f"[eval] {v['name']} ...", flush=True)
        arr = per_image_iou(v, cfg, device)
        iou[v["name"]] = arr
        json.dump(arr.tolist(), open(OUT / f"per_image_{v['name']}.json", "w"))
        print(f"   mean IoU = {arr.mean():.4f}  (n={len(arr)})")

    base = iou["neither"]
    rows = []
    for name in ["ofcv_only", "brf_only", "ofcv_brf"]:
        diff = iou[name] - base
        obs, lo, hi, p, equiv = paired_bootstrap(diff)
        rows.append({
            "comparison": f"{name} - neither", "mean_iou_variant": float(iou[name].mean()),
            "mean_iou_neither": float(base.mean()), "mean_diff": obs,
            "ci95_low": lo, "ci95_high": hi, "bootstrap_p": p,
            "tost_margin": MARGIN, "equivalent_within_margin": equiv,
        })
        print(f"  {name}-neither: dIoU={obs:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"p={p:.3f}  equivalent(+-{MARGIN})={equiv}")

    summary = {"split": SPLIT, "n_images": int(len(base)), "n_bootstrap": N_BOOT,
               "seed": SEED, "tost_margin": MARGIN, "comparisons": rows}
    json.dump(summary, open(OUT / "ci_summary.json", "w"), indent=2)
    print(f"[done] wrote {OUT/'ci_summary.json'}")


if __name__ == "__main__":
    main()
