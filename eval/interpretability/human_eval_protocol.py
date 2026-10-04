"""
spectra/eval/interpretability/human_eval_protocol.py

Generate a human evaluation package: a folder of image+overlay+saliency triplets
plus a CSV form for annotators to score "Does this saliency map highlight
physically meaningful transparent regions?"

Output (default: results/human_eval/):
    images/<id>.png        side-by-side: input | prediction | OFCV | Grad-CAM
    forms/protocol.md      detailed instructions
    forms/responses.csv    pre-filled empty CSV ready for the annotators
    forms/IAA_protocol.md  inter-annotator agreement procedure

The scoring task:
    For each image rate:
      Q1. Does the model's prediction match the transparent regions? (1-5)
      Q2. Does the OFCV map highlight the physically correct regions? (1-5)
      Q3. Does the Grad-CAM saliency overlap with transparent regions? (1-5)
      Q4. Are the highlighted boundaries consistent with refractive edges? (1-5)

We deliberately decouple Q1 (correctness) from Q2-Q4 (interpretability) so
poor segmentation does not contaminate the interpretability score.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_config, to_device, load_checkpoint
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_eval_dataset


PROTOCOL_MD = """# SPECTRA human-evaluation protocol

## Task
Rate each image triplet on a 1-5 Likert scale.

You will see three figures side-by-side:
  1. Input image
  2. SPECTRA's predicted transparent mask overlaid in orange
  3. OFCV physics map (heat-coloured)
  4. Grad-CAM saliency map

## Questions
1. **Q1 — Segmentation quality**:
   "The orange overlay closely matches the actual transparent objects."
2. **Q2 — OFCV physical fidelity**:
   "The OFCV map highlights regions where light should refract/distort
    (i.e. the transparent surface boundaries and bulk)."
3. **Q3 — Saliency localisation**:
   "The Grad-CAM saliency highlights the transparent regions, not background."
4. **Q4 — Boundary consistency**:
   "Boundaries highlighted by OFCV correspond to refractive edges
    (rim of glass, water surface, etc.) — not arbitrary edges."

Scale (for each):
  1  Strongly disagree
  2  Disagree
  3  Neutral
  4  Agree
  5  Strongly agree

## Sample sizes
- N = 100 random images from Trans10K test
- 3 independent annotators

## Inter-annotator agreement (IAA)
We compute Krippendorff's α and Spearman ρ between annotators.
A reliable result requires α ≥ 0.6.

## Hostile-reviewer test
We deliberately include 5 *failure* cases (chosen by IoU bottom decile) and
5 *easy* cases (top decile) as sanity checks. Annotators should rate the
easy cases ≥ 4 and the hard cases noticeably lower; if not, they should
re-read the protocol.

## Output
Fill in `forms/responses.csv` (one row per image per annotator).
"""

IAA_MD = """# Inter-annotator agreement procedure

After all annotators submit responses.csv files:

```
python eval/interpretability/iaa.py \\
    --responses ann1.csv ann2.csv ann3.csv \\
    --output results/human_eval/iaa.json
```

This computes:
  - Krippendorff's α (ordinal)
  - mean pairwise Spearman ρ
  - per-question agreement

Report all three numbers in the paper's interpretability section.
"""


def make_panel(img_rgb, pred, ofcv, cam, out_path):
    H, W = img_rgb.shape[:2]
    panels = []

    panels.append(img_rgb)

    overlay = img_rgb.copy()
    p_bin = (pred > 0.5).astype(np.uint8)
    overlay[p_bin > 0] = (overlay[p_bin > 0] * 0.5 +
                          np.array([255, 140, 0], np.uint8) * 0.5).astype(np.uint8)
    panels.append(overlay)

    panels.append((cv2.applyColorMap((ofcv * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)))
    panels.append((cv2.applyColorMap((cam  * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)))

    out = np.concatenate(panels, axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def build_package(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = cfg.data.image_size

    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    load_checkpoint(args.checkpoint, model, device=str(device))

    ds = build_eval_dataset(name=args.dataset, root=cfg.data.root, split=args.split,
                            image_size=(H, W))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n_images, len(ds)), replace=False)

    out_root = Path(args.output)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "forms").mkdir(parents=True, exist_ok=True)

    # GradCAM
    from eval.interpretability.saliency_overlap import GradCAM
    gradcam = GradCAM(model)

    rows = []
    try:
        for rank, i in enumerate(tqdm(idxs, desc="human-eval-pack")):
            sample = ds[int(i)]
            img = sample["image"].unsqueeze(0).to(device)
            t1  = sample["image_t1"].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img, t1, return_intermediates=True)
            cam = gradcam(img, t1)

            prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear",
                                 align_corners=False).squeeze().detach().cpu().numpy()
            ofcv = out["ofcv_map"]
            if ofcv is not None:
                ofcv = F.interpolate(ofcv, size=(H, W), mode="bilinear",
                                     align_corners=False).squeeze().detach().cpu().numpy()
            else:
                ofcv = np.zeros((H, W), np.float32)
            cam_np = cam.detach().squeeze().cpu().numpy()

            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_rgb = (sample["image"].numpy() * std + mean) * 255
            img_rgb = img_rgb.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

            img_id = f"img_{rank:04d}"
            make_panel(img_rgb, prob, ofcv, cam_np,
                       out_root / "images" / f"{img_id}.png")
            rows.append({"image_id": img_id,
                         "source_path": sample["image_path"]})
    finally:
        gradcam.close()

    # Write CSV form
    csv_path = out_root / "forms" / "responses.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "annotator", "image_id", "Q1_seg", "Q2_ofcv", "Q3_cam", "Q4_bnd",
        ])
        for r in rows:
            w.writerow(["", r["image_id"], "", "", "", ""])

    with open(out_root / "forms" / "protocol.md", "w") as f:
        f.write(PROTOCOL_MD)
    with open(out_root / "forms" / "IAA_protocol.md", "w") as f:
        f.write(IAA_MD)
    with open(out_root / "manifest.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} images + form to {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    p.add_argument("--dataset",    default="trans10k")
    p.add_argument("--split",      default="test")
    p.add_argument("--n-images",   type=int, default=100)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default="results/human_eval")
    args = p.parse_args()
    build_package(args)
