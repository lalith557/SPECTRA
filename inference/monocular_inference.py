"""
spectra/inference/monocular_inference.py

Command-line monocular inference for SPECTRA-Mono. Takes a single image and
outputs a transparency mask without needing a second frame.

Example:
    python inference/monocular_inference.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/spectra_mono_best.pth \
        --image my_photo.jpg \
        --output prediction.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import load_config, load_checkpoint
from models.spectra_mono import SPECTRA_Mono


def preprocess(img: np.ndarray, size):
    H, W = size
    img = cv2.resize(img, (W, H))
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = (img / 255.0 - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image",      required=True)
    p.add_argument("--output",     default="prediction.png")
    p.add_argument("--warp-mode",  default="learned",
                   choices=["learned", "homography", "flow_warp"])
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = cfg.data.image_size
    x = preprocess(img_rgb, (H, W)).to(device)

    model = SPECTRA_Mono(cfg, warp_mode=args.warp_mode, use_gnn=False).to(device).eval()
    if Path(args.checkpoint).exists():
        load_checkpoint(args.checkpoint, model, device=str(device))

    with torch.no_grad():
        out = model(x, image_t1=None)
        prob = F.interpolate(out["seg_prob"], size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().cpu().numpy()

    pred = (prob * 255).astype(np.uint8)
    cv2.imwrite(args.output, pred)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
