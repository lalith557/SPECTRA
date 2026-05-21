"""
Generate the hero before/after pair shown on the landing page.
Runs SPECTRA on a chosen input and saves:
  web/public/hero/before.jpg   (resized input)
  web/public/hero/after.jpg    (segmentation overlay)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils import load_config
from models.spectra_model import SPECTRA

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUT = Path("web/public/hero"); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("demo/examples/example_glass_bottle.jpg")


@torch.no_grad()
def main():
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    ckpt = torch.load("weights/spectra_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    img_pil = Image.open(SRC).convert("RGB")
    img_rgb = np.array(img_pil)
    H_orig, W_orig = img_rgb.shape[:2]

    H, W = cfg.data.image_size
    resized = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    out = model(tensor, tensor, return_intermediates=False)
    prob = F.interpolate(out["seg_prob"], size=(H_orig, W_orig), mode="bilinear", align_corners=False
        ).squeeze().cpu().float().numpy()

    # Crop hero to 16:9 around object centre
    target_ratio = 16 / 9
    if W_orig / H_orig > target_ratio:
        new_w = int(H_orig * target_ratio); offset = (W_orig - new_w) // 2
        img_rgb = img_rgb[:, offset:offset + new_w]
        prob = prob[:, offset:offset + new_w]
    else:
        new_h = int(W_orig / target_ratio); offset = (H_orig - new_h) // 2
        img_rgb = img_rgb[offset:offset + new_h, :]
        prob = prob[offset:offset + new_h, :]
    img_rgb = cv2.resize(img_rgb, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
    prob    = cv2.resize(prob,    (1280, 720), interpolation=cv2.INTER_LINEAR)

    # Save BEFORE (untouched input)
    cv2.imwrite(str(OUT / "before.jpg"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Build AFTER (overlay + outline + soft glow)
    mask = (prob > 0.5).astype(np.uint8)
    overlay = img_rgb.copy().astype(np.float32)
    color = np.array([90, 110, 235], dtype=np.float32)  # SPECTRA indigo (in RGB)
    overlay = np.where(mask[..., None].astype(bool),
                       overlay * 0.45 + color * 0.55, overlay)

    # Edge outline
    edges = cv2.Canny((prob * 255).astype(np.uint8), 80, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    overlay[edges > 0] = [255, 255, 255]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "after.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Save a smaller poster too for the OG image
    poster = cv2.resize(overlay, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "poster.jpg"),
                cv2.cvtColor(poster, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 86])

    print(f"saved: {OUT}/before.jpg, after.jpg, poster.jpg")


if __name__ == "__main__":
    main()
