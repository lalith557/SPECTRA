"""
spectra/eval/visualize.py
Visualization script for SPECTRA predictions.
Produces side-by-side panels:
  Input Image | GT Mask | Predicted Mask | Boundary Map | Material Map
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config
from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized (C,H,W) tensor back to (H,W,3) uint8 image."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def to_heatmap(tensor: torch.Tensor, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a single-channel (1,H,W) tensor to a coloured heatmap (H,W,3)."""
    arr = tensor.squeeze().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(arr, colormap)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def visualize(cfg_path: str, checkpoint: str, n_samples: int = 8, split: str = "val",
              save_dir: str = "visualizations"):
    cfg    = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = SPECTRA(cfg, use_gnn=False).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Build dataloader
    _, val_loader = build_dataloaders(cfg)
    loader = val_loader

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in loader:
        if count >= n_samples:
            break

        # Move to device
        image    = batch["image"].to(device)
        image_t1 = batch["image_t1"].to(device)
        mask_gt  = batch["mask"].cpu().numpy()  # (B, H, W)

        # Forward pass with intermediates
        outputs = model(
            image=image,
            image_t1=image_t1,
            return_intermediates=True,
        )

        B = image.shape[0]
        for b in range(B):
            if count >= n_samples:
                break

            # --- Input image ---
            img_vis = denormalize(image[b])

            # --- GT mask ---
            gt = mask_gt[b]  # (H, W)
            gt_vis = (gt * 255).astype(np.uint8)
            gt_vis = cv2.applyColorMap(gt_vis, cv2.COLORMAP_BONE)
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)

            # --- Predicted segmentation ---
            seg_prob = outputs["seg_prob"][b]  # (1, H, W)
            pred_mask = (seg_prob.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
            pred_vis = (pred_mask * 255).astype(np.uint8)
            pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_BONE)
            pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB)

            # --- Seg probability heatmap ---
            seg_heat = to_heatmap(seg_prob)

            # --- BRF boundary map ---
            brf_map = outputs.get("brf_map")
            if brf_map is not None:
                brf_vis = to_heatmap(brf_map[b], colormap=cv2.COLORMAP_INFERNO)
            else:
                brf_vis = np.zeros_like(img_vis)

            # --- OFCV map ---
            ofcv_map = outputs.get("ofcv_map")
            if ofcv_map is not None:
                ofcv_vis = to_heatmap(ofcv_map[b], colormap=cv2.COLORMAP_VIRIDIS)
            else:
                ofcv_vis = np.zeros_like(img_vis)

            # --- Material map ---
            mat_logits = outputs.get("mat_logits")
            if mat_logits is not None:
                mat_pred = torch.argmax(mat_logits[b], dim=0).cpu().numpy()  # (H, W)
                n_classes = mat_logits.shape[1]
                mat_norm = (mat_pred.astype(np.float32) / max(n_classes - 1, 1) * 255).astype(np.uint8)
                mat_vis = cv2.applyColorMap(mat_norm, cv2.COLORMAP_TURBO)
                mat_vis = cv2.cvtColor(mat_vis, cv2.COLOR_BGR2RGB)
            else:
                mat_vis = np.zeros_like(img_vis)

            # --- Plot 2x3 grid ---
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"SPECTRA Prediction — Sample {count + 1}", fontsize=16, fontweight="bold")

            panels = [
                (axes[0, 0], img_vis,   "Input Image"),
                (axes[0, 1], gt_vis,    "GT Mask"),
                (axes[0, 2], pred_vis,  "Predicted Mask (>0.5)"),
                (axes[1, 0], seg_heat,  "Seg Probability"),
                (axes[1, 1], brf_vis,   "BRF Boundary Map"),
                (axes[1, 2], ofcv_vis,  "OFCV Violation Map"),
            ]

            for ax, vis, title in panels:
                ax.imshow(vis)
                ax.set_title(title, fontsize=12)
                ax.axis("off")

            plt.tight_layout()
            out_path = save_dir / f"pred_{count + 1:03d}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

            count += 1

    print(f"\nDone! {count} visualizations saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SPECTRA predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_epoch001.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument("--save-dir", default="visualizations",
                        help="Directory to write visualization PNGs to")
    args = parser.parse_args()
    visualize(args.config, args.checkpoint, args.n_samples, save_dir=args.save_dir)
