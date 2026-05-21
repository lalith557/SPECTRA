"""
spectra/eval/visual_quality_eval.py
Runs inference on validation subset and categorises visual results
(thin objects, complex boundaries, failure cases) for W&B logging.
"""
import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.spectra_model import SPECTRA
from data.trans10k_dataset import build_dataloaders
from eval.metrics import calculate_iou
from utils import load_config, to_device

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def calculate_iou_numpy(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    if np.sum(union) == 0:
        return 1.0
    return np.sum(intersection) / np.sum(union)


def categorise_sample(mask, iou):
    """
    Categorise based on GT mask stats and IoU.
    mask: numpy array (H, W) binary {0, 1}
    """
    if iou < 0.5:
        return "Failure Case"
        
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "Background / No Object"
        
    area = np.sum(mask)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    
    if area == 0:
        return "Empty"
        
    complexity = perimeter / np.sqrt(area)
    
    if complexity > 6.0:
        return "Thin/Complex Boundaries"
    else:
        return "Standard"


@torch.no_grad()
def visual_quality_analysis(cfg_path: str, ckpt_baseline: str, ckpt_brf: str, ckpt_full: str, num_samples: int = 50):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if WANDB_AVAILABLE and cfg.logging.get("wandb_project"):
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            name=f"visual_quality_{cfg.experiment}",
            config=cfg.__dict__,
            job_type="eval"
        )

    _, val_loader = build_dataloaders(cfg)

    # Load all three models
    def load_model(ckpt_path, **kwargs):
        m = SPECTRA(cfg, **kwargs).to(device)
        print(f"Loading checkpoint {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}. Using untrained weights.")
        m.eval()
        return m

    model_baseline = load_model(ckpt_baseline, use_ofcv=False, use_brf=False, use_gnn=False)
    model_brf = load_model(ckpt_brf, use_ofcv=False, use_brf=True, use_gnn=False)
    model_full = load_model(ckpt_full, use_ofcv=True, use_brf=True, use_gnn=True)

    save_dir = Path("results/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    panels = []
    
    dataset_size = len(val_loader.dataset)
    indices_to_eval = set(random.sample(range(dataset_size), min(num_samples, dataset_size)))
    
    sample_idx = 0
    for batch in val_loader:
        if sample_idx not in indices_to_eval:
            sample_idx += batch["image"].size(0)
            continue
            
        batch = to_device(batch, device)
        
        with torch.amp.autocast("cuda"):
            out_base = model_baseline(image=batch["image"], image_t1=batch["image_t1"])
            out_brf = model_brf(image=batch["image"], image_t1=batch["image_t1"])
            out_full = model_full(image=batch["image"], image_t1=batch["image_t1"])
            
        probs_base = out_base["seg_prob"].squeeze(1).cpu().numpy()
        probs_brf = out_brf["seg_prob"].squeeze(1).cpu().numpy()
        probs_full = out_full["seg_prob"].squeeze(1).cpu().numpy()
        masks = batch["mask"].cpu().numpy()
        images = batch["image"].cpu().numpy()
        
        for b in range(images.shape[0]):
            if sample_idx not in indices_to_eval:
                sample_idx += 1
                continue
                
            img = images[b].transpose(1, 2, 0)
            # Unnormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            mask = masks[b]
            prob_base = probs_base[b]
            prob_brf = probs_brf[b]
            prob_full = probs_full[b]
            
            pred = (prob_full > 0.5).astype(np.uint8)
            iou = calculate_iou_numpy(pred, mask)
            category = categorise_sample(mask, iou)
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(f"Category: {category} | Full IoU: {iou:.4f}")
            
            axes[0].imshow(img)
            axes[0].set_title("Input")
            axes[0].axis("off")
            
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("GT Mask")
            axes[1].axis("off")
            
            axes[2].imshow(prob_base, cmap="jet")
            axes[2].set_title("Baseline")
            axes[2].axis("off")
            
            axes[3].imshow(prob_brf, cmap="jet")
            axes[3].set_title("+BRF")
            axes[3].axis("off")
            
            axes[4].imshow(prob_full, cmap="jet")
            axes[4].set_title("Full SPECTRA")
            axes[4].axis("off")
            
            plt.tight_layout()
            
            save_path = save_dir / f"sample_{sample_idx}_{category.replace(' ', '_').replace('/', '_')}.png"
            plt.savefig(save_path)
            
            if WANDB_AVAILABLE and wandb.run:
                panels.append(wandb.Image(str(save_path), caption=f"{category} | IoU: {iou:.4f}"))
                
            plt.close(fig)
            sample_idx += 1
            
            if len(panels) >= num_samples:
                break
        if len(panels) >= num_samples:
            break

    if WANDB_AVAILABLE and wandb.run:
        wandb.log({"Visual Quality Panels": panels})
        wandb.finish()
        
    print(f"Generated {len(panels)} visual quality panels in {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt_baseline", default="results/checkpoints/checkpoint_ablation_dinov2.pth")
    parser.add_argument("--ckpt_brf", default="results/checkpoints/checkpoint_ablation_brf.pth")
    parser.add_argument("--ckpt_full", default="results/checkpoints/checkpoint_ablation_full.pth")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    
    visual_quality_analysis(args.config, args.ckpt_baseline, args.ckpt_brf, args.ckpt_full, args.num_samples)
