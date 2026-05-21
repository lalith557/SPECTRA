"""
spectra/eval/robustness_eval.py
Lighting and corruption robustness evaluation.

Evaluates SPECTRA under systematically degraded conditions:
  1. Brightness shift      — over/underexposed frames
  2. Low light             — simulate nighttime / dim indoor
  3. Strong sunlight glare — blown-out highlights on glass
  4. Motion blur           — fast camera pan
  5. JPEG compression      — streaming/webcam artefacts
  6. Gaussian noise        — sensor noise in low light
  7. Fog/haze              — outdoor conditions
  8. Colour jitter         — white balance shift

For each condition × severity level, computes IoU, F-measure, MAE.
Generates a robustness curve figure for the paper appendix.

Why this matters:
  - "Only works in clean lab conditions" is the #1 criticism of transparent
    object detection papers. This script disproves it systematically.
  - Shows which conditions SPECTRA handles well (physics signals are
    lighting-invariant) vs which degrade it (motion blur kills OFCV)

Usage:
    python eval/robustness_eval.py --config configs/config.yaml \
        --checkpoint checkpoints/spectra_best.pth
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Callable, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config, get_logger, to_device
from eval.metrics import TransparentObjectMetrics

logger = get_logger("spectra.robustness")


# ---------------------------------------------------------------------------
# Corruption functions (applied to images at test time, no re-training)
# ---------------------------------------------------------------------------

def corrupt_brightness(img: np.ndarray, severity: int) -> np.ndarray:
    """Increase brightness (overexposure). severity 1-5."""
    factor = 1.0 + severity * 0.3
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def corrupt_low_light(img: np.ndarray, severity: int) -> np.ndarray:
    """Reduce brightness to simulate low-light / nighttime. severity 1-5."""
    factor = 1.0 - severity * 0.15
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def corrupt_glare(img: np.ndarray, severity: int) -> np.ndarray:
    """
    Add sunlight glare patches — saturated white regions.
    Simulates direct sunlight on glass surfaces.
    """
    out = img.copy().astype(np.float32)
    H, W = img.shape[:2]
    n_patches = severity * 2
    for _ in range(n_patches):
        cx, cy = np.random.randint(W // 4, 3 * W // 4), np.random.randint(H // 4, 3 * H // 4)
        rx, ry = np.random.randint(W // 20, W // (8 - severity)), np.random.randint(H // 20, H // (8 - severity))
        y, x   = np.ogrid[:H, :W]
        mask   = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) <= 1
        strength = 0.5 + severity * 0.08
        out[mask] = out[mask] * (1 - strength) + 255 * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Horizontal motion blur — simulates fast camera pan. severity 1-5."""
    ksize  = 5 + severity * 6
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel)


def corrupt_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    """JPEG compression artefacts. severity 1-5 → quality 80→20."""
    quality = 80 - severity * 12
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(5, quality)]
    _, buf   = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def corrupt_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Gaussian sensor noise. severity 1-5."""
    sigma = severity * 12.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def corrupt_fog(img: np.ndarray, severity: int) -> np.ndarray:
    """Fog/haze overlay — reduces contrast and adds white veil."""
    alpha = 0.1 + severity * 0.12
    fog   = np.full_like(img, 220, dtype=np.float32)
    return np.clip(img.astype(np.float32) * (1 - alpha) + fog * alpha, 0, 255).astype(np.uint8)


def corrupt_colour_jitter(img: np.ndarray, severity: int) -> np.ndarray:
    """White balance / colour temperature shift."""
    out = img.astype(np.float32)
    shifts = np.array([1.0 + severity * 0.08, 1.0, 1.0 - severity * 0.06])
    np.random.shuffle(shifts)
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] * shifts[c], 0, 255)
    return out.astype(np.uint8)


CORRUPTIONS: Dict[str, Callable] = {
    "brightness":    corrupt_brightness,
    "low_light":     corrupt_low_light,
    "glare":         corrupt_glare,
    "motion_blur":   corrupt_motion_blur,
    "jpeg_compress": corrupt_jpeg,
    "gaussian_noise": corrupt_noise,
    "fog":           corrupt_fog,
    "colour_jitter": corrupt_colour_jitter,
}

SEVERITIES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Robustness evaluation runner
# ---------------------------------------------------------------------------

def evaluate_under_corruption(
    model,
    dataset,
    corruption_fn:   Callable,
    severity:        int,
    device:          torch.device,
    cfg,
    n_samples:       int = 200,
) -> Dict:
    """
    Run inference with a specific corruption applied to all test images.

    Args:
        corruption_fn: function(img_np, severity) → corrupted img_np
        severity:      1 (mild) to 5 (severe)
        n_samples:     number of test images to evaluate (subset for speed)

    Returns:
        metrics dict with iou, f_measure, mae, ber
    """
    from data.augmentation import get_val_transforms
    from torch.utils.data import DataLoader

    metrics = TransparentObjectMetrics()
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model.eval()
    count = 0
    MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    STD   = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    H, W  = cfg.data.image_size

    for batch in loader:
        if count >= n_samples:
            break
        batch = to_device(batch, device)

        # Denorm image → apply corruption → re-norm
        img_t = batch["image"]                            # (1, 3, H, W)
        img_np = ((img_t * STD + MEAN) * 255).squeeze(0).permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        img_corrupted = corruption_fn(img_np, severity)

        # Re-normalise
        img_norm  = (img_corrupted.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(img_tensor, img_tensor)

        metrics.update(out["seg_prob"], batch["mask"])
        count += 1

    return metrics.compute()


def run_robustness_eval(
    config_path:     str,
    checkpoint_path: str,
    output_dir:      str = "outputs/robustness",
    n_samples:       int = 200,
    severities:      List[int] = SEVERITIES,
):
    from models.spectra_model import SPECTRA
    from data.trans10k_dataset import Trans10KDataset

    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W   = cfg.data.image_size
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model = SPECTRA(cfg, use_gnn=False).to(device).eval()
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Clean baseline
    dataset = Trans10KDataset(root=cfg.data.root, split="test", image_size=(H, W))
    logger.info(f"Evaluating on {min(n_samples, len(dataset))} test images")

    all_results: Dict[str, Dict] = {"clean": {}}

    # Clean baseline first
    logger.info("\n--- Clean baseline ---")
    clean_metrics = evaluate_under_corruption(
        model, dataset, lambda img, s: img, 1, device, cfg, n_samples
    )
    all_results["clean"] = clean_metrics
    logger.info(f"Clean IoU: {clean_metrics['iou']:.4f} | F: {clean_metrics['f_measure']:.4f}")

    # Each corruption × severity
    for corruption_name, fn in CORRUPTIONS.items():
        all_results[corruption_name] = {}
        logger.info(f"\n--- Corruption: {corruption_name} ---")

        for sev in severities:
            result = evaluate_under_corruption(
                model, dataset, fn, sev, device, cfg, n_samples
            )
            all_results[corruption_name][str(sev)] = result
            logger.info(
                f"  sev={sev} | IoU={result['iou']:.4f} | "
                f"F={result['f_measure']:.4f} | MAE={result['mae']:.4f}"
            )

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = Path(output_dir) / "robustness_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {out_json}")

    # ── Robustness curve figure ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    clean_iou = all_results["clean"]["iou"]

    colors = plt.cm.tab10(np.linspace(0, 1, len(CORRUPTIONS)))

    for i, (corruption_name, col) in enumerate(zip(CORRUPTIONS, colors)):
        ax   = axes[i]
        ious = [all_results[corruption_name][str(s)]["iou"] for s in severities]

        ax.plot(severities, ious, "o-", color=col, linewidth=2, markersize=6)
        ax.axhline(clean_iou, color="#888780", linestyle="--", linewidth=1,
                   label=f"Clean ({clean_iou:.3f})")
        ax.set_title(corruption_name.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Severity", fontsize=9)
        ax.set_ylabel("IoU", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(severities)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("SPECTRA Robustness Evaluation — IoU vs Corruption Type & Severity",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = Path(output_dir) / "robustness_curves.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Robustness figure saved: {fig_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"{'Corruption':<20} | {'sev1':>6} | {'sev3':>6} | {'sev5':>6} | {'drop':>6}")
    logger.info("-"*60)
    for name in CORRUPTIONS:
        r = all_results[name]
        iou1 = r.get("1", {}).get("iou", 0)
        iou3 = r.get("3", {}).get("iou", 0)
        iou5 = r.get("5", {}).get("iou", 0)
        drop = clean_iou - iou5
        logger.info(f"{name:<20} | {iou1:>6.3f} | {iou3:>6.3f} | {iou5:>6.3f} | {drop:>+6.3f}")
    logger.info("="*60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    parser.add_argument("--output",     default="outputs/robustness")
    parser.add_argument("--n-samples",  type=int, default=200)
    args = parser.parse_args()
    run_robustness_eval(args.config, args.checkpoint, args.output, args.n_samples)
