"""
spectra/inference/vis_utils.py
Visualisation utilities for SPECTRA.
Produces publication-quality figures and real-time overlays.

Outputs:
  - Segmentation mask overlay on RGB
  - Optical flow residual heatmap
  - BRF boundary heatmap
  - OFCV violation map
  - Physics side panel for video
  - Uncertainty / entropy map
  - GradCAM-style attention visualisation
  - HUD overlay for real-time inference
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

CMAP_TRANSPARENT = np.array([
    [0,   150, 255],   # blue — transparent regions
], dtype=np.uint8)

CMAP_HEAT = cv2.COLORMAP_INFERNO    # for residual / BRF maps
CMAP_VIRIDIS = cv2.COLORMAP_VIRIDIS
CMAP_JET = cv2.COLORMAP_JET


# ---------------------------------------------------------------------------
# Core overlay
# ---------------------------------------------------------------------------

def overlay_mask_on_image(
    image_bgr: np.ndarray,   # (H, W, 3) uint8
    mask_bin:  np.ndarray,   # (H, W) uint8 {0, 1}
    prob_map:  Optional[np.ndarray] = None,  # (H, W) float [0, 1]
    alpha:     float = 0.45,
    colour_bgr: Tuple[int, int, int] = (255, 150, 0),  # cyan-ish
) -> np.ndarray:
    """
    Draw transparent object mask over the original image.

    If prob_map provided, uses continuous opacity (more transparent = more overlay).
    Falls back to binary mask otherwise.
    """
    out  = image_bgr.copy().astype(np.float32)
    overlay = np.zeros_like(out)
    overlay[:] = colour_bgr

    if prob_map is not None:
        weight = (prob_map * alpha)[..., np.newaxis]   # continuous blending
    else:
        weight = (mask_bin.astype(np.float32) * alpha)[..., np.newaxis]

    out = out * (1 - weight) + overlay * weight

    # Draw contour at the mask boundary for crisp edges
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out.astype(np.uint8), contours, -1, (0, 255, 150), 1)

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def tensor_to_heatmap(
    t:        torch.Tensor,   # (1, 1, H, W) or (H, W) tensor
    cmap:     int = CMAP_HEAT,
    target_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert a 2D tensor to a BGR heatmap (H, W, 3) uint8."""
    if t is None:
        return None

    if isinstance(t, torch.Tensor):
        arr = t.squeeze().cpu().float().numpy()
    else:
        arr = t.squeeze()

    # Normalise to [0, 255]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = (arr * 255).astype(np.uint8)
    heat      = cv2.applyColorMap(arr_uint8, cmap)

    if target_hw is not None:
        heat = cv2.resize(heat, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)

    return heat   # (H, W, 3) BGR


def colorise_probability_map(
    prob: np.ndarray,   # (H, W) float [0, 1]
) -> np.ndarray:
    """Map probability to a blue→red diverging colourmap."""
    arr_uint8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(arr_uint8, cv2.COLORMAP_COOL)


# ---------------------------------------------------------------------------
# Physics side panel (video mode)
# ---------------------------------------------------------------------------

def make_physics_panel(
    frame_bgr:   np.ndarray,
    prob_map:    Optional[np.ndarray],
    ofcv_map:    Optional[torch.Tensor],
    brf_map:     Optional[torch.Tensor],
    height:      int,
) -> np.ndarray:
    """
    Compose a vertical stack of physics signal heatmaps into a panel
    the same height as the main video frame.

    Panel layout (top→bottom):
      1. Probability map (continuous glass prediction)
      2. OFCV violation map
      3. BRF boundary map
    """
    W = frame_bgr.shape[1] // 2   # panel is half the width of original
    h3 = height // 3

    strips = []

    # 1. Probability
    if prob_map is not None:
        strip = colorise_probability_map(prob_map)
        strip = cv2.resize(strip, (W, h3))
        strip = _add_label(strip, "Seg probability")
    else:
        strip = np.zeros((h3, W, 3), dtype=np.uint8)
    strips.append(strip)

    # 2. OFCV
    if ofcv_map is not None:
        strip = tensor_to_heatmap(ofcv_map, CMAP_HEAT, (h3, W))
        strip = _add_label(strip, "OFCV violation (C1)")
    else:
        strip = np.zeros((h3, W, 3), dtype=np.uint8)
    strips.append(strip)

    # 3. BRF
    if brf_map is not None:
        strip = tensor_to_heatmap(brf_map, CMAP_VIRIDIS, (height - 2 * h3, W))
        strip = _add_label(strip, "BRF boundary (C2)")
    else:
        strip = np.zeros((height - 2 * h3, W, 3), dtype=np.uint8)
    strips.append(strip)

    panel = np.concatenate(strips, axis=0)
    return panel


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a small text label in the top-left corner of an image."""
    out = img.copy()
    cv2.putText(
        out, text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# HUD overlay for video
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    stats: Dict[str, float],
) -> np.ndarray:
    """
    Draw semi-transparent HUD with inference statistics.
    """
    out  = frame.copy()
    h, w = out.shape[:2]

    # Background bar
    bar_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    fps   = stats.get("fps", 0)
    lat   = stats.get("latency_ms", 0)
    pct   = stats.get("transparency_pct", 0)
    frame_n = stats.get("frame", 0)

    # FPS colour: green ≥ 20, amber ≥ 10, red < 10
    fps_col = (0, 220, 60) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 50, 220)

    def put(text, x, col=(200, 200, 200)):
        cv2.putText(out, text, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    put(f"FPS: {fps:.1f}", 10, fps_col)
    put(f"Latency: {lat:.0f}ms", 110, (200, 200, 200))
    put(f"Transparent: {pct:.1f}%", 260, (255, 180, 60))
    put(f"Frame: {frame_n}", w - 120, (140, 140, 140))
    put("SPECTRA", w // 2 - 35, (180, 255, 200))

    return out


# ---------------------------------------------------------------------------
# Publication figures (matplotlib)
# ---------------------------------------------------------------------------

def save_prediction_figure(
    image_rgb:    np.ndarray,       # (H, W, 3) uint8 RGB
    gt_mask:      Optional[np.ndarray],  # (H, W) uint8
    pred_prob:    np.ndarray,       # (H, W) float [0, 1]
    ofcv_map:     Optional[np.ndarray] = None,
    brf_map:      Optional[np.ndarray] = None,
    residual_map: Optional[np.ndarray] = None,
    save_path:    str = "outputs/prediction.pdf",
    title:        str = "",
):
    """
    Generate a multi-panel publication figure.

    Layout (left→right):
      Input | GT mask | Prediction | OFCV | BRF | Residual
    """
    panels = [("Input", image_rgb, None)]

    if gt_mask is not None:
        gt_vis = np.zeros_like(image_rgb)
        gt_vis[gt_mask > 0] = [0, 255, 150]
        panels.append(("Ground truth", gt_vis, "Greens"))

    pred_vis = cm.inferno(pred_prob)[..., :3]   # (H, W, 3) float
    panels.append(("SPECTRA prediction", pred_vis, None))

    if ofcv_map is not None:
        panels.append(("OFCV violation (C1)", ofcv_map, "hot"))
    if brf_map is not None:
        panels.append(("BRF boundary (C2)", brf_map, "viridis"))
    if residual_map is not None:
        panels.append(("Flow residual", residual_map, "plasma"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img, cmap) in zip(axes, panels):
        if cmap and img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        elif img.dtype == np.float32 or img.max() <= 1.0:
            ax.imshow(np.clip(img, 0, 1))
        else:
            ax.imshow(img)
        ax.set_title(label, fontsize=10, pad=6)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_ablation_figure(
    rows: List[Dict],   # [{'name': str, 'iou': float, 'f': float, ...}, ...]
    save_path: str = "outputs/ablation.pdf",
):
    """
    Bar chart comparing ablation rows.
    Each row: {'name': str, 'iou': float, 'f_measure': float}
    """
    names = [r["name"] for r in rows]
    ious  = [r["iou"] for r in rows]
    fms   = [r.get("f_measure", 0) for r in rows]

    x    = np.arange(len(names))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    bars1 = ax.bar(x - w/2, ious, w, label="IoU",       color="#534AB7", alpha=0.85)
    bars2 = ax.bar(x + w/2, fms,  w, label="F-measure", color="#0F6E56", alpha=0.85)

    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SPECTRA ablation study", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_label_efficiency_curve(
    fractions:   List[float],   # [0.1, 0.25, 0.5, 1.0]
    spectra_ious: List[float],
    baseline_ious: List[float],
    sota_full:   float,
    save_path:   str = "outputs/label_efficiency.pdf",
):
    """
    Plot the key result: SPECTRA with X% labels vs baseline with 100% labels.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    x_pct = [f * 100 for f in fractions]

    ax.plot(x_pct, spectra_ious,  "o-", color="#534AB7", linewidth=2,
            markersize=7, label="SPECTRA (ours)")
    ax.plot(x_pct, baseline_ious, "s--", color="#888780", linewidth=1.5,
            markersize=6, label="Supervised baseline")
    ax.axhline(sota_full, color="#D85A30", linestyle=":", linewidth=1.5,
               label=f"Prior SOTA @ 100% labels ({sota_full:.3f})")

    # Annotate crossover point
    for i, (f, s) in enumerate(zip(fractions, spectra_ious)):
        if s >= sota_full:
            ax.annotate(
                f"SPECTRA matches SOTA\nwith {int(f*100)}% labels",
                xy=(f * 100, s),
                xytext=(f * 100 + 5, s - 0.04),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#D85A30"),
                color="#D85A30",
            )
            break

    ax.set_xlabel("Fraction of labelled training data (%)", fontsize=11)
    ax.set_ylabel("IoU on Trans10K test set", fontsize=11)
    ax.set_title("Label efficiency — SPECTRA vs supervised baseline", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Uncertainty / entropy map
# ---------------------------------------------------------------------------

def compute_entropy_map(prob: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """
    Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p).
    High entropy = uncertain prediction.

    Args:
        prob: (B, 1, H, W) or (H, W) probability tensor

    Returns:
        entropy_map: (H, W) float [0, 1] — 1 = maximally uncertain
    """
    p   = prob.squeeze().clamp(eps, 1 - eps)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    ent = ent / np.log(2)   # normalise to [0, 1] (max entropy = 1 bit)
    return ent.cpu().float().numpy()


def overlay_uncertainty(
    image_bgr: np.ndarray,
    entropy_map: np.ndarray,   # (H, W) float [0, 1]
    threshold: float = 0.7,
    colour_bgr: Tuple[int, int, int] = (0, 0, 255),   # red for uncertain
) -> np.ndarray:
    """
    Overlay high-uncertainty regions on image.
    Low confidence = model is unsure — shown in red.
    """
    uncertain = (entropy_map > threshold).astype(np.uint8)
    return overlay_mask_on_image(
        image_bgr, uncertain,
        prob_map=entropy_map * (entropy_map > threshold),
        alpha=0.5,
        colour_bgr=colour_bgr,
    )
