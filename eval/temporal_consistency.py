"""
spectra/eval/temporal_consistency.py

Metrics for temporal consistency of segmentation across consecutive frames.

Reported metrics:
  - TC-IoU       : IoU of pred(t) and pred(t-1) warped via GT flow (or RAFT flow if not annotated)
  - TC-MAE       : Mean absolute pixel-level difference between aligned predictions
  - Flicker      : Per-pixel variance across the window, averaged spatially
  - Boundary jitter: mean Hausdorff distance between consecutive mask boundaries
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Warp helpers
# ---------------------------------------------------------------------------

def warp_flow(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp tensor x of shape (B, C, H, W) by flow (B, 2, H, W) using bilinear sampling.
    Flow is expected as (dx, dy) in *pixel* units.
    """
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device), torch.arange(W, device=x.device),
        indexing="ij",
    )
    grid_x = (xx.unsqueeze(0) + flow[:, 0]).float()
    grid_y = (yy.unsqueeze(0) + flow[:, 1]).float()
    grid_x = 2 * grid_x / (W - 1) - 1
    grid_y = 2 * grid_y / (H - 1) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


# ---------------------------------------------------------------------------
# Per-window metrics
# ---------------------------------------------------------------------------

def temporal_iou(pred_t: torch.Tensor, pred_t_warped: torch.Tensor,
                 thresh: float = 0.5) -> float:
    """IoU between binarised pred(t) and pred(t-1) warped into frame t."""
    a = (pred_t > thresh).float()
    b = (pred_t_warped > thresh).float()
    inter = (a * b).sum().item()
    union = (a + b).clamp_max(1.0).sum().item()
    return inter / max(union, 1e-8)


def temporal_mae(pred_t: torch.Tensor, pred_t_warped: torch.Tensor) -> float:
    return (pred_t - pred_t_warped).abs().mean().item()


def flicker(pred_window: torch.Tensor) -> float:
    """Per-pixel variance over T frames, averaged spatially.
    pred_window: (T, 1, H, W) in [0, 1]."""
    return pred_window.var(dim=0).mean().item()


def boundary_jitter(pred_window: torch.Tensor, thresh: float = 0.5) -> float:
    """Average per-frame change in boundary length (proxy for jitter)."""
    bins = (pred_window > thresh).float()
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred_window.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(bins, sobel_x, padding=1)
    gy = F.conv2d(bins, sobel_y, padding=1)
    boundary = (gx.abs() + gy.abs()) > 0.1
    lengths = boundary.float().sum(dim=(-1, -2)).squeeze()   # (T,)
    if lengths.numel() < 2:
        return 0.0
    return lengths.diff().abs().mean().item()


# ---------------------------------------------------------------------------
# Sequence-level aggregator
# ---------------------------------------------------------------------------

def aggregate_window(
    pred_window: torch.Tensor,      # (T, 1, H, W) probabilities
    flow_fwd_window: torch.Tensor,  # (T-1, 2, H, W) flow t-1 -> t
) -> Dict[str, float]:
    T = pred_window.shape[0]
    tc_ious, tc_maes = [], []
    for t in range(1, T):
        prev = pred_window[t-1:t]                               # (1, 1, H, W)
        flow = flow_fwd_window[t-1:t]                            # (1, 2, H, W)
        prev_warp = warp_flow(prev, flow)
        tc_ious.append(temporal_iou(pred_window[t:t+1], prev_warp))
        tc_maes.append(temporal_mae(pred_window[t:t+1], prev_warp))

    return {
        "tc_iou_mean":      float(np.mean(tc_ious)) if tc_ious else 0.0,
        "tc_mae_mean":      float(np.mean(tc_maes)) if tc_maes else 0.0,
        "flicker":          flicker(pred_window),
        "boundary_jitter":  boundary_jitter(pred_window),
    }
