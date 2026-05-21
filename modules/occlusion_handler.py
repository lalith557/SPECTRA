"""
spectra/modules/occlusion_handler.py
Occlusion-aware transparent object detection.

Problem: In real scenes, transparent objects are often:
  - Partially hidden behind opaque objects (a glass behind a vase)
  - Stacked/overlapping with other transparent objects
  - Truncated at image boundaries

Standard segmentation models fail here because they rely on complete
object appearance. SPECTRA's physics signals (OFCV, BRF) are local —
they fire at each detected boundary/residual independently. But they
still need to be completed into full object masks even when parts are occluded.

Approach:
  1. Amodal completion: predict the full extent of transparent objects
     including occluded portions, using context from visible parts.
  2. Occlusion order estimation: determine which transparent object is
     in front of which occluder, so depth/grasping systems get correct ordering.
  3. Boundary confidence weighting: down-weight predictions near occluding
     object boundaries where the physics signal is contaminated.

Architecture:
  OcclusionHandler(
    visible_mask  → amodal_mask + occlusion_map + order_map
  )
  Uses a recurrent refinement loop (3 iterations) that propagates
  predictions from high-confidence visible regions into occluded gaps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Amodal completion via iterative diffusion
# ---------------------------------------------------------------------------

class AmodalDiffusion(nn.Module):
    """
    Iteratively complete occluded transparent object masks.

    Algorithm:
      1. Start from the visible segmentation probability map
      2. Detect high-confidence transparent regions (seeds)
      3. Diffuse predictions from seeds into uncertain (occluded) regions
         weighted by BRF boundary signals (don't cross hard boundaries)
      4. Repeat for N_iters iterations

    This is analogous to Random Walk segmentation but guided by physics signals.
    """

    def __init__(self, n_iters: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.n_iters = n_iters

        # Learned diffusion kernel (predicts how to spread predictions spatially)
        self.diffuse_conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Boundary stop gate: prevent diffusion across opaque object boundaries
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seg_prob:    Tensor,    # (B, 1, H, W) initial seg prediction
        brf_map:     Tensor,    # (B, 1, H, W) boundary field (stops diffusion)
        ofcv_map:    Tensor,    # (B, 1, H, W) physics violation (seeds diffusion)
        occluder_mask: Optional[Tensor] = None,  # (B, 1, H, W) detected occluders
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            amodal_mask:     (B, 1, H, W) completed mask including occluded regions
            confidence_map:  (B, 1, H, W) per-pixel completion confidence
        """
        # Boundary gate: high BRF = true glass boundary = diffusion should STOP
        # Low BRF in uncertain region = gap due to occlusion = diffusion should CONTINUE
        gate_input  = torch.cat([brf_map, 1.0 - brf_map], dim=1)
        boundary_stop = self.boundary_gate(gate_input)   # (B, 1, H, W)

        current = seg_prob.clone()
        confidence = torch.ones_like(seg_prob)

        for _ in range(self.n_iters):
            # Physics-guided diffusion input
            diffuse_in = torch.cat([current, ofcv_map, brf_map], dim=1)
            delta      = self.diffuse_conv(diffuse_in)   # (B, 1, H, W)

            # Gate: only spread where boundary signal is low (not a real boundary)
            spread = delta * (1.0 - boundary_stop)

            # Don't override high-confidence predictions
            update_mask = (current < 0.4).float()   # update only uncertain regions

            # Optional: don't spread into confirmed opaque regions
            if occluder_mask is not None:
                update_mask = update_mask * (1.0 - occluder_mask)

            current    = current + spread * update_mask
            current    = current.clamp(0, 1)
            confidence = confidence * (1.0 - spread * update_mask * 0.1)  # decay confidence

        return current, confidence.clamp(0, 1)


# ---------------------------------------------------------------------------
# Occlusion order estimator
# ---------------------------------------------------------------------------

class OcclusionOrderEstimator(nn.Module):
    """
    Estimates the layering order of transparent objects in a scene.

    Key insight: if glass A is in front of glass B:
      - A's refraction distorts the view of B
      - The OFCV residual at A's region should be higher (two refractions)
      - The BRF boundary of A should be sharper (closer to camera)

    Outputs a relative depth order map: higher value = closer to camera.
    This is used by robotic grasping systems to know which object to grasp first.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.order_head = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seg_prob:  Tensor,    # (B, 1, H, W)
        ofcv_map:  Tensor,    # (B, 1, H, W) — higher = more refraction layers
        brf_map:   Tensor,    # (B, 1, H, W) — sharper boundary = closer
    ) -> Tensor:
        """
        Returns:
            order_map: (B, 1, H, W) in [0, 1]
                       High = transparent object closer to camera
                       Low  = transparent object further away / behind another
        """
        feat = torch.cat([seg_prob, ofcv_map, brf_map], dim=1)
        return self.order_head(feat)


# ---------------------------------------------------------------------------
# Full occlusion handler module
# ---------------------------------------------------------------------------

class OcclusionHandler(nn.Module):
    """
    Complete occlusion-aware extension for SPECTRA.

    Wraps amodal completion + occlusion order estimation into a single
    module that can be appended to the main SPECTRA pipeline.

    Args:
        n_iters:    diffusion iterations for amodal completion
        hidden_dim: internal channel width
    """

    def __init__(self, n_iters: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.amodal    = AmodalDiffusion(n_iters=n_iters, hidden_dim=hidden_dim)
        self.order_est = OcclusionOrderEstimator(hidden_dim=hidden_dim)

    def detect_occluders(
        self,
        seg_prob: Tensor,
        brf_map:  Tensor,
    ) -> Tensor:
        """
        Heuristically detect occluding objects.
        An occluder is a region where:
          - seg_prob is low (not transparent)
          - brf_map is low (not a glass boundary)
          - but surrounded by transparent regions (gap in the middle of a transparent region)

        Returns: (B, 1, H, W) occluder probability map
        """
        # Dilate the transparent region to find expected coverage
        dilated = F.max_pool2d(seg_prob, kernel_size=31, stride=1, padding=15)

        # Occluder = inside expected coverage but not predicted transparent and not a boundary
        occluder = dilated * (1.0 - seg_prob) * (1.0 - brf_map)
        return occluder.clamp(0, 1)

    def forward(
        self,
        seg_prob:  Tensor,    # (B, 1, H, W) initial segmentation
        ofcv_map:  Tensor,    # (B, 1, H, W)
        brf_map:   Tensor,    # (B, 1, H, W)
    ) -> Dict[str, Tensor]:
        """
        Returns:
            amodal_mask:     (B, 1, H, W) completed mask (includes occluded parts)
            visible_mask:    (B, 1, H, W) original prediction (visible parts only)
            occluder_map:    (B, 1, H, W) detected occluding regions
            order_map:       (B, 1, H, W) relative depth order (higher = closer)
            confidence_map:  (B, 1, H, W) completion confidence
        """
        occluder_map = self.detect_occluders(seg_prob, brf_map)

        amodal_mask, confidence = self.amodal(
            seg_prob, brf_map, ofcv_map, occluder_map
        )

        order_map = self.order_est(seg_prob, ofcv_map, brf_map)

        return {
            "amodal_mask":    amodal_mask,
            "visible_mask":   seg_prob,
            "occluder_map":   occluder_map,
            "order_map":      order_map,
            "confidence_map": confidence,
        }
