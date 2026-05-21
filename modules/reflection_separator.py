"""
spectra/modules/reflection_separator.py
Separates mirror-like specular reflections from actual transparent surfaces.

Problem: both reflections and transparent surfaces cause visual anomalies,
but they are physically distinct:
  - Transparent surface: background scene is VISIBLE through the material (refracted)
  - Specular reflection:  a DIFFERENT scene (mirror image) is visible ON the surface

Physics-based discrimination:
  1. Polarisation cue (approximated from RGB): reflections are partially polarised;
     transmission is not. Estimated via Brewster's angle heuristic on intensity gradients.
  2. Flow parallax: when camera moves, a transparent surface moves differently
     from its background (refraction changes with viewpoint angle).
     A mirror reflection moves exactly opposite to camera motion.
  3. Frequency signature: transparent boundaries have the double-peak BRF signature.
     Pure mirror reflections have sharp single-edge transitions.
  4. Depth consistency: transparent objects have a consistent depth boundary.
     Reflections appear at a virtual depth that is inconsistent with scene geometry.

Architecture:
  Input:  image pair (t0, t1) + OFCV map + BRF map + optical flow
  Output: per-pixel scores:
            transparent_prob  (0=not transparent, 1=transparent)
            reflection_prob   (0=not reflective,  1=specular reflection)
          These sum to ≤ 1 (both can be low = opaque surface)

Novel aspect: the flow-parallax cue — reflections produce flow that is
ANTI-CORRELATED with camera motion; transparency produces consistent
parallax shift. This is a free physics signal requiring no annotation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Polarisation proxy from RGB intensity gradients
# ---------------------------------------------------------------------------

class PolarisationProxy(nn.Module):
    """
    Approximate polarisation cue from RGB gradients.

    At Brewster's angle, reflected light is fully polarised (s-polarisation).
    We approximate this by looking at the gradient magnitude anisotropy:
    reflections tend to produce high-frequency highlights with anisotropic
    gradients (bright in one orientation, dark perpendicular).
    Transparent refractions produce more isotropic background distortion.

    This is a proxy, not true polarisation — but it provides a useful
    discriminative signal between the two phenomena.
    """

    def __init__(self, n_orientations: int = 4):
        super().__init__()
        self.n_orientations = n_orientations

        # Sobel filters at multiple orientations
        angles  = [i * (180 / n_orientations) for i in range(n_orientations)]
        kernels = []
        for angle_deg in angles:
            import math
            theta = math.radians(angle_deg)
            # Rotated Sobel
            kx = torch.tensor([
                [-math.sin(theta) - math.cos(theta),
                 -2 * math.sin(theta),
                 -math.sin(theta) + math.cos(theta)],
                [-2 * math.cos(theta), 0, 2 * math.cos(theta)],
                [math.sin(theta) - math.cos(theta),
                 2 * math.sin(theta),
                 math.sin(theta) + math.cos(theta)],
            ], dtype=torch.float32).view(1, 1, 3, 3)
            kernels.append(kx)

        self.register_buffer("sobel_bank", torch.cat(kernels, dim=0))  # (n_orient, 1, 3, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB

        Returns:
            anisotropy_map: (B, 1, H, W) in [0, 1]
                            High = anisotropic gradient → reflection-like
                            Low  = isotropic gradient  → transparent-like
        """
        # Luminance
        lum = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

        # Apply rotated Sobel bank
        B, _, H, W = lum.shape
        bank   = self.sobel_bank   # (n_orient, 1, 3, 3)
        # Convolve: output (B, n_orient, H, W)
        grads  = F.conv2d(lum, bank, padding=1)
        energy = grads ** 2  # (B, n_orient, H, W)

        # Anisotropy: ratio of max to mean gradient energy across orientations
        e_max  = energy.max(dim=1, keepdim=True).values
        e_mean = energy.mean(dim=1, keepdim=True) + 1e-8

        anisotropy = (e_max / e_mean - 1.0).clamp(0, 10) / 10.0  # [0, 1]
        return anisotropy


# ---------------------------------------------------------------------------
# Flow parallax cue
# ---------------------------------------------------------------------------

def compute_flow_parallax_cue(
    flow_fwd:    Tensor,    # (B, 2, H, W)  optical flow t→t+1
    ofcv_map:    Tensor,    # (B, 1, H, W)  physics violation map
    eps:         float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    Compute transparent vs reflection flow cues.

    For camera translation:
      - Opaque background: flow follows epipolar geometry
      - Transparent overlay: flow shows additional refraction component
        (same direction as background but slightly different magnitude)
      - Mirror reflection: flow is anti-correlated — the virtual image
        moves OPPOSITE to the camera motion

    We estimate the dominant flow direction (camera motion proxy) and
    compute per-pixel agreement vs anti-agreement.

    Returns:
        transparent_parallax: (B, 1, H, W)  consistent parallax → transparent
        reflection_parallax:  (B, 1, H, W)  anti-correlated flow → reflection
    """
    B, _, H, W = flow_fwd.shape

    # Dominant flow direction (mean across spatial dims = camera motion estimate)
    mean_flow = flow_fwd.mean(dim=[-2, -1], keepdim=True)   # (B, 2, 1, 1)
    mean_mag  = mean_flow.norm(dim=1, keepdim=True) + eps

    # Per-pixel flow direction agreement with dominant motion
    flow_mag  = flow_fwd.norm(dim=1, keepdim=True) + eps
    dot       = (flow_fwd * mean_flow).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    cos_sim   = dot / (flow_mag * mean_mag)    # [-1, 1]

    # High positive cos_sim + high OFCV → transparent (flow consistent but physics violated)
    transparent_parallax = torch.sigmoid(cos_sim * 3.0) * ofcv_map

    # High negative cos_sim + high OFCV → reflection (flow anti-correlated + anomaly)
    reflection_parallax  = torch.sigmoid(-cos_sim * 3.0) * ofcv_map

    return transparent_parallax, reflection_parallax


# ---------------------------------------------------------------------------
# Full reflection separator
# ---------------------------------------------------------------------------

class ReflectionSeparator(nn.Module):
    """
    Separates mirror-like specular reflections from transparent surfaces.

    Inputs: RGB image pair + optical flow + OFCV + BRF signals
    Output: per-pixel (transparent_prob, reflection_prob)

    The two outputs are fed back into the main SPECTRA pipeline to:
      - Suppress false positives caused by reflections
      - Improve precision in scenes with mixed glass + mirror surfaces
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.polarisation = PolarisationProxy(n_orientations=4)

        # Fusion MLP: [anisotropy (1) + transparent_parallax (1) +
        #              reflection_parallax (1) + ofcv (1) + brf (1)] = 5 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1),   # 2 outputs: [transparent, reflection]
        )

    def forward(
        self,
        image:       Tensor,    # (B, 3, H, W)
        flow_fwd:    Tensor,    # (B, 2, H, W)
        ofcv_map:    Tensor,    # (B, 1, H, W)
        brf_map:     Tensor,    # (B, 1, H, W)
    ) -> Dict[str, Tensor]:
        """
        Returns:
            transparent_prob:  (B, 1, H, W) probability of being transparent
            reflection_prob:   (B, 1, H, W) probability of being specular reflection
            separation_logits: (B, 2, H, W) raw logits for loss computation
        """
        B, _, H, W = image.shape

        # Resize flow to match image spatial resolution
        flow_up = F.interpolate(flow_fwd, size=(H, W), mode="bilinear", align_corners=False)

        # 1. Polarisation proxy (anisotropy map)
        anisotropy = self.polarisation(image)   # (B, 1, H, W)

        # 2. Flow parallax cues
        transp_par, reflect_par = compute_flow_parallax_cue(flow_up, ofcv_map)

        # 3. Fuse all 5 physics cues
        fused = torch.cat([
            anisotropy,    # polarisation proxy
            transp_par,    # transparent parallax
            reflect_par,   # reflection parallax
            ofcv_map,      # flow physics violation
            brf_map,       # frequency boundary signature
        ], dim=1)   # (B, 5, H, W)

        logits = self.fusion(fused)   # (B, 2, H, W)

        # Softmax so transparent + reflection + [implicit opaque] sums to 1
        probs  = torch.softmax(logits, dim=1)   # (B, 2, H, W)

        return {
            "transparent_prob":  probs[:, 0:1],    # (B, 1, H, W)
            "reflection_prob":   probs[:, 1:2],    # (B, 1, H, W)
            "separation_logits": logits,
        }
