"""
spectra/pretrain/refraction_aug.py
Novel Contribution C4 — Physics-Contrastive Self-Supervised Pre-training.

Synthesises transparent object regions on arbitrary images using Snell's law
pixel displacement. No manual labels required — the refractive index n is the
supervisory signal.

Snell's law: n1 * sin(θ1) = n2 * sin(θ2)
For a flat glass surface, this produces a lateral pixel shift proportional
to the thickness and refractive index of the material.
"""
import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ofcv - optical flow consistency violation, dinov2 (self-DIstillation with NO labels) extracts high level semantic features
    # 

# ---------------------------------------------------------------------------
# Snell's law warp
# ---------------------------------------------------------------------------

def snell_displacement_field(
    H: int,
    W: int,
    mask: Tensor,          # (H, W) binary — region to apply refraction
    n:    float = 1.5,     # refractive index (air=1.0, water=1.33, glass=1.5)
    light_dir: Tuple[float, float] = (0.0, 1.0),   # (dx, dy) incoming light direction (normalised)
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Compute per-pixel displacement field for a transparent region.

    Physics:
        At the boundary, refraction bends the light ray. For an incident
        ray at angle θ1 to the surface normal, the refracted angle θ2 satisfies:
            sin(θ2) = sin(θ1) / n    (n1=1.0 air, n2=n glass)

        The lateral shift of the background seen through the glass is:
            Δx = t * (tan(θ1) - tan(θ2))
        where t is the effective glass thickness (approximated as a constant).

        We model this as a spatially-varying grid displacement within the mask.

    Returns:
        flow: (2, H, W) displacement in pixels [dx, dy]
    """
    n1 = 1.0   # air
    n2 = float(n)

    # Incident angle from light direction
    dx, dy = light_dir
    # Angle of incidence w.r.t. surface normal (assume horizontal surface → normal=(0,1))
    theta1 = math.atan2(abs(dx), abs(dy) + 1e-8)

    # Snell's law — check for total internal reflection
    sin_theta2 = (n1 / n2) * math.sin(theta1)
    sin_theta2 = min(sin_theta2, 0.9999)
    theta2     = math.asin(sin_theta2)

    # Lateral shift magnitude (in pixels; scale by image size for realism)
    thickness = H * 0.05   # approximate glass thickness = 5% of image height
    shift_x   = thickness * (math.tan(theta1) - math.tan(theta2)) * math.copysign(1, dx)
    shift_y   = thickness * 0.1 * (math.tan(theta1) - math.tan(theta2)) * math.copysign(1, dy)

    # Apply shift only inside the mask — smooth at boundaries via erosion
    flow = torch.zeros(2, H, W, device=device)
    flow[0] = mask.float() * shift_x   # dx
    flow[1] = mask.float() * shift_y   # dy

    # Smooth the displacement at mask boundaries to avoid hard artefacts
    kernel = torch.ones(1, 1, 7, 7, device=device) / 49.0
    flow = F.conv2d(flow.unsqueeze(1), kernel, padding=3).squeeze(1)   # per-channel smooth

    return flow


def apply_snell_warp(
    img: Tensor,     # (3, H, W) float [0, 1]
    mask: Tensor,    # (H, W) binary
    n: float = 1.5,
    light_dir: Tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """
    Apply Snell's law refractive warp to the background visible through the
    transparent region defined by mask.

    Returns:
        warped: (3, H, W) image with synthetic transparent region
    """
    _, H, W = img.shape
    device  = img.device

    flow = snell_displacement_field(H, W, mask, n=n, light_dir=light_dir, device=device)

    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=0)   # (2, H, W)

    # Normalise flow to grid units
    flow_norm = flow.clone()
    flow_norm[0] = flow_norm[0] / (W - 1) * 2.0
    flow_norm[1] = flow_norm[1] / (H - 1) * 2.0

    sample_grid = (base_grid + flow_norm).permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)

    warped_bg = F.grid_sample(
        img.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)   # (3, H, W)

    # Blend: inside mask → warped background; outside → original
    # Also add a thin specular highlight at the boundary
    boundary_highlight = _compute_boundary_highlight(mask, strength=0.15)
    transparent_region = warped_bg * 0.85 + img * 0.15 + boundary_highlight

    out = img.clone()
    mask_3ch = mask.float().unsqueeze(0)   # (1, H, W)
    out = out * (1 - mask_3ch) + transparent_region * mask_3ch

    return out.clamp(0, 1)


def _compute_boundary_highlight(mask: Tensor, strength: float = 0.15) -> Tensor:
    """Thin bright highlight at mask boundary — simulates glass edge caustic."""
    _, H, W = mask.shape if mask.dim() == 3 else (1, *mask.shape)
    mask_4d = mask.float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    # Erode and dilate to get boundary
    kernel   = torch.ones(1, 1, 5, 5, device=mask.device) / 25.0
    smoothed = F.conv2d(mask_4d, kernel, padding=2)
    boundary = (smoothed > 0.1).float() * (smoothed < 0.9).float()

    return boundary.squeeze(0) * strength   # (1, H, W) → broadcast over 3 channels


# ---------------------------------------------------------------------------
# Random mask generators
# ---------------------------------------------------------------------------

def random_ellipse_mask(H: int, W: int) -> np.ndarray:
    """Generate a random ellipse binary mask."""
    mask = np.zeros((H, W), dtype=np.float32)
    cx   = random.randint(W // 4, 3 * W // 4)
    cy   = random.randint(H // 4, 3 * H // 4)
    rx   = random.randint(W // 8, W // 3)
    ry   = random.randint(H // 8, H // 3)

    y, x = np.ogrid[:H, :W]
    ellipse = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
    mask[ellipse] = 1.0
    return mask


def random_polygon_mask(H: int, W: int, n_vertices: int = 6) -> np.ndarray:
    """Generate a random convex polygon binary mask."""
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    angles   = sorted(random.uniform(0, 2 * math.pi) for _ in range(n_vertices))
    cx, cy   = W // 2 + random.randint(-W // 6, W // 6), H // 2 + random.randint(-H // 6, H // 6)
    r        = random.randint(min(H, W) // 6, min(H, W) // 3)
    pts      = np.array([
        [int(cx + r * math.cos(a)), int(cy + r * math.sin(a))]
        for a in angles
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(np.float32)


def random_blob_mask(H: int, W: int) -> np.ndarray:
    """Smooth random blob via low-frequency noise thresholding."""
    # Low-resolution random noise, upsampled → organic blob shape
    small = np.random.randn(H // 16 + 1, W // 16 + 1).astype(np.float32)
    noise_t = torch.from_numpy(small).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(noise_t, size=(H, W), mode="bilinear", align_corners=False)
    blob = (upsampled.squeeze().numpy() > 0.3).astype(np.float32)
    return blob


MASK_GENERATORS = [random_ellipse_mask, random_polygon_mask, random_blob_mask]


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

class PhysicsContrastiveAugmentor:
    """
    Generates (anchor, positive, n_value) triplets for contrastive pre-training.

    For each image:
      - anchor:   original image
      - positive: image with synthetic transparent region (different n)
      - n_value:  refractive index used (float label — used for weighted loss)

    No manual annotation required.

    Args:
        n_min:  minimum refractive index (water ≈ 1.33)
        n_max:  maximum refractive index (crystal glass ≈ 1.90)
    """

    def __init__(self, n_min: float = 1.33, n_max: float = 1.90):
        self.n_min = n_min
        self.n_max = n_max

    def __call__(
        self,
        img: Tensor,   # (3, H, W) float [0, 1]
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Returns:
            anchor:   (3, H, W) — original
            positive: (3, H, W) — with synthetic transparent region
            mask:     (H, W)    — binary region of synthetic glass
            n_value:  float     — refractive index used
        """
        _, H, W = img.shape
        n_value  = random.uniform(self.n_min, self.n_max)

        # Random light direction (slight angle variation)
        angle     = random.uniform(-0.3, 0.3)   # radians from vertical
        light_dir = (math.sin(angle), math.cos(angle))

        # Random mask shape
        gen  = random.choice(MASK_GENERATORS)
        mask_np = gen(H, W)
        mask = torch.from_numpy(mask_np).to(img.device)

        positive = apply_snell_warp(img, mask, n=n_value, light_dir=light_dir)

        return img, positive, mask, n_value


# ---------------------------------------------------------------------------
# Physics NT-Xent loss (C4 contrastive loss)
# ---------------------------------------------------------------------------

class PhysicsNTXentLoss(nn.Module):
    """
    Modified NT-Xent contrastive loss for physics-contrastive pre-training.

    Key modification over standard SimCLR NT-Xent:
      - Temperature τ is scaled by |n_i - n_j|: pairs with larger refractive
        index difference are harder negatives (sharper contrast in embedding space)
      - Positive pair: (anchor_region, positive_region) at same spatial location
        but different n values → same material, different appearance
      - Negative pair: (transparent_region, opaque_region) → different material

    Args:
        base_temperature: base τ (standard SimCLR uses 0.07)
        n_scale:          scaling factor for n-difference weighting
    """

    def __init__(self, base_temperature: float = 0.07, n_scale: float = 2.0):
        super().__init__()
        self.base_temp = base_temperature
        self.n_scale   = n_scale

    def forward(
        self,
        z_anchor:   Tensor,     # (B, D) — anchor embeddings
        z_positive: Tensor,     # (B, D) — positive embeddings (synthetic transparent)
        n_values:   Tensor,     # (B,)   — refractive indices used per sample
    ) -> Tensor:
        """
        Computes physics-weighted contrastive loss.
        """
        B, D = z_anchor.shape
        device = z_anchor.device

        # L2 normalise
        z_a = F.normalize(z_anchor,   dim=-1)   # (B, D)
        z_p = F.normalize(z_positive, dim=-1)   # (B, D)

        # Compute |n_i - n_j| matrix for temperature weighting
        n_diff = torch.abs(n_values.unsqueeze(0) - n_values.unsqueeze(1))   # (B, B)
        # Scale temperature: larger n difference → lower τ → sharper separation
        tau = self.base_temp / (1.0 + self.n_scale * n_diff)   # (B, B) in (0, base_temp]

        # Similarity matrix: [z_a; z_p] × [z_a; z_p]^T
        z_all = torch.cat([z_a, z_p], dim=0)   # (2B, D)
        sim   = torch.mm(z_all, z_all.T)        # (2B, 2B)

        # Tile tau to match 2B × 2B
        tau_full = tau.repeat(2, 2)              # (2B, 2B)
        sim_scaled = sim / tau_full.clamp(min=1e-4)

        # Mask diagonal (self-similarity)
        mask_diag = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim_scaled = sim_scaled.masked_fill(mask_diag, -9e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=device)
        labels = torch.cat([labels + B, labels])   # (2B,)

        loss = F.cross_entropy(sim_scaled, labels)
        return loss


# ---------------------------------------------------------------------------
# Pre-training model wrapper
# ---------------------------------------------------------------------------

class SPECTRAPretrainModel(nn.Module):
    """
    Encoder + projection head for contrastive pre-training.

    During pre-training, we only train the projection head.
    The encoder (DINOv2 backbone) is optionally fine-tuned with a very small LR.

    Args:
        encoder:     DINOv2Backbone instance
        embed_dim:   backbone output dim
        proj_dim:    contrastive projection head output dim
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 1024,
        proj_dim:  int = 128,
    ):
        super().__init__()
        self.encoder = encoder

        # MLP projection head (SimCLR-style: 3 layers with BN)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, proj_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            z: (B, proj_dim) projected embedding
        """
        enc_out = self.encoder(x)
        patch_tokens = enc_out["patch_tokens"]   # (B, C, h, w)

        # Global average pool → single vector per image
        pooled = patch_tokens.mean(dim=[-2, -1])  # (B, C)

        z = self.projector(pooled)    # (B, proj_dim)
        return z
