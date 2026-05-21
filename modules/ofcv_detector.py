"""
spectra/modules/ofcv_detector.py
Novel Contribution C1 — Optical Flow Consistency Violation (OFCV) Detector.

The key insight: transparent objects violate the brightness constancy assumption
of optical flow (the Lambertian assumption). Light refracts through glass, so
the apparent colour of background pixels visible through the glass changes as
the camera or object moves — creating a systematically high photometric residual
that a trained head can localise.

Architecture:
  Input : [DINOv2 patch features (C, H, W)] + [flow residual map (1, H, W)]
          + [flow consistency map (1, H, W)]
  Output: violation map V ∈ [0,1]^(H, W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention using cross-channel statistics."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        attn    = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class OFCVDetector(nn.Module):
    """
    Physics-informed violation localiser.

    The input residual map R and consistency map Fc are physics signals
    that require zero manual annotation. The DINOv2 features provide
    semantic context (e.g. knowing this region is likely a window frame
    helps suppress false positives at window edges).

    Forward pass:
      1. Project DINOv2 features to hidden_dim
      2. Concatenate residual + consistency as additional channels
      3. CBAM attention over fused features
      4. Lightweight MLP decoder → sigmoid output

    Args:
        in_channels:  DINOv2 embed dim (1024 for ViT-L)
        hidden_dim:   internal channel width
        n_heads:      multi-head attention heads in cross-attention block
        dropout:      dropout probability
    """

    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim:  int = 256,
        n_heads:     int = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Project DINOv2 features down
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Physics signal encoder — residual (1) + consistency (1) = 2 channels
        self.physics_proj = nn.Sequential(
            nn.Conv2d(2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Fusion: semantic (hidden_dim) + physics (hidden_dim) = 2*hidden_dim
        fused_dim = hidden_dim * 2

        self.cbam = CBAM(fused_dim)

        # Cross-attention: physics queries semantic context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),  # logit map
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,    # (B, C, h, w)  DINOv2 features
        residual_map: torch.Tensor,    # (B, 1, H, W)  photometric residual
        consistency_map: torch.Tensor, # (B, 1, H, W)  flow consistency
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            violation_map: (B, 1, h, w) sigmoid probability of violation
            logit_map:     (B, 1, h, w) raw logits (for loss computation)
        """
        B, C, h, w = patch_tokens.shape

        # Resize physics maps to match patch token spatial resolution
        res_down  = F.interpolate(residual_map,    size=(h, w), mode="bilinear", align_corners=False)
        cons_down = F.interpolate(consistency_map, size=(h, w), mode="bilinear", align_corners=False)

        # 1. Semantic branch
        sem = self.feat_proj(patch_tokens)   # (B, hidden_dim, h, w)

        # 2. Physics branch
        physics_in = torch.cat([res_down, cons_down], dim=1)  # (B, 2, h, w)
        phy = self.physics_proj(physics_in)                    # (B, hidden_dim, h, w)

        # 3. Cross-attention: physics-informed queries over semantic keys
        # Flatten spatial dims for attention
        phy_flat = rearrange(phy, "b c h w -> b (h w) c")   # (B, hw, C)
        sem_flat = rearrange(sem, "b c h w -> b (h w) c")   # (B, hw, C)

        attn_out, _ = self.cross_attn(
            query=phy_flat,
            key=sem_flat,
            value=sem_flat,
        )
        attn_out = self.attn_norm(attn_out + phy_flat)       # residual + norm
        phy_attended = rearrange(attn_out, "b (h w) c -> b c h w", h=h, w=w)

        # 4. Concatenate and apply CBAM
        fused = torch.cat([sem, phy_attended], dim=1)        # (B, 2*hidden_dim, h, w)
        fused = self.cbam(fused)

        # 5. Decode
        logits = self.decoder(fused)                          # (B, 1, h, w)
        violation_map = torch.sigmoid(logits)

        return violation_map, logits
