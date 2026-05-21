"""
spectra/modules/signal_fusion.py
Cross-modal signal fusion head.
Fuses DINOv2 features, OFCV violation map, and BRF map via cross-attention
then decodes through a Feature Pyramid Network (FPN)-style decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# FPN Decoder
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder that fuses multi-scale backbone features
    with the OFCV and BRF signals to produce high-resolution predictions.

    Args:
        in_channels_list: channel counts of each FPN level (coarse→fine)
        fusion_channels:  channels after cross-modal fusion injection
        out_channels:     final output channels before heads
        num_upsample:     total upsampling factor (log2 of stride from patch to pixel)
    """

    def __init__(
        self,
        in_channels_list: List[int] = [256, 256, 256, 256],
        fusion_channels:  int = 256,
        out_channels:     int = 128,
    ):
        super().__init__()

        # Lateral 1x1 convs to unify channels across FPN levels
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

        # Top-down 3x3 convs after addition
        self.smoothers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.GELU(),
            )
            for _ in in_channels_list
        ])



        # Final upsampling conv to pixel resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        fpn_features: List[torch.Tensor],   # 4 levels, coarse→fine: each (B, 256, h_i, w_i)
        target_size:  Tuple[int, int],       # (H, W) of original image
    ) -> torch.Tensor:
        """
        Returns:
            out: (B, out_channels, H, W) fused feature map at full resolution
        """
        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.laterals, fpn_features)]

        # Top-down pathway: start from finest (last) and work to coarsest
        # laterals ordered coarse→fine so we reverse for top-down
        laterals_rev = laterals[::-1]   # fine→coarse for top-down

        td = laterals_rev[0]
        top_down = [td]
        for lat in laterals_rev[1:]:
            up = F.interpolate(top_down[-1], size=lat.shape[-2:], mode="bilinear", align_corners=False)
            top_down.append(lat + up)

        # Take finest resolution feature
        finest = top_down[-1]   # (B, fusion_channels, h, w)
        finest = self.smoothers[-1](finest)

        # Upsample to target (image) resolution
        fused_full = F.interpolate(finest, size=target_size, mode="bilinear", align_corners=False)
        out = self.final_conv(fused_full)   # (B, out_channels, H, W)

        return out


# ---------------------------------------------------------------------------
# Cross-modal attention fusion
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Full fusion head (combines cross-modal attn + FPN decoder + output heads)
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """
    Complete signal fusion and prediction head.

    Inputs : DINOv2 multi-scale features + OFCV map + BRF map
    Outputs: segmentation logit map + material classification logit map
             (both at full image resolution)

    Args:
        semantic_dim:  DINOv2 patch token dim (1024 for ViT-L)
        hidden_dim:    internal channel width
        num_classes:   number of material classes
        image_size:    spatial resolution of the original image (H, W)
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        hidden_dim:   int = 256,
        num_classes:  int = 5,
        image_size:   Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.image_size = image_size

        # Token projection
        self.token_proj = nn.Conv2d(semantic_dim, hidden_dim, kernel_size=1, bias=False)

        # FPN decoder
        self.fpn_decoder = FPNDecoder(
            in_channels_list=[256, 256, 256, 256],
            fusion_channels=hidden_dim,
            out_channels=hidden_dim // 2,
        )

        # Segmentation head (binary: transparent / not)
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
        )

        # Material classification head
        self.mat_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, num_classes, kernel_size=1),
        )

    def forward(
        self,
        patch_tokens:  torch.Tensor,          # (B, C, h, w)  final DINOv2 tokens
        fpn_features:  List[torch.Tensor],    # 4 scales from backbone FPN
        ofcv_map:      torch.Tensor,          # (B, 1, h, w)  physics residual gating
        brf_map:       torch.Tensor,          # (B, 1, H, W)  structural boundary signal
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            seg_logits: (B, 1, H, W)
            mat_logits: (B, num_classes, H, W)
            seg_prob:   (B, 1, H, W)  after sigmoid
        """
        B, C, h, w = patch_tokens.shape

        # ── 1. Causal OFCV Modulation ─────────────────────────────────────
        # Instead of being a passive branch, OFCV now gates the semantic tokens.
        # This forces the model to justify every semantic detection with optical evidence.
        gated_tokens = patch_tokens * torch.sigmoid(ofcv_map)

        # Project tokens to hidden dimension
        proj_tokens = self.token_proj(gated_tokens)   # (B, hidden_dim, h, w)

        # ── 2. Structural BRF Integration ──────────────────────────────────
        # BRF is added directly to the feature space to shape the boundaries.
        # Downsample full-res BRF to patch resolution
        brf_down = F.interpolate(brf_map, size=(h, w), mode="bilinear", align_corners=False)
        fused_tokens = proj_tokens + brf_down

        # Override finest FPN level with directly modulated patch tokens
        fpn_with_fusion = list(fpn_features)
        fpn_with_fusion[-1] = fused_tokens

        # FPN decoder → pixel-resolution features
        pixel_feats = self.fpn_decoder(
            fpn_with_fusion, target_size=self.image_size
        )   # (B, hidden//2, H, W)

        seg_logits = self.seg_head(pixel_feats)    # (B, 1, H, W)
        mat_logits = self.mat_head(pixel_feats)    # (B, num_classes, H, W)
        seg_prob   = torch.sigmoid(seg_logits)

        return {
            "seg_logits": seg_logits,
            "mat_logits": mat_logits,
            "seg_prob":   seg_prob,
        }
