"""
spectra/models/backbone.py
DINOv2 ViT backbone wrapper.
Returns multi-scale patch token features suitable for dense prediction.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Backbone(nn.Module):
    """
    Wraps the pretrained DINOv2 ViT encoder (loaded via torch.hub).
    Extracts patch-level features at the specified intermediate layers
    to form a multi-scale feature pyramid.

    Args:
        model_name:   dinov2_vitl14 | dinov2_vitb16
        frozen:       freeze all backbone weights
        out_indices:  which transformer blocks to tap for features
                      (counted from 0; last block = -1 equivalent)
    """

    # Embedding dimensions per model
    EMBED_DIMS = {
        "dinov2_vitl14": 1024,
        "dinov2_vitb16": 768,
        "dinov2_vits14": 384,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        frozen: bool = True,
        out_indices: Optional[List[int]] = None,
        image_size: int = 512,
    ):
        super().__init__()
        self.model_name = model_name
        self.frozen     = frozen
        
        if out_indices is None:
            if "vits" in model_name or "vitb" in model_name:
                self.out_indices = [2, 5, 8, 11]
            else:
                self.out_indices = [9, 16, 20, 23]
        else:
            self.out_indices = list(out_indices)
        self.embed_dim   = self.EMBED_DIMS.get(model_name, 1024)
        self.patch_size  = 14

        # Load pretrained DINOv2 from torch.hub
        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
        )

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        else:
            # Partially freeze early layers
            for name, param in self.encoder.named_parameters():
                if "blocks.0." in name or "blocks.1." in name:
                    param.requires_grad_(False)

        # Compute spatial resolution of patch tokens
        self.grid_h = image_size // self.patch_size
        self.grid_w = image_size // self.patch_size

        # Lateral 1×1 conv projections to unify channel dims across scales
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False)
            for _ in self.out_indices
        ])

        # FPN top-down pathway — upsample and fuse
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU(),
            )
            for _ in self.out_indices
        ])

    def _extract_intermediate_layers(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Run the ViT encoder and collect patch tokens at out_indices.
        DINOv2's forward_features returns CLS + patch tokens.
        We use get_intermediate_layers for multi-scale output.
        """
        features = self.encoder.get_intermediate_layers(
            x, n=self.out_indices, return_class_token=False
        )
        # Each element: (B, n_patches, C) — reshape to (B, C, H, W)
        spatial = []
        for feat in features:
            B, N, C = feat.shape
            feat_2d = feat.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
            spatial.append(feat_2d)
        return spatial

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) normalised RGB

        Returns:
            dict with:
              'features': list of (B, 256, h, w) FPN feature maps, coarsest→finest
              'patch_tokens': (B, C, H/14, W/14) final-layer patch tokens (full dim)
        """
        # Multi-scale intermediate features
        if self.frozen:
            with torch.no_grad():
                spatial_feats = self._extract_intermediate_layers(x)
        else:
            spatial_feats = self._extract_intermediate_layers(x)

        # FPN: lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, spatial_feats)]

        # FPN: top-down fusion (coarse → fine)
        # Start from the coarsest (index 0 = earliest layer)
        fpn_out = [laterals[-1]]  # finest scale first (last out_index = deepest = semantically richest)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(fpn_out[-1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            fpn_out.append(self.fpn_convs[i](laterals[i] + up))

        fpn_out = fpn_out[::-1]  # reorder: coarse→fine

        # Full-dim patch tokens from final layer (for downstream heads)
        patch_tokens = spatial_feats[-1]  # (B, C, H/14, W/14)

        return {
            "features":     fpn_out,       # list of 4 scales, each (B,256,h,w)
            "patch_tokens": patch_tokens,  # (B, C, H/14, W/14)
        }
