"""
spectra/models/spectra_mono.py

SPECTRA-Mono: monocular variant that accepts a *single* input image and
synthesises a second frame internally — using either:
    1. A handcrafted homography / flow warp (data/synthetic_t1.py)
    2. A learned displacement field (small CNN — trained via the OFCV loss)

The synthesized t1 feeds the OFCV pipeline so SPECTRA still gets a
temporal cue, even when the deployment scenario has no real video.

This converts a multi-frame video-only model into a single-image inference
system — a big practical-impact step for robotics / phone-camera applications.
"""
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectra_model import SPECTRA
from data.synthetic_t1 import make_synthetic_pair_tensor


class LearnedMonocularWarper(nn.Module):
    """
    Predicts a small per-pixel displacement field from a single image,
    then warps the image to produce a synthetic image_t1.
    """

    def __init__(self, max_disp: float = 4.0):
        super().__init__()
        self.max_disp = max_disp
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        flow = self.net(x)
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow = torch.tanh(flow) * self.max_disp

        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device),
            indexing="ij",
        )
        gx = 2 * (xx.unsqueeze(0) + flow[:, 0]) / (W - 1) - 1
        gy = 2 * (yy.unsqueeze(0) + flow[:, 1]) / (H - 1) - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)


class SPECTRA_Mono(SPECTRA):
    """
    Monocular SPECTRA: takes a single image and produces a transparency mask.

    Args:
        cfg:        SPECTRA config
        warp_mode:  'homography' | 'flow_warp' | 'learned'
        deterministic_eval: if True at eval time, fix a single homography seed
    """

    def __init__(self, cfg, warp_mode: str = "learned", **kwargs):
        super().__init__(cfg, **kwargs)
        self.warp_mode = warp_mode
        if warp_mode == "learned":
            self.warper = LearnedMonocularWarper()
        else:
            self.warper = None

    def forward(self, image: torch.Tensor,
                image_t1: Optional[torch.Tensor] = None,
                return_intermediates: bool = False):
        if image_t1 is None:
            image_t1 = self._make_t1(image)
        return super().forward(image, image_t1,
                               return_intermediates=return_intermediates)

    def _make_t1(self, image: torch.Tensor) -> torch.Tensor:
        if self.warper is not None:
            return self.warper(image)
        B = image.shape[0]
        out = []
        rng = random.Random(0 if not self.training else None)
        for i in range(B):
            _, t1 = make_synthetic_pair_tensor(image[i], mode=self.warp_mode, rng=rng)
            out.append(t1)
        return torch.stack(out, dim=0)
