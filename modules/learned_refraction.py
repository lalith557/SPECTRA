"""
spectra/modules/learned_refraction.py

C1+: Learned refraction estimator that replaces the hand-crafted OFCV cue.
Predicts a per-pixel refraction field — the apparent pixel-displacement induced
by light bending through a transparent medium — and uses it to *warp* the
background image. The warp residual (what the model couldn't explain via
the predicted refraction) becomes a learned, differentiable transparency cue.

Architectural sketch:
    [DINOv2 features, RAFT flow, residual] → small U-Net → (dx, dy, n_index)
    The refraction field (dx, dy) is then used to warp the original image.
    Pixels whose ground-truth motion can only be explained by adding (dx, dy)
    are flagged as transparent.

This module is differentiable end-to-end, can be trained jointly with
SPECTRA, and yields a *learned physics prior* — an estimate of the
refractive index map alongside the displacement field, which is novel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight encoder-decoder used as the displacement-field predictor
# ---------------------------------------------------------------------------

class _SmallUNet(nn.Module):
    def __init__(self, in_ch: int, mid: int = 64, out_ch: int = 3):
        super().__init__()
        self.enc1 = self._block(in_ch, mid)
        self.enc2 = self._block(mid, mid * 2)
        self.enc3 = self._block(mid * 2, mid * 4)
        self.bot  = self._block(mid * 4, mid * 4)
        self.dec3 = self._block(mid * 8, mid * 2)
        self.dec2 = self._block(mid * 4, mid)
        self.dec1 = self._block(mid * 2, mid)
        self.head = nn.Conv2d(mid, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        b  = F.interpolate(b, size=e3.shape[-2:], mode="bilinear",
                           align_corners=False)
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                           align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Learned refraction module
# ---------------------------------------------------------------------------

class LearnedRefractionEstimator(nn.Module):
    """
    Predicts a per-pixel refraction field (dx, dy, n_index) and uses it to
    explain photometric residual.

    Args:
        feat_dim:    DINOv2 channels (e.g. 384)
        mid:         internal channel width
        max_disp:    soft cap on per-pixel displacement (in normalised coords)
        learn_n:     if True, also predict refractive index in [n_min, n_max]
        n_min, n_max: clamp range for refractive index
    """

    def __init__(
        self,
        feat_dim: int = 384,
        mid: int = 64,
        max_disp: float = 0.05,
        learn_n: bool = True,
        n_min: float = 1.0,
        n_max: float = 2.0,
    ):
        super().__init__()
        # Inputs: DINOv2 features (down-projected), flow (2), residual (1)
        self.feat_proj = nn.Conv2d(feat_dim, 32, 1)
        in_ch = 32 + 2 + 1
        self.net = _SmallUNet(in_ch=in_ch, mid=mid, out_ch=3 if learn_n else 2)
        self.max_disp = max_disp
        self.learn_n = learn_n
        self.n_min   = n_min
        self.n_max   = n_max

    def forward(
        self,
        patch_tokens: torch.Tensor,   # (B, C, h, w)
        flow_fwd:     torch.Tensor,   # (B, 2, H, W)
        residual:     torch.Tensor,   # (B, 1, H, W)
        image_t0:     torch.Tensor,   # (B, 3, H, W)
        image_t1:     torch.Tensor,   # (B, 3, H, W)
    ):
        B, _, H, W = image_t0.shape
        feat = self.feat_proj(patch_tokens)
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([feat, flow_fwd, residual], dim=1)
        out = self.net(x)

        disp = torch.tanh(out[:, :2]) * self.max_disp
        if self.learn_n:
            n_norm = torch.sigmoid(out[:, 2:3])
            n_index = self.n_min + (self.n_max - self.n_min) * n_norm
        else:
            n_index = torch.ones((B, 1, H, W), device=image_t0.device)

        # Warp image_t1 by (predicted_flow + predicted_refraction_disp)
        # Convert normalised disp back to pixels: disp \in [-1,1] normalised
        # We treat tanh disp as fraction of image size.
        disp_pix = torch.stack([disp[:, 0] * W, disp[:, 1] * H], dim=1)
        warped = self._warp(image_t1, flow_fwd + disp_pix)

        # Residual after compensating for predicted refraction
        residual_refined = (warped - image_t0).abs().mean(dim=1, keepdim=True)
        # Confidence: original residual we *managed* to explain
        explained = (residual - residual_refined).clamp(min=0)
        # Sigmoid normalisation so it can be used as a transparency cue
        transparency_cue = torch.sigmoid(explained * 8 - 2)

        return {
            "disp":              disp_pix,
            "n_index":           n_index,
            "residual_refined":  residual_refined,
            "transparency_cue":  transparency_cue,
        }

    @staticmethod
    def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device), torch.arange(W, device=x.device),
            indexing="ij",
        )
        gx = (xx.unsqueeze(0).float() + flow[:, 0])
        gy = (yy.unsqueeze(0).float() + flow[:, 1])
        gx = 2 * gx / (W - 1) - 1
        gy = 2 * gy / (H - 1) - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)
