"""
spectra/modules/brf.py
Novel Contribution C2 — Boundary Resonance Field (BRF).

Transparent object edges produce a characteristic double-edge signature:
you simultaneously see the boundary of the glass object and the background
edge refracted through it, at a phase-shifted location. BRF exploits this
by computing Gabor responses across multiple orientations/scales and detecting
co-occurring double-peak signatures in the frequency domain.

This is a fully differentiable single-frame signal — complementary to OFCV
which requires temporal pairs.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable Gabor filter bank
# ---------------------------------------------------------------------------

def build_gabor_kernel(
    ksize: int,
    sigma: float,
    theta: float,
    lam: float,
    gamma: float,
    psi: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a single Gabor kernel as a 2D tensor.

    Args:
        ksize:  kernel spatial size (odd integer)
        sigma:  Gaussian envelope standard deviation
        theta:  orientation in radians
        lam:    wavelength of the sinusoidal factor
        gamma:  spatial aspect ratio
        psi:    phase offset

    Returns:
        kernel: (1, 1, ksize, ksize) real Gabor filter
    """
    half = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Rotated coordinates
    x_rot =  x * math.cos(theta) + y * math.sin(theta)
    y_rot = -x * math.sin(theta) + y * math.cos(theta)

    gauss    = torch.exp(-(x_rot ** 2 + gamma ** 2 * y_rot ** 2) / (2 * sigma ** 2))
    sinusoid = torch.cos(2 * math.pi * x_rot / lam + psi)
    kernel   = gauss * sinusoid

    # Normalise to zero mean
    kernel = kernel - kernel.mean()
    kernel = kernel / (kernel.std() + 1e-8)

    return kernel.view(1, 1, ksize, ksize)


def build_gabor_bank(
    n_orientations: int = 8,
    n_scales: int = 3,
    base_sigma: float = 2.0,
    base_lam: float = 6.0,
    gamma: float = 0.5,
    ksize: int = 15,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a bank of Gabor filters covering multiple orientations and scales.

    Args:
        n_orientations: number of evenly-spaced orientations [0, π)
        n_scales:       number of scale levels (sigma doubles each level)
        base_sigma:     smallest Gaussian sigma
        base_lam:       base wavelength (λ = 2σ for optimal bandwidth)
        gamma:          aspect ratio (< 1 → elongated)
        ksize:          kernel spatial size (should be odd)

    Returns:
        bank: (n_orientations * n_scales, 1, ksize, ksize)
    """
    kernels = []
    for s in range(n_scales):
        sigma = base_sigma * (2 ** s)
        lam   = base_lam   * (2 ** s)
        for o in range(n_orientations):
            theta = o * math.pi / n_orientations
            k = build_gabor_kernel(
                ksize=ksize, sigma=sigma, theta=theta,
                lam=lam, gamma=gamma, device=device,
            )
            kernels.append(k)

    bank = torch.cat(kernels, dim=0)   # (n_filters, 1, ksize, ksize)
    return bank


# ---------------------------------------------------------------------------
# Double-peak detector
# ---------------------------------------------------------------------------

def detect_double_peak(
    response: torch.Tensor,
    window: int = 15,
    min_peak_dist: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    For each spatial location and orientation, detect if there are two
    distinct energy peaks within 'window' pixels along the orientation axis.
    This is the hallmark of a transparent boundary (glass object edge +
    refracted background edge at a phase-shifted location).

    Args:
        response: (B, n_filters, H, W) absolute Gabor responses
        window:   neighbourhood to search for double peaks
        min_peak_dist: minimum separation between two peaks (in pixels)

    Returns:
        brf_map: (B, 1, H, W) boundary resonance energy in [0, 1]
    """
    B, num_filters, H, W = response.shape

    # Use max-pool to find local maxima
    # A pixel is a local max if it equals its neighbourhood maximum
    pool_out = F.max_pool2d(
        response,
        kernel_size=window,
        stride=1,
        padding=window // 2,
    )

    # Local maxima indicator
    is_local_max = (response >= pool_out - eps).float()   # (B, F, H, W)

    # For double-peak detection: dilate local maxima and check
    # if there are ≥ 2 peaks within the window
    # We use average pooling on the local max map — if the average
    # is above 2/(window*window) we have candidate double peaks
    avg_peaks = F.avg_pool2d(
        is_local_max,
        kernel_size=window,
        stride=1,
        padding=window // 2,
    )  # (B, F, H, W)

    double_peak_threshold = 2.0 / (window * window)
    double_peak_mask = (avg_peaks > double_peak_threshold).float()  # (B, F, H, W)

    # Weight double-peak mask by response magnitude
    weighted = response * double_peak_mask   # (B, F, H, W)

    # Max over filter orientations → scalar BRF per pixel
    brf_raw, _ = weighted.max(dim=1, keepdim=True)   # (B, 1, H, W)

    # Normalise per-image
    flat = brf_raw.flatten(2)  # (B, 1, H*W)
    min_val = flat.min(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    max_val = flat.max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    brf_norm = (brf_raw - min_val) / (max_val - min_val + eps)

    return brf_norm   # (B, 1, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# BRF Module (nn.Module wrapper)
# ---------------------------------------------------------------------------

class BoundaryResonanceField(nn.Module):
    """
    Full BRF computation module — differentiable and GPU-accelerated.

    Pipeline:
      1. Convert input to grayscale (Gabor works on luminance)
      2. Apply Gabor bank (grouped convolution — efficient)
      3. Compute absolute energy response
      4. Detect double-peak signature → scalar BRF map
      5. Learnable refinement head to suppress false positives

    Args:
        n_orientations: Gabor bank orientations
        n_scales:       Gabor bank scales
        peak_window:    spatial window for double-peak search
        refine_channels: channels in the refinement CNN
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_scales:       int = 3,
        peak_window:    int = 15,
        refine_channels: int = 32,
    ):
        super().__init__()
        self.n_orientations  = n_orientations
        self.n_scales        = n_scales
        self.peak_window     = peak_window
        self.n_filters       = n_orientations * n_scales

        # Register Gabor bank as a buffer (non-trainable, moves with .to(device))
        bank = build_gabor_bank(
            n_orientations=n_orientations,
            n_scales=n_scales,
            ksize=15,
        )  # (n_filters, 1, 15, 15)
        self.register_buffer("gabor_bank", bank)

        # Learnable refinement: BRF map → refined BRF
        self.refine = nn.Sequential(
            nn.Conv2d(1, refine_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.GELU(),
            nn.Conv2d(refine_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) RGB to (B, 1, H, W) luminance."""
        # ITU-R BT.601 coefficients
        r_w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype)
        return (x * r_w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised RGB

        Returns:
            brf_map:      (B, 1, H, W) raw BRF (double-peak energy)
            brf_refined:  (B, 1, H, W) learnable-refined BRF (used in downstream)
        """
        B, _, H, W = x.shape

        gray = self._to_gray(x)   # (B, 1, H, W)

        # Grouped convolution: treat batch*1 as a single channel batch
        # Gabor bank: (n_filters, 1, kH, kW)
        pad = self.gabor_bank.shape[-1] // 2
        responses = F.conv2d(
            gray,
            self.gabor_bank,
            padding=pad,
        )   # (B, n_filters, H, W)

        # Absolute energy
        energy = torch.abs(responses)   # (B, n_filters, H, W)

        # Double-peak detection → BRF map
        brf_raw = detect_double_peak(
            energy,
            window=self.peak_window,
        )   # (B, 1, H, W)

        # Learnable refinement
        brf_refined = self.refine(brf_raw)   # (B, 1, H, W)

        return brf_raw, brf_refined
