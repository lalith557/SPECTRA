"""
spectra/modules/differentiable_optics.py

Tiny differentiable optics layer:
  Given a depth map z(x, y), a refractive index field n(x, y), and the camera
  intrinsics, compute the per-pixel apparent displacement of the background
  via Snell's law (small-angle approximation).

This isn't a full ray tracer — it's a closed-form first-order approximation
that gives us a *physics-grounded* differentiable cue. The point is that
the gradient flows through the optical model, allowing the network to refine
its depth / index predictions to match observed flow.

Equation (small-angle):
    Δx ≈ (1 - 1/n) · ∂z/∂x · t
    Δy ≈ (1 - 1/n) · ∂z/∂y · t
where t is medium thickness (we use a learned constant per image).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableOptics(nn.Module):
    """
    Computes apparent background displacement under a refracting surface.

    forward(depth, n_index)
        depth:   (B, 1, H, W) — depth/thickness map (in arbitrary units)
        n_index: (B, 1, H, W) — refractive index map

    returns:
        disp:    (B, 2, H, W) — apparent (dx, dy) at each pixel
    """

    def __init__(self, thickness: float = 1.0):
        super().__init__()
        # Learnable global scaling
        self.t = nn.Parameter(torch.tensor(float(thickness)))

    def forward(self, depth: torch.Tensor, n_index: torch.Tensor) -> torch.Tensor:
        # Sobel gradient of depth
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sy = sx.transpose(-1, -2)
        gx = F.conv2d(depth, sx, padding=1)
        gy = F.conv2d(depth, sy, padding=1)

        coeff = (1 - 1.0 / n_index.clamp(min=1.0)) * self.t
        dx = coeff * gx
        dy = coeff * gy
        return torch.cat([dx, dy], dim=1)
