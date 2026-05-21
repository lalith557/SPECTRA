"""
spectra/flow/raft_wrapper.py + warp_utils.py (combined)
RAFT optical flow inference wrapper and backward warp utilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Backward warp (differentiable)
# ---------------------------------------------------------------------------

def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp img at time t+1 back to time t using the predicted forward flow.

    Args:
        img:  (B, C, H, W) — frame at t+1
        flow: (B, 2, H, W) — flow from t to t+1 (dx, dy in pixels)

    Returns:
        warped: (B, C, H, W) — img warped to align with frame t
    """
    B, C, H, W = img.shape
    device = img.device

    # Build normalised grid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    grid = grid.expand(B, -1, -1, -1)                          # (B, 2, H, W)

    # Convert pixel flow to normalised coordinates
    flow_norm = flow.clone()
    flow_norm[:, 0, :, :] = flow_norm[:, 0, :, :] / (W - 1) * 2.0  # dx
    flow_norm[:, 1, :, :] = flow_norm[:, 1, :, :] / (H - 1) * 2.0  # dy

    # Displaced sampling grid
    sample_grid = (grid + flow_norm).permute(0, 2, 3, 1)  # (B, H, W, 2)

    warped = F.grid_sample(
        img,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped


def compute_warp_residual(
    img_t0: torch.Tensor,
    img_t1: torch.Tensor,
    flow:   torch.Tensor,
    eps:    float = 1e-8,
) -> torch.Tensor:
    """
    Compute the per-pixel photometric residual between the
    backward-warped t+1 frame and the actual t frame.
    High residual = Lambertian brightness constancy violated
                  = potential transparent/reflective region.

    Args:
        img_t0: (B, 3, H, W)  frame at time t
        img_t1: (B, 3, H, W)  frame at time t+1
        flow:   (B, 2, H, W)  flow t→t+1

    Returns:
        residual: (B, 1, H, W) in [0, ∞)
    """
    warped_t1 = backward_warp(img_t1, flow)
    diff      = torch.abs(img_t0 - warped_t1)       # (B, 3, H, W)
    residual  = diff.mean(dim=1, keepdim=True)        # (B, 1, H, W)
    return residual


def compute_flow_consistency(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    alpha1:   float = 0.01,
    alpha2:   float = 0.5,
) -> torch.Tensor:
    """
    Forward-backward consistency check.
    Points where fwd+bwd flow disagrees are likely object boundaries
    or regions with complex appearance (transparent objects).

    Returns:
        consistency: (B, 1, H, W) in [0, 1], high = consistent (opaque/static)
    """
    flow_bwd_warped = backward_warp(flow_bwd, flow_fwd)
    diff = flow_fwd + flow_bwd_warped                          # should be ~0 if consistent
    sq   = (diff ** 2).sum(dim=1, keepdim=True)                # (B, 1, H, W)
    mag  = (flow_fwd ** 2).sum(dim=1, keepdim=True) \
         + (flow_bwd_warped ** 2).sum(dim=1, keepdim=True)     # (B, 1, H, W)

    threshold = alpha1 * mag + alpha2
    consistency = (sq < threshold).float()
    return consistency


# ---------------------------------------------------------------------------
# RAFT wrapper
# ---------------------------------------------------------------------------

class RAFTWrapper(nn.Module):
    """
    Thin wrapper around torchvision's RAFT implementation.
    Handles normalisation and provides a clean dict-based interface.

    Note: RAFT expects images in [-1, 1] range.
    Our pipeline uses ImageNet-normalised images, so we un-normalise
    before passing to RAFT.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_name: str = "raft_large", iters: int = 20):
        super().__init__()
        try:
            from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights
            if model_name == "raft_large":
                weights = Raft_Large_Weights.DEFAULT
                self.raft = raft_large(weights=weights).eval()
            else:
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                weights = Raft_Small_Weights.DEFAULT
                self.raft = raft_small(weights=weights).eval()
        except ImportError:
            raise ImportError(
                "torchvision >= 0.16 required for RAFT. "
                "Install with: pip install torchvision>=0.16.0"
            )

        self.iters = iters
        # Freeze RAFT — we only use it as a feature extractor
        for param in self.raft.parameters():
            param.requires_grad_(False)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ImageNet-normalised tensor back to [0, 255] for RAFT."""
        mean = self.IMAGENET_MEAN.to(x.device)
        std  = self.IMAGENET_STD.to(x.device)
        x    = x * std + mean                    # [0, 1]
        return (x * 255.0).clamp(0, 255)         # [0, 255]

    @torch.no_grad()
    def forward(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_t0, img_t1: (B, 3, H, W) ImageNet-normalised

        Returns:
            flow_fwd: (B, 2, H, W)  flow from t0 → t1
            flow_bwd: (B, 2, H, W)  flow from t1 → t0
        """
        t0_255 = self._denorm(img_t0)
        t1_255 = self._denorm(img_t1)

        # RAFT returns a list of flow estimates; take the finest
        flow_fwd_list = self.raft(t0_255, t1_255, num_flow_updates=self.iters)
        flow_bwd_list = self.raft(t1_255, t0_255, num_flow_updates=self.iters)

        flow_fwd = flow_fwd_list[-1]   # (B, 2, H, W)
        flow_bwd = flow_bwd_list[-1]   # (B, 2, H, W)

        return flow_fwd, flow_bwd
