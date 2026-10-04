"""
spectra/modules/neural_physics_prior.py

A neural physics *prior*: a small generative head that, given DINOv2 features,
predicts (depth, refractive_index, surface_orientation) jointly. Combined with
the differentiable optics layer, these predictions induce an apparent flow that
the loss compares against the observed RAFT flow.

The training signal is unsupervised (no depth/index labels needed!):
    L_prior = || flow_observed - flow_predicted ||_1  (within transparent mask)
Pretrained on the synthetic refraction dataset used in pretrain/.

This is the differentiable-physics replacement for the hand-crafted OFCV cue —
a research-level contribution worth a section in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .differentiable_optics import DifferentiableOptics


class NeuralPhysicsPrior(nn.Module):
    """
    Predicts (depth, n_index, orientation) from DINOv2 features, then
    uses differentiable optics to produce an *expected* flow field which
    can be compared to observed RAFT flow as a self-supervised signal.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden:   int = 128,
        n_min:    float = 1.0,
        n_max:    float = 2.0,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4, 1),   # depth (1) + n (1) + orientation (2)
        )
        self.optics = DifferentiableOptics()
        self.n_min  = n_min
        self.n_max  = n_max

    def forward(self, patch_tokens: torch.Tensor, img_hw):
        H, W = img_hw
        out = self.head(patch_tokens)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        depth_logits, n_logits = out[:, 0:1], out[:, 1:2]
        ori = out[:, 2:4]

        depth = F.softplus(depth_logits)                  # >= 0
        n_norm = torch.sigmoid(n_logits)
        n_index = self.n_min + (self.n_max - self.n_min) * n_norm

        disp = self.optics(depth, n_index)                # (B, 2, H, W)

        return {
            "depth":     depth,
            "n_index":   n_index,
            "orientation": ori,
            "predicted_flow": disp,
        }

    @staticmethod
    def prior_loss(predicted_flow: torch.Tensor,
                   observed_flow:  torch.Tensor,
                   mask:           torch.Tensor) -> torch.Tensor:
        """
        L_prior = mean | f_obs - f_pred | over the transparent mask.
        If mask is None, fall back to global mean (unconditional pretraining).
        """
        err = (predicted_flow - observed_flow).abs().mean(dim=1, keepdim=True)
        if mask is not None:
            mask = mask.float()
            denom = mask.sum().clamp(min=1)
            return (err * mask).sum() / denom
        return err.mean()
