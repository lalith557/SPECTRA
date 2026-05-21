"""
spectra/train/losses.py
SPECTRA multi-task loss: segmentation + material classification + boundary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs   = torch.sigmoid(logits)
        flat_p  = probs.view(-1)
        flat_t  = targets.float().view(-1)
        inter   = (flat_p * flat_t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)


class BoundaryLoss(nn.Module):
    """
    Upweights pixels near mask boundaries using a distance-transform-based weight map.
    Transparent objects are defined by their edges — this loss focuses training on them.
    """

    def __init__(self, theta: float = 10.0):
        super().__init__()
        self.theta = theta

    def _get_boundary_weight(self, mask: Tensor) -> Tensor:
        """
        Args:
            mask: (B, H, W) binary GT mask
        Returns:
            weights: (B, H, W) higher near boundaries
        """
        mask_4d = mask.float().unsqueeze(1)   # (B, 1, H, W)
        kernel  = torch.ones(1, 1, 3, 3, device=mask.device) / 9.0

        # Erode and dilate to get boundary band
        eroded  = F.conv2d(mask_4d, kernel, padding=1)
        dilated = F.conv2d(1 - mask_4d, kernel, padding=1)

        inner_boundary = (eroded < 1.0).float() * mask_4d
        outer_boundary = (dilated < 1.0).float() * (1 - mask_4d)

        boundary = (inner_boundary + outer_boundary).squeeze(1).clamp(0, 1)  # (B, H, W)
        weights  = 1.0 + self.theta * boundary
        return weights

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  (B, 1, H, W)
            targets: (B, H, W)
        """
        weights  = self._get_boundary_weight(targets)               # (B, H, W)
        bce      = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets.float(), reduction="none"
        )   # (B, H, W)
        return (bce * weights).mean()


# ---------------------------------------------------------------------------
# Combined SPECTRA loss
# ---------------------------------------------------------------------------

def gradient_smoothness_loss(x: Tensor, mask: Tensor) -> Tensor:
    """
    Penalises high gradients in x inside the object mask.
    Encourages physically smooth optical fields in transparent regions.
    """
    if x.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest").squeeze(1)

    mask_4d = mask.unsqueeze(1).float()
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    
    # Apply mask to gradients
    loss = (dx * mask_4d[:, :, :, :-1]).mean() + (dy * mask_4d[:, :, :-1, :]).mean()
    return loss


# ---------------------------------------------------------------------------
# Combined SPECTRA loss
# ---------------------------------------------------------------------------

class SPECTRALoss(nn.Module):
    """
    Multi-task loss:
        L = λ_seg * L_seg + λ_bnd * L_bnd + λ_consist * L_consist + λ_refl * L_refl

    (L_mat is disabled temporarily for causal transition)
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_mat: float = 0.0,  # Disabled
        lambda_bnd: float = 0.5,
        lambda_consist: float = 0.1,
        lambda_refl: float = 0.1,
        aux_var_weight: float = 0.1,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_mat = lambda_mat
        self.lambda_bnd = lambda_bnd
        self.lambda_consist = lambda_consist
        self.lambda_refl = lambda_refl
        self.aux_var_weight = aux_var_weight

        self.dice_loss = DiceLoss()
        self.bnd_loss  = BoundaryLoss()

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets:     Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        seg_logits = predictions["seg_logits"]   # (B, 1, H, W)
        mat_logits = predictions["mat_logits"]   # (B, C, H, W)
        mask       = targets["mask"]             # (B, H, W)
        material   = targets["material"]         # (B,)

        # --- 1. Segmentation loss (Dice + BCE) ---
        bce_seg = F.binary_cross_entropy_with_logits(
            seg_logits.squeeze(1), mask.float()
        )
        dice_seg = self.dice_loss(seg_logits.squeeze(1), mask)
        L_seg    = bce_seg + dice_seg

        # --- 2. Material classification loss (LOGGED BUT NOT IN TOTAL) ---
        mat_global = mat_logits.mean(dim=[-2, -1])
        L_mat = F.cross_entropy(mat_global, material)

        # --- 3. Boundary loss ---
        L_bnd = self.bnd_loss(seg_logits, mask)

        # --- 4. Transparency Consistency Loss (NEW) ---
        L_consist = torch.tensor(0.0, device=seg_logits.device)
        if "ofcv_map" in predictions and predictions["ofcv_map"] is not None:
            L_consist = gradient_smoothness_loss(predictions["ofcv_map"], mask)

        # --- 5. Reflection Suppression Loss (NEW) ---
        # Penalise predicted foreground in regions with high texture but NO flow violation
        # (Using a simple heuristic: prob * (1 - ofcv) * texture_score)
        L_refl = torch.tensor(0.0, device=seg_logits.device)
        if "ofcv_map" in predictions and predictions["ofcv_map"] is not None:
            prob = torch.sigmoid(seg_logits)
            ofcv = F.interpolate(predictions["ofcv_map"], size=prob.shape[-2:], mode="bilinear")
            # Heuristic: high confidence + low ofcv = likely reflection false positive
            L_refl = (prob * (1.0 - ofcv)).mean()

        # --- Total ---
        # We explicitly remove L_mat to force the model to rely on causal physics signals.
        total = (
            self.lambda_seg * L_seg
            + self.lambda_bnd * L_bnd
            + self.lambda_consist * L_consist
            + self.lambda_refl * L_refl
        )

        # --- OFCV Variance regularisation ---
        L_var = torch.tensor(0.0, device=seg_logits.device)
        if "ofcv_map" in predictions and predictions["ofcv_map"] is not None:
            ofcv_map = predictions["ofcv_map"]
            var = ofcv_map.var(dim=[1, 2, 3]).mean()
            L_var = torch.exp(-var)
            total = total + self.aux_var_weight * L_var

        return {
            "total": total,
            "seg":   L_seg,
            "mat":   L_mat,
            "bnd":   L_bnd,
            "consist": L_consist,
            "refl":  L_refl,
            "var":   L_var,
        }
