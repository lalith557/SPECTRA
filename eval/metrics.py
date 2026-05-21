"""
spectra/eval/metrics.py
Standard transparent object detection metrics:
  - IoU (Intersection over Union)
  - F-measure (β=0.3 as used in Trans10K paper)
  - MAE (Mean Absolute Error)
  - BER (Balance Error Rate)
  - Weighted F-measure
"""
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Optional


class TransparentObjectMetrics:
    """
    Stateful metric accumulator.
    Call update() after each batch, then compute() for epoch-level numbers.

    All inputs expected as torch.Tensor on any device.
    """

    def __init__(self, threshold: float = 0.5, beta_sq: float = 0.3):
        self.threshold = threshold
        self.beta_sq   = beta_sq
        self.reset()

    def reset(self):
        self._tp  = 0.0
        self._fp  = 0.0
        self._fn  = 0.0
        self._tn  = 0.0
        self._mae = 0.0
        self._n   = 0

    @torch.no_grad()
    def update(self, pred_prob: Tensor, gt_mask: Tensor):
        """
        Args:
            pred_prob: (B, 1, H, W) or (B, H, W) — predicted probability
            gt_mask:   (B, H, W) — binary ground truth {0, 1}
        """
        if pred_prob.dim() == 4:
            pred_prob = pred_prob.squeeze(1)

        pred_bin = (pred_prob >= self.threshold).float()
        gt       = gt_mask.float()

        B = pred_bin.shape[0]
        for i in range(B):
            p = pred_bin[i].cpu()
            g = gt[i].cpu()
            r = pred_prob[i].cpu()

            self._tp += (p * g).sum().item()
            self._fp += (p * (1 - g)).sum().item()
            self._fn += ((1 - p) * g).sum().item()
            self._tn += ((1 - p) * (1 - g)).sum().item()
            self._mae += (r - g).abs().mean().item()

        self._n += B

    def compute(self) -> Dict[str, float]:
        eps = 1e-8

        precision = self._tp / (self._tp + self._fp + eps)
        recall    = self._tp / (self._tp + self._fn + eps)

        # F-measure with β²=0.3 (precision-weighted, standard in SOD papers)
        f_measure = (1 + self.beta_sq) * precision * recall / (
            self.beta_sq * precision + recall + eps
        )

        # IoU
        iou = self._tp / (self._tp + self._fp + self._fn + eps)

        # MAE
        mae = self._mae / max(self._n, 1)

        # BER (Balance Error Rate)
        pos_err = self._fn / (self._tp + self._fn + eps)
        neg_err = self._fp / (self._fp + self._tn + eps)
        ber     = 0.5 * (pos_err + neg_err)

        return {
            "iou":       round(iou,       4),
            "f_measure": round(f_measure, 4),
            "mae":       round(mae,       4),
            "ber":       round(ber,       4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
        }


# ---------------------------------------------------------------------------
# Standalone functions for quick evaluation
# ---------------------------------------------------------------------------

def compute_iou(pred_bin: Tensor, gt: Tensor, eps: float = 1e-8) -> float:
    pred = pred_bin.float().view(-1)
    gt   = gt.float().view(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + eps)).item()


def compute_mae(pred_prob: Tensor, gt: Tensor) -> float:
    return (pred_prob.float() - gt.float()).abs().mean().item()
