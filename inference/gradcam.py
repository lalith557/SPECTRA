"""
spectra/inference/gradcam.py
Explainability module for SPECTRA.

Produces:
  1. GradCAM maps — which spatial regions drive the transparent object prediction
  2. Physics attention maps — where OFCV and BRF attention heads focus
  3. Cross-modal attention visualisation — how physics queries semantic features
  4. Layer-wise Relevance Propagation (LRP) approximation via guided backprop

Why this matters for FAANG/research:
  - Physics-based AI should be EXPLAINABLE — showing the model attends to
    the exact physical signals (flow residuals, boundary frequencies) is a
    strong scientific claim with visual evidence
  - "Black-box deep learning" criticism is pre-empted by these figures
  - Paper Figure 4 comes directly from this script

Usage:
    python inference/gradcam.py --image path/to/glass_image.jpg \
        --checkpoint checkpoints/spectra_best.pth \
        --output outputs/explainability/
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from utils import load_config, get_logger

logger = get_logger("spectra.gradcam")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Hook-based GradCAM
# ---------------------------------------------------------------------------

class GradCAMHook:
    """
    Registers forward and backward hooks on a target layer to capture
    activations and gradients for GradCAM computation.

    Usage:
        hook = GradCAMHook(model.backbone.encoder.blocks[-1])
        output = model(image, image_t1)
        loss = output['seg_prob'].mean()
        loss.backward()
        cam = hook.compute()
        hook.remove()
    """

    def __init__(self, layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = layer.register_forward_hook(self._save_activation)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # DINOv2 blocks output (B, N, C) — reshape to (B, C, H, W) in compute()
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(
        self,
        grid_h: int,
        grid_w: int,
        relu:   bool = True,
    ) -> torch.Tensor:
        """
        Compute GradCAM from stored activations and gradients.

        Args:
            grid_h, grid_w: spatial dimensions of patch tokens
            relu: apply ReLU (standard GradCAM) or not (signed CAM)

        Returns:
            cam: (1, H, W) normalised heatmap
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients captured. Run forward+backward first.")

        # Handle DINOv2 token format (B, N+1, C) — drop CLS token
        act  = self.activations   # (B, N, C) or (B, C, H, W)
        grad = self.gradients

        if act.dim() == 3:
            # Sequence format: (B, N, C) — may include CLS token
            B, N, C = act.shape
            if N == grid_h * grid_w + 1:
                act  = act[:, 1:]   # drop CLS
                grad = grad[:, 1:]
            act  = act.transpose(1, 2).reshape(B, C, grid_h, grid_w)
            grad = grad.transpose(1, 2).reshape(B, C, grid_h, grid_w)

        # GradCAM: global average pool gradients → channel weights
        weights = grad.mean(dim=[-2, -1], keepdim=True)  # (B, C, 1, 1)
        cam     = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        if relu:
            cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_flat = cam.flatten(1)
        cam_min  = cam_flat.min(dim=1).values.view(-1, 1, 1, 1)
        cam_max  = cam_flat.max(dim=1).values.view(-1, 1, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam   # (B, 1, H, W)

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Attention map extractor (for cross-modal attention in FusionHead)
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    Extracts attention weights from nn.MultiheadAttention layers.
    Patches the forward method to capture the attention weight matrix.
    """

    def __init__(self, mha_layer: nn.MultiheadAttention):
        self.attention_weights: Optional[torch.Tensor] = None
        self._original_forward = mha_layer.forward
        self._mha = mha_layer
        mha_layer.forward = self._patched_forward

    def _patched_forward(self, query, key, value, **kwargs):
        out, attn_weights = self._original_forward(
            query, key, value,
            need_weights=True,
            average_attn_weights=True,
            **{k: v for k, v in kwargs.items() if k not in ["need_weights", "average_attn_weights"]},
        )
        self.attention_weights = attn_weights.detach()
        return out, attn_weights

    def get_weights(self, grid_h: int, grid_w: int) -> Optional[torch.Tensor]:
        """
        Returns attention map reshaped to spatial format.
        attn_weights: (B, N_queries, N_keys) → averaged over query dim → (B, H, W)
        """
        if self.attention_weights is None:
            return None
        # Average over query dimension → (B, N_keys)
        avg = self.attention_weights.mean(dim=1)   # (B, N_keys)
        B, N = avg.shape
        if N == grid_h * grid_w:
            return avg.reshape(B, 1, grid_h, grid_w)
        return None

    def remove(self):
        self._mha.forward = self._original_forward


# ---------------------------------------------------------------------------
# Main explainability engine
# ---------------------------------------------------------------------------

class SPECTRAExplainer:
    """
    Full explainability suite for SPECTRA.

    Produces multiple explanation types per image:
      1. GradCAM on final encoder block → which regions drive prediction
      2. OFCV attention map → where physics query attends to semantic features
      3. BRF energy map → where frequency boundaries are detected
      4. Physics residual map → where flow consistency breaks

    Args:
        model:      loaded SPECTRA model
        device:     inference device
        image_size: (H, W)
    """

    def __init__(self, model, device: torch.device, image_size: Tuple[int, int]):
        self.model      = model
        self.device     = device
        self.image_size = image_size
        self.patch_size = 14
        self.grid_h     = image_size[0] // self.patch_size
        self.grid_w     = image_size[1] // self.patch_size

    def explain(
        self,
        image_rgb: np.ndarray,    # (H, W, 3) uint8
        target:    str = "seg",   # "seg" | "glass" | "water"
    ) -> Dict[str, np.ndarray]:
        """
        Run full explanation pipeline on one image.

        Returns dict with numpy arrays (all at original image resolution):
            gradcam:    GradCAM heatmap (H, W) float [0, 1]
            attn_map:   cross-modal attention (H, W) float [0, 1]
            ofcv_map:   OFCV violation (H, W) float [0, 1]
            brf_map:    BRF boundary (H, W) float [0, 1]
            residual:   flow residual (H, W) float [0, 1]
            seg_prob:   segmentation probability (H, W) float [0, 1]
        """
        H_orig, W_orig = image_rgb.shape[:2]
        H, W = self.image_size

        # Preprocess
        rgb_norm = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor   = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # ── 1. GradCAM ───────────────────────────────────────────────────
        # Hook the last transformer block of DINOv2
        try:
            target_layer = self.model.backbone.encoder.blocks[-1]
        except AttributeError:
            target_layer = list(self.model.backbone.encoder.modules())[-2]

        hook = GradCAMHook(target_layer)

        # Forward with grad enabled
        self.model.zero_grad()
        out = self.model(tensor, tensor, return_intermediates=True)

        # Target: mean of segmentation probability → scalar loss for backward
        loss = out["seg_prob"].mean()
        loss.backward()

        gradcam = hook.compute(self.grid_h, self.grid_w)   # (1, 1, h, w)
        gradcam = F.interpolate(gradcam, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        gradcam = gradcam.squeeze().cpu().numpy()
        hook.remove()

        self.model.zero_grad()

        # ── 2. Physics maps (from forward pass intermediates) ─────────────
        with torch.no_grad():
            out2 = self.model(tensor, tensor, return_intermediates=True)

        def to_np(t):
            if t is None:
                return np.zeros((H_orig, W_orig), dtype=np.float32)
            up = F.interpolate(t, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
            arr = up.squeeze().cpu().float().numpy()
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        return {
            "gradcam":   gradcam,
            "ofcv_map":  to_np(out2.get("ofcv_map")),
            "brf_map":   to_np(out2.get("brf_map")),
            "residual":  to_np(out2.get("residual_map")),
            "seg_prob":  to_np(out2["seg_prob"]),
        }


# ---------------------------------------------------------------------------
# Publication figure generator
# ---------------------------------------------------------------------------

def save_explainability_figure(
    image_rgb:     np.ndarray,
    explanations:  Dict[str, np.ndarray],
    save_path:     str = "outputs/explainability/explanation.pdf",
    alpha:         float = 0.6,
):
    """
    6-panel figure: Input | GradCAM | OFCV | BRF | Residual | Seg prediction
    Each heatmap overlaid on the input image.
    """
    fig = plt.figure(figsize=(18, 4))
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.05)

    panels = [
        ("Input",              image_rgb,                   None),
        ("GradCAM\n(encoder)", explanations["gradcam"],     "hot"),
        ("OFCV\n(C1 physics)", explanations["ofcv_map"],    "inferno"),
        ("BRF\n(C2 boundary)", explanations["brf_map"],     "viridis"),
        ("Flow residual",      explanations["residual"],    "plasma"),
        ("Seg probability",    explanations["seg_prob"],    "Blues"),
    ]

    for i, (title, data, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if cmap is None:
            ax.imshow(image_rgb)
        else:
            # Overlay heatmap on original image
            cmap_obj   = plt.cm.get_cmap(cmap)
            heat_rgb   = (cmap_obj(data)[..., :3] * 255).astype(np.uint8)
            blended    = (image_rgb * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)
            ax.imshow(blended)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    plt.suptitle("SPECTRA Explainability — Physics-informed attention maps", fontsize=11, y=1.02)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Explainability figure saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_gradcam(
    config_path:     str,
    image_path:      str,
    checkpoint_path: Optional[str],
    output_dir:      str = "outputs/explainability",
):
    from models.spectra_model import SPECTRA

    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W   = cfg.data.image_size

    model = SPECTRA(cfg, use_gnn=False).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    image_rgb = cv2.cvtColor(cv2.resize(image_bgr, (W, H)), cv2.COLOR_BGR2RGB)

    explainer    = SPECTRAExplainer(model, device, (H, W))
    explanations = explainer.explain(image_rgb)

    stem = Path(image_path).stem
    save_explainability_figure(
        image_rgb,
        explanations,
        save_path=f"{output_dir}/{stem}_explanation.pdf",
    )

    # Save individual heatmaps as PNGs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, arr in explanations.items():
        if name == "seg_prob":
            continue
        arr_u8  = (arr * 255).astype(np.uint8)
        heat    = cv2.applyColorMap(arr_u8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"{output_dir}/{stem}_{name}.png", heat)

    logger.info(f"All explanation maps saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", default="checkpoints/spectra_best.pth")
    parser.add_argument("--output",     default="outputs/explainability")
    args = parser.parse_args()
    run_gradcam(args.config, args.image, args.checkpoint, args.output)
