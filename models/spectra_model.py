"""
spectra/models/spectra_model.py
Full SPECTRA model — integrates all 4 novel contributions end-to-end.

Forward pass:
  1. DINOv2 backbone → multi-scale FPN features + patch tokens
  2. RAFT optical flow → flow_fwd, flow_bwd
  3. Warp residual computation → photometric residual map
  4. Flow consistency check → consistency map
  5. OFCV detector → violation map (C1)
  6. BRF → boundary resonance field (C2)
  7. Fusion head → pixel-level predictions
  8. [Optional] Superpixel graph → MBP-GNN refinement (C3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.backbone import DINOv2Backbone
from flow.raft_wrapper import RAFTWrapper, compute_warp_residual, compute_flow_consistency
from modules.ofcv_detector import OFCVDetector
from modules.brf import BoundaryResonanceField
from modules.signal_fusion import FusionHead
from graph.superpixel import SuperpixelGraphBuilder, unproject_to_pixels
from graph.mbp_gnn import MBPGNN


class SPECTRA(nn.Module):
    """
    Full SPECTRA pipeline for transparent object detection.

    Args:
        cfg: Config object (from load_config)
        use_gnn: whether to run MBP-GNN refinement (slightly slower at inference)
    """

    def __init__(self, cfg, use_gnn: bool = True, use_ofcv: bool = True, use_brf: bool = True):
        super().__init__()
        self.cfg     = cfg
        self.use_gnn = use_gnn
        self.use_ofcv = use_ofcv
        self.use_brf = use_brf
        H, W         = cfg.data.image_size

        # ── C0: Backbone ──────────────────────────────────────────────────
        self.backbone = DINOv2Backbone(
            model_name=cfg.model.backbone,
            frozen=cfg.model.backbone_frozen,
            image_size=H,
        )
        embed_dim = cfg.model.embed_dim   # 1024 for ViT-L/14

        # ── C0: Optical flow (frozen, inference only) ─────────────────────
        self.flow_model = RAFTWrapper(
            model_name=cfg.flow.model,
            iters=cfg.flow.iters,
        )

        # ── C1: OFCV Detector ─────────────────────────────────────────────
        self.ofcv = OFCVDetector(
            in_channels=embed_dim,
            hidden_dim=cfg.ofcv.hidden_dim,
            n_heads=cfg.ofcv.n_heads,
            dropout=cfg.ofcv.dropout,
        )

        # ── C2: BRF ───────────────────────────────────────────────────────
        self.brf = BoundaryResonanceField(
            n_orientations=cfg.brf.n_orientations,
            n_scales=cfg.brf.n_scales,
            peak_window=cfg.brf.peak_window,
        )

        # ── Fusion head ───────────────────────────────────────────────────
        self.fusion = FusionHead(
            semantic_dim=embed_dim,
            hidden_dim=256,
            num_classes=cfg.model.num_classes + 1,  # +1 for background
            image_size=(H, W),
        )

        # ── C3: MBP-GNN (optional refinement) ────────────────────────────
        if use_gnn:
            node_in_dim = 256 + 1 + 1 + 3 + 2   # proj_dino + ofcv + brf + rgb + xy
            self.graph_builder = SuperpixelGraphBuilder(
                n_segments=cfg.graph.n_segments,
                compactness=cfg.graph.compactness,
                feat_proj_dim=256,
                dino_dim=embed_dim,
            )
            self.gnn = MBPGNN(
                node_in_dim=node_in_dim,
                hidden_dim=cfg.graph.gnn_hidden,
                n_layers=cfg.graph.gnn_layers,
                num_classes=cfg.model.num_classes + 1,
                dropout=cfg.graph.gnn_dropout,
            )

            # Learnable blend weight: fusion output vs GNN output
            self.gnn_blend = nn.Parameter(torch.tensor(0.3))

    # ─────────────────────────────────────────────────────────────────────

    def _run_flow(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run RAFT and compute residual + consistency maps."""
        flow_fwd, flow_bwd = self.flow_model(img_t0, img_t1)

        residual    = compute_warp_residual(img_t0, img_t1, flow_fwd)      # (B,1,H,W)
        consistency = compute_flow_consistency(flow_fwd, flow_bwd)          # (B,1,H,W)

        return flow_fwd, residual, consistency

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        image:    torch.Tensor,           # (B, 3, H, W) — frame t
        image_t1: torch.Tensor,           # (B, 3, H, W) — frame t+1
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            seg_prob:   (B, 1, H, W)  transparent probability map
            mat_logits: (B, C, H, W)  material class logits
            seg_logits: (B, 1, H, W)  raw seg logits (for loss)
            [if return_intermediates]:
                ofcv_map, brf_map, residual_map, consistency_map
        """
        B, _, H, W = image.shape

        # ── 1. Backbone ───────────────────────────────────────────────────
        backbone_out  = self.backbone(image)
        fpn_features  = backbone_out["features"]     # list of 4 (B,256,h,w)
        patch_tokens  = backbone_out["patch_tokens"] # (B, C, h, w)

        # ── 2. Optical flow + residual ────────────────────────────────────
        # No gradients through RAFT — it's frozen physics infrastructure
        with torch.no_grad():
            flow_fwd, residual, consistency = self._run_flow(image, image_t1)

        # ── 3. C1: OFCV ───────────────────────────────────────────────────
        if self.use_ofcv:
            ofcv_map, ofcv_logits = self.ofcv(
                patch_tokens=patch_tokens,
                residual_map=residual,
                consistency_map=consistency,
            )   # both (B, 1, h, w) and (B, 1, H, W) — h is patch resolution
            
            # Upsample OFCV to full image resolution for downstream modules
            ofcv_full = F.interpolate(ofcv_map, size=(H, W), mode="bilinear", align_corners=False)
        else:
            h_patch, w_patch = patch_tokens.shape[2], patch_tokens.shape[3]
            ofcv_map = torch.zeros(B, 1, h_patch, w_patch, device=image.device)
            ofcv_full = torch.zeros(B, 1, H, W, device=image.device)
            ofcv_logits = None

        # ── 4. C2: BRF ────────────────────────────────────────────────────
        if self.use_brf:
            brf_raw, brf_refined = self.brf(image)   # both (B, 1, H, W)
        else:
            brf_refined = torch.zeros(B, 1, H, W, device=image.device)

        # ── 5. Fusion head ────────────────────────────────────────────────
        fusion_out = self.fusion(
            patch_tokens=patch_tokens,
            fpn_features=fpn_features,
            ofcv_map=ofcv_map,      # patch-resolution gating (B, 1, h, w)
            brf_map=brf_refined,    # full-resolution structural signal
        )
        # fusion_out: {'seg_logits': (B,1,H,W), 'mat_logits': (B,C,H,W), 'seg_prob': (B,1,H,W)}

        # ── 6. C3: MBP-GNN refinement (optional) ─────────────────────────
        if self.use_gnn and self.training is False:
            # GNN runs on CPU-friendly batched graph — only at inference or val
            # (Too slow for every training step; used for final eval and demo)
            gnn_seg = self._gnn_refinement(
                images=image,
                dino_feats=patch_tokens,
                ofcv_maps=ofcv_full,
                brf_maps=brf_refined,
                flow_cons_maps=consistency,
                H=H, W=W,
            )
            # Blend fusion + GNN predictions with learned weight
            alpha = torch.sigmoid(self.gnn_blend)
            final_prob = (1 - alpha) * fusion_out["seg_prob"] + alpha * gnn_seg
        else:
            final_prob = fusion_out["seg_prob"]

        outputs = {
            "seg_prob":   final_prob,
            "seg_logits": fusion_out["seg_logits"],
            "mat_logits": fusion_out["mat_logits"],
        }

        if return_intermediates:
            outputs.update({
                "ofcv_map":       ofcv_full,
                "brf_map":        brf_refined,
                "residual_map":   residual,
                "consistency_map": consistency,
                "ofcv_logits":    ofcv_logits,
            })

        return outputs

    def _gnn_refinement(
        self,
        images, dino_feats, ofcv_maps, brf_maps, flow_cons_maps, H, W
    ) -> torch.Tensor:
        """Run MBP-GNN and return pixel-level seg probability (B, 1, H, W)."""
        B = images.shape[0]
        device = images.device

        batch_graph = self.graph_builder(
            images=images,
            dino_feats=dino_feats,
            ofcv_maps=ofcv_maps,
            brf_maps=brf_maps,
            flow_cons_maps=flow_cons_maps,
        )

        seg_logits_node, _ = self.gnn(
            x=batch_graph.x,
            edge_index=batch_graph.edge_index,
            edge_attr=batch_graph.edge_attr,
            batch=batch_graph.batch,
        )   # (N_total, 1)

        seg_prob_node = torch.sigmoid(seg_logits_node).squeeze(-1)   # (N_total,)

        # Unproject to pixel maps for each image in the batch
        pixel_maps = []
        for i in range(B):
            node_mask = batch_graph.batch == i
            node_probs = seg_prob_node[node_mask]
            segments_i = batch_graph.segments[
                (batch_graph.ptr[i]):(batch_graph.ptr[i+1])
                if hasattr(batch_graph, 'ptr') else slice(None)
            ] if hasattr(batch_graph, 'ptr') else batch_graph.segments

            # Fallback: use stored segment map from graph builder
            # In practice, we store segments per graph in Data.segments
            data_i = batch_graph.get_example(i)
            seg_map = data_i.segments   # (H, W)
            pixel_map = unproject_to_pixels(node_probs, seg_map)  # (H, W)
            pixel_maps.append(pixel_map.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)

        return torch.cat(pixel_maps, dim=0)   # (B, 1, H, W)
