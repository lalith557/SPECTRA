"""
spectra/models/spectra_learned_physics.py

SPECTRA-LP: SPECTRA with the hand-crafted OFCV cue replaced by the
learned-physics stack (LearnedRefractionEstimator + NeuralPhysicsPrior).

Drop-in replacement for the main SPECTRA model, with the same forward/return
signature plus optional extra losses if intermediates are requested.

Usage:
    model = SPECTRA_LP(cfg, use_gnn=False)
    out   = model(image, image_t1, return_intermediates=True)
    # out adds: 'learned_disp', 'learned_n', 'predicted_flow'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectra_model import SPECTRA
from modules.learned_refraction import LearnedRefractionEstimator
from modules.neural_physics_prior import NeuralPhysicsPrior


class SPECTRA_LP(SPECTRA):
    def __init__(self, cfg, use_gnn: bool = False, use_brf: bool = True):
        # Build base SPECTRA without OFCV — we replace it
        super().__init__(cfg, use_gnn=use_gnn, use_ofcv=False, use_brf=use_brf)
        self.learned_refraction = LearnedRefractionEstimator(
            feat_dim=cfg.model.embed_dim,
            mid=64, max_disp=0.05, learn_n=True,
            n_min=cfg.pretrain.get("n_min", 1.0),
            n_max=cfg.pretrain.get("n_max", 2.0),
        )
        self.neural_physics_prior = NeuralPhysicsPrior(
            feat_dim=cfg.model.embed_dim, hidden=128,
            n_min=cfg.pretrain.get("n_min", 1.0),
            n_max=cfg.pretrain.get("n_max", 2.0),
        )

    def forward(self, image, image_t1, return_intermediates: bool = False):
        B, _, H, W = image.shape
        backbone_out  = self.backbone(image)
        patch_tokens  = backbone_out["patch_tokens"]
        fpn_features  = backbone_out["features"]

        with torch.no_grad():
            flow_fwd, residual, consistency = self._run_flow(image, image_t1)

        # Learned refraction cue replaces OFCV
        refr = self.learned_refraction(
            patch_tokens, flow_fwd, residual, image, image_t1
        )
        learned_cue = refr["transparency_cue"]   # (B, 1, H, W)

        # Downsample to patch resolution for fusion
        h, w = patch_tokens.shape[-2:]
        learned_cue_small = F.interpolate(
            learned_cue, size=(h, w), mode="bilinear", align_corners=False
        )

        if self.use_brf:
            brf_raw, brf_refined = self.brf(image)
        else:
            brf_refined = torch.zeros(B, 1, H, W, device=image.device)

        fusion_out = self.fusion(
            patch_tokens=patch_tokens, fpn_features=fpn_features,
            ofcv_map=learned_cue_small,    # learned cue takes OFCV slot
            brf_map=brf_refined,
        )

        # Neural physics prior runs separately — used for auxiliary loss
        nphys = self.neural_physics_prior(patch_tokens, img_hw=(H, W))

        outputs = {
            "seg_prob":   fusion_out["seg_prob"],
            "seg_logits": fusion_out["seg_logits"],
            "mat_logits": fusion_out["mat_logits"],
            "learned_cue": learned_cue,
        }
        if return_intermediates:
            outputs.update({
                "learned_disp":    refr["disp"],
                "learned_n":       refr["n_index"],
                "predicted_flow":  nphys["predicted_flow"],
                "predicted_depth": nphys["depth"],
                "residual_map":    residual,
                "consistency_map": consistency,
                "brf_map":         brf_refined,
                "flow_fwd":        flow_fwd,
            })
        return outputs

    def physics_prior_loss(self, predicted_flow, observed_flow, mask=None):
        return self.neural_physics_prior.prior_loss(predicted_flow, observed_flow, mask)
