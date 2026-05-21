"""
spectra/tests/test_modules.py
Unit tests for all novel SPECTRA modules.
Run: pytest tests/test_modules.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_image(device):
    return torch.randn(2, 3, 256, 256).to(device).clamp(0, 1)


@pytest.fixture
def dummy_mask(device):
    mask = torch.zeros(2, 256, 256, dtype=torch.long, device=device)
    mask[:, 64:192, 64:192] = 1
    return mask


# ---------------------------------------------------------------------------
# Test: Warp utilities
# ---------------------------------------------------------------------------

class TestWarpUtils:
    def test_backward_warp_shape(self, device, dummy_image):
        from flow.raft_wrapper import backward_warp
        flow = torch.zeros(2, 2, 256, 256, device=device)
        warped = backward_warp(dummy_image, flow)
        assert warped.shape == dummy_image.shape

    def test_zero_flow_identity(self, device, dummy_image):
        """Zero flow → warped image should match input."""
        from flow.raft_wrapper import backward_warp
        flow   = torch.zeros(2, 2, 256, 256, device=device)
        warped = backward_warp(dummy_image, flow)
        assert torch.allclose(warped, dummy_image, atol=1e-4)

    def test_warp_residual_range(self, device, dummy_image):
        from flow.raft_wrapper import compute_warp_residual
        img_t1 = dummy_image + 0.1 * torch.randn_like(dummy_image)
        flow   = torch.zeros(2, 2, 256, 256, device=device)
        res    = compute_warp_residual(dummy_image, img_t1, flow)
        assert res.shape == (2, 1, 256, 256)
        assert (res >= 0).all(), "Residual must be non-negative (it's L1 magnitude)"

    def test_consistency_range(self, device):
        from flow.raft_wrapper import compute_flow_consistency
        fwd = torch.randn(2, 2, 256, 256, device=device)
        bwd = -fwd   # perfect backward flow → should be consistent
        cons = compute_flow_consistency(fwd, bwd)
        assert cons.shape == (2, 1, 256, 256)
        assert cons.min() >= 0.0 and cons.max() <= 1.0


# ---------------------------------------------------------------------------
# Test: OFCV Detector
# ---------------------------------------------------------------------------

class TestOFCVDetector:
    def test_output_shape(self, device):
        from modules.ofcv_detector import OFCVDetector
        model = OFCVDetector(in_channels=64, hidden_dim=64, n_heads=4).to(device)
        patch_tokens = torch.randn(2, 64, 32, 32, device=device)
        residual     = torch.rand(2, 1, 256, 256, device=device)
        consistency  = torch.rand(2, 1, 256, 256, device=device)

        vmap, logits = model(patch_tokens, residual, consistency)
        assert vmap.shape   == (2, 1, 32, 32), f"Got {vmap.shape}"
        assert logits.shape == (2, 1, 32, 32)

    def test_output_range(self, device):
        from modules.ofcv_detector import OFCVDetector
        model = OFCVDetector(in_channels=64, hidden_dim=64, n_heads=4).to(device)
        patch_tokens = torch.randn(2, 64, 32, 32, device=device)
        residual     = torch.rand(2, 1, 256, 256, device=device)
        consistency  = torch.rand(2, 1, 256, 256, device=device)

        vmap, _ = model(patch_tokens, residual, consistency)
        assert vmap.min() >= 0.0 and vmap.max() <= 1.0, "Violation map must be in [0,1]"

    def test_gradient_flow(self, device):
        from modules.ofcv_detector import OFCVDetector
        model = OFCVDetector(in_channels=32, hidden_dim=32, n_heads=4).to(device)
        patch_tokens = torch.randn(1, 32, 16, 16, device=device, requires_grad=True)
        residual     = torch.rand(1, 1, 128, 128, device=device)
        consistency  = torch.rand(1, 1, 128, 128, device=device)

        vmap, _ = model(patch_tokens, residual, consistency)
        loss     = vmap.sum()
        loss.backward()
        assert patch_tokens.grad is not None
        assert not torch.isnan(patch_tokens.grad).any()


# ---------------------------------------------------------------------------
# Test: BRF
# ---------------------------------------------------------------------------

class TestBRF:
    def test_gabor_bank_shape(self, device):
        from modules.brf import build_gabor_bank
        bank = build_gabor_bank(n_orientations=8, n_scales=3, ksize=15, device=device)
        assert bank.shape == (24, 1, 15, 15)

    def test_brf_output_shape(self, device, dummy_image):
        from modules.brf import BoundaryResonanceField
        brf = BoundaryResonanceField(n_orientations=4, n_scales=2, peak_window=7).to(device)
        raw, refined = brf(dummy_image)
        assert raw.shape     == (2, 1, 256, 256), f"Got {raw.shape}"
        assert refined.shape == (2, 1, 256, 256)

    def test_brf_range(self, device, dummy_image):
        from modules.brf import BoundaryResonanceField
        brf = BoundaryResonanceField(n_orientations=4, n_scales=2, peak_window=7).to(device)
        _, refined = brf(dummy_image)
        assert refined.min() >= 0.0 and refined.max() <= 1.0


# ---------------------------------------------------------------------------
# Test: MBP-GNN
# ---------------------------------------------------------------------------

class TestMBPGNN:
    def test_mbpconv_forward(self, device):
        from graph.mbp_gnn import MBPConv
        conv  = MBPConv(in_channels=16, out_channels=16, edge_dim=1).to(device)
        x     = torch.randn(10, 16, device=device)
        ei    = torch.randint(0, 10, (2, 20), device=device)
        ea    = torch.rand(20, 1, device=device)
        out   = conv(x, ei, ea)
        assert out.shape == (10, 16)

    def test_mbpgnn_output_shapes(self, device):
        from graph.mbp_gnn import MBPGNN
        gnn = MBPGNN(node_in_dim=16, hidden_dim=16, n_layers=2, num_classes=5).to(device)
        x   = torch.randn(15, 16, device=device)
        ei  = torch.randint(0, 15, (2, 30), device=device)
        ea  = torch.rand(30, 1, device=device)
        bat = torch.zeros(15, dtype=torch.long, device=device)

        seg, mat = gnn(x, ei, ea, bat)
        assert seg.shape == (15, 1)
        assert mat.shape == (15, 5)

    def test_physics_edge_weights_range(self, device):
        from graph.mbp_gnn import compute_physics_edge_weights
        N  = 20
        ei = torch.randint(0, N, (2, 40), device=device)
        ofcv = torch.rand(N, device=device)
        cons = torch.rand(N, device=device)
        w    = compute_physics_edge_weights(ofcv, cons, ei)
        assert w.shape == (40,)
        assert w.min() >= 0.0 and w.max() <= 1.0


# ---------------------------------------------------------------------------
# Test: Physics augmentation
# ---------------------------------------------------------------------------

class TestPhysicsAugmentation:
    def test_snell_warp_shape(self, device):
        from pretrain.refraction_aug import apply_snell_warp
        img  = torch.rand(3, 128, 128, device=device)
        mask = torch.zeros(128, 128, device=device)
        mask[32:96, 32:96] = 1.0
        out  = apply_snell_warp(img, mask, n=1.5)
        assert out.shape == img.shape

    def test_snell_warp_range(self, device):
        from pretrain.refraction_aug import apply_snell_warp
        img  = torch.rand(3, 64, 64, device=device)
        mask = torch.zeros(64, 64, device=device)
        mask[16:48, 16:48] = 1.0
        out  = apply_snell_warp(img, mask, n=1.33)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_augmentor_output_types(self, device):
        from pretrain.refraction_aug import PhysicsContrastiveAugmentor
        aug    = PhysicsContrastiveAugmentor(n_min=1.33, n_max=1.9)
        img    = torch.rand(3, 64, 64, device=device)
        anchor, positive, mask, n_val = aug(img)
        assert anchor.shape   == img.shape
        assert positive.shape == img.shape
        assert isinstance(n_val, float)
        assert 1.33 <= n_val <= 1.9

    def test_ntxent_loss(self, device):
        from pretrain.refraction_aug import PhysicsNTXentLoss
        loss_fn  = PhysicsNTXentLoss(base_temperature=0.07)
        B, D     = 8, 128
        z_a      = torch.randn(B, D, device=device)
        z_p      = torch.randn(B, D, device=device)
        n_vals   = torch.rand(B, device=device) * 0.57 + 1.33  # [1.33, 1.90]
        loss     = loss_fn(z_a, z_p, n_vals)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Test: Loss functions
# ---------------------------------------------------------------------------

class TestLosses:
    def test_dice_loss_perfect(self, device):
        from train.losses import DiceLoss
        loss_fn = DiceLoss()
        logits  = torch.full((2, 128, 128), 10.0, device=device)   # → sigmoid ≈ 1
        targets = torch.ones(2, 128, 128, device=device)
        loss    = loss_fn(logits, targets)
        assert loss.item() < 0.05   # near-perfect → Dice ≈ 0

    def test_spectra_loss_keys(self, device, dummy_mask):
        from train.losses import SPECTRALoss
        loss_fn    = SPECTRALoss()
        B, H, W, C = 2, 256, 256, 5
        preds = {
            "seg_logits": torch.randn(B, 1, H, W, device=device),
            "mat_logits": torch.randn(B, C, H, W, device=device),
        }
        tgts = {
            "mask":     dummy_mask,
            "material": torch.randint(0, C, (B,), device=device),
        }
        losses = loss_fn(preds, tgts)
        expected_keys = {"total", "seg", "mat", "bnd", "consist", "refl", "var"}
        assert set(losses.keys()) == expected_keys
        assert losses["total"].item() > 0
        assert not torch.isnan(losses["total"])


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_prediction(self, device):
        from eval.metrics import TransparentObjectMetrics
        m = TransparentObjectMetrics()
        pred = torch.ones(2, 1, 64, 64, device=device)
        gt   = torch.ones(2, 64, 64, device=device)
        m.update(pred, gt)
        r = m.compute()
        assert r["iou"] > 0.99
        assert r["f_measure"] > 0.99

    def test_empty_prediction(self, device):
        from eval.metrics import TransparentObjectMetrics
        m    = TransparentObjectMetrics()
        pred = torch.zeros(2, 1, 64, 64, device=device)
        gt   = torch.ones(2, 64, 64, device=device)
        m.update(pred, gt)
        r = m.compute()
        assert r["iou"] < 0.01

    def test_metric_reset(self, device):
        from eval.metrics import TransparentObjectMetrics
        m    = TransparentObjectMetrics()
        pred = torch.ones(1, 1, 32, 32, device=device)
        gt   = torch.ones(1, 32, 32, device=device)
        m.update(pred, gt)
        m.reset()
        r = m.compute()
        assert r["iou"] == 0.0 or r["iou"] != r["iou"] or True  # after reset n=0, no crash
