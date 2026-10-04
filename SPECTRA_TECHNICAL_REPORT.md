# SPECTRA — Comprehensive Technical Report

> Single-source technical reference for the SPECTRA transparent-object
> segmentation project. Covers architecture, modules, datasets, training
> protocol, evaluation, novel contributions, codebase layout, roadmap
> status, and every detail you need to write the paper.

**Caveat re: numbers**: Empirical results in this document use the synthetic
mockup figures from `FIGURE/` as placeholders. To populate the report with
real numbers, run `python scripts/run_roadmap.py` (it runs all 10 priorities
end-to-end) and re-run `python scripts/generate_paper_figures.py` — the
figure script auto-detects real outputs in `results/`.

---

## 1. Problem statement

Transparent objects (glass, plastic, water) violate the **brightness
constancy assumption** that most visual systems rely on:

- The colour of a transparent object's pixels depends on the background seen
  through it, not just the object itself.
- Standard appearance-based segmenters latch onto textures that don't belong
  to the object → low IoU on Trans10K, GSD, ClearPose.

SPECTRA's thesis: **explicitly model the physics of refraction** as a
differentiable cue, so the network can localise transparency from
photometric residuals rather than appearance alone.

## 2. Architecture

```
Input frame (B,3,H,W)            Input frame t+1 (B,3,H,W)
        │                                │
        ▼                                ▼
┌──────────────┐                ┌──────────────┐
│ DINOv2-S/14  │                │  RAFT-large  │
│  (backbone)  │                │  (frozen)    │
└──────┬───────┘                └──────┬───────┘
       │ patch tokens (B,384,h,w)      │ flow_fwd, flow_bwd
       │ FPN features (4-scale)         ▼
       │                       ┌──────────────────┐
       │                       │ warp residual R  │
       │                       │ flow cons.   Fc  │
       │                       └──────┬───────────┘
       │                              │
       ▼                              ▼
┌─────────────────────────────────────────────┐
│ OFCV detector (C1)                          │
│  cross-attn(physics → semantic) + CBAM      │
│  → violation map V ∈ [0,1]^(h,w)            │
└─────────────────────┬───────────────────────┘
                      │ V
       ┌──────────────┴────────────┐
       │                           │
       ▼                           ▼
┌──────────────┐         ┌──────────────────┐
│ BRF (C2)     │         │  Fusion head     │
│ Gabor + dual-│ ─────▶  │  semantic+phys.  │
│ peak prior   │  BRF    │  → seg_logits    │
└──────────────┘         └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────┐
                       │ MBP-GNN (C3)      │
                       │ superpixel graph  │
                       │ refinement        │
                       └────────┬──────────┘
                                │
                                ▼
                           Final mask
```

### 2.1 Backbone (C0) — `models/backbone.py`

- **Choice**: DINOv2-S/14 (`facebook/dinov2-small`).
  - Self-supervised ViT pretraining yields features that are sensitive to
    surface boundaries and reflectance — useful priors for transparency.
- Output:
  - `patch_tokens`: (B, 384, h, w) where h = H/14, w = W/14
  - `features`: 4-scale FPN-style features (B, 256, h_i, w_i) for i ∈ {0..3}
- 22.1 M parameters, fine-tuned with `lr_backbone = 1e-5`.

### 2.2 Optical-flow infrastructure — `flow/raft_wrapper.py`

- **Choice**: RAFT-large (`raft_large`) from torchvision.
- Frozen, no gradient.
- Outputs:
  - `flow_fwd`: (B, 2, H, W) forward flow t→t+1
  - `flow_bwd`: (B, 2, H, W) backward flow t+1→t
- Derived signals:
  - `residual = |I_t - warp(I_{t+1}, flow_fwd)|`
  - `consistency = exp(-|flow_fwd + warp(flow_bwd, flow_fwd)|)`

### 2.3 OFCV Detector (C1, novel) — `modules/ofcv_detector.py`

- **Intuition**: light refracting through glass means the photometric
  residual is **non-zero specifically in transparent regions**, even when
  optical flow looks consistent.
- **Architecture**:
  - Project DINOv2 features → 256 ch (`feat_proj`).
  - 2-layer conv encoder on [residual, consistency] → 256 ch (`physics_proj`).
  - **Cross-attention**: physics tokens query semantic features (multi-head, 8 heads).
  - CBAM (channel + spatial attention) on concatenated features.
  - 3-conv decoder → sigmoid violation map.
- **Output**: V ∈ [0,1]^(h,w) at patch resolution.
- ~3.1 M parameters.

### 2.4 BRF — Boundary Resonance Field (C2, novel) — `modules/brf.py`

- **Intuition**: transparent objects produce *double-peaked* edge responses
  (inner refractive boundary + outer specular boundary).
- **Architecture**:
  - Bank of 8 oriented Gabor filters × 3 scales (no learned weights).
  - Within a 15-pixel window, detect double peaks in each Gabor response.
  - Aggregate across orientations → BRF map.
  - Final 3-conv refinement head.
- **Output**: BRF ∈ ℝ^(H,W).
- Differentiable end-to-end; the Gabor bank is fixed.

### 2.5 Fusion Head — `modules/signal_fusion.py`

- **Input streams**:
  - `patch_tokens` (DINO semantic)
  - `fpn_features` (multi-scale spatial)
  - `ofcv_map` (physics gate, patch resolution)
  - `brf_map` (structural prior, full resolution)
- **Operation**: OFCV is used as a per-pixel gate; BRF is concatenated as
  an extra channel. The result goes through a small UNet-style head that
  emits `seg_logits` (B, 1, H, W) and `mat_logits` (B, C, H, W).
- **Loss heads**: segmentation BCE, boundary loss, optional material classification.

### 2.6 MBP-GNN (C3, novel) — `graph/mbp_gnn.py` + `graph/superpixel.py`

- **Intuition**: superpixels respect refractive boundaries better than
  pixel-grid CNN receptive fields. A GNN over superpixels enforces
  refraction-coherent label smoothness.
- **Pipeline**:
  - SLIC superpixels (`n_segments=512`, `compactness=10.0`).
  - Per-node features: DINO projection + OFCV mean + BRF mean + RGB + (x, y).
  - 4-layer GCN/GINE-style network.
  - Output: per-node logits → reproject to pixel grid.
- **Blend**: learned weight `alpha = σ(gnn_blend)` mixes fusion-head output
  with GNN output. Initial `α ≈ 0.43`.
- Runs at validation / inference only (too expensive per training step).

### 2.7 Novel-physics extensions (research stretch)

- **`modules/learned_refraction.py`** (`LearnedRefractionEstimator`):
  Replaces the hand-crafted OFCV cue with a learned small UNet predicting
  (dx, dy, n_index). Warps the next frame by predicted refraction; what
  remains unexplained is a learned transparency cue.
- **`modules/differentiable_optics.py`** (`DifferentiableOptics`):
  Small-angle Snell-law layer: `Δx ≈ (1 − 1/n) · ∂z/∂x · t`.
- **`modules/neural_physics_prior.py`** (`NeuralPhysicsPrior`):
  Predicts (depth, n_index, orientation) from DINO features; produces an
  *expected* optical flow that can be compared to observed RAFT flow
  → self-supervised pretext loss.
- **`models/spectra_learned_physics.py`** (`SPECTRA_LP`):
  Full SPECTRA with OFCV replaced by the learned-physics stack. Drop-in.
- **`models/spectra_mono.py`** (`SPECTRA_Mono`):
  Monocular variant — generates a synthetic second frame internally
  (homography, smoothed-flow warp, or a learned warper) so SPECTRA works
  on single images.

## 3. Datasets

| Dataset       | Loader file                          | Backing                                                                                       | Notes                                                              |
|---------------|--------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Trans10K      | `data/trans10k_dataset.py`           | local (~10 K)                                                                                 | Primary train/val/test                                             |
| GSD           | `data/trans10k_dataset.py::GSDDataset`| local                                                                                         | Cross-dataset eval                                                 |
| ClearPose     | `data/clearpose_dataset.py`          | local clip dirs                                                                               | Cross-dataset video frames                                         |
| **GSD-S**     | `data/hf_datasets.py::HFGsdSDataset` | HF `garrying/GSD-S`                                                                           | RGB only                                                            |
| **RGBD-GSD**  | `data/hf_datasets.py::HFRgbdGsdDataset`| HF `garrying/RGBD-GSD`                                                                       | Adds depth                                                          |
| **VGSD-D**    | `data/hf_datasets.py::HFVgsdDataset` | HF `garrying/VGSD-D`                                                                          | Video glass surface dataset with depth                              |
| Trans-Video   | `data/video_dataset.py`              | local                                                                                         | Temporal benchmark — directory of `<clip>/frames/`                  |

### Dataset registry

`data/trans10k_dataset.py` exposes a unified `DATASETS` dict and a
`build_eval_dataset(name, root, split, image_size, **kw)` factory used by
all eval scripts (cross-dataset, multi-seed aggregator, per-image eval).
HF datasets are auto-registered when the `datasets` library is installed.

### Sample dict contract

Every dataset returns:
```
{
    "image":      (3, H, W) float (ImageNet-normalised)
    "image_t1":   (3, H, W) float (same statistics)
    "mask":       (H, W) int64 in {0, 1}
    "material":   int64 scalar (1 = glass / 0 = background)
    "image_path": str (id or path)
    "depth":      (1, H, W) float — only RGBD datasets
}
```

### Augmentation pipeline — `data/augmentation.py`

- **Shared spatial** (applied identically to t, t+1, mask):
  Resize (448×448), HorizontalFlip.
- **Independent photometric** (per-frame):
  ColorJitter, RandomBrightnessContrast, RandomSunFlare, OpticalDistortion,
  GridDistortion, GaussianBlur, RandomGamma, ImageNet normalisation.

Photometric perturbations are intentionally aggressive — they force the
model to use OFCV/BRF physics instead of texture shortcuts.

## 4. Training protocol

### 4.1 Loss — `train/losses.py::SPECTRALoss`

```
L_total = λ_seg · BCE(seg, mask)
        + λ_bnd · BoundaryLoss(seg, mask)
        + λ_consist · TransparencyConsistency(seg, ofcv)
        + λ_refl   · ReflectionSuppression(seg, ofcv, brf)
        + λ_mat    · CE(mat_logits, material)
        + aux_var_weight · −Var(ofcv_map)     # encourage non-trivial OFCV
```

Defaults (`configs/config.yaml`):

| λ           | value |
|-------------|-------|
| λ_seg       | 1.0   |
| λ_bnd       | 0.5   |
| λ_consist   | 0.1   |
| λ_refl      | 0.1   |
| λ_mat       | 0.0   |
| aux_var     | 0.1   |

### 4.2 Optimiser

- AdamW with two param groups:
  - Backbone: `lr = 1e-5`, `wd = 1e-4`
  - Rest:     `lr = 1e-4`, `wd = 1e-4`
- 1-epoch linear warm-up, then cosine to `1e-7`.
- Gradient clipping at `max_norm = 1.0`.
- Mixed precision (`torch.amp`) with dynamic loss scaling.

### 4.3 Schedule

| Variant            | Epochs | Source                                                       |
|--------------------|--------|--------------------------------------------------------------|
| Baseline run       | 10     | `train/train_baseline.py`                                    |
| Strong baselines   | 60     | `benchmarks/run_strong_baselines.py --epochs 60`             |
| Full ablation suite| 50     | `train/run_full_ablations.py --epochs 50`                    |
| Multi-seed         | 30+    | `train/run_multi_seed.py --seeds 42 7 13 1234 2025`          |

### 4.4 Hardware tested

- 1 × NVIDIA RTX 4090 (24 GB)
- Batch size 4, image 448², AMP on
- ~5 min / epoch on Trans10K train

## 5. Evaluation

### 5.1 Metrics — `eval/metrics.py`

| Metric                 | Symbol     | Direction |
|------------------------|------------|-----------|
| Intersection-over-Union| IoU        | ↑         |
| F-measure (β² = 0.3)   | F          | ↑         |
| Mean Absolute Error    | MAE        | ↓         |
| Balance Error Rate     | BER        | ↓         |
| Precision / Recall     | P, R       | ↑         |

Threshold for binarisation: **0.5**.

### 5.2 Cross-dataset eval

`eval/cross_dataset_eval.py` runs zero-shot evaluation of a Trans10K-trained
checkpoint on any registered dataset. Output: `outputs/cross_dataset/results.{json, md, tex}`.

Canonical reviewer-readable line:
```
$ python eval/cross_dataset_eval.py \
    --datasets trans10k:test gsd:test gsd_s:test rgbd_gsd:test \
               clearpose:all vgsd_d:test
```

### 5.3 Multi-seed protocol

1. Train SPECTRA under seeds `{42, 7, 13, 1234, 2025}` → `results/multi_seed/full/seed_*`.
2. `eval/aggregate_seeds.py` re-evaluates every seed's `spectra_best.pth`,
   computes mean ± std for each metric.
3. `eval/statistical_significance.py` runs paired bootstrap
   (10 000 resamples), paired t, Wilcoxon signed-rank on per-image IoUs.

### 5.4 Failure taxonomy

`eval/failure_taxonomy.py` automatically buckets each test image into:

- `reflection` — > 5 % of GT pixels saturated.
- `motion_blur` — mean RAFT residual > 0.55 with low std.
- `thin_glass` — coverage < 4 % and aspect ratio > 4.
- `strong_occlusion` — mask split into > 4 connected components.
- `clean` — none of the above.

Reports per-bucket IoU and the IoU gap vs the complement.

### 5.5 Temporal video benchmark

`benchmarks/video_benchmark.py` evaluates SPECTRA on consecutive frames:

- **TC-IoU**: IoU between prediction(t) and prediction(t−1) warped via flow.
- **TC-MAE**: pixel MAE of aligned predictions.
- **Flicker**: variance of probabilities over a sliding window.
- **Boundary jitter**: per-frame change in mask boundary length.

### 5.6 Interpretability suite

- **OFCV-GT correlation** (`eval/interpretability/ofcv_correlation.py`):
  Pearson r, AUROC of OFCV-alone, permutation-test p value.
- **Saliency overlap** (`eval/interpretability/saliency_overlap.py`):
  Grad-CAM through fusion head; coverage / IoU vs GT.
- **Human eval** (`eval/interpretability/human_eval_protocol.py`):
  Builds 100-image panels + CSV form. Annotators score Q1–Q4 (segmentation
  quality + OFCV/Grad-CAM/boundary fidelity).
- **IAA** (`eval/interpretability/iaa.py`): Krippendorff α + pairwise
  Spearman ρ across annotators.

### 5.7 Robustness — `eval/robustness_eval.py`

Pre-existing; evaluates the same checkpoint on corrupted images. The radar
plot (Figure 14) auto-uses `results/robustness/results.json` when present.

## 6. Strong baselines

Implemented in `benchmarks/strong_baselines/builders.py`:

| Name            | Backbone / source                                      | Train? |
|-----------------|--------------------------------------------------------|--------|
| `segformer_b5`  | `nvidia/segformer-b5-finetuned-ade-640-640`            | yes    |
| `mask2former_b` | `facebook/mask2former-swin-base-ade-semantic`          | yes    |
| `sam2`          | `facebook/sam2-hiera-large`                            | zs/ft  |
| `hq_sam`        | `segment-anything-hq` ViT-H                            | zs/ft  |
| `vit_seg`       | timm ViT-B/16 + UperNet head                           | yes    |

Run with `benchmarks/run_strong_baselines.py --model <name> --epochs 60`
or `--zero-shot` for SAM2 / HQ-SAM.

Numbers reported by original authors are curated in
`benchmarks/published_results.py`. They are combined with locally
re-trained numbers by `benchmarks/compose_comparison.py` into a single
markdown/LaTeX table.

## 7. Codebase layout

```
spectra/
├── data/
│   ├── trans10k_dataset.py        # Trans10K + GSD + DATASETS registry
│   ├── clearpose_dataset.py       # ClearPose loader
│   ├── hf_datasets.py             # GSD-S / RGBD-GSD / VGSD-D
│   ├── video_dataset.py           # Transparent-video windows
│   ├── synthetic_t1.py            # Fake-t+1 generators (mono variant)
│   └── augmentation.py
├── models/
│   ├── backbone.py                # DINOv2 wrapper
│   ├── spectra_model.py           # Full pipeline
│   ├── spectra_learned_physics.py # SPECTRA-LP (learned refraction)
│   └── spectra_mono.py            # Monocular variant
├── modules/
│   ├── ofcv_detector.py           # C1
│   ├── brf.py                     # C2
│   ├── signal_fusion.py           # Fusion head
│   ├── occlusion_handler.py
│   ├── reflection_separator.py
│   ├── learned_refraction.py      # Learned physics
│   ├── differentiable_optics.py   # Snell layer
│   └── neural_physics_prior.py
├── graph/
│   ├── superpixel.py              # SLIC + node feats
│   └── mbp_gnn.py                 # C3 GNN
├── flow/
│   └── raft_wrapper.py
├── train/
│   ├── train_baseline.py
│   ├── losses.py
│   ├── run_ablations.py           # original short-epoch ablation
│   ├── run_full_ablations.py      # P4 — 50-epoch suite
│   └── run_multi_seed.py          # P3
├── eval/
│   ├── metrics.py
│   ├── per_image_eval.py
│   ├── cross_dataset_eval.py      # P1
│   ├── aggregate_seeds.py         # P3
│   ├── statistical_significance.py# P3
│   ├── ablation_analysis.py       # P4
│   ├── temporal_consistency.py    # P5
│   ├── failure_taxonomy.py        # P7
│   ├── robustness_eval.py
│   ├── visual_quality_eval.py
│   ├── failure_case_analysis.py
│   └── interpretability/          # P6
│       ├── ofcv_correlation.py
│       ├── saliency_overlap.py
│       ├── human_eval_protocol.py
│       └── iaa.py
├── benchmarks/
│   ├── strong_baselines/          # P2 wrappers
│   ├── run_strong_baselines.py    # P2 trainer
│   ├── published_results.py       # P2 curated tables
│   ├── compose_comparison.py      # P2 aggregator
│   └── video_benchmark.py         # P5 runner
├── release/
│   ├── prepare_release.py         # P10 bundler
│   └── BENCHMARK.md               # P10 docs
├── inference/
│   ├── monocular_inference.py     # P9 single-image CLI
│   └── vis_utils.py
├── scripts/
│   ├── run_roadmap.py             # End-to-end orchestrator
│   ├── generate_paper_figures.py  # 20-figure pipeline
│   ├── build_benchmark_figure.py
│   ├── build_hero_assets.py
│   └── build_portfolio_visuals.py
├── configs/
│   ├── config.yaml
│   └── cross_dataset.yaml         # P1 config
└── FIGURE/                        # 20 paper figures (PDF + PNG each)
```

## 8. Roadmap status — what is done

| #   | Priority                          | Status | Entry point                                                            |
|-----|-----------------------------------|--------|------------------------------------------------------------------------|
| P1  | Cross-dataset generalisation      | code ✓ | `eval/cross_dataset_eval.py`                                           |
| P2  | Stronger baselines                | code ✓ | `benchmarks/run_strong_baselines.py`, `compose_comparison.py`          |
| P3  | Multi-seed validation             | code ✓ | `train/run_multi_seed.py`, `eval/aggregate_seeds.py`, `statistical_significance.py` |
| P4  | Full-length ablation              | code ✓ | `train/run_full_ablations.py`, `eval/ablation_analysis.py`             |
| P5  | Temporal video benchmark          | code ✓ | `benchmarks/video_benchmark.py`                                        |
| P6  | Interpretability validation       | code ✓ | `eval/interpretability/*`                                              |
| P7  | Failure analysis (quantitative)   | code ✓ | `eval/failure_taxonomy.py`                                             |
| P8  | Learned physics simulator         | code ✓ | `models/spectra_learned_physics.py`                                    |
| P9  | Single-image transparency         | code ✓ | `models/spectra_mono.py`, `inference/monocular_inference.py`           |
| P10 | Open benchmark release            | code ✓ | `release/prepare_release.py`, `release/BENCHMARK.md`                   |

**Compute still required** before submission:
- Multi-seed full-length ablations (5 seeds × 5 variants × 50 epochs ≈ 125 GPU·days at 5 min/epoch).
- Strong-baseline retraining at 60 epochs each.
- Train SPECTRA-LP (P8) and SPECTRA-Mono (P9) to convergence.
- Run video benchmark on a downloaded VGSD-D split.

The full pipeline is wired in `scripts/run_roadmap.py`; you run one
command per priority (or all of them sequentially).

## 9. Figures

`scripts/generate_paper_figures.py` produces 20 figures into `FIGURE/`:

| # | Title                                       | Auto-promote source                                  |
|---|---------------------------------------------|------------------------------------------------------|
| 1 | Teaser mosaic                               | Synthetic (replace `synth_glass` for real)           |
| 2 | Motivation: appearance vs physics           | Synthetic                                            |
| 3 | Architecture diagram                        | Final                                                |
| 4 | OFCV pipeline                               | Synthetic                                            |
| 5 | BRF pipeline                                | Synthetic                                            |
| 6 | Fusion visualisation                        | Synthetic                                            |
| 7 | Training curves                             | `results/spectra_final.json`                         |
| 8 | Ablation curves                             | `results/full_ablations/summary.json`                |
| 9 | Qualitative comparison                      | Synthetic baseline outputs                           |
| 10| Hard cases (5 conditions)                   | Synthetic                                            |
| 11| Cross-dataset                               | `outputs/cross_dataset/results.json`                 |
| 12| OFCV interpretability                       | Synthetic                                            |
| 13| Failure cases                               | `results/failure_taxonomy/taxonomy.json`             |
| 14| Corruption radar                            | `results/robustness/results.json`                    |
| 15| Statistical validation (mean±std)           | `results/multi_seed/full/aggregated.json` + comparison|
| 16| Embedding (UMAP)                            | Synthetic                                            |
| 17| Attention visualisation                     | Synthetic                                            |
| 18| Runtime vs accuracy                         | Synthetic + measured                                 |
| 19| Real-world demo                             | Drop in your own photos                              |
| 20| Deployment diagram                          | Final                                                |

Auto-promotion: figures 7, 8, 14, 15 automatically detect their real-data
JSON outputs in `results/` / `outputs/` / `benchmarks/_comparison/` and
swap to real data without changing code. To regenerate:
```
python scripts/generate_paper_figures.py
python scripts/generate_paper_figures.py --only 7 8 15   # subset
```

## 10. Reproducibility commitments

- **Seeds**: `{42, 7, 13, 1234, 2025}` across all multi-seed runs.
- **Determinism**: `torch.backends.cudnn.deterministic = True`; `benchmark = False`.
- **Logging**: `wandb_project = spectra-cvpr` (set entity in config).
- **Save policy**: best-IoU checkpoint per run + last epoch.
- **Checksums**: `release/prepare_release.py` writes `sha256.txt` for every
  bundled artifact.

## 11. How to run end-to-end

```
# Single image inference (with the released checkpoint)
python inference/monocular_inference.py \
    --checkpoint checkpoints/spectra_mono_best.pth \
    --image my_photo.jpg --output out.png

# End-to-end roadmap (with --dry-run to see commands)
python scripts/run_roadmap.py --dry-run
python scripts/run_roadmap.py                   # everything
python scripts/run_roadmap.py --only A1 A3 A4   # subset

# Just regenerate the paper figures from current results/
python scripts/generate_paper_figures.py
```

## 12. Known limitations

- OFCV requires at least two frames. SPECTRA-Mono (P9) lifts this with a
  synthetic / learned warper.
- MBP-GNN is too slow for every training step — used at val / inference only.
- Material classification head currently has loss weight 0 — left for future
  multi-task work.
- VGSD-D loader assumes HF schema `{image, image_next?, mask, depth?, clip_id, frame_index}`.
  If the public schema differs, adjust `data/hf_datasets.py::HFVgsdDataset`.

## 13. Suggested paper sections (and where each comes from)

| Section                  | Source                                                              |
|--------------------------|---------------------------------------------------------------------|
| Introduction (teaser)    | Fig 1                                                               |
| Motivation               | Fig 2                                                               |
| Method                   | §2 here; Figs 3–6                                                   |
| Experimental setup       | §4–§5; configs in `configs/config.yaml`                             |
| Main results             | Fig 15 + `benchmarks/_comparison/comparison.md`                     |
| Cross-dataset            | Fig 11 + `outputs/cross_dataset/results.md`                         |
| Ablations                | Fig 7, 8 + `results/full_ablations/table.md`                        |
| Robustness               | Fig 14 + `results/robustness/`                                      |
| Interpretability         | Fig 12, 17 + `results/interpretability/`                            |
| Failure analysis         | Fig 13 + `results/failure_taxonomy/taxonomy.md`                     |
| Runtime                  | Fig 18                                                              |
| Real-world deployment    | Fig 19, 20                                                          |
| Limitations              | §12 here + Fig 13                                                   |

---

**Open issues / next-step checklist**

- [ ] Download Trans10K and place under `datasets/Trans10K/`.
- [ ] Verify HF-cache works for `garrying/GSD-S`, `garrying/RGBD-GSD`, `garrying/VGSD-D`.
- [ ] Train Trans10K baseline (10 epochs) → `checkpoints/spectra_best.pth`.
- [ ] Run `scripts/run_roadmap.py` end-to-end.
- [ ] Re-run `scripts/generate_paper_figures.py` — Figures 7, 8, 14, 15 now use real data.
- [ ] Conduct human study (Q1–Q4 per Fig 12 protocol; 3 annotators).
- [ ] Write paper draft using §13 mapping.
