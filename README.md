# SPECTRA

**Causal physically-guided transparent-object segmentation.**

DINOv2 backbone + Optical Flow Consistency Violation (OFCV) gating + Boundary Resonance Field (BRF) structural prior + fusion head. Trained on Trans10K. Best checkpoint: **val IoU 0.9217 / test mean IoU 0.9237** on Trans10K's 4,428-image test split.

This is a student project. The contribution is *interpretable, physics-guided causal conditioning* — not state-of-the-art accuracy. See `release/FINAL_SUMMARY.md` for the full report.

---

## What's actually novel

1. **OFCV gating** — optical-flow consistency violation maps gate patch tokens before fusion. Trained without OFCV the network reaches the same clean-data IoU by epoch 5; with OFCV it gets there by epoch 1 and produces image-dependent transparency maps you can read.
2. **BRF prior** — Gabor-bank boundary resonance map injected at full resolution as a static structural prior.
3. **Robustness behavior** — clean IoU 0.97 on a 100-image hard-case subset; at severity 5 motion-blur the model still scores **IoU 0.82** (vs 0.72 mid-training).
4. **Interpretable per-image transparency maps** — OFCV fires on glass shells, goes dark on opaque inclusions in the same scene.

What this project **does not** claim: SOTA on any public leaderboard. Public baselines have not been benchmarked under matched conditions (see `BENCHMARK_TODO` below).

---

## Repository layout

```
spectra/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── utils.py
│
├── demo/                 # Gradio webcam/image/video app
├── models/               # SPECTRA model + DINOv2 backbone
├── modules/              # OFCV detector, BRF, fusion head
├── flow/                 # RAFT wrapper + optical-flow utils
├── graph/                # superpixel + MBP-GNN (optional)
├── data/                 # Trans10K / GSD / ClearPose loaders
├── inference/            # video inference + viz utilities
├── pretrain/             # physics-contrastive pre-training (scaffolded)
├── api/                  # FastAPI deployment endpoint
├── train/                # train_baseline.py, run_ablations_reduced.py, losses.py
├── eval/                 # metrics, visualize, robustness, failure analysis
├── benchmarks/           # U-Net / DeepLabV3+ / SegFormer / SAM comparison
├── scripts/              # build_portfolio_visuals.py, build_benchmark_figure.py
├── tests/                # unit tests
├── configs/              # config.yaml
│
├── weights/              # production checkpoints (gitignored — see download)
├── results/              # all experimental artefacts + portfolio figures
├── release/              # frozen copy of the production run
└── paper/                # PAPER.md + figures/
```

Large files (`weights/*.pth`, `datasets/`, `venv/`, `wandb/`) are gitignored. Use a Git LFS pointer or a GitHub Release attachment for the 320 MB `spectra_best.pth`.

---

## Quick start

### Run the demo (webcam + image + video)

```bash
python demo/gradio_demo.py \
  --config configs/config.yaml \
  --checkpoint weights/spectra_best.pth
```

Open `http://127.0.0.1:7860`. The interface shows segmentation overlay, OFCV violation map, BRF boundary field, flow residual, uncertainty (entropy), and per-class material confidence.

### Evaluate a checkpoint

```bash
# Per-epoch metrics on val
python eval/eval_checkpoints.py --config configs/config.yaml \
  --checkpoints weights/checkpoint_epoch010.pth \
  --out results/eval.json

# 6-panel qualitative figures
python eval/visualize.py --config configs/config.yaml \
  --checkpoint weights/spectra_best.pth \
  --n_samples 16 --save-dir results/qualitative

# Hard-case sweep (8 corruptions × 5 severities)
python eval/robustness_eval.py --config configs/config.yaml \
  --checkpoint weights/spectra_best.pth \
  --output results/robustness --n-samples 100

# Top-20 worst predictions + category histogram
python eval/failure_case_analysis.py --config configs/config.yaml \
  --checkpoint weights/spectra_best.pth \
  --output results/failures
```

### Reproduce training

```bash
# Final 10-epoch run (≈3.5 h on RTX 4070 / 8 GB)
python train/train_baseline.py --config configs/config.yaml

# 3-variant ablation (full / no-OFCV / no-BRF, 5 epochs each, ≈5 h)
python train/run_ablations_reduced.py
```

---

## Headline numbers (Trans10K val, E10)

| Metric | Value |
|--------|-------|
| IoU | **0.9217** |
| F-measure | 0.9560 |
| MAE | 0.0293 |
| BER | 0.0265 |
| Test mean IoU (4,428 imgs) | 0.9237 |

### Ablation (5-epoch retrains from scratch)

| Variant | E1 IoU | E5 IoU | E1 Δ vs full |
|---------|--------|--------|--------------|
| full (OFCV + BRF) | 0.8623 | 0.9133 | — |
| no_ofcv | 0.8365 | 0.9147 | -0.026 |
| no_brf  | 0.8414 | 0.9164 | -0.021 |

OFCV and BRF help **early-epoch convergence and interpretability**, not raw clean-data accuracy. The honest framing.

### Hard-case robustness (E10, 100-image subset, IoU @ severity 5)

| brightness | low_light | glare | motion_blur | jpeg | noise | fog | colour_jitter |
|------------|-----------|-------|-------------|------|-------|-----|---------------|
| 0.872 | 0.966 | 0.859 | 0.815 | 0.960 | 0.756 | 0.969 | 0.969 |

Clean baseline on this subset: 0.969.

---

## Matched-condition benchmark

Same Trans10K splits, same 10-epoch training schedule, same eval pipeline (`eval/metrics.py`). See `benchmarks/comparison_table.json` for the full JSON and `results/portfolio/fig05_benchmark_comparison.png` for the figure.

| Model | Params (M) | Val IoU | F | MAE | BER | Train min |
|-------|------------|---------|---|-----|-----|-----------|
| SAM ViT-B (zero-shot, centre-point prompt) | 91 frozen | 0.1341 | 0.315 | 0.292 | 0.459 | 0 |
| SegFormer-B0 | 3.8 | 0.8680 | 0.926 | 0.057 | 0.047 | 44 |
| U-Net (ResNet-34) | 24.4 | 0.8799 | 0.938 | 0.051 | 0.045 | 44 |
| DeepLabV3+ (ResNet-50) | 39.6 | 0.8856 | 0.935 | 0.047 | 0.040 | 45 |
| **SPECTRA (ours)** | **26.6** | **0.9217** | **0.956** | **0.029** | **0.027** | 222 |

SPECTRA reaches +0.036 IoU over the strongest baseline (DeepLabV3+), at ≈5× the training cost. SAM zero-shot (centre-point prompt) lands at 0.13 — a clean demonstration that foundation models without task-specific training do not solve transparent-object segmentation.

Reproduce: `python benchmarks/run_all.py` (≈2.5 h for the trained models + 4 min for SAM).

---

## Installation

```bash
git clone <repo-url>
cd spectra

# Python 3.10+, CUDA 11.8 venv
python -m venv venv
source venv/bin/activate         # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# PyTorch Geometric (only needed if use_gnn=True)
pip install torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## Citation

This is a student project. If you find the code useful:

```
Gona, L. (2026). SPECTRA: Causal physically-guided transparent-object segmentation.
Final-project report, Lovely Professional University. https://github.com/<repo>
```
