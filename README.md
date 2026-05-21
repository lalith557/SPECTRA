<div align="center">

# SPECTRA

### Causal Physically-Guided Transparent-Object Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](#citation)
[![Trans10K IoU](https://img.shields.io/badge/Trans10K_val_IoU-0.9217-success.svg)](#headline-numbers-trans10k-val-e10)

**Trans10K val IoU 0.9217 · test mean IoU 0.9237 · +0.036 over DeepLabV3+ baseline**

</div>

---

## About the Project

**SPECTRA** is a transparent-object segmentation system that combines a frozen **DINOv2** vision backbone with two physics-inspired modules:

- **OFCV** (Optical Flow Consistency Violation) — gates patch tokens by detecting refraction-induced motion that violates rigid-body flow assumptions.
- **BRF** (Boundary Resonance Field) — injects a Gabor-bank structural prior at full resolution, sharpening glass edges.

These signals are fused with a lightweight head to produce segmentation masks that are not only accurate but **interpretable** — you can see *why* the network thinks a region is glass.

> This is a final-year student project. The contribution is **interpretable, physics-guided causal conditioning** — not state-of-the-art accuracy. See [`release/FINAL_SUMMARY.md`](release/FINAL_SUMMARY.md) for the full report.

---

## What's Actually Novel

| # | Contribution | Evidence |
|---|---|---|
| 1 | **OFCV gating** — optical-flow consistency violation maps gate patch tokens before fusion | Without OFCV the network needs 5 epochs to hit the same IoU; with OFCV it gets there by epoch 1 |
| 2 | **BRF prior** — Gabor-bank boundary resonance map injected at full resolution as a static structural prior | Improves early-epoch convergence and edge fidelity |
| 3 | **Robustness behavior** — survives heavy corruption | Clean IoU 0.97; at severity-5 motion-blur still scores **IoU 0.82** |
| 4 | **Per-image transparency maps** — interpretable signals you can read | OFCV fires on glass shells, goes dark on opaque inclusions in the same scene |

> **What this project does NOT claim:** SOTA on any public leaderboard. Public baselines have not been benchmarked under matched conditions outside this repo.

---

## Headline Numbers — Trans10K val (E10)

| Metric | Value |
|--------|------:|
| **IoU** | **0.9217** |
| F-measure | 0.9560 |
| MAE | 0.0293 |
| BER | 0.0265 |
| Test mean IoU (4,428 imgs) | **0.9237** |

### Matched-Condition Benchmark

Same Trans10K splits, same 10-epoch schedule, same eval pipeline.

| Model | Params (M) | Val IoU | F | MAE | BER | Train (min) |
|-------|-----------:|--------:|--:|----:|----:|------------:|
| SAM ViT-B (zero-shot) | 91 frozen | 0.1341 | 0.315 | 0.292 | 0.459 | 0 |
| SegFormer-B0 | 3.8 | 0.8680 | 0.926 | 0.057 | 0.047 | 44 |
| U-Net (ResNet-34) | 24.4 | 0.8799 | 0.938 | 0.051 | 0.045 | 44 |
| DeepLabV3+ (ResNet-50) | 39.6 | 0.8856 | 0.935 | 0.047 | 0.040 | 45 |
| **SPECTRA (ours)** | **26.6** | **0.9217** | **0.956** | **0.029** | **0.027** | 222 |

SPECTRA reaches **+0.036 IoU** over the strongest baseline (DeepLabV3+), at ~5× the training cost. SAM zero-shot lands at 0.13 — a clean demonstration that foundation models without task-specific training do not solve transparent-object segmentation.

### Ablation (5-epoch retrains from scratch)

| Variant | E1 IoU | E5 IoU | E1 Δ vs full |
|---------|-------:|-------:|-------------:|
| full (OFCV + BRF) | 0.8623 | 0.9133 | — |
| no_ofcv | 0.8365 | 0.9147 | -0.026 |
| no_brf  | 0.8414 | 0.9164 | -0.021 |

OFCV and BRF help **early-epoch convergence and interpretability**, not raw clean-data accuracy. The honest framing.

### Hard-Case Robustness (E10, 100-image subset, IoU @ severity 5)

| brightness | low_light | glare | motion_blur | jpeg | noise | fog | colour_jitter |
|-----------:|----------:|------:|------------:|-----:|------:|----:|--------------:|
| 0.872 | 0.966 | 0.859 | 0.815 | 0.960 | 0.756 | 0.969 | 0.969 |

Clean baseline on this subset: **0.969**.

---

## Quick Start

### Installation

```bash
git clone https://github.com/lalith557/SPECTRA.git
cd SPECTRA

# Python 3.10+, CUDA 11.8
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: PyTorch Geometric (only if use_gnn=True)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

> **Note:** Pretrained weights (`spectra_best.pth`, ~320 MB) are not stored in this repo. Host them on HuggingFace, a GitHub Release, or Git LFS and update the path in `configs/config.yaml`.

### Run the Demo

```bash
python demo/gradio_demo.py \
  --config configs/config.yaml \
  --checkpoint weights/spectra_best.pth
```

Open http://127.0.0.1:7860. The interface shows segmentation overlay, OFCV violation map, BRF boundary field, flow residual, uncertainty (entropy), and per-class material confidence.

### Evaluate a Checkpoint

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

### Reproduce Training

```bash
# Final 10-epoch run (~3.5 h on RTX 4070 / 8 GB)
python train/train_baseline.py --config configs/config.yaml

# 3-variant ablation (full / no-OFCV / no-BRF, 5 epochs each, ~5 h)
python train/run_ablations_reduced.py
```

---

## Repository Layout

```
spectra/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── utils.py
│
├── demo/                 # Gradio webcam / image / video app
├── models/               # SPECTRA model + DINOv2 backbone
├── modules/              # OFCV detector, BRF, fusion head
├── flow/                 # RAFT wrapper + optical-flow utils
├── graph/                # superpixel + MBP-GNN (optional)
├── data/                 # Trans10K / GSD / ClearPose loaders          [gitignored]
├── inference/            # video inference + viz utilities
├── pretrain/             # physics-contrastive pre-training (scaffolded)
├── api/                  # FastAPI deployment endpoint
├── train/                # train_baseline.py, run_ablations_reduced.py, losses.py
├── eval/                 # metrics, visualize, robustness, failure analysis
├── benchmarks/           # U-Net / DeepLabV3+ / SegFormer / SAM         [gitignored]
├── scripts/              # build_portfolio_visuals.py, build_benchmark_figure.py
├── tests/                # unit tests
├── configs/              # config.yaml
│
├── weights/              # production checkpoints                       [gitignored]
├── results/              # experimental artefacts + figures             [gitignored]
├── release/              # frozen copy of the production run            [gitignored]
└── paper/                # PAPER.md + figures
```

Large folders (`venv/`, `data/`, `datasets/`, `results/`, `weights/`, `checkpoints/`, `release/`, `benchmarks/`, `web/`) are excluded via `.gitignore`. Host model weights via Git LFS or a GitHub Release attachment.

---

## Tech Stack

- **Backbone:** DINOv2 (frozen ViT)
- **Physics modules:** OFCV (RAFT optical flow), BRF (Gabor filter bank)
- **Framework:** PyTorch 2.1 + CUDA 11.8
- **Training:** Trans10K (4,428-image test split)
- **Optional:** PyTorch Geometric (MBP-GNN), Segmentation Models PyTorch (baselines), SAM (zero-shot baseline)
- **Serving:** Gradio (demo) + FastAPI (API)

---

## Citation

This is a student project. If you find the code useful:

```bibtex
@misc{gona2026spectra,
  author       = {Gona, Lalith},
  title        = {SPECTRA: Causal Physically-Guided Transparent-Object Segmentation},
  year         = {2026},
  howpublished = {Final-project report, Lovely Professional University},
  url          = {https://github.com/lalith557/SPECTRA}
}
```

---

<div align="center">

**Built as a final-year project · Lovely Professional University · 2026**

</div>
