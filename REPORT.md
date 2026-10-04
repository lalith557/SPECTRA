<div align="center">

# SPECTRA
### Causal, Physically-Guided Segmentation of Transparent Objects

**Final-Year Project Report**
**Lalith Gona** · Lovely Professional University · 2026

`gonalalith2005@gmail.com`

---

> *"Glass is invisible to a camera — and to every model trained on what a camera sees.
> SPECTRA teaches a network to look at **what physics says is impossible**, not just what is present."*

---

</div>

## 1. Executive Summary

Transparent objects — glass bottles, windows, clear plastics — defeat conventional segmentation networks because the pixels behind them look almost identical to the pixels around them. The texture, colour, and edges of a transparent surface are a function of *what is behind it*, not *what it is*. This breaks the brightness-constancy assumption that virtually every deep segmentation pipeline silently depends on.

**SPECTRA** is a final-year research system that addresses this problem from a *causal*, physics-first standpoint. Rather than treating glass as a harder texture-classification problem, SPECTRA exposes the **physical fingerprint of refraction** — the disagreement it produces between consecutive video frames — and uses it to *condition* a DINOv2 vision backbone before the segmentation decision is made.

| Headline metric (Trans10K val, E10) | Value |
|---|---:|
| **Intersection-over-Union (IoU)** | **0.9217** |
| F-measure | 0.9560 |
| Mean Absolute Error | 0.0293 |
| Balanced Error Rate | 0.0265 |
| Test mean IoU (4,428 images) | **0.9237** |
| Improvement over DeepLabV3+ baseline | **+0.036 IoU** |
| Improvement over SAM ViT-B (zero-shot) | **+0.79 IoU** |

The contribution of this work is not raw accuracy but **interpretable, physics-guided causal conditioning** — the network produces per-image *transparency maps* that a human can read, audit, and verify.

---

## 2. The Problem in One Picture

> *Why does this matter?* A self-driving car cannot brake for a glass door if it has segmented the corridor behind it instead. A robot arm cannot grasp a beaker if it has reached for the wall behind it. A AR/VR pipeline cannot composite light correctly around a window if it cannot tell the window exists.

Conventional CNN/ViT segmentation models trained on Trans10K plateau around **IoU 0.87–0.89** because their entire signal comes from texture and edges — both of which transparent objects *borrow* from their background. SPECTRA breaks that ceiling by giving the network a *new* signal that opaque surfaces physically cannot produce.

---

## 3. Core Idea — Two Physics-Derived Signals

| # | Signal | Physical principle | Behaviour |
|---|---|---|---|
| 1 | **OFCV** — Optical Flow Consistency Violation | Snell's law: refraction bends light, so motion behind a transparent surface no longer matches rigid-body camera motion | **Fires** on glass; **stays dark** on opaque objects, even in the same scene |
| 2 | **BRF** — Boundary Resonance Field | Glass produces a characteristic *double-edge* frequency response (inner + outer surface) detectable by a Gabor filter bank | Static structural prior that sharpens glass boundaries at full resolution |

These are not learned features re-discovering known physics — they are the physics, **handed to the network directly**, then verified to be non-collapsing and image-dependent across 10 epochs of training.

---

## 4. System Design

### 4.1 End-to-End Architecture

```
                                ┌──────────────────────────────────────────────┐
                                │              SPECTRA PIPELINE                │
                                └──────────────────────────────────────────────┘

  RGB frame I_t  ─────────────► DINOv2 ViT-S/14 (frozen)
                                        │
                                        ▼
                                Patch tokens (B, 384, 32, 32)
                                        │
                                        ▼
                                FPN multi-scale features
                                        │
                                        │  ┌─────────────────────────────────┐
  RGB frame I_t+1 ──► RAFT ─────────────┼─►│  OFCV DETECTOR                  │
                      (frozen)          │  │  • photometric residual r_t     │
                                        │  │  • fwd-bwd flow consistency c_t │
                                        │  │  • small transformer block      │
                                        │  └────────────────┬────────────────┘
                                        │                   │
                                        │             ofcv_map (B,1,32,32)
                                        │                   │
  RGB frame I_t ──► Gabor bank ─────────┼────► BRF map (B,1,448,448)
                    (8 orient × 3 scales)│            │
                                        ▼            ▼
                                ┌──────────────────────────────┐
                                │       FUSION HEAD            │
                                │  tokens × sigmoid(ofcv_map)  │
                                │      ⊕ BRF projection         │
                                └──────────────┬───────────────┘
                                               ▼
                                       seg_logits  +  mat_logits
                                       (segmentation) (material)
```

### 4.2 Module-Level Responsibilities

| Module | File | Role |
|---|---|---|
| Backbone | [models/backbone.py](spectra/models/backbone.py) | Frozen DINOv2 ViT-S/14 — provides strong, self-supervised patch features |
| OFCV Detector | [modules/ofcv_detector.py](spectra/modules/ofcv_detector.py) | Computes per-patch consistency-violation score from RAFT flow |
| BRF | [modules/brf.py](spectra/modules/brf.py) | Non-trainable Gabor bank → boundary resonance map |
| Signal Fusion | [modules/signal_fusion.py](spectra/modules/signal_fusion.py) | Gates patch tokens by σ(OFCV), concatenates BRF, decodes masks |
| SPECTRA Model | [models/spectra_model.py](spectra/models/spectra_model.py) | End-to-end wrapper, training/eval forward |
| Training | [train/train_baseline.py](spectra/train/train_baseline.py) | 10-epoch schedule, warmup + cosine decay, AMP, anti-collapse regulariser |
| Evaluation | [eval/](spectra/eval/) | IoU/F/MAE/BER, robustness sweep, failure-case analysis |
| Deployment | [demo/gradio_demo.py](spectra/demo/gradio_demo.py), [api/](spectra/api/) | Interactive Gradio app + FastAPI endpoint |

### 4.3 Deployment View

```
   ┌──────────┐    HTTPS     ┌────────────────┐     ┌──────────────────┐
   │  Browser │ ───────────► │  FastAPI / API │ ──► │  SPECTRA model   │
   │  Gradio  │              │  (api/main.py) │     │  + RAFT + Gabor  │
   └──────────┘              └────────────────┘     └──────────────────┘
        ▲                              │
        │       6-panel visualisation  │
        └──────────────────────────────┘
        seg overlay · OFCV map · BRF · flow residual · uncertainty · material confidence
```

Containerised via [Dockerfile](spectra/Dockerfile); deployable to Railway via [railway.json](spectra/railway.json).

---

## 5. Training Configuration (Reproducibility)

| Setting | Value |
|---|---|
| Dataset | Trans10K (5,000 train / 1,000 val / 4,428 test) |
| Image size | 448 × 448 |
| Batch size | 4 |
| Optimizer | AdamW (backbone LR 1e-5, head LR 1e-4) |
| Schedule | 1-epoch linear warmup → cosine decay → 1e-7 |
| Epochs | 10 (~3.5 h on RTX 4070 Laptop, 8 GB) |
| Precision | Mixed-precision AMP |
| Loss | CE + Dice + boundary-aware + OFCV variance regulariser |

All checkpoints, configs, logs, and metrics are frozen under [release/](spectra/release/).

---

## 6. Results

### 6.1 Matched-Condition Benchmark — SPECTRA vs Four Standard Baselines

All four baselines were trained on **the identical Trans10K splits**, with **the identical 10-epoch schedule**, and evaluated through **the identical metric pipeline**. No cherry-picking.

| Model | Params (M) | Val IoU | F | MAE | BER | Train (min) |
|---|---:|---:|---:|---:|---:|---:|
| SAM ViT-B (zero-shot, centre prompt) | 91 frozen | 0.1341 | 0.315 | 0.292 | 0.459 | 0 |
| SegFormer-B0 | 3.8 | 0.8680 | 0.926 | 0.057 | 0.047 | 44 |
| U-Net (ResNet-34) | 24.4 | 0.8799 | 0.938 | 0.051 | 0.045 | 44 |
| DeepLabV3+ (ResNet-50) | 39.6 | 0.8856 | 0.935 | 0.047 | 0.040 | 45 |
| **SPECTRA (ours)** | **26.6** | **0.9217** | **0.956** | **0.029** | **0.027** | 222 |

- **+0.036 IoU** over the strongest baseline (DeepLabV3+) — a **+4.1 % relative** improvement.
- **-38 % MAE** vs the same baseline.
- **+0.79 IoU** over SAM zero-shot — confirming foundation models without task-specific training **do not** solve transparent-object segmentation.
- Cost: **≈5× training time per epoch** (the honest sticker price of the physics module).

### 6.2 Training Trajectory — Monotone, No Collapse

| Epoch | IoU | F | MAE | BER |
|---|---:|---:|---:|---:|
| 1 | 0.8544 | 0.9221 | 0.0997 | 0.0548 |
| 5 | 0.9076 | 0.9444 | 0.0363 | 0.0297 |
| **10** | **0.9217** | **0.9560** | **0.0293** | **0.0265** |

Critically, **OFCV variance grew from 6.3 × 10⁻³ at step 50 to 2.3 × 10⁻¹ by epoch 10 and stayed there**. The gate did *not* collapse to a constant — the most common failure mode of learned gating mechanisms.

### 6.3 Hard-Case Robustness (Severity 5, 100-image subset)

Clean baseline on this subset: **IoU 0.969**.

| Corruption | brightness | low_light | glare | motion_blur | jpeg | gauss noise | fog | colour jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **IoU @ sev 5** | 0.872 | 0.966 | **0.859** | **0.815** | 0.960 | 0.756 | 0.969 | 0.969 |

The two corruptions explicitly named in the original architecture spec as hardest — **glare** and **motion blur** — degrade gracefully and remain above **0.80**. Gaussian noise is the only genuine weak point.

### 6.4 Honest Ablation (3 variants × 5 epochs from scratch)

| Variant | E1 IoU | E5 IoU | Δ vs full @ E1 |
|---|---:|---:|---:|
| full (OFCV + BRF) | 0.8623 | 0.9133 | — |
| no_ofcv | 0.8365 | 0.9147 | -0.026 |
| no_brf | 0.8414 | 0.9164 | -0.021 |

**Headline finding (stated openly):** removing OFCV or BRF costs ≈0.02 IoU at epoch 1, but the DINOv2 + FusionHead path catches up by epoch 5. The novel modules buy **convergence speed**, **hard-case robustness**, and **interpretability** — not raw clean-data accuracy. Any framing of "we got more accurate because of physics" would be misleading, and this report does not make that claim.

---

## 7. Why the Professor Should Care — Interpretability as the Real Result

The single most striking qualitative result is this:

> In a scene containing **a glass bottle with a wooden roller inside it**, the OFCV map fires brightly on the glass shell — and **darkens within the opaque inclusion**. The network has learned, from physics alone, to tell glass from wood *in the same image, without supervision on that distinction*.

This is the **causal signal** a hand-crafted physics model would produce, recovered automatically and visualisable as a heatmap. The Gradio demo renders six panels per input — segmentation overlay, OFCV map, BRF field, flow residual, prediction entropy, and per-class material confidence — turning the model from a black box into something a domain expert can audit.

For a final-year project, this is the contribution worth defending: **a segmentation system whose internal reasoning is physically grounded and human-readable**.

---

## 8. Limitations (Stated Up Front)

| Limitation | What it means |
|---|---|
| **Single-dataset evaluation** | All comparisons are on Trans10K. Cross-dataset generalisation to GSD / ClearPose not yet measured. |
| **Two-frame requirement** | OFCV needs $I_t, I_{t+1}$. On single stills the signal degenerates; deployment needs a video buffer or a learned synthetic second frame. |
| **BRF is more "prior" than "signal"** | Mean BRF activation is stable around 0.51 across val; useful structurally but not strongly per-image discriminating. |
| **5-epoch ablation, not 30** | A full 30-epoch ablation per variant (~50 GPU-hours) would tighten the convergence story. Not yet run. |

---

## 9. Future Work

1. **Cross-dataset generalisation** — evaluate the Trans10K checkpoint on GSD and ClearPose without further fine-tuning.
2. **Synthetic second frame** — learn a plausible $I_{t+1}$ from $I_t$ alone so OFCV works on single stills.
3. **Full 30-epoch ablation** to characterise late-epoch behaviour.
4. **Richer material head** — currently 4-way and under-supervised.
5. **Stronger baselines** — larger SegFormer variants and SAM with box prompts to stress-test the +0.036 IoU gap.

---

## 10. Conclusion

SPECTRA shows that **a small, well-chosen physical prior can change how a vision network reasons** about a problem its standard inductive biases were never designed for. On Trans10K, it reaches **val IoU 0.9217 and test mean IoU 0.9237** — exceeding U-Net, DeepLabV3+, SegFormer-B0, and zero-shot SAM under matched conditions — while producing per-image transparency maps that a human can interpret, debug, and trust.

The accuracy headline is real and reproducible. But the **lasting contribution** of this project is the demonstration that **causal, physics-guided conditioning works in practice**: the OFCV gate stays alive across training, responds to scene content, and distinguishes glass from opaque objects in the same image — exactly as the underlying physics says it should.

For a final-year project, that is a more honest, more defensible, and more interesting result than a 0.01-IoU bump on a leaderboard.

---

<div align="center">

### Repository

[`github.com/lalith557/SPECTRA`](https://github.com/lalith557/SPECTRA)

**All code, configs, logs, checkpoints, and figures are reproducible from this repository.**

---

*Lalith Gona · B.Tech Computer Science & Engineering · Lovely Professional University · 2026*

</div>
