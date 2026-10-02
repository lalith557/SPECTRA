# SPECTRA — Cloud GPU Setup & Execution

Everything needed to reproduce the SPECTRA experiments on a dedicated cloud GPU. The local
RTX 4070 Laptop thermally throttles (83 °C, ~53% clocks), turning a ~3.3 h run into ~10 h; use
cloud for all remaining training. **All commands below run on the cloud GPU, not the laptop.**

## 1. Recommended GPU class
A single modern data-center/prosumer GPU: **L4 24 GB, A10 24 GB, RTX 4090 24 GB, A5000 24 GB, or A100 40 GB**. Any one is enough; no multi-GPU needed.

## 2. Minimum VRAM
**8 GB** (P1 at 448/bs4 peaked at ~6.8 GB locally). Tight but works.

## 3. Recommended VRAM
**24 GB** — comfortable headroom at 448, fast, no fragmentation risk. (Do **not** raise batch size — it is part of the frozen protocol.)

## 4. Required CUDA / PyTorch environment
- CUDA **12.8** runtime (driver supporting cu128).
- **torch==2.11.0+cu128**, **torchvision==0.26.0+cu128** (install from the cu128 wheel index).
- cuDNN 9.x (bundled with the wheels).

## 5. Python version
**3.13.x** (verified 3.13.12). 3.10–3.13 acceptable.

## 6. Dependency installation
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
pip install -r requirements-cloud.txt
```
Do **not** install xFormers (DINOv2 runs without it; a mismatched build can break torch).

## 7. Dataset setup
Datasets are **git-ignored / not in the repo** — transfer them out-of-band (rsync/scp/cloud bucket) into the repo root with this exact layout:
```
datasets/Trans10K/{train,validation,test}/{images,masks}/     # 5000 / 1000 / 4428 pairs
VSGD_dataset/test/<clip>/JPEGImages/*.jpg + SegmentationClassPNG/*.png   # 105 clips  (P3 + cross-dataset)
clearpose_dataset/.../scene*/{id}-color.png + {id}-label.png            # 12 scenes  (P1.5 cross-dataset)
```
P1/P2 need only Trans10K. P1.5 needs ClearPose + VGSD. P3 needs VGSD.

## 8. Pretrained checkpoint setup
- **No checkpoint needed to START training** — P1/P2/P3 train from scratch (DINOv2 pretrained backbone + random heads).
- **DINOv2 ViT-S/14**: fetched at runtime via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` → needs internet on first run (or a pre-warmed `~/.cache/torch/hub`).
- **RAFT-large**: `torchvision Raft_Large_Weights.DEFAULT` → ~20 MB auto-download on first use.
- The legacy `checkpoints/spectra_best.pth` (broken-RAFT, 0.9217) is **reference only** — do not use it to initialize corrected training.

## 9. Repository setup
```bash
git clone <repo-url> spectra && cd spectra
git checkout 5ca9f0f            # base commit
# Apply the corrected-RAFT working-tree change (it is NOT committed). Either:
#   (a) copy flow/raft_wrapper.py from the local machine, or
#   (b) commit the fix locally and push a branch, then checkout here.
# Verify (see 12).
```
Also transfer `configs/config.yaml` (has `wandb_project: null`) if not committed.

## 10. Environment verification
```bash
python scripts/cloud_preflight.py            # imports, CUDA, GPU/VRAM, config, outputs
python scripts/cloud_preflight.py --load-models   # also instantiate DINOv2+RAFT (downloads weights)
```
Must print `PREFLIGHT PASSED`.

## 11. Dataset verification
`cloud_preflight.py` already checks dataset counts (5000/1000/4428, 105 VGSD clips, ClearPose scenes). It FAILS if any count is wrong.

## 12. RAFT verification
```bash
python scripts/raft_sanity_test.py
```
Confirms corrected `[-1,1]` active, old `[0,255]` absent, identical-frame flow ≈ 0, real-pair flow > 0. Must print `RAFT SANITY PASSED`.

## 13. Tiny smoke test (optional, ~a few min)
Run P1 for 1 epoch to a throwaway dir to confirm end-to-end before the full run:
```bash
# (edit a COPY of config if you want 1 epoch; do NOT change the real config for the real run)
# Simplest: trust preflight + raft sanity; the real P1 validates at E1 within ~20 min on cloud.
```

## 14. P1 — corrected final SPECTRA headline (MANDATORY)
```bash
WANDB_MODE=offline HF_HUB_OFFLINE=0 python scripts/run_corrected_headline.py
# -> EXPERIMENTS/13_corrected_headline_2026-10-02/checkpoints/spectra_best.pth  (best-val)
```
Full SPECTRA (ofcv+brf, no gnn), Trans10K, 10 ep, 448, bs4, seed42, corrected RAFT. (Already completed locally under throttling = 0.9229; re-run on cloud only if you want clean-timing provenance.)

## 15. P1.5 — cross-dataset refresh on the NEW corrected checkpoint (inference only)
```bash
python eval/local_cross_dataset_eval.py \
  --checkpoint EXPERIMENTS/13_corrected_headline_2026-10-02/checkpoints/spectra_best.pth \
  --max 1500 --bs 4 \
  --out EXPERIMENTS/13_corrected_headline_2026-10-02/crossdataset_corrected.json
# -> ClearPose (n=1500), VGSD video + static (n=1500 each)
```

## 16. P2 — corrected Trans10K ablation (30 ep)  [cloud; authorized to run ONLY ofcv_only + ofcv_brf]
`neither` & `brf_only` are RAFT-independent → **reuse** `EXPERIMENTS/03_ablation_30ep` (do NOT rerun). The existing runner already supports a variant filter and validates every epoch (exact protocol that produced `03`):
```bash
WANDB_MODE=offline python train/run_full_ablations.py \
  --epochs 30 --variants ofcv_only ofcv_brf \
  --output EXPERIMENTS/17_corrected_trans10k_ablation_2026-10-02
# -> per-variant result.json (per-epoch val IoU, best/final, duration) + summary.json/summary.md
```
Only scientific change vs `03` = corrected RAFT (working-tree). Do NOT pass `neither`/`brf_only`/`full`.

## 17. P3 — corrected VGSD ablation (12 ep, 336)  [cloud; ONLY ofcv_only + ofcv_brf]
`neither` (0.8232) & `brf_only` (0.7862) reuse `EXPERIMENTS/10_vgsd_native` (do NOT rerun). Run the written driver (reuses `train_variant` from `run_vgsd_ablation.py` verbatim):
```bash
WANDB_MODE=offline python scripts/run_vgsd_corrected.py
# -> EXPERIMENTS/18_corrected_vgsd_ablation_2026-10-02/vgsd_4way/{ofcv_only,ofcv_brf}/result.json (+ meta.json)
```

## 18. P4 — optional minimal multi-seed (3 seeds, 10 ep)  [optional; do not run without authorization]
```bash
python train/run_multi_seed.py --variant neither --seeds 42 123 3407 --output EXPERIMENTS/15_multiseed_optional
python train/run_multi_seed.py --variant full    --seeds 42 123 3407 --output EXPERIMENTS/15_multiseed_optional
python eval/aggregate_seeds.py   # mean±std
```

## 19. Expected output locations
- P1: `EXPERIMENTS/13_corrected_headline_2026-10-02/{checkpoints,meta.json,run.log}`
- P1.5: `EXPERIMENTS/13_.../crossdataset_corrected.json`
- P2: `EXPERIMENTS/17_corrected_trans10k_ablation_2026-10-02/` (per the P2 command in §16)
- P3: `EXPERIMENTS/18_corrected_vgsd_ablation_2026-10-02/vgsd_4way/` (per the written driver)
- P4: `EXPERIMENTS/15_multiseed_optional/`

## 20. Failure handling
- Preflight/RAFT-sanity FAIL → fix before any training; never "work around" by changing the protocol.
- OOM at 448/bs4 on <8 GB → use a ≥24 GB GPU; do **not** lower batch size (frozen protocol). Report instead.
- DINOv2/RAFT download blocked → pre-warm `~/.cache/torch/hub` and set `HF_HUB_OFFLINE`/torch hub offline.
- Any crash → STOP, save the log, report; do not auto-modify the protocol.

## 21. Safe terminate / restart
- Each run writes `checkpoint_epochNNN.pth` every epoch; a killed run leaves a clean partial dir — **start a fresh timestamped dir for the re-run; never resume into a different protocol**, and never overwrite a completed run.
- `train_baseline.py` supports `--resume <ckpt>` for same-protocol resumption, but for publication cleanliness prefer a fresh from-scratch run.
- Never overwrite `checkpoints/spectra_best.pth` (legacy reference).
