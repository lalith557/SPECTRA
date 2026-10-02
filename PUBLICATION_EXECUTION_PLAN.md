# SPECTRA — Publication Execution Plan (provenance ledger)

Separates **LEGACY / broken-RAFT** evidence from **VALID / corrected** evidence so no headline
table silently mixes protocols. See also `EXPERIMENTS/RAFT_BUG_AUDIT.md` and
`EXPERIMENTS/PUBLICATION_INTEGRITY_AUDIT.md`. **Do not write the paper yet.**

## RAFT protocol definition
- **Broken (legacy):** `_denorm` → `(x*255.0).clamp(0,255)` fed to RAFT (expects [-1,1]).
- **Corrected:** `_denorm` → `(x*2.0-1.0).clamp(-1,1)` (== torchvision `weights.transforms()`). Working-tree-only (uncommitted) on base `5ca9f0f`; must be preserved exactly.

## LEGACY / BROKEN-RAFT — do NOT present as corrected final
| Result | Value | Where | Status |
|---|---|---|---|
| Original SPECTRA headline (10ep, full, OFCV+BRF) | 0.9217 | `checkpoints/spectra_best.pth` | **Superseded** by corrected P1 (0.9229). Reference/provenance only. |
| Trans10K ablation `ofcv_only` | 0.9350 | `EXPERIMENTS/03_ablation_30ep` | Legacy; replace via P2. |
| Trans10K ablation `ofcv_brf` | 0.9348 | `EXPERIMENTS/03_ablation_30ep` | Legacy; replace via P2. |
| Trans10K ablation `full`/GNN | 0.9325 | `EXPERIMENTS/03_ablation_30ep` | Legacy; replace via P2 (or drop GNN). |
| VGSD `ofcv_only` | 0.7937 | `EXPERIMENTS/10_vgsd_native` | Legacy; replace via P3. |
| VGSD `ofcv_brf` (stale/interrupted) | 0.8633 / E10 0.78 | `EXPERIMENTS/10_vgsd_native` | Legacy/incomplete; replace via P3. |
| Legacy 10-ep OFCV/`full` multi-seed | σ=0.0008 | `results/multi_seed` | Legacy; optional P4 if needed. |
| Cross-dataset ClearPose 0.5223 / VGSD 0.9071 | — | `EXPERIMENTS/06_crossdataset` | Inference on legacy ckpt (BN-adapted). Refresh via P1.5 on corrected ckpt. |

## VALID / CORRECTED — usable as final (with provenance)
| Result | Value | Where | Status |
|---|---|---|---|
| **Corrected SPECTRA headline** (P1) | **0.9229** val IoU (E10) | `EXPERIMENTS/13_corrected_headline_2026-10-02/` | **Completed under thermal throttling — numerically valid** (throttling slows clocks only). Pending user acceptance vs optional clean-timing cloud re-run. |
| Backbone "neither" (30ep) | 0.9322 | `EXPERIMENTS/03_ablation_30ep` | RAFT-independent → valid as-is. |
| `brf_only` (30ep) | 0.9317 | `EXPERIMENTS/03_ablation_30ep` | RAFT-independent → valid. |
| VGSD `neither` / `brf_only` | 0.8232 / 0.7862 | `EXPERIMENTS/10_vgsd_native` | RAFT-independent → valid. |
| OFCV mechanistic intervention | REAL≈SHUF≈CONST≈RAND | `results/diagnostics/gate_intervention.json` | Corrected ckpt (D). OFCV inert. |
| BRF mechanistic intervention | REAL≈SHUF≈CONST≈RAND | `results/diagnostics/brf_intervention.json` | RAFT-independent question. BRF inert. |
| OFCV temporal (real vs dup) | ΔOFCV≈0, Δpred≈0 | `results/diagnostics/phase3_4_ofcv_temporal.json` | Inference, BN-adapted flow (B). OFCV inert. |

## Corrected headline vs legacy (context only — do NOT merge)
Legacy broken-RAFT **0.9217** → corrected **0.9229** (**+0.0012**, within noise). Present the corrected 0.9229 as the final headline; cite 0.9217 only as "pre-correction reference."

## Required provenance for EVERY final-paper number
experiment ID · checkpoint path · dataset + split · seed · full protocol (res/epochs/bs/opt/sched/loss/aug) · RAFT protocol (corrected) · result file · git state. Template lives in each experiment dir's `README.md` / `meta.json`.

## Execution order (all on cloud; each manually authorized)
P1 ✅ (done locally, valid-under-throttle; optional clean re-run) → P1.5 (cross-dataset refresh, inference) → P2 (Trans10K OFCV ablation, 30ep) → P3 (VGSD OFCV ablation, 12ep/336) → P4 (optional multi-seed). Freeze all results, THEN write the paper in one pass.

## Paper tables needing a provenance label or replacement
- Headline/matched-budget table (0.9217 → 0.9229 corrected).
- Main ablation table (OFCV/`full` cells → P2 corrected; neither/brf_only keep).
- VGSD ablation table (OFCV cells → P3 corrected; neither/brf_only keep).
- Cross-dataset table (→ P1.5 corrected-ckpt refresh).
- Multi-seed table (legacy → optional P4 corrected).
- Mechanistic-intervention tables (already corrected/valid).
