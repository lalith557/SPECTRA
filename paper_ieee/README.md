# SPECTRA — IEEE Conference Manuscript (Overleaf project)

Honest empirical-study framing. Every number in `main.tex` is a real measurement
traceable to a JSON in this repo (see **Provenance** below). No fabricated
results, no leaderboard numbers from incompatible protocols.

## Files
- `main.tex` — IEEEtran conference paper, **self-contained**: the 45-reference
  bibliography is embedded with `\begin{thebibliography}` (no `.bib`, no bibtex
  pass), numbered 1..45 in ascending order of first appearance.
- `references.bib` — optional; the same references in BibTeX form (not needed to
  compile `main.tex`).
- `figures/` — the 7 figures referenced by the paper (copied from `../FIGURE/`).

## Build on Overleaf (recommended — no local LaTeX needed)
1. Create a new Overleaf project → **Upload Project** → zip this `paper_ieee/`
   folder (must include `figures/`).
2. Set the compiler to **pdfLaTeX** (Menu → Compiler).
3. Overleaf provides `IEEEtran.cls` automatically.
4. Recompile once. **No bibtex pass is required** — the bibliography is embedded.

## Build locally (if you install MiKTeX/TeX Live)
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Provenance — where each headline number comes from
| Claim in paper | Source file |
|---|---|
| Matched 10-epoch comparison (Tab. III) | `release/metrics/comparison_table.json` |
| 30-epoch ablation, 5 variants (Tab. IV) | `results/full_ablations/summary.json` |
| Multi-seed test IoU (Tab. V) | `results/multi_seed/full/aggregated.json` |
| 5-epoch early-convergence ablation (Tab. VI) | `release/metrics/ablation_table.json` |
| Robustness 8×5 (Tab. VII) | `release/metrics/final_robustness.json` |
| Failure taxonomy (Tab. VIII) | `release/metrics/failure_summary.json` |
| OFCV non-collapse trajectory | `release/metrics/per_epoch_metrics.json` |
| Cross-dataset (ClearPose/VSGD) + real-video control | `outputs/cross_dataset/local_results.json` (via `eval/local_cross_dataset_eval.py`) |

## Integrity notes (read before submitting)
- **No published-SOTA table.** Prior Trans10K numbers are multi-class mean-IoU;
  ours is pooled binary foreground IoU. The paper states this explicitly in
  Sec. "Results" and deliberately does not tabulate them together.
- **The ablation is the finding.** The paper's thesis is that the physics
  modules add no accuracy beyond noise — this is what your own data shows.
  Do not re-introduce "physics improves accuracy" language.
- **Cross-dataset is listed as a limitation**, because `outputs/cross_dataset/`
  only produced valid numbers for Trans10K (others returned NaN).
