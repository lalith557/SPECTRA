"""
scripts/consolidate_results.py

Gather every real result of the SPECTRA project into ONE folder: EXPERIMENTS/.
Idempotent: safe to re-run. Copies (does not move) so originals stay in place.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "EXPERIMENTS"

# (subfolder, list of source paths to copy in)
PLAN = {
    "01_matched_comparison": [
        "release/metrics/comparison_table.json",
        "benchmarks/_comparison/comparison.json",
    ],
    "02_converged_baseline": [
        "benchmarks/deeplabv3p/results.json",
        "benchmarks/deeplabv3p_converged.log",
    ],
    "03_ablation_30ep": [
        "results/full_ablations/summary.json",
        "results/full_ablations/table.md",
        "results/full_ablations/table.tex",
        "results/full_ablations/convergence.pdf",
    ],
    "04_multiseed": [
        "results/multi_seed/full/aggregated.json",
        "results/multi_seed/full/summary.json",
    ],
    "06_crossdataset": [
        "outputs/cross_dataset/local_results.json",
        "outputs/cross_dataset/results.json",
    ],
    "07_robustness": [
        "release/metrics/final_robustness.json",   # 100-image subset
    ],
    "08_failure": [
        "release/metrics/failure_summary.json",
    ],
    "00_training_curves": [
        "release/metrics/per_epoch_metrics.json",
        "release/metrics/ablation_table.json",
    ],
}


def copy_in(sub, srcs):
    d = DST / sub
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in srcs:
        sp = ROOT / s
        if sp.exists():
            shutil.copy2(sp, d / sp.name)
            n += 1
        else:
            print(f"   (missing, skipped) {s}")
    return n


def main():
    DST.mkdir(exist_ok=True)
    total = 0
    for sub, srcs in PLAN.items():
        c = copy_in(sub, srcs)
        total += c
        print(f"[{sub}] copied {c} file(s)")
    # placeholders for pending experiments so the structure is visible
    for sub in ["05_bootstrap_ci", "09_segformer_b5", "10_vgsd_native"]:
        (DST / sub).mkdir(parents=True, exist_ok=True)
    print(f"consolidated {total} files into {DST}")


if __name__ == "__main__":
    main()
