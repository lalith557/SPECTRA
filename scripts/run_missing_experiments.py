"""
scripts/run_missing_experiments.py

Runs the GPU experiments that were missing, SEQUENTIALLY (one GPU job at a time),
storing every result under the consolidated EXPERIMENTS/ folder.

Steps (each is skippable / resumable):
  7) Full-test robustness on all 4,428 images   (~hours, no training)
  1) Multi-seed 5x3: the variants/seeds not already on disk (~many hours)
  2) SegFormer-B5 baseline under the binary protocol (batch 2 for 8 GB)

Usage:
  venv/Scripts/python.exe scripts/run_missing_experiments.py            # all
  venv/Scripts/python.exe scripts/run_missing_experiments.py --only 7   # subset
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
EXP = ROOT / "EXPERIMENTS"


def run(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {' '.join(cmd)}\n    log -> {logfile}", flush=True)
    with open(logfile, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"    returncode={rc}", flush=True)
    return rc


def step_robustness():
    out = EXP / "07_robustness" / "full_4428"
    run([PY, "eval/robustness_eval.py", "--checkpoint", "checkpoints/spectra_best.pth",
         "--output", str(out), "--n-samples", "4428"], out / "run.log")


def step_multiseed():
    # full already has seeds 7,13,42; the other 4 variants only have seed 42.
    todo = {"neither": [7, 13], "ofcv_only": [7, 13],
            "brf_only": [7, 13], "ofcv_brf": [7, 13]}
    for variant, seeds in todo.items():
        run([PY, "train/run_multi_seed.py", "--variant", variant,
             "--seeds", *map(str, seeds), "--output", "results/multi_seed"],
            EXP / "04_multiseed" / f"{variant}_seeds.log")
    # re-aggregate all variants/seeds
    run([PY, "eval/aggregate_seeds.py"], EXP / "04_multiseed" / "aggregate.log")


def step_segformer_b5():
    out = EXP / "09_segformer_b5"
    run([PY, "benchmarks/run_strong_baselines.py", "--model", "segformer_b5",
         "--epochs", "40", "--batch-size", "2"], out / "run.log")


STEPS = {7: step_robustness, 1: step_multiseed, 2: step_segformer_b5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, default=[7, 1, 2],
                    help="which step numbers to run (default: 7 1 2)")
    args = ap.parse_args()
    for n in args.only:
        if n in STEPS:
            STEPS[n]()
        else:
            print(f"unknown step {n}")
    print("\nAll requested steps finished.")


if __name__ == "__main__":
    main()
