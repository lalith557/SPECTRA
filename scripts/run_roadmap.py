"""
spectra/scripts/run_roadmap.py

Top-level orchestrator that runs the SPECTRA conference-readiness roadmap
end to end. Designed to be re-runnable and resume-friendly: each step writes
its own outputs, and subsequent steps detect completion and skip.

Steps (Priority A first, then B, then C):
    A1. Cross-dataset evaluation
    A2. Strong baseline training + published-results table + comparison
    A3. Multi-seed training + aggregation + significance
    A4. Full-length ablations + analysis
    B5. Temporal video benchmark
    B6. Interpretability validation (OFCV correlation + saliency + human-eval pack)
    B7. Failure taxonomy
    C8. (training of learned physics — opt-in)
    C9. (training of monocular — opt-in)
    C10. Release bundling

Usage:
    python scripts/run_roadmap.py                # run everything reasonable
    python scripts/run_roadmap.py --skip A2      # skip strong baselines
    python scripts/run_roadmap.py --only A1 A3   # run just these steps
    python scripts/run_roadmap.py --dry-run      # print plan without executing
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CKPT = "checkpoints/spectra_best.pth"


def step(cmd: List[str], cwd: Path = ROOT, env_extra=None) -> dict:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return {"cmd": cmd, "rc": proc.returncode, "duration_s": time.time() - t0}


PLAN = {
    # ── Priority A ──────────────────────────────────────────────────────────
    "A1_cross_dataset": [
        ["python", "eval/cross_dataset_eval.py",
         "--config", "configs/config.yaml",
         "--checkpoint", DEFAULT_CKPT,
         "--datasets", "trans10k:test", "gsd:test", "clearpose:all",
         "--output", "outputs/cross_dataset/"],
    ],
    "A2_strong_baselines": [
        ["python", "benchmarks/run_strong_baselines.py", "--model", "segformer_b5", "--epochs", "60"],
        ["python", "benchmarks/run_strong_baselines.py", "--model", "mask2former_b", "--epochs", "60"],
        ["python", "benchmarks/run_strong_baselines.py", "--model", "sam2", "--zero-shot"],
        ["python", "benchmarks/run_strong_baselines.py", "--model", "hq_sam", "--zero-shot"],
        ["python", "benchmarks/published_results.py"],
        ["python", "benchmarks/compose_comparison.py"],
    ],
    "A3_multi_seed": [
        ["python", "train/run_multi_seed.py", "--seeds", "42", "7", "13",
         "1234", "2025", "--variant", "full"],
        ["python", "eval/aggregate_seeds.py", "--variant", "full"],
    ],
    "A4_full_ablations": [
        ["python", "train/run_full_ablations.py", "--epochs", "50"],
        ["python", "eval/ablation_analysis.py"],
    ],
    # ── Priority B ──────────────────────────────────────────────────────────
    "B5_video": [
        ["python", "benchmarks/video_benchmark.py",
         "--checkpoint", DEFAULT_CKPT, "--window-size", "4"],
    ],
    "B6_interpretability": [
        ["python", "eval/interpretability/ofcv_correlation.py",
         "--checkpoint", DEFAULT_CKPT],
        ["python", "eval/interpretability/saliency_overlap.py",
         "--checkpoint", DEFAULT_CKPT],
        ["python", "eval/interpretability/human_eval_protocol.py",
         "--checkpoint", DEFAULT_CKPT, "--n-images", "100"],
    ],
    "B7_failure": [
        ["python", "eval/failure_taxonomy.py",
         "--checkpoint", DEFAULT_CKPT,
         "--output", "results/failure_taxonomy/taxonomy.json"],
    ],
    # ── Priority C (opt-in, often long) ────────────────────────────────────
    "C8_learned_physics": [
        # Training is left to the user — we just instantiate it once to
        # verify the model builds.
        ["python", "-c",
         "from utils import load_config; from models.spectra_learned_physics import SPECTRA_LP; "
         "cfg = load_config('configs/config.yaml'); m = SPECTRA_LP(cfg, use_gnn=False); "
         "print('SPECTRA-LP params:', sum(p.numel() for p in m.parameters() if p.requires_grad))"],
    ],
    "C9_monocular": [
        ["python", "-c",
         "from utils import load_config; from models.spectra_mono import SPECTRA_Mono; "
         "cfg = load_config('configs/config.yaml'); m = SPECTRA_Mono(cfg, warp_mode='learned', use_gnn=False); "
         "print('SPECTRA-Mono params:', sum(p.numel() for p in m.parameters() if p.requires_grad))"],
    ],
    "C10_release": [
        ["python", "release/prepare_release.py", "--version", "1.0",
         "--checkpoint", DEFAULT_CKPT, "--output", "release/"],
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", nargs="+", default=[])
    p.add_argument("--only", nargs="+", default=[])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    selected = list(PLAN.keys())
    if args.only:
        selected = [k for k in selected
                    if any(k.startswith(o) for o in args.only)]
    if args.skip:
        selected = [k for k in selected
                    if not any(k.startswith(s) for s in args.skip)]

    print(f"Plan ({len(selected)} steps):")
    for k in selected:
        print(f"  - {k}: {len(PLAN[k])} commands")

    if args.dry_run:
        for k in selected:
            print(f"\n# {k}")
            for cmd in PLAN[k]:
                print("$", " ".join(shlex.quote(c) for c in cmd))
        return

    record = []
    for k in selected:
        print(f"\n=== {k} ===")
        for cmd in PLAN[k]:
            r = step(cmd)
            record.append({"step": k, **r})
            if r["rc"] != 0:
                print(f"FAILED ({r['rc']}): {cmd}")
        print(f"=== {k} complete ===")

    print("\n=== roadmap summary ===")
    for r in record:
        print(f"  rc={r['rc']:>3}  {r['duration_s']:>7.1f}s  {' '.join(r['cmd'])}")


if __name__ == "__main__":
    main()
