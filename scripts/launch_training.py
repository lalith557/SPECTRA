"""
spectra/scripts/launch_training.py

One-command launcher for the SPECTRA training plan. Designed for a single
RTX 4070 (12 GB VRAM) running unattended overnight / over a weekend.

Features
--------
1. GPU detection + VRAM check + auto batch-size tuning.
2. Sequential plan: A1 (eval) -> A4 (ablations) -> A3 (multi-seed) -> A2 (baselines).
3. Per-step state file (`results/_run_state.json`) -> resumable on Ctrl+C / crash.
4. Live ETA based on first epoch's wall-clock.
5. Logs to console *and* `results/_launcher.log`.
6. Graceful shutdown — pressing Ctrl+C twice cancels cleanly.
7. Skips steps already marked done (`--force` to re-run them).

Usage
-----
    # Run the whole plan (overnight + weekend friendly)
    python scripts/launch_training.py

    # Just one phase
    python scripts/launch_training.py --only A4
    python scripts/launch_training.py --only A3 A1

    # Restart a step from scratch (clears its state)
    python scripts/launch_training.py --force A4

    # See what would run without executing
    python scripts/launch_training.py --dry-run

Tips
----
- Open `results/_launcher.log` in a second terminal to tail progress:
    Get-Content -Wait results/_launcher.log
- The script keeps the workstation busy but should leave 1-2 GB VRAM headroom
  on a 12 GB 4070. Close Chrome / other GPU apps before launching.
- Resume after reboot: just re-run the same command — completed steps are
  skipped automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "results" / "_run_state.json"
LOG_PATH   = ROOT / "results" / "_launcher.log"
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"),
              flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h: return f"{h:d}h{m:02d}m"
    if m: return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# ---------------------------------------------------------------------------
# State persistence (resumability)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"steps": {}}


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# GPU detection + VRAM-aware batch size
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    name: str = "cpu"
    vram_gb: float = 0.0
    cuda: bool = False


def detect_gpu() -> GpuInfo:
    try:
        import torch
        if not torch.cuda.is_available():
            return GpuInfo()
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return GpuInfo(
            name=props.name,
            vram_gb=props.total_memory / (1024 ** 3),
            cuda=True,
        )
    except Exception as e:
        _log(f"GPU detection failed: {e}")
        return GpuInfo()


def pick_batch_size(gpu: GpuInfo, requested: int) -> int:
    if not gpu.cuda:
        return min(requested, 2)
    # SPECTRA at 448² with DINOv2-S + RAFT ~6-7 GB per sample with AMP on.
    # Headroom margins below leave 1-2 GB for the OS + Chrome / VS Code.
    if gpu.vram_gb < 9:    return min(requested, 1)   # 8 GB laptop GPUs
    if gpu.vram_gb < 13:   return min(requested, 2)   # 12 GB (desktop 4070)
    if gpu.vram_gb < 20:   return min(requested, 3)   # 16 GB
    return requested                                  # 24 GB+


# ---------------------------------------------------------------------------
# Training plan
# ---------------------------------------------------------------------------

@dataclass
class Step:
    id: str
    title: str
    cmd: List[str]
    est_minutes: int           # rough estimate on a 4070 (~65% of 4090)
    requires_checkpoint: bool = False


def build_plan(batch_size: int, ablation_epochs: int, multi_seed_epochs: int,
               baseline_epochs: int, seeds: List[int],
               include_baseline_training: bool) -> List[Step]:
    bs = str(batch_size)
    # Important — most of the heavy commands accept the same args. We pass
    # `--config` and let each script override what it needs.
    return [
        # ── A1: cross-dataset eval (cheap, needs checkpoint) ────────────────
        Step(
            id="A1_cross_dataset",
            title="A1 — Cross-dataset eval (Trans10K / GSD / GSD-S / RGBD-GSD / ClearPose)",
            cmd=[sys.executable,"eval/cross_dataset_eval.py",
                 "--config", "configs/config.yaml",
                 "--checkpoint", "checkpoints/spectra_best.pth",
                 "--datasets", "trans10k:test", "gsd:test",
                              "gsd_s:test", "rgbd_gsd:test",
                              "clearpose:all",
                 "--batch-size", bs,
                 "--output", "outputs/cross_dataset/"],
            est_minutes=15,
            requires_checkpoint=True,
        ),
        # ── A4: full ablations (longest single step, ~32 h on 4070) ─────────
        Step(
            id="A4_full_ablations",
            title=f"A4 — Full-length ablations ({ablation_epochs} epochs × 5 variants)",
            cmd=[sys.executable,"train/run_full_ablations.py",
                 "--config", "configs/config.yaml",
                 "--epochs", str(ablation_epochs),
                 "--output", "results/full_ablations"],
            # 4070 laptop @ bs=1 (8 GB): ~14 min/ep on Trans10K, 5 variants
            est_minutes=int(ablation_epochs * 5 * 14),
        ),
        # ── A4 post: ablation analysis (figures / tables) ───────────────────
        Step(
            id="A4_analysis",
            title="A4 — Ablation analysis (curves + table)",
            cmd=[sys.executable,"eval/ablation_analysis.py",
                 "--input", "results/full_ablations/summary.json",
                 "--output-dir", "results/full_ablations"],
            est_minutes=2,
        ),
        # ── A3: multi-seed (one seed at a time, resumable) ──────────────────
        *[
            Step(
                id=f"A3_seed_{seed}",
                title=f"A3 — Multi-seed training (seed {seed}, {multi_seed_epochs} ep)",
                cmd=[sys.executable, "train/run_multi_seed.py",
                     "--config", "configs/config.yaml",
                     "--seeds", str(seed), "--variant", "full",
                     "--output", "results/multi_seed"],
                # 4070 laptop @ bs=1: ~14 min/ep
                est_minutes=int(multi_seed_epochs * 14),
            )
            for seed in seeds
        ],
        Step(
            id="A3_aggregate",
            title="A3 — Aggregate seeds + significance",
            cmd=[sys.executable,"eval/aggregate_seeds.py",
                 "--config", "configs/config.yaml",
                 "--variant", "full"],
            est_minutes=10,
        ),
        # ── A2: strong baselines ────────────────────────────────────────────
        # Trainable baselines are opt-in (--include-baseline-training).
        # By default we rely on published numbers + zero-shot SAM2/HQ-SAM.
        *(
            [
                Step(
                    id="A2_segformer_b5",
                    title=f"A2 — SegFormer-B5 ({baseline_epochs} ep)",
                    cmd=[sys.executable, "benchmarks/run_strong_baselines.py",
                         "--model", "segformer_b5",
                         "--config", "configs/config.yaml",
                         "--epochs", str(baseline_epochs),
                         "--batch-size", bs],
                    est_minutes=int(baseline_epochs * 18),     # ~18 min/ep on 8GB
                ),
                Step(
                    id="A2_mask2former_b",
                    title=f"A2 — Mask2Former-Swin-B ({baseline_epochs} ep)",
                    cmd=[sys.executable, "benchmarks/run_strong_baselines.py",
                         "--model", "mask2former_b",
                         "--config", "configs/config.yaml",
                         "--epochs", str(baseline_epochs),
                         "--batch-size", bs],
                    est_minutes=int(baseline_epochs * 22),     # ~22 min/ep on 8GB
                ),
            ] if include_baseline_training else []
        ),
        Step(
            id="A2_sam2_zs",
            title="A2 — SAM2 zero-shot eval",
            cmd=[sys.executable,"benchmarks/run_strong_baselines.py",
                 "--model", "sam2", "--zero-shot"],
            est_minutes=45,
        ),
        Step(
            id="A2_hq_sam_zs",
            title="A2 — HQ-SAM zero-shot eval",
            cmd=[sys.executable,"benchmarks/run_strong_baselines.py",
                 "--model", "hq_sam", "--zero-shot"],
            est_minutes=60,
        ),
        Step(
            id="A2_published",
            title="A2 — Published baseline numbers table",
            cmd=[sys.executable,"benchmarks/published_results.py"],
            est_minutes=1,
        ),
        Step(
            id="A2_compose",
            title="A2 — Compose baseline comparison",
            cmd=[sys.executable,"benchmarks/compose_comparison.py"],
            est_minutes=1,
        ),
        # ── Figures (data-aware — auto-promotes to real numbers) ────────────
        Step(
            id="figures",
            title="Regenerate 20 paper figures (real-data aware)",
            cmd=[sys.executable,"scripts/generate_paper_figures.py"],
            est_minutes=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Runner with state + interrupt handling
# ---------------------------------------------------------------------------

_INTERRUPTED = {"count": 0}


def _sigint(signum, frame):
    _INTERRUPTED["count"] += 1
    if _INTERRUPTED["count"] == 1:
        _log("Ctrl+C received — finishing current step, then exiting. "
             "Press Ctrl+C again to cancel immediately.")
    else:
        _log("Second Ctrl+C — terminating now.")
        sys.exit(130)


def run_step(step: Step, state: dict, dry: bool) -> bool:
    rec = state["steps"].setdefault(step.id, {})
    if rec.get("status") == "completed":
        _log(f"[skip] {step.id} (already completed at {rec['finished_at']})")
        return True
    if dry:
        _log(f"[plan] {step.id}: {' '.join(shlex.quote(c) for c in step.cmd)}")
        return True

    started = time.time()
    rec.update({"status": "running", "started_at":
                datetime.now().isoformat(timespec="seconds"),
                "estimate_minutes": step.est_minutes})
    save_state(state)

    eta_end = datetime.now() + timedelta(minutes=step.est_minutes)
    _log(f"=== {step.title} ===")
    _log(f"  cmd: {' '.join(shlex.quote(c) for c in step.cmd)}")
    _log(f"  estimated: {step.est_minutes} min (~ done by "
         f"{eta_end.strftime('%a %H:%M')})")

    # Stream subprocess stdout to both our log and the console
    log_f = open(LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(
        step.cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            log_f.write(line); log_f.flush()
    finally:
        proc.wait()
        log_f.close()

    duration = time.time() - started
    if proc.returncode == 0:
        rec.update({"status": "completed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_s": duration})
        _log(f"--- {step.id} OK in {_hms(duration)} ---")
        save_state(state)
        return True
    else:
        rec.update({"status": "failed", "returncode": proc.returncode,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_s": duration})
        _log(f"!!! {step.id} FAILED (rc={proc.returncode}) after {_hms(duration)} !!!")
        save_state(state)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only",  nargs="+", default=None,
                   help="Run only steps whose ID starts with these prefixes.")
    p.add_argument("--skip",  nargs="+", default=[],
                   help="Skip steps whose ID starts with these prefixes.")
    p.add_argument("--force", nargs="+", default=[],
                   help="Re-run completed steps matching these prefixes.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-size",       type=int, default=4)
    p.add_argument("--ablation-epochs",   type=int, default=30,
                   help="Was 50 — reduced to 30 as the laptop-friendly 'full' schedule.")
    p.add_argument("--multi-seed-epochs", type=int, default=30)
    p.add_argument("--baseline-epochs",   type=int, default=60)
    p.add_argument("--include-baseline-training", action="store_true",
                   help="Also retrain SegFormer-B5 + Mask2Former-B (adds ~38 h). "
                        "Default plan uses published numbers + SAM2/HQ-SAM zero-shot only.")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13],
                   help="Multi-seed list (was 5; defaults to 3 — minimum for mean±std).")
    p.add_argument("--ignore-checkpoint-missing", action="store_true",
                   help="Don't bail if checkpoints/spectra_best.pth is absent.")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _sigint)

    # GPU + batch size sanity
    gpu = detect_gpu()
    if gpu.cuda:
        _log(f"GPU: {gpu.name}  VRAM: {gpu.vram_gb:.1f} GB")
    else:
        _log("WARNING: no CUDA — running on CPU will be impossibly slow. "
             "Press Ctrl+C now if that's not intentional.")
    bs = pick_batch_size(gpu, args.batch_size)
    if bs != args.batch_size:
        _log(f"Auto-tuned batch size: {args.batch_size} -> {bs} "
             f"(VRAM {gpu.vram_gb:.1f} GB).")

    plan = build_plan(
        batch_size=bs,
        ablation_epochs=args.ablation_epochs,
        multi_seed_epochs=args.multi_seed_epochs,
        baseline_epochs=args.baseline_epochs,
        seeds=args.seeds,
        include_baseline_training=args.include_baseline_training,
    )

    # Subset selection
    if args.only:
        plan = [s for s in plan if any(s.id.startswith(o) for o in args.only)]
    if args.skip:
        plan = [s for s in plan if not any(s.id.startswith(sk) for sk in args.skip)]
    if not plan:
        _log("Nothing to run after --only / --skip filters."); return

    # Clear forced step states so they re-run
    state = load_state()
    for f in args.force:
        for sid in list(state["steps"]):
            if sid.startswith(f):
                _log(f"--force: clearing prior state for {sid}")
                state["steps"].pop(sid, None)
    save_state(state)

    # Checkpoint check for steps that need it
    ckpt = ROOT / "checkpoints" / "spectra_best.pth"
    needs_ckpt = [s for s in plan if s.requires_checkpoint]
    if needs_ckpt and not ckpt.exists() and not args.ignore_checkpoint_missing:
        _log(f"NOTE: {len(needs_ckpt)} steps require {ckpt} which is missing. "
             "These will be deferred — train SPECTRA first (or pass "
             "--ignore-checkpoint-missing to attempt anyway).")
        plan = [s for s in plan if not s.requires_checkpoint] + needs_ckpt

    # Show plan + ETA
    total = sum(s.est_minutes for s in plan if state["steps"].get(s.id, {}).get("status") != "completed")
    _log(f"Plan: {len(plan)} steps, ~{_hms(total*60)} of compute remaining "
         f"(ETA {(datetime.now()+timedelta(minutes=total)).strftime('%a %d %b %H:%M')})")
    for s in plan:
        st = state["steps"].get(s.id, {}).get("status", "pending")
        _log(f"  [{st:9}] {s.id}  (~{s.est_minutes} min)  {s.title}")

    if args.dry_run:
        _log("Dry run — exiting.")
        return

    # Execute
    plan_start = time.time()
    for i, step in enumerate(plan, 1):
        if _INTERRUPTED["count"]:
            _log("Interrupted — stopping before next step.")
            break
        _log(f"\n>>> step {i}/{len(plan)} <<<")
        ok = run_step(step, state, dry=False)
        if not ok and not step.id.startswith("A2_") and not step.id.startswith("A1_"):
            # Critical training steps — abort the plan
            _log("Critical step failed. Stopping. Re-run to resume after fixing.")
            break

    _log(f"\n=== launcher done. total wall: {_hms(time.time()-plan_start)} ===")
    _log(f"state: {STATE_PATH}")
    _log(f"log:   {LOG_PATH}")


if __name__ == "__main__":
    main()
