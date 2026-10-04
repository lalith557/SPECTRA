"""
spectra/eval/statistical_significance.py

Test statistical significance of SPECTRA vs. a baseline using:
  - paired bootstrap (recommended for segmentation metrics on the same images)
  - paired t-test on per-image IoU
  - Wilcoxon signed-rank test (non-parametric)

The input is a JSON of per-image metrics (one record per image) for each model.
Generate per-image records with eval/per_image_eval.py.

Usage:
    python eval/statistical_significance.py \
        --ours results/spectra_per_image.json \
        --baseline results/segformer_b5_per_image.json \
        --metric iou
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_per_image(path: str) -> Dict[str, float]:
    """JSON file: [{image_path, iou, f_measure, mae, ber}, ...]"""
    with open(path) as f:
        data = json.load(f)
    return {r["image_path"]: r for r in data}


def paired_records(a: Dict, b: Dict, metric: str) -> Tuple[List[float], List[float]]:
    keys = sorted(set(a) & set(b))
    xa = [a[k][metric] for k in keys]
    xb = [b[k][metric] for k in keys]
    return xa, xb


def paired_t(x: List[float], y: List[float]) -> Tuple[float, float]:
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("inf") if mean_d != 0 else 0.0, 0.0
    t = mean_d / (sd / math.sqrt(n))
    # Two-sided p, normal approximation
    # (for higher precision, use scipy.stats.t if present)
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    except ImportError:
        # Normal approximation
        p = 2 * (1 - _phi(abs(t)))
    return t, p


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilcoxon(x: List[float], y: List[float]) -> Tuple[float, float]:
    try:
        from scipy.stats import wilcoxon as scipy_wilcox
        stat, p = scipy_wilcox(x, y, zero_method="wilcox")
        return float(stat), float(p)
    except ImportError:
        return float("nan"), float("nan")


def paired_bootstrap(
    x: List[float], y: List[float],
    n_resamples: int = 10000, seed: int = 0,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Returns (observed_mean_diff, two_sided_p, 95% CI).
    Tests H0: mean(x) == mean(y).
    """
    rng = random.Random(seed)
    n = len(x)
    diffs = [a - b for a, b in zip(x, y)]
    observed = sum(diffs) / n

    # Centre diffs to enforce H0 (mean(diffs) = 0)
    centred = [d - observed for d in diffs]

    count_more_extreme = 0
    boot_means = []
    for _ in range(n_resamples):
        sample = [centred[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        boot_means.append(m)
        if abs(m) >= abs(observed):
            count_more_extreme += 1
    p_value = (count_more_extreme + 1) / (n_resamples + 1)

    # CI: percentile of raw diffs (uncentred)
    boot_raw = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_raw.append(sum(sample) / n)
    boot_raw.sort()
    lo = boot_raw[int(0.025 * n_resamples)]
    hi = boot_raw[int(0.975 * n_resamples)]
    return observed, p_value, (lo, hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours",     required=True, help="Per-image JSON for SPECTRA")
    p.add_argument("--baseline", required=True, help="Per-image JSON for baseline")
    p.add_argument("--metric",   default="iou",
                   choices=["iou", "f_measure", "mae", "ber"])
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--output",   default="results/significance_test.json")
    args = p.parse_args()

    a = load_per_image(args.ours)
    b = load_per_image(args.baseline)
    x, y = paired_records(a, b, args.metric)

    if len(x) == 0:
        raise RuntimeError("No paired images between the two files.")

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    t_stat, t_p = paired_t(x, y)
    w_stat, w_p = wilcoxon(x, y)
    obs, bp, ci = paired_bootstrap(x, y, n_resamples=args.n_bootstrap)

    result = {
        "ours_file":     args.ours,
        "baseline_file": args.baseline,
        "metric":        args.metric,
        "n_paired":      len(x),
        "mean_ours":     mean_x,
        "mean_baseline": mean_y,
        "mean_diff":     obs,
        "t_stat":        t_stat, "t_p":  t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "bootstrap_p":   bp,
        "bootstrap_95ci": list(ci),
        "significant_at_0.05": bp < 0.05,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
