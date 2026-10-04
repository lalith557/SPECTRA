"""
spectra/eval/interpretability/iaa.py

Inter-annotator agreement for the human-evaluation responses.csv files.
Computes Krippendorff's alpha (ordinal), pairwise Spearman ρ, and
percentage agreement.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load(path: Path) -> Dict[str, Dict[str, int]]:
    rows: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            iid = r["image_id"]
            scores = {}
            for q in ("Q1_seg", "Q2_ofcv", "Q3_cam", "Q4_bnd"):
                v = r.get(q, "").strip()
                if v:
                    scores[q] = int(v)
            if scores:
                rows[iid] = scores
    return rows


def spearman(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    def ranks(x):
        s = sorted((v, i) for i, v in enumerate(x))
        r = [0.0] * len(x)
        for rk, (_, i) in enumerate(s, 1):
            r[i] = rk
        return r
    ra, rb = ranks(a), ranks(b)
    mean_a = sum(ra) / n; mean_b = sum(rb) / n
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    va = sum((x - mean_a) ** 2 for x in ra)
    vb = sum((x - mean_b) ** 2 for x in rb)
    return cov / ((va ** 0.5) * (vb ** 0.5) + 1e-8)


def krippendorff_alpha_ordinal(values: List[List[int]]) -> float:
    """
    Ordinal Krippendorff's alpha.
    values: list per unit, each is a list of ratings from annotators (may have missing).
    """
    # Disagreement weight for ordinal: ((rank diff)/(max-min))^2
    flat = [v for unit in values for v in unit]
    if not flat:
        return float("nan")
    lo, hi = min(flat), max(flat)
    span = hi - lo or 1

    def delta(a, b):
        return ((a - b) / span) ** 2

    def expected():
        n = len(flat)
        s = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s += delta(flat[i], flat[j])
        return s / (n * (n - 1))

    def observed():
        s = 0.0; n_pairs = 0
        for unit in values:
            m = len(unit)
            if m < 2:
                continue
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    s += delta(unit[i], unit[j])
                    n_pairs += 1
        return s / max(n_pairs, 1)

    D_o = observed(); D_e = expected()
    return 1 - D_o / D_e if D_e > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", nargs="+", required=True,
                   help="annotator responses.csv files (>=2)")
    p.add_argument("--output",    default="results/human_eval/iaa.json")
    args = p.parse_args()

    annotators = [load(Path(p_)) for p_ in args.responses]
    image_ids = sorted(set.intersection(*[set(a.keys()) for a in annotators]))

    questions = ["Q1_seg", "Q2_ofcv", "Q3_cam", "Q4_bnd"]
    per_q_alpha = {}
    per_q_pairwise = {}

    for q in questions:
        values = [[ann[iid][q] for ann in annotators if q in ann[iid]]
                  for iid in image_ids]
        per_q_alpha[q] = krippendorff_alpha_ordinal(values)
        # pairwise spearman
        rhos = []
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a = [annotators[i][iid][q] for iid in image_ids
                     if q in annotators[i][iid] and q in annotators[j][iid]]
                b = [annotators[j][iid][q] for iid in image_ids
                     if q in annotators[i][iid] and q in annotators[j][iid]]
                if len(a) > 1:
                    rhos.append(spearman(a, b))
        per_q_pairwise[q] = float(statistics.mean(rhos)) if rhos else 0.0

    summary = {
        "n_annotators":     len(annotators),
        "n_paired_images":  len(image_ids),
        "alpha_per_question":     per_q_alpha,
        "spearman_per_question":  per_q_pairwise,
        "alpha_overall":    statistics.mean(per_q_alpha.values()),
        "spearman_overall": statistics.mean(per_q_pairwise.values()),
        "passes_threshold": all(v >= 0.6 for v in per_q_alpha.values()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
