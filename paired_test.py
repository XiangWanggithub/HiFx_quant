#!/usr/bin/env python3
"""
Paired Test for Model Comparison (Wilcoxon Signed-Rank & Paired t-test)

Compares two models by computing per-question accuracy rates and running
paired statistical tests. Most useful with repeats > 1, where each question
has multiple responses and the per-question accuracy rate is a continuous
value (k/N correct out of N repeats).

With repeats=1 (binary outcomes), prefer mcnemar_test.py instead.

Usage:
    python paired_test.py <run_dir_a> <run_dir_b> [--metric acc] [--alpha 0.05]

Example:
    python test_evalscope.py --model-path /home/models/Qwen3-0.6B --dataset arc --repeats 20
    python test_evalscope.py --model-path /home/models/Qwen3-1.7B --dataset arc --repeats 20
    python paired_test.py outputs/arc_eval/20260211_150547 outputs/arc_eval/20260213_120000
"""

import argparse
import glob
import json
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired tests (Wilcoxon + t-test) comparing two models on the same benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_dir_a', help='Timestamped run directory for model A')
    parser.add_argument('run_dir_b', help='Timestamped run directory for model B')
    parser.add_argument('--metric', default='acc', help='Metric key to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    return parser.parse_args()


def load_reviews(run_dir, metric):
    """Load per-question results from review JSONL files in a run directory.

    Groups results by group_id to handle repeats.

    Returns:
        model_name: str
        results: dict mapping subset_name -> dict mapping group_id -> list[float]
    """
    reviews_dir = os.path.join(run_dir, 'reviews')
    if not os.path.isdir(reviews_dir):
        print(f"Error: No reviews directory found at {reviews_dir}", file=sys.stderr)
        sys.exit(1)

    model_dirs = [d for d in os.listdir(reviews_dir)
                  if os.path.isdir(os.path.join(reviews_dir, d))]
    if len(model_dirs) == 0:
        print(f"Error: No model directories found in {reviews_dir}", file=sys.stderr)
        sys.exit(1)
    if len(model_dirs) > 1:
        print(f"Warning: Multiple model directories found: {model_dirs}. Using first.", file=sys.stderr)
    model_name = model_dirs[0]

    model_reviews_dir = os.path.join(reviews_dir, model_name)
    jsonl_files = glob.glob(os.path.join(model_reviews_dir, '*.jsonl'))
    if not jsonl_files:
        print(f"Error: No JSONL review files found in {model_reviews_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for fpath in sorted(jsonl_files):
        subset_name = os.path.splitext(os.path.basename(fpath))[0]
        subset_results = {}
        skipped = 0
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                score_data = entry['sample_score']
                score_value = score_data['score']['value']
                if metric not in score_value:
                    skipped += 1
                    continue
                gid = score_data.get('group_id')
                if gid is None:
                    gid = entry['index']
                subset_results.setdefault(gid, []).append(score_value[metric])
        if not subset_results and skipped > 0:
            print(f"Warning: No entries with metric '{metric}' in {fpath}", file=sys.stderr)
        elif skipped > 0:
            print(f"Warning: Skipped {skipped} entries without metric '{metric}' in {os.path.basename(fpath)}", file=sys.stderr)
        results[subset_name] = subset_results

    return model_name, results


def normal_sf(z):
    """Two-sided p-value for standard normal: P(|Z| >= |z|)."""
    return math.erfc(abs(z) / math.sqrt(2))


def wilcoxon_signed_rank(diffs):
    """Wilcoxon signed-rank test on paired differences.

    Returns:
        (W_plus, W_minus, z, p_value, n_nonzero)
        Uses normal approximation with tie correction.
    """
    # Remove zero differences
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)

    if n == 0:
        return 0.0, 0.0, 0.0, 1.0, 0

    abs_d = [abs(d) for d in nonzero]
    signs = [1 if d > 0 else -1 for d in nonzero]

    # Rank |d| with tie averaging
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    tie_sizes = []
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based
        tie_sizes.append(j - i)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_plus = sum(ranks[i] for i in range(n) if signs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if signs[i] < 0)

    # Normal approximation
    mean_w = n * (n + 1) / 4
    var_w = n * (n + 1) * (2 * n + 1) / 24
    # Tie correction
    tie_correction = sum(t ** 3 - t for t in tie_sizes) / 48
    var_w -= tie_correction

    if var_w <= 0:
        return w_plus, w_minus, 0.0, 1.0, n

    # Use W+ for direction: z > 0 means model A tends to score higher
    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = normal_sf(z)

    return w_plus, w_minus, z, p, n


def paired_t_test(diffs):
    """Paired t-test on differences.

    Returns:
        (mean_diff, t_stat, p_value)
        Uses normal approximation for p-value (valid for large n).
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 1.0

    mean_d = sum(diffs) / n
    if n == 1:
        return mean_d, 0.0, 1.0

    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n)
    if se == 0:
        if mean_d == 0:
            return 0.0, 0.0, 1.0
        else:
            return mean_d, float('inf'), 0.0

    t = mean_d / se
    p = normal_sf(t)  # Normal approx, accurate for large n
    return mean_d, t, p


def print_comparison(subset_name, model_a, model_b, results_a, results_b, alpha):
    """Compare two models on a single subset using paired tests.

    Returns:
        list of per-question differences (rate_a - rate_b), or None if no overlap.
    """
    common = set(results_a.keys()) & set(results_b.keys())
    only_a = set(results_a.keys()) - set(results_b.keys())
    only_b = set(results_b.keys()) - set(results_a.keys())

    if not common:
        print(f"  {subset_name}: No overlapping questions found.")
        return None

    if only_a or only_b:
        print(f"  Note: {len(only_a)} questions only in {model_a}, {len(only_b)} only in {model_b}")

    # Compute per-question accuracy rates
    max_rep_a = max(len(results_a[idx]) for idx in common)
    max_rep_b = max(len(results_b[idx]) for idx in common)

    rates_a = []
    rates_b = []
    diffs = []
    for idx in sorted(common):
        sa = results_a[idx]
        sb = results_b[idx]
        ra = sum(s for s in sa) / len(sa)
        rb = sum(s for s in sb) / len(sb)
        rates_a.append(ra)
        rates_b.append(rb)
        diffs.append(ra - rb)

    n = len(common)
    mean_a = sum(rates_a) / n * 100
    mean_b = sum(rates_b) / n * 100

    rep_str = f", {max_rep_a} repeats" if max(max_rep_a, max_rep_b) > 1 else ""
    print(f"\n  Subset: {subset_name} ({n} questions{rep_str})")
    print(f"    {model_a} mean accuracy: {mean_a:.2f}%")
    print(f"    {model_b} mean accuracy: {mean_b:.2f}%")
    print(f"    Mean difference (A - B): {sum(diffs) / n * 100:+.2f}%")

    n_zero = sum(1 for d in diffs if d == 0.0)
    n_a_better = sum(1 for d in diffs if d > 0)
    n_b_better = sum(1 for d in diffs if d < 0)
    print(f"    Per-question breakdown: {model_a} better on {n_a_better}, "
          f"{model_b} better on {n_b_better}, tied on {n_zero}")
    print()

    # Wilcoxon signed-rank test
    w_plus, w_minus, z, p_w, n_nonzero = wilcoxon_signed_rank(diffs)
    print(f"    Wilcoxon Signed-Rank Test:")
    if n_nonzero == 0:
        print(f"      All differences are zero. No test needed.")
    else:
        print(f"      W+ = {w_plus:.1f}, W- = {w_minus:.1f} ({n_nonzero} non-zero pairs)")
        print(f"      z = {z:.4f}, p = {p_w:.4e}")
        if n_nonzero < 20:
            print(f"      (Note: n={n_nonzero} is small; normal approximation may be imprecise)")
        if p_w < alpha:
            better = model_a if z > 0 else model_b
            print(f"      Result: SIGNIFICANT (p < {alpha})")
            print(f"      -> {better} is significantly better")
        else:
            print(f"      Result: NOT significant (p >= {alpha})")
    print()

    # Paired t-test
    mean_d, t, p_t = paired_t_test(diffs)
    print(f"    Paired t-test:")
    if n <= 1:
        print(f"      Not enough data points.")
    else:
        print(f"      t = {t:.4f}, p = {p_t:.4e} (df = {n - 1})")
        if p_t < alpha:
            better = model_a if t > 0 else model_b
            print(f"      Result: SIGNIFICANT (p < {alpha})")
            print(f"      -> {better} is significantly better")
        else:
            print(f"      Result: NOT significant (p >= {alpha})")

    return diffs


def main():
    args = parse_args()

    for d in [args.run_dir_a, args.run_dir_b]:
        if not os.path.isdir(d):
            print(f"Error: Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    model_a, results_a = load_reviews(args.run_dir_a, args.metric)
    model_b, results_b = load_reviews(args.run_dir_b, args.metric)

    print("=" * 70)
    print(f"Paired Test: {model_a} vs {model_b}")
    print(f"  Run A: {args.run_dir_a}")
    print(f"  Run B: {args.run_dir_b}")
    print(f"  Metric: {args.metric}, Alpha: {args.alpha}")
    print("=" * 70)

    subsets_a = set(results_a.keys())
    subsets_b = set(results_b.keys())
    common_subsets = sorted(subsets_a & subsets_b)

    if not common_subsets:
        print("\nError: No matching dataset subsets found between the two runs.")
        print(f"  Run A subsets: {sorted(subsets_a)}")
        print(f"  Run B subsets: {sorted(subsets_b)}")
        sys.exit(1)

    only_in_a = subsets_a - subsets_b
    only_in_b = subsets_b - subsets_a
    if only_in_a:
        print(f"\n  Warning: Subsets only in {model_a}: {sorted(only_in_a)}")
    if only_in_b:
        print(f"\n  Warning: Subsets only in {model_b}: {sorted(only_in_b)}")

    all_diffs = []
    for subset in common_subsets:
        diffs = print_comparison(
            subset, model_a, model_b,
            results_a[subset], results_b[subset], args.alpha
        )
        if diffs:
            all_diffs.extend(diffs)

    # Overall comparison (if multiple subsets)
    if len(common_subsets) > 1 and all_diffs:
        n = len(all_diffs)
        mean_diff = sum(all_diffs) / n * 100

        print()
        print("-" * 70)
        print(f"  OVERALL ({n} questions)")
        print(f"    Mean difference (A - B): {mean_diff:+.2f}%")
        print()

        w_plus, w_minus, z, p_w, n_nonzero = wilcoxon_signed_rank(all_diffs)
        print(f"    Wilcoxon Signed-Rank Test:")
        print(f"      W+ = {w_plus:.1f}, W- = {w_minus:.1f} ({n_nonzero} non-zero pairs)")
        print(f"      z = {z:.4f}, p = {p_w:.4e}")
        if p_w < args.alpha:
            better = model_a if z > 0 else model_b
            print(f"      Result: SIGNIFICANT (p < {args.alpha})")
            print(f"      -> {better} is significantly better")
        else:
            print(f"      Result: NOT significant (p >= {args.alpha})")
        print()

        mean_d, t, p_t = paired_t_test(all_diffs)
        print(f"    Paired t-test:")
        print(f"      t = {t:.4f}, p = {p_t:.4e} (df = {n - 1})")
        if p_t < args.alpha:
            better = model_a if t > 0 else model_b
            print(f"      Result: SIGNIFICANT (p < {args.alpha})")
            print(f"      -> {better} is significantly better")
        else:
            print(f"      Result: NOT significant (p >= {args.alpha})")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
