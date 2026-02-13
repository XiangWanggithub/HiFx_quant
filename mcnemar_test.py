#!/usr/bin/env python3
"""
McNemar's Test for Model Comparison

Compares two models' per-question correctness on the same benchmark using
McNemar's test. Reads evalscope review JSONL files from two evaluation runs.

Usage:
    python mcnemar_test.py <run_dir_a> <run_dir_b> [--metric acc] [--alpha 0.05]

Example:
    python test_evalscope.py --model-path /home/models/Qwen3-0.6B --dataset arc
    python test_evalscope.py --model-path /home/models/Qwen3-1.7B --dataset arc
    python mcnemar_test.py outputs/arc_eval/20260211_150547 outputs/arc_eval/20260213_120000
"""

import argparse
import glob
import json
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="McNemar's test comparing two models on the same benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_dir_a', help='Timestamped run directory for model A')
    parser.add_argument('run_dir_b', help='Timestamped run directory for model B')
    parser.add_argument('--metric', default='acc', help='Metric key to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    return parser.parse_args()


def chi2_sf(x, df=1):
    """Survival function (1 - CDF) for chi-squared distribution with df=1.

    Uses the regularized incomplete gamma function approximation.
    For df=1, chi2_sf(x) = erfc(sqrt(x/2)).
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


def binom_cdf(k, n, p=0.5):
    """CDF of binomial distribution: P(X <= k) for X ~ Binom(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    # Use log to avoid overflow for large n
    log_pmf = n * math.log(1 - p)  # log(P(X=0))
    total += math.exp(log_pmf)
    for i in range(1, k + 1):
        log_pmf += math.log(n - i + 1) - math.log(i) + math.log(p) - math.log(1 - p)
        total += math.exp(log_pmf)
    return min(total, 1.0)


def load_reviews(run_dir, metric):
    """Load per-question results from review JSONL files in a run directory.

    Returns:
        model_name: str
        results: dict mapping subset_name -> dict mapping question_index -> score
    """
    reviews_dir = os.path.join(run_dir, 'reviews')
    if not os.path.isdir(reviews_dir):
        print(f"Error: No reviews directory found at {reviews_dir}", file=sys.stderr)
        sys.exit(1)

    # Find model name (subdirectory under reviews/)
    model_dirs = [d for d in os.listdir(reviews_dir)
                  if os.path.isdir(os.path.join(reviews_dir, d))]
    if len(model_dirs) == 0:
        print(f"Error: No model directories found in {reviews_dir}", file=sys.stderr)
        sys.exit(1)
    if len(model_dirs) > 1:
        print(f"Warning: Multiple model directories found: {model_dirs}. Using first.", file=sys.stderr)
    model_name = model_dirs[0]

    # Find all review JSONL files
    model_reviews_dir = os.path.join(reviews_dir, model_name)
    jsonl_files = glob.glob(os.path.join(model_reviews_dir, '*.jsonl'))
    if not jsonl_files:
        print(f"Error: No JSONL review files found in {model_reviews_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for fpath in sorted(jsonl_files):
        subset_name = os.path.splitext(os.path.basename(fpath))[0]  # e.g., "arc_ARC-Challenge"
        subset_results = {}
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                idx = entry['index']
                score_value = entry['sample_score']['score']['value']
                if metric not in score_value:
                    available = list(score_value.keys())
                    print(f"Error: Metric '{metric}' not found. Available: {available}", file=sys.stderr)
                    sys.exit(1)
                subset_results[idx] = score_value[metric]
        results[subset_name] = subset_results

    return model_name, results


def mcnemar_test(b, c):
    """Perform McNemar's test given discordant pair counts.

    Args:
        b: count where model A correct, model B wrong
        c: count where model A wrong, model B correct

    Returns:
        (method, statistic, p_value)
    """
    n = b + c
    if n == 0:
        return 'none', 0.0, 1.0

    if n >= 25:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = chi2_sf(chi2)
        return 'chi2', chi2, p
    else:
        # Exact binomial test (two-sided)
        k = min(b, c)
        p = 2 * binom_cdf(k, n, 0.5)
        p = min(p, 1.0)
        return 'exact', k, p


def print_comparison(subset_name, model_a, model_b, results_a, results_b, alpha):
    """Compare two models on a single subset and print results.

    Returns:
        (a, b, c, d) contingency counts for aggregation, or None if no overlap.
    """
    # Find common question indices
    common = set(results_a.keys()) & set(results_b.keys())
    only_a = set(results_a.keys()) - set(results_b.keys())
    only_b = set(results_b.keys()) - set(results_a.keys())

    if not common:
        print(f"  {subset_name}: No overlapping questions found.")
        return None

    if only_a or only_b:
        print(f"  Note: {len(only_a)} questions only in {model_a}, {len(only_b)} only in {model_b}")

    # Build contingency table
    a = b = c = d = 0
    for idx in common:
        a_correct = results_a[idx] >= 0.5
        b_correct = results_b[idx] >= 0.5
        if a_correct and b_correct:
            a += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
        else:
            d += 1

    n = len(common)
    acc_a = (a + b) / n * 100
    acc_b = (a + c) / n * 100

    print(f"\n  Subset: {subset_name} ({n} questions)")
    print(f"    {model_a} accuracy: {acc_a:.2f}%")
    print(f"    {model_b} accuracy: {acc_b:.2f}%")
    print()
    print(f"    Contingency Table:")
    print(f"    {'':20s} {model_b}")
    print(f"    {'':20s} {'Correct':>8s}  {'Wrong':>8s}")
    print(f"    {model_a}")
    print(f"      {'Correct':12s}    {a:>8d}  {b:>8d}")
    print(f"      {'Wrong':12s}    {c:>8d}  {d:>8d}")
    print()

    method, stat, p = mcnemar_test(b, c)
    if method == 'none':
        print(f"    Models agree on every question. No test needed.")
    elif method == 'chi2':
        print(f"    Discordant pairs: b={b}, c={c}")
        print(f"    McNemar chi2 = {stat:.4f}, p = {p:.4e}")
    else:
        print(f"    Discordant pairs: b={b}, c={c} (small sample, exact binomial test)")
        print(f"    p = {p:.4e}")

    if method != 'none':
        if p < alpha:
            better = model_b if c > b else model_a
            print(f"    Result: SIGNIFICANT (p < {alpha})")
            print(f"    -> {better} is significantly better")
        else:
            print(f"    Result: NOT significant (p >= {alpha})")
            print(f"    -> No significant difference between models")

    return a, b, c, d


def main():
    args = parse_args()

    # Validate directories
    for d in [args.run_dir_a, args.run_dir_b]:
        if not os.path.isdir(d):
            print(f"Error: Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Load reviews
    model_a, results_a = load_reviews(args.run_dir_a, args.metric)
    model_b, results_b = load_reviews(args.run_dir_b, args.metric)

    print("=" * 70)
    print(f"McNemar's Test: {model_a} vs {model_b}")
    print(f"  Run A: {args.run_dir_a}")
    print(f"  Run B: {args.run_dir_b}")
    print(f"  Metric: {args.metric}, Alpha: {args.alpha}")
    print("=" * 70)

    # Match subsets across the two runs
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

    # Per-subset comparison
    total_a = total_b = total_c = total_d = 0
    for subset in common_subsets:
        result = print_comparison(
            subset, model_a, model_b,
            results_a[subset], results_b[subset], args.alpha
        )
        if result:
            a, b, c, d = result
            total_a += a
            total_b += b
            total_c += c
            total_d += d

    # Overall comparison (if multiple subsets)
    if len(common_subsets) > 1:
        total_n = total_a + total_b + total_c + total_d
        acc_a = (total_a + total_b) / total_n * 100
        acc_b = (total_a + total_c) / total_n * 100

        print()
        print("-" * 70)
        print(f"  OVERALL ({total_n} questions)")
        print(f"    {model_a} accuracy: {acc_a:.2f}%")
        print(f"    {model_b} accuracy: {acc_b:.2f}%")
        print()
        print(f"    Contingency Table:")
        print(f"    {'':20s} {model_b}")
        print(f"    {'':20s} {'Correct':>8s}  {'Wrong':>8s}")
        print(f"    {model_a}")
        print(f"      {'Correct':12s}    {total_a:>8d}  {total_b:>8d}")
        print(f"      {'Wrong':12s}    {total_c:>8d}  {total_d:>8d}")
        print()

        method, stat, p = mcnemar_test(total_b, total_c)
        if method == 'none':
            print(f"    Models agree on every question. No test needed.")
        elif method == 'chi2':
            print(f"    Discordant pairs: b={total_b}, c={total_c}")
            print(f"    McNemar chi2 = {stat:.4f}, p = {p:.4e}")
        else:
            print(f"    Discordant pairs: b={total_b}, c={total_c}")
            print(f"    p = {p:.4e}")

        if method != 'none':
            if p < args.alpha:
                better = model_b if total_c > total_b else model_a
                print(f"    Result: SIGNIFICANT (p < {args.alpha})")
                print(f"    -> {better} is significantly better")
            else:
                print(f"    Result: NOT significant (p >= {args.alpha})")
                print(f"    -> No significant difference between models")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
