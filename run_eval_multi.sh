#!/bin/bash
# Run evalscope evaluation multiple times and compute mean/std of accuracy.
# Supports multiple datasets — reports per-dataset statistics.
#
# Usage:
#   ./run_eval_multi.sh [--runs N] [-- extra args for test_evalscope.py]
#
# Examples:
#   ./run_eval_multi.sh                                    # 10 runs, arc, sampling
#   ./run_eval_multi.sh --runs 5                           # 5 runs
#   ./run_eval_multi.sh -- --dataset gsm8k --limit 100     # pass extra args
#   ./run_eval_multi.sh --runs 3 -- --dataset arc gsm8k    # 3 runs on arc + gsm8k

set -e

NUM_RUNS=10
EXTRA_ARGS=("--dataset" "arc" "--sampling")

# Parse script arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--runs N] [-- extra args for test_evalscope.py]"
            exit 1
            ;;
    esac
done

# Collect all scores as "dataset:score" pairs
ALL_RESULTS=()

echo "========================================================================"
echo "Multi-Run Evaluation"
echo "========================================================================"
echo "Number of runs: $NUM_RUNS"
echo "Args passed to test_evalscope.py: ${EXTRA_ARGS[*]}"
echo "========================================================================"
echo ""

for i in $(seq 1 "$NUM_RUNS"); do
    echo "-------- Run $i / $NUM_RUNS --------"

    # Record timestamp before running
    BEFORE_TS=$(date +%s.%N)

    # Run evaluation
    python test_evalscope.py "${EXTRA_ARGS[@]}"

    # Find all report JSONs created after BEFORE_TS
    REPORT_FILES=$(find outputs/ -name "*.json" -path "*/reports/*" -newermt "@${BEFORE_TS}" 2>/dev/null)
    # REPORT_FILES=$(find outputs/ -name "*.json" -path "*/reports/*" 2>/dev/null)

    if [[ -z "$REPORT_FILES" ]]; then
        echo "ERROR: No report files found after run $i"
        exit 1
    fi

    # Extract dataset name and score from each report
    while IFS= read -r REPORT_FILE; do
        RESULT=$(python3 -c "
import json
with open('$REPORT_FILE') as f:
    report = json.load(f)
print(f\"{report['dataset_name']}:{report['score']}\")
")
        ALL_RESULTS+=("$RESULT")
        DS_NAME="${RESULT%%:*}"
        DS_SCORE="${RESULT##*:}"
        DS_PCT=$(python3 -c "print(f'{float('$DS_SCORE') * 100:.2f}')")
        echo "  Run $i  $DS_NAME: ${DS_PCT}%"
    done <<< "$REPORT_FILES"

    echo ""
done

echo "========================================================================"
echo "RESULTS"
echo "========================================================================"

# Calculate per-dataset mean and std
python3 -c "
import sys
from collections import defaultdict

results = defaultdict(list)
for entry in sys.argv[1:]:
    ds, score = entry.split(':')
    results[ds].append(float(score))

for ds in sorted(results):
    scores = results[ds]
    pct = [s * 100 for s in scores]
    n = len(pct)
    mean = sum(pct) / n
    if n > 1:
        variance = sum((s - mean) ** 2 for s in pct) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0

    print(f'Dataset: {ds}')
    print(f'  Scores (%): {[round(s, 2) for s in pct]}')
    print(f'  Num runs:   {n}')
    print(f'  Mean:       {mean:.2f}%')
    print(f'  Std:        {std:.2f}%')
    print(f'  Min:        {min(pct):.2f}%')
    print(f'  Max:        {max(pct):.2f}%')
    print()
" "${ALL_RESULTS[@]}"

echo "========================================================================"
