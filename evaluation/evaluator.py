"""
Evaluator — compare pipeline predictions against ground truth.
Run after processing sample_claims.csv (which has ground truth columns).

Usage:
    python evaluation/evaluator.py \
        --predictions outputs/sample_output.csv \
        --ground-truth data/sample_claims.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_metrics, MetricResult


def load_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_evaluation(predictions_path: str, ground_truth_path: str) -> MetricResult:
    print(f"\nLoading predictions : {predictions_path}")
    print(f"Loading ground truth: {ground_truth_path}\n")

    predictions  = load_csv(predictions_path)
    ground_truth = load_csv(ground_truth_path)

    print(f"Predictions : {len(predictions)} rows")
    print(f"Ground truth: {len(ground_truth)} rows\n")

    metrics = compute_metrics(predictions, ground_truth)
    print(metrics.summary())
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VisionClaim predictions against ground truth."
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to predicted output.csv",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        help="Path to ground truth CSV (sample_claims.csv)",
    )
    args = parser.parse_args()

    run_evaluation(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
