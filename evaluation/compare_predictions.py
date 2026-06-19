"""
Side-by-side comparison of predictions vs. ground truth.
Highlights mismatches per claim for debugging.

Usage:
    python evaluation/compare_predictions.py \
        --predictions outputs/sample_output.csv \
        --ground-truth data/sample_claims.csv \
        --field claim_status
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


COMPARE_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "valid_image",
    "severity",
    "issue_type",
    "object_part",
    "risk_flags",
]


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_image_paths(paths_str: str) -> str:
    parts = []
    for p in paths_str.split(";"):
        p = p.strip().replace("\\", "/").lower()
        if p.startswith("data/"):
            p = p[5:]
        if p:
            parts.append(p)
    return ";".join(sorted(parts))


def _make_key(row: dict) -> str:
    uid = row.get('user_id', '').strip()
    paths = _normalize_image_paths(row.get('image_paths', ''))
    return f"{uid}|{paths}"


def compare(
    predictions_path: str,
    ground_truth_path: str,
    field: str | None = None,
    show_all: bool = False,
) -> None:
    preds = load_csv(predictions_path)
    gts   = load_csv(ground_truth_path)

    gt_index = {_make_key(r): r for r in gts}
    fields = [field] if field else COMPARE_FIELDS

    total = 0
    mismatches = 0

    print(f"\n{'=' * 70}")
    print(f"  PREDICTION COMPARISON REPORT")
    print(f"  Predictions : {predictions_path}")
    print(f"  Ground Truth: {ground_truth_path}")
    print(f"{'=' * 70}\n")

    for pred in preds:
        key = _make_key(pred)
        gt  = gt_index.get(key)
        if gt is None:
            continue

        total += 1
        row_mismatches = []

        for f in fields:
            pv = str(pred.get(f, "")).strip().lower()
            gv = str(gt.get(f, "")).strip().lower()
            if pv != gv:
                row_mismatches.append((f, pv, gv))

        if row_mismatches or show_all:
            if row_mismatches:
                mismatches += 1

            user_id = pred.get("user_id", "?")
            obj     = pred.get("claim_object", "?")
            status  = "MISMATCH [X]" if row_mismatches else "CORRECT [OK]"

            print(f"  [{status}] user={user_id} | object={obj}")
            print(f"    Claim: {pred.get('user_claim', '')[:80]}...")

            if row_mismatches:
                for fname, pval, gval in row_mismatches:
                    print(f"    {fname:30s} | PRED: {pval!r:25s} | GT: {gval!r}")
            print()

    print(f"{'-' * 70}")
    print(f"  Total evaluated : {total}")
    print(f"  Mismatches      : {mismatches}")
    print(f"  Correct         : {total - mismatches}")
    print(f"  Accuracy        : {(total - mismatches) / total:.1%}" if total else "  Accuracy: N/A")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of predictions vs. ground truth."
    )
    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--ground-truth", "-g", required=True)
    parser.add_argument(
        "--field", "-f",
        help="Compare only this field (e.g. claim_status). Default: all fields.",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all rows, not just mismatches.",
    )
    args = parser.parse_args()
    compare(args.predictions, args.ground_truth, args.field, args.show_all)


if __name__ == "__main__":
    main()
