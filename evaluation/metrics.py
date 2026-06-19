"""
Evaluation metrics for VisionClaim Investigator.
Compares predicted output.csv against ground truth sample_claims.csv.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    """Holds computed evaluation metrics."""
    total: int = 0
    # Per-field accuracy
    claim_status_correct: int = 0
    severity_correct: int = 0
    evidence_met_correct: int = 0
    valid_image_correct: int = 0
    issue_type_correct: int = 0
    object_part_correct: int = 0
    # Risk flag metrics
    risk_flag_tp: int = 0   # True positive (flag correctly predicted)
    risk_flag_fp: int = 0   # False positive (flag predicted but not in GT)
    risk_flag_fn: int = 0   # False negative (flag in GT but not predicted)

    @property
    def claim_status_accuracy(self) -> float:
        return self.claim_status_correct / self.total if self.total else 0.0

    @property
    def severity_accuracy(self) -> float:
        return self.severity_correct / self.total if self.total else 0.0

    @property
    def evidence_met_accuracy(self) -> float:
        return self.evidence_met_correct / self.total if self.total else 0.0

    @property
    def valid_image_accuracy(self) -> float:
        return self.valid_image_correct / self.total if self.total else 0.0

    @property
    def issue_type_accuracy(self) -> float:
        return self.issue_type_correct / self.total if self.total else 0.0

    @property
    def object_part_accuracy(self) -> float:
        return self.object_part_correct / self.total if self.total else 0.0

    @property
    def risk_flag_precision(self) -> float:
        denom = self.risk_flag_tp + self.risk_flag_fp
        return self.risk_flag_tp / denom if denom else 0.0

    @property
    def risk_flag_recall(self) -> float:
        denom = self.risk_flag_tp + self.risk_flag_fn
        return self.risk_flag_tp / denom if denom else 0.0

    @property
    def risk_flag_f1(self) -> float:
        p, r = self.risk_flag_precision, self.risk_flag_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        lines = [
            f"{'-' * 50}",
            f"  Total claims evaluated : {self.total}",
            f"{'-' * 50}",
            f"  Claim Status Accuracy  : {self.claim_status_accuracy:.1%}",
            f"  Evidence Met Accuracy  : {self.evidence_met_accuracy:.1%}",
            f"  Valid Image Accuracy   : {self.valid_image_accuracy:.1%}",
            f"  Severity Accuracy      : {self.severity_accuracy:.1%}",
            f"  Issue Type Accuracy    : {self.issue_type_accuracy:.1%}",
            f"  Object Part Accuracy   : {self.object_part_accuracy:.1%}",
            f"{'-' * 50}",
            f"  Risk Flag Precision    : {self.risk_flag_precision:.1%}",
            f"  Risk Flag Recall       : {self.risk_flag_recall:.1%}",
            f"  Risk Flag F1           : {self.risk_flag_f1:.1%}",
            f"{'-' * 50}",
        ]
        return "\n".join(lines)


def compute_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
) -> MetricResult:
    """
    Compute evaluation metrics by comparing predictions to ground truth.

    Args:
        predictions:  List of dicts from the predicted output CSV.
        ground_truth: List of dicts from sample_claims.csv (with GT columns).

    Returns:
        MetricResult with all metrics computed.
    """
    # Index ground truth by user_id + image_paths for matching
    gt_index: dict[str, dict] = {}
    for row in ground_truth:
        key = _make_key(row)
        gt_index[key] = row

    result = MetricResult()

    for pred in predictions:
        key = _make_key(pred)
        gt = gt_index.get(key)
        if gt is None:
            continue  # No matching GT row

        result.total += 1

        # claim_status
        if _norm(pred.get("claim_status")) == _norm(gt.get("claim_status")):
            result.claim_status_correct += 1

        # evidence_standard_met
        if _norm(pred.get("evidence_standard_met")) == _norm(gt.get("evidence_standard_met")):
            result.evidence_met_correct += 1

        # valid_image
        if _norm(pred.get("valid_image")) == _norm(gt.get("valid_image")):
            result.valid_image_correct += 1

        # severity (treat "unknown" vs GT "unknown" as correct)
        if _norm(pred.get("severity")) == _norm(gt.get("severity")):
            result.severity_correct += 1

        # issue_type
        if _norm(pred.get("issue_type")) == _norm(gt.get("issue_type")):
            result.issue_type_correct += 1

        # object_part
        if _norm(pred.get("object_part")) == _norm(gt.get("object_part")):
            result.object_part_correct += 1

        # risk_flags — set-based comparison
        pred_flags = _parse_flags(pred.get("risk_flags", "none"))
        gt_flags   = _parse_flags(gt.get("risk_flags", "none"))

        tp = len(pred_flags & gt_flags)
        fp = len(pred_flags - gt_flags)
        fn = len(gt_flags - pred_flags)

        result.risk_flag_tp += tp
        result.risk_flag_fp += fp
        result.risk_flag_fn += fn

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Create a match key from user_id and image_paths."""
    uid   = str(row.get("user_id", "")).strip()
    paths = _normalize_image_paths(str(row.get("image_paths", "")))
    return f"{uid}|{paths}"


def _norm(val) -> str:
    """Normalize string value for comparison."""
    if val is None:
        return ""
    return str(val).strip().lower()


def _parse_flags(flags_str: str) -> set[str]:
    """Parse semicolon-separated flags into a set, excluding 'none'."""
    flags = {f.strip().lower() for f in flags_str.split(";") if f.strip()}
    flags.discard("none")
    return flags
