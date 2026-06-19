"""
CSV handler — reads claims.csv and writes output.csv
with the exact 14-column schema required by the problem statement.
"""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from models.schemas import ClaimRecord, Verdict

logger = logging.getLogger(__name__)

# Exact output column order (must match output.csv header)
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def load_claims(csv_path: str | Path, data_dir: str | Path) -> list[ClaimRecord]:
    """
    Read claims.csv and return a list of ClaimRecord objects.

    Args:
        csv_path:  Path to the claims CSV file.
        data_dir:  Root directory for resolving relative image paths.

    Returns:
        List of ClaimRecord (image_paths already split and path-resolved).
    """
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Claims CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.replace('"', '')

    records: list[ClaimRecord] = []
    for _, row in df.iterrows():
        raw_paths = str(row.get("image_paths", "")).strip()
        # Split on semicolons, resolve relative to data_dir
        paths = []
        for p in raw_paths.split(";"):
            p = p.strip()
            if p:
                resolved = data_dir / p
                paths.append(str(resolved))

        record = ClaimRecord(
            user_id=str(row.get("user_id", "")).strip(),
            image_paths=paths,
            user_claim=str(row.get("user_claim", "")).strip(),
            claim_object=str(row.get("claim_object", "")).strip().lower(),
        )
        records.append(record)

    logger.info(f"Loaded {len(records)} claims from {csv_path}")
    return records


def load_claims_with_history(
    csv_path: str | Path, data_dir: str | Path
) -> tuple[list[ClaimRecord], dict[str, list[ClaimRecord]]]:
    """
    Load all claims and build a user_id → [ClaimRecord, ...] history map.
    Used by the History Risk Agent.

    Returns:
        (all_claims, history_map)
    """
    all_claims = load_claims(csv_path, data_dir)
    history_map: dict[str, list[ClaimRecord]] = {}
    for claim in all_claims:
        history_map.setdefault(claim.user_id, []).append(claim)
    return all_claims, history_map


def write_verdicts(verdicts: list[Verdict], output_path: str | Path) -> None:
    """
    Write verdict list to output.csv with the exact required column order.

    Args:
        verdicts:     List of Verdict objects.
        output_path:  Destination CSV path (created if not exists).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [v.to_csv_row() for v in verdicts]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(verdicts)} verdicts to {output_path}")


def get_image_id(image_path: str | Path) -> str:
    """
    Extract image ID from a path.
    e.g. "/data/images/test/case_001/img_2.jpg" → "img_2"
    """
    return Path(image_path).stem
