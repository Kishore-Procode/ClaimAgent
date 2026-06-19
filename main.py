"""
VisionClaim Investigator — Main Entry Point
Uses OpenRouter API (no local GPU required).

Setup:
    1. Get a free API key at https://openrouter.ai
    2. Set env var: $env:OPENROUTER_API_KEY = "sk-or-..."
    3. pip install -r requirements.txt

Usage:
    # Full test run
    python main.py --input data/claims.csv --output outputs/output.csv --data-dir data/

    # Sample run (has ground truth for evaluation)
    python main.py --input data/sample_claims.csv --output outputs/sample_output.csv --data-dir data/

    # Pass API key directly (instead of env var)
    python main.py --input data/claims.csv --output outputs/output.csv --api-key sk-or-...

    # Use a different free model
    python main.py --input data/claims.csv --output outputs/output.csv \
        --model "google/gemma-3-27b-it:free"

    # Disable bonus agents for speed
    python main.py --input data/claims.csv --output outputs/output.csv \
        --no-authenticity --no-duplicate --no-critique
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

# ── Set up logging before anything else ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        ),
    ],
)
logger = logging.getLogger("main")

import config
from models.model_manager import ModelManager
from pipeline.investigator import InvestigationPipeline
from utils.csv_handler import load_claims_with_history, write_verdicts
from models.schemas import Verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionClaim Investigator — AI Claims Verification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=str(config.DEFAULT_INPUT_CSV),
        help="Path to input claims CSV (default: data/claims.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(config.DEFAULT_OUTPUT_CSV),
        help="Path to output CSV (default: outputs/output.csv)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=str(config.DEFAULT_DATA_DIR),
        help="Root directory for resolving image paths (default: data/)",
    )
    parser.add_argument(
        "--model",
        default=config.VISION_MODEL_ID,
        help=f"OpenRouter model ID (default: {config.VISION_MODEL_ID})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--no-authenticity",
        action="store_true",
        help="Disable authenticity bonus agent",
    )
    parser.add_argument(
        "--no-duplicate",
        action="store_true",
        help="Disable duplicate detection bonus agent",
    )
    parser.add_argument(
        "--no-critique",
        action="store_true",
        help="Disable self-critique bonus agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and validate setup without running the model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N claims (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  VisionClaim Investigator")
    logger.info("=" * 60)
    logger.info(f"  Input   : {args.input}")
    logger.info(f"  Output  : {args.output}")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Model   : {args.model}")
    logger.info(f"  Via     : OpenRouter API (no local GPU)")
    logger.info("=" * 60)

    # ── Step 1: Load claims ───────────────────────────────────────────────────
    logger.info("Loading claims...")
    all_claims, history_map = load_claims_with_history(args.input, args.data_dir)

    if args.limit:
        all_claims = all_claims[:args.limit]
        logger.info(f"Limiting to first {args.limit} claims.")

    logger.info(f"Loaded {len(all_claims)} claims. Users: {len(history_map)}")

    if args.dry_run:
        logger.info("DRY RUN — skipping model loading and inference.")
        logger.info(f"Sample claim: {all_claims[0] if all_claims else 'none'}")
        return

    # ── Step 2: Initialize model and pipeline ────────────────────────────────
    logger.info("Initializing OpenRouter model manager...")
    api_key = args.api_key or config.OPENROUTER_API_KEY
    mm = ModelManager(model_id=args.model, api_key=api_key)
    mm.load()

    pipeline = InvestigationPipeline(
        model_manager=mm,
        history_map=history_map,
        enable_authenticity=not args.no_authenticity,
        enable_duplicate=not args.no_duplicate,
        enable_critique=not args.no_critique,
    )
    pipeline.setup()

    # ── Step 3: Process all claims ───────────────────────────────────────────
    verdicts: list[Verdict] = []
    start_time = time.time()
    errors = 0

    for i, claim in enumerate(all_claims, 1):
        logger.info(f"\n[{i}/{len(all_claims)}] Processing: user={claim.user_id}")
        try:
            verdict = pipeline.investigate(claim)
            verdicts.append(verdict)
            logger.info(
                f"  → {verdict.claim_status} | severity={verdict.severity} | "
                f"flags={verdict.risk_flags}"
            )
        except Exception as e:
            logger.error(f"  CRITICAL ERROR for {claim.user_id}: {e}", exc_info=True)
            errors += 1
            # Write a fallback row so output is complete
            verdicts.append(pipeline._emergency_fallback(claim))

    total_time = time.time() - start_time

    # ── Step 4: Write output ──────────────────────────────────────────────────
    logger.info(f"\nWriting {len(verdicts)} verdicts to {args.output}...")
    write_verdicts(verdicts, args.output)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Claims processed : {len(verdicts)}")
    logger.info(f"  Errors           : {errors}")
    logger.info(f"  Total time       : {total_time:.1f}s")
    logger.info(f"  Avg per claim    : {total_time / len(verdicts):.1f}s" if verdicts else "")
    logger.info(f"  Output file      : {args.output}")

    # Decision breakdown
    from collections import Counter
    status_counts = Counter(v.claim_status for v in verdicts)
    logger.info("\n  Decision breakdown:")
    for status, count in sorted(status_counts.items()):
        pct = count / len(verdicts) * 100 if verdicts else 0
        logger.info(f"    {status:35s}: {count:3d} ({pct:.0f}%)")
    logger.info("=" * 60)

    # ── Step 6: Cleanup ───────────────────────────────────────────────────────
    pipeline.teardown()
    logger.info("Done.")


if __name__ == "__main__":
    main()
