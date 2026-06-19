"""
Bonus Agent 9 — Duplicate Evidence Agent (Python-only, no AI)
Detects if submitted images were used in previous claims (cross-claim reuse).
Uses perceptual hashing (imagehash.phash).
"""
from __future__ import annotations
import logging
from pathlib import Path

import imagehash

from agents.base_agent import BaseAgent
from models.schemas import DuplicateResult, InvestigationContext
from utils.image_utils import compute_phash, images_are_similar
import config

logger = logging.getLogger(__name__)


class DuplicateAgent(BaseAgent):
    """Bonus Agent 9: Cross-claim image duplicate detection."""

    def __init__(self, model_manager, hash_store: dict[str, str] | None = None):
        """
        Args:
            model_manager: Shared model manager.
            hash_store:    Pre-computed {image_path: hex_hash} from prior claims.
                           Built incrementally by the pipeline.
        """
        super().__init__(model_manager)
        self.hash_store: dict[str, str] = hash_store or {}

    def run(self, context: InvestigationContext) -> InvestigationContext:
        image_paths = context.claim.image_paths
        self.log(f"Checking {len(image_paths)} image(s) for duplicates")

        duplicate_risk = "low"
        similar_claim_ids: list[str] = []

        for img_path in image_paths:
            current_hash = compute_phash(img_path)
            if current_hash is None:
                continue

            # Compare against all stored hashes
            for stored_path, stored_hex in self.hash_store.items():
                if stored_path == img_path:
                    continue
                try:
                    stored_hash = imagehash.hex_to_hash(stored_hex)
                    if images_are_similar(current_hash, stored_hash, config.DUPLICATE_HASH_THRESHOLD):
                        similar_claim_ids.append(Path(stored_path).parent.name)
                        duplicate_risk = "high"
                        self.warn(f"Duplicate image detected: {img_path} ≈ {stored_path}")
                except Exception:
                    continue

            # Store this image's hash for future comparisons
            self.hash_store[img_path] = str(current_hash)

        if similar_claim_ids and duplicate_risk == "low":
            duplicate_risk = "medium"

        context.duplicate = DuplicateResult(
            duplicate_risk=duplicate_risk,
            similar_claim_ids=list(set(similar_claim_ids)),
        )

        self.log(f"Duplicate risk={duplicate_risk}, similar={similar_claim_ids}")
        return context
