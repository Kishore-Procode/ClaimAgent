"""
Bonus Agent 8 — Authenticity Agent (Python-only, no AI)
Checks EXIF metadata to flag possible image manipulation.
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import AuthenticityResult, InvestigationContext
from utils.image_utils import extract_exif, compute_authenticity_score
import config

logger = logging.getLogger(__name__)


class AuthenticityAgent(BaseAgent):
    """Bonus Agent 8: EXIF-based authenticity check."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        image_paths = context.claim.image_paths
        self.log(f"Checking authenticity for {len(image_paths)} image(s)")

        # Aggregate scores across all images — take the minimum (weakest link)
        min_score = 100.0
        all_flags: list[str] = []
        any_exif = False
        first_ts = None
        first_cam = None
        any_gps = False

        for img_path in image_paths:
            exif = extract_exif(img_path)
            score, flags = compute_authenticity_score(exif)
            min_score = min(min_score, score)
            all_flags.extend(flags)
            if exif.get("exif_present"):
                any_exif = True
            if exif.get("timestamp") and first_ts is None:
                first_ts = exif["timestamp"]
            if exif.get("camera_model") and first_cam is None:
                first_cam = exif["camera_model"]
            if exif.get("gps_present"):
                any_gps = True

        # Deduplicate flags
        unique_flags = list(dict.fromkeys(all_flags))

        context.authenticity = AuthenticityResult(
            score=min_score,
            exif_present=any_exif,
            timestamp=first_ts,
            gps_present=any_gps,
            camera_model=first_cam,
            flags=unique_flags,
        )

        self.log(f"Authenticity score={min_score:.0f}, flags={unique_flags}")
        return context
