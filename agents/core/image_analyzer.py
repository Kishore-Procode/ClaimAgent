"""
Agent 3 — Image Analyzer
Performs a comprehensive single-pass VL analysis of each image:
  - Object identification
  - Visible parts detection
  - Damage detection & severity
  - Image quality assessment
  - Prompt injection detection
Results are merged across all images into a unified ImageAnalysis.
"""
from __future__ import annotations
import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.schemas import (
    SingleImageAnalysis, ImageAnalysis, InvestigationContext
)
from utils.prompt_builder import image_analyzer_prompt, SYSTEM_VISION
from utils.image_utils import get_image_id, load_image

logger = logging.getLogger(__name__)


class ImageAnalyzerAgent(BaseAgent):
    """Agent 3: Analyze all images for this claim."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        claim = context.claim
        extracted = context.extracted

        object_part = extracted.object_part if extracted else "unknown"
        issue_type  = extracted.issue_type  if extracted else "unknown"

        self.log(f"Analyzing {len(claim.image_paths)} image(s) for {claim.claim_object}")

        per_image_results: list[SingleImageAnalysis] = []

        for img_path in claim.image_paths:
            img_id = get_image_id(img_path)
            result = self._analyze_single_image(
                img_path, img_id, claim.claim_object, object_part, issue_type
            )
            per_image_results.append(result)

        # Merge across all images
        merged = self._merge_results(per_image_results)
        context.image_analysis = merged
        self.log(
            f"Merged: valid={merged.valid_image}, "
            f"severity={merged.overall_severity}, flags={merged.flags}"
        )
        return context

    def _analyze_single_image(
        self,
        img_path: str,
        img_id: str,
        claim_object: str,
        object_part: str,
        issue_type: str,
    ) -> SingleImageAnalysis:
        """Run VL analysis on a single image."""

        # Check image exists
        if not Path(img_path).exists():
            self.warn(f"Image not found: {img_path}")
            return SingleImageAnalysis(
                image_id=img_id,
                image_path=img_path,
                detected_object="not_found",
                object_matches_claim=False,
                visible_parts=[],
                damages=[],
                overall_severity="unknown",
                image_quality="wrong_angle",
                valid_image=False,
                contains_text_instruction=False,
                raw_response="IMAGE_NOT_FOUND",
            )

        prompt = image_analyzer_prompt(claim_object, object_part, issue_type)

        try:
            messages = self.mm.build_vision_messages(
                image_paths=[img_path],
                prompt=prompt,
                system_prompt=SYSTEM_VISION,
            )
            raw = self.mm.generate_vision(messages)
            logger.info(f"RAW_VISION_RESPONSE={raw}")
            data = self.parse_json(raw, fallback={})
            logger.info(
                f"VALID_DEBUG: visible={data.get('visible_parts')}, "
                f"damage={data.get('damages')}, valid={data.get('valid_image')}"
            )

            return SingleImageAnalysis(
                image_id=img_id,
                image_path=img_path,
                detected_object=self.safe_str(data.get("detected_object"), "unknown"),
                object_matches_claim=self.safe_bool(data.get("object_matches_claim"), False),
                visible_parts=self.safe_list(data.get("visible_parts")),
                damages=self.safe_list(data.get("damages")),
                overall_severity=self.safe_str(data.get("overall_severity"), "unknown"),
                image_quality=self.safe_str(data.get("image_quality"), "good"),
                valid_image=self.safe_bool(data.get("valid_image"), True),
                contains_text_instruction=self.safe_bool(data.get("contains_text_instruction"), False),
                raw_response=raw[:500],
            )

        except Exception as e:
            self.warn(f"Image analysis failed for {img_id}: {e}")
            return SingleImageAnalysis(
                image_id=img_id,
                image_path=img_path,
                detected_object="error",
                object_matches_claim=False,
                visible_parts=[],
                damages=[],
                overall_severity="unknown",
                image_quality="good",
                valid_image=False,
                contains_text_instruction=False,
                raw_response=str(e),
            )

    def _merge_results(self, results: list[SingleImageAnalysis]) -> ImageAnalysis:
        """Merge per-image analyses into a single unified result."""
        if not results:
            return ImageAnalysis(
                per_image=[],
                merged_visible_parts=[],
                merged_damages=[],
                best_image_id=None,
                overall_severity="unknown",
                valid_image=False,
                flags=["damage_not_visible"],
            )

        # Merge all visible parts (union)
        all_parts: set[str] = set()
        all_damages: list[dict] = []
        flags: set[str] = set()
        any_valid = False
        best_image: SingleImageAnalysis | None = None
        contains_text_instruction = False

        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0, "none": 0}
        best_severity = 0

        for r in results:
            all_parts.update(r.visible_parts)
            all_damages.extend(r.damages)

            if r.valid_image:
                any_valid = True

            if r.contains_text_instruction:
                contains_text_instruction = True
                flags.add("text_instruction_present")

            # Image quality flags
            if r.image_quality == "blurry":
                flags.add("blurry_image")
            elif r.image_quality == "wrong_angle":
                flags.add("wrong_angle")
            elif r.image_quality == "partial":
                flags.add("cropped_or_obstructed")

            # Track best image (valid + highest severity)
            rank = severity_rank.get(r.overall_severity, 0)
            if r.valid_image and rank >= best_severity:
                best_severity = rank
                best_image = r

        # If no valid image was found, mark all images as invalid
        if not any_valid:
            flags.add("damage_not_visible")

        # Deduplicate damages by (type, part)
        seen = set()
        unique_damages = []
        for d in all_damages:
            key = (d.get("type", ""), d.get("part", ""))
            if key not in seen:
                seen.add(key)
                unique_damages.append(d)

        # Compute merged overall severity (max across all images)
        merged_severity = "unknown"
        for r in results:
            if severity_rank.get(r.overall_severity, 0) > severity_rank.get(merged_severity, 0):
                merged_severity = r.overall_severity

        return ImageAnalysis(
            per_image=results,
            merged_visible_parts=list(all_parts),
            merged_damages=unique_damages,
            best_image_id=best_image.image_id if best_image else None,
            overall_severity=merged_severity,
            valid_image=any_valid,
            flags=list(flags),
        )
