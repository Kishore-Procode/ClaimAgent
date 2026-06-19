"""
Agent 4 — Coverage Analyzer ⭐ (Key Differentiator)
Cross-references what the claim REQUIRES to be visible
against what the images actually SHOW.

Produces:
  - evidence_standard_met (bool)
  - evidence_standard_met_reason (text)
  - coverage_map {"screen": True, "hinge": False}
  - supporting_image_ids
  - next_best_evidence recommendations
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import CoverageAnalysis, InvestigationContext
import config

logger = logging.getLogger(__name__)

# Separate alias sets for packages to avoid treating package contents as package box evidence
PACKAGE_ALIASES = {
    "package",
    "parcel",
    "box",
    "shipping box",
    "mailer",
    "outer box",
    "cardboard box",
    "cardboard",
    "packaging"
}

PACKAGE_CONTENT_ALIASES = {
    "earbuds",
    "charging case",
    "contents",
    "item",
    "product",
    "device",
    "inside",
    "interior",
    "inner",
    "headphone",
    "earbud tip"
}

# Keyword synonyms for flexible part matching
# Maps required part keywords to sets of visible part aliases
PART_ALIASES: dict[str, set[str]] = {
    "rear_bumper":      {"rear bumper", "back bumper", "rear", "bumper"},
    "front_bumper":     {"front bumper", "front", "bumper"},
    "windshield":       {"windshield", "front glass", "glass", "windscreen"},
    "hood":             {"hood", "bonnet"},
    "door":             {"door", "door panel", "side door"},
    "door_panel":       {"door", "door panel"},
    "side_mirror":      {"side mirror", "mirror", "wing mirror"},
    "headlight":        {"headlight", "head light", "front light"},
    "taillight":        {"taillight", "tail light", "rear light", "back light"},
    "screen":           {"screen", "display", "lcd", "panel"},
    "display":          {"screen", "display", "lcd"},
    "hinge":            {"hinge", "hinge area"},
    "hinge_area":       {"hinge", "hinge area"},
    "keyboard":         {"keyboard", "keys", "keypad"},
    "trackpad":         {"trackpad", "touchpad", "palm rest"},
    "palm_rest":        {"palm rest", "trackpad area"},
    "corner":           {"corner", "body corner"},
    "laptop_body":      {"body", "chassis", "outer shell", "corner"},
    "outer_lid":        {"lid", "outer lid", "top cover"},
    "laptop_lid":       {"lid", "top", "outer panel"},
    "package_corner":   {"corner", "box corner", "package corner"},
    "box_corner":       {"corner", "box corner"},
    "package_seal":     {"seal", "tape", "flap", "opening"},
    "box_opening":      {"opening", "seal", "flap"},
    "package_interior": PACKAGE_CONTENT_ALIASES,
    "contents_area":    PACKAGE_CONTENT_ALIASES,
    "contents":         PACKAGE_CONTENT_ALIASES,
    "package":          PACKAGE_ALIASES,
    "outer_box":        PACKAGE_ALIASES,
    "shipping_label":   {"label", "shipping label", "sticker"},
    "package_surface":  {"surface", "side", "package side"},
    "package_side":     {"side", "package surface"},
}


class CoverageAnalyzerAgent(BaseAgent):
    """Agent 4: Verify evidence coverage of the claim."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        requirements = context.requirements
        image_analysis = context.image_analysis

        if requirements is None or image_analysis is None:
            self.warn("Missing requirements or image analysis — cannot assess coverage.")
            context.coverage = CoverageAnalysis(
                evidence_standard_met=False,
                evidence_standard_met_reason="Coverage could not be assessed due to missing data.",
                coverage_map={},
                coverage_score=0.0,
                missing_evidence=[],
                supporting_image_ids=[],
                next_best_evidence=[],
            )
            return context

        required_parts = requirements.required_parts
        visible_parts_raw = image_analysis.merged_visible_parts  # from VL model

        self.log(
            f"Required: {required_parts} | "
            f"Visible (raw): {visible_parts_raw}"
        )

        # Build coverage map
        coverage_map: dict[str, bool] = {}
        for req_part in required_parts:
            coverage_map[req_part] = self._is_part_visible(req_part, visible_parts_raw)

        # Determine which images support the claim
        supporting_ids = self._find_supporting_images(image_analysis, required_parts)

        # Calculate coverage score
        if required_parts:
            covered = sum(1 for v in coverage_map.values() if v)
            coverage_score = covered / len(required_parts)
        else:
            coverage_score = 1.0

        # Identify missing evidence
        missing = [part for part, visible in coverage_map.items() if not visible]

        # Evidence standard is met if coverage ≥ threshold AND at least one valid image
        evidence_met = (
            coverage_score >= config.COVERAGE_THRESHOLD
            and image_analysis.valid_image
        )

        # Build human-readable reason
        reason = self._build_reason(
            coverage_map, missing, image_analysis, evidence_met, coverage_score
        )

        # Next-best-evidence recommendations
        next_best = self._suggest_next_evidence(
            missing, context.claim.claim_object
        )

        context.coverage = CoverageAnalysis(
            evidence_standard_met=evidence_met,
            evidence_standard_met_reason=reason,
            coverage_map=coverage_map,
            coverage_score=coverage_score,
            missing_evidence=missing,
            supporting_image_ids=supporting_ids,
            next_best_evidence=next_best,
        )

        self.log(
            f"Coverage: {coverage_score:.0%}, met={evidence_met}, "
            f"missing={missing}, supporting={supporting_ids}"
        )
        return context

    def _is_part_visible(self, required_part: str, visible_parts: list[str]) -> bool:
        """
        Check if a required part appears in the visible parts list,
        using alias expansion for flexible matching.
        """
        req_lower = required_part.lower().replace("_", " ")
        visible_lower = {p.lower().replace("_", " ") for p in visible_parts}

        # Direct match
        if req_lower in visible_lower:
            return True

        # Alias match
        aliases = PART_ALIASES.get(required_part.lower(), set())
        if aliases & visible_lower:
            return True

        # Substring match (e.g. "bumper" matches "rear_bumper")
        for vis in visible_lower:
            if req_lower in vis or vis in req_lower:
                return True
            # Check aliases against visible parts
            for alias in aliases:
                if alias in vis or vis in alias:
                    return True

        return False

    def _find_supporting_images(
        self,
        image_analysis: "ImageAnalysis",
        required_parts: list[str],
    ) -> list[str]:
        """Return image IDs that show at least one required part."""
        supporting: list[str] = []
        for img in image_analysis.per_image:
            if not img.valid_image:
                continue
            for req_part in required_parts:
                if self._is_part_visible(req_part, img.visible_parts):
                    if img.image_id not in supporting:
                        supporting.append(img.image_id)
        return supporting

    def _build_reason(
        self,
        coverage_map: dict[str, bool],
        missing: list[str],
        image_analysis: "ImageAnalysis",
        evidence_met: bool,
        coverage_score: float,
    ) -> str:
        """Build a human-readable evidence standard reason."""
        visible = [part for part, v in coverage_map.items() if v]

        if not image_analysis.valid_image:
            return "No valid images were submitted for review."

        if evidence_met:
            if missing:
                covered_str = ", ".join(visible)
                return (
                    f"The {covered_str} {'is' if len(visible) == 1 else 'are'} visible "
                    f"and the damage can be partially verified from the submitted image(s)."
                )
            else:
                covered_str = ", ".join(visible)
                return (
                    f"The {covered_str} {'is' if len(visible) == 1 else 'are'} visible "
                    f"and the claimed damage can be verified from the submitted image(s)."
                )
        else:
            if missing:
                missing_str = ", ".join(missing)
                return (
                    f"The image does not show the {missing_str}, "
                    f"so the claimed damage cannot be verified."
                )
            return "The submitted image does not provide sufficient evidence to assess the claim."

    def _suggest_next_evidence(
        self, missing_parts: list[str], claim_object: str
    ) -> list[str]:
        """Suggest what photos to upload to complete the evidence."""
        suggestions = []
        for part in missing_parts:
            readable = part.replace("_", " ")
            suggestions.append(f"Upload a clear photo of the {readable}")
        return suggestions
