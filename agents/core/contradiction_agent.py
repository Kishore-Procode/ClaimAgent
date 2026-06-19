"""
Agent 6 — Contradiction Agent
Detects mismatches between the claim and visual evidence:
  1. Severity exaggeration (claimed "shattered", image shows "scratch")
  2. Wrong part (claimed hinge, only screen visible)
  3. Prompt injection text in images (text_instruction_present flag)
  4. Claim object mismatch (submitted wrong item photo)
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import ContradictionResult, InvestigationContext
from utils.prompt_builder import contradiction_prompt, SYSTEM_INVESTIGATOR

logger = logging.getLogger(__name__)

# Severity hierarchy for exaggeration detection
SEVERITY_RANK = {
    "none": 0, "unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4
}

# Damage type groupings for semantic contradiction
DAMAGE_GROUPS: dict[str, set[str]] = {
    "structural": {"crack", "shatter", "broken_part", "missing_contents"},
    "cosmetic":   {"scratch", "stain", "water_damage"},
    "physical":   {"dent", "crushed_packaging", "torn_packaging"},
}


class ContradictionAgent(BaseAgent):
    """Agent 6: Detect contradictions between claim and visual evidence."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        claim    = context.claim
        extracted = context.extracted
        img_analysis = context.image_analysis

        if extracted is None or img_analysis is None:
            self.warn("Missing extracted claim or image analysis — skipping contradiction check.")
            context.contradiction = ContradictionResult(
                has_contradiction=False,
                contradiction_flags=[],
                alignment_score=1.0,
                contradiction_reason="",
            )
            return context

        contradiction_flags: list[str] = []
        alignment_score = 1.0

        # ── Check 1: Prompt injection in image ────────────────────────────────
        if img_analysis.flags and "text_instruction_present" in img_analysis.flags:
            contradiction_flags.append("text_instruction_present")
            alignment_score -= 0.1  # flag but don't penalize heavily

        # ── Check 2: Object mismatch ──────────────────────────────────────────
        object_mismatches = [
            img for img in img_analysis.per_image
            if not img.object_matches_claim and img.valid_image
        ]
        if len(object_mismatches) == len(img_analysis.per_image) and img_analysis.per_image:
            # ALL valid images show wrong object
            contradiction_flags.append("claim_mismatch")
            alignment_score -= 0.4

        # ── Check 3: Damage type contradiction ────────────────────────────────
        observed_types = {d.get("type", "").lower() for d in img_analysis.merged_damages}
        claimed_type = extracted.issue_type.lower()

        if observed_types and claimed_type not in observed_types:
            # Check if they're in the same damage group (semantic similarity)
            claimed_group = self._get_damage_group(claimed_type)
            obs_groups = {self._get_damage_group(t) for t in observed_types}

            if claimed_group and claimed_group not in obs_groups:
                contradiction_flags.append("claim_mismatch")
                alignment_score -= 0.3

        # ── Check 4: Severity exaggeration ────────────────────────────────────
        # If claimed damage type implies high severity but observed is low
        high_severity_claims = {"shatter", "broken_part", "crushed_packaging", "missing_contents"}
        low_severity_observed = img_analysis.overall_severity in ("low",)

        if claimed_type in high_severity_claims and low_severity_observed and observed_types:
            if "claim_mismatch" not in contradiction_flags:
                contradiction_flags.append("claim_mismatch")
            alignment_score -= 0.2

        # ── Check 5: Negative evidence — claimed part IS visible, NO damage found ──
        # If the coverage shows the claimed part is visible AND no damage was detected
        # at all, that is active contradiction, not "not enough information".
        damage_claims = {
            "crack", "shatter", "broken_part", "scratch", "stain",
            "water_damage", "dent", "crushed_packaging", "missing_contents",
            "torn_packaging",
        }
        coverage = context.coverage
        claimed_part = extracted.object_part.lower() if extracted.object_part else ""
        claimed_part_is_visible = (
            coverage is not None
            and any(
                claimed_part in p.lower()
                for p in (coverage.visible_parts or [])
            )
        )
        no_damage_detected = not observed_types  # image analysis found zero damage

        if (
            claimed_type in damage_claims
            and claimed_part_is_visible
            and no_damage_detected
            and "claim_mismatch" not in contradiction_flags
        ):
            # The part the user says is damaged is clearly visible but looks fine
            contradiction_flags.append("claim_mismatch")
            alignment_score -= 0.35
            logger.info(
                "Negative-evidence contradiction: '%s' is visible but no damage detected "
                "(claimed: %s)", claimed_part, claimed_type
            )

        # ── Finalize ──────────────────────────────────────────────────────────
        alignment_score = max(0.0, min(1.0, alignment_score))
        has_contradiction = bool(contradiction_flags)

        # Build contradiction reason
        reason = ""
        if "claim_mismatch" in contradiction_flags:
            if no_damage_detected and claimed_part_is_visible:
                reason = (
                    f"The {claimed_part} is clearly visible in the submitted image "
                    f"but shows no signs of {claimed_type}. "
                    f"The absence of detectable damage contradicts the claim."
                )
            else:
                observed_str = ", ".join(observed_types) if observed_types else "no damage"
                reason = (
                    f"The claim states {claimed_type} but the image shows {observed_str}. "
                )
        if "text_instruction_present" in contradiction_flags:
            reason += "The image contains embedded text instructions which were ignored."

        # Use LLM for nuanced cases where rule logic alone is uncertain
        if alignment_score < 0.7 and alignment_score > 0.3:
            reason = self._llm_contradiction_check(context, reason) or reason

        context.contradiction = ContradictionResult(
            has_contradiction=has_contradiction,
            contradiction_flags=contradiction_flags,
            alignment_score=alignment_score,
            contradiction_reason=reason.strip(),
        )

        self.log(
            f"Contradiction: {has_contradiction}, flags={contradiction_flags}, "
            f"alignment={alignment_score:.2f}"
        )
        return context

    def _get_damage_group(self, damage_type: str) -> str | None:
        for group, members in DAMAGE_GROUPS.items():
            if damage_type in members:
                return group
        return None

    def _llm_contradiction_check(
        self, context: InvestigationContext, existing_reason: str
    ) -> str | None:
        """Use LLM for nuanced contradiction analysis when rules are uncertain."""
        extracted = context.extracted
        img = context.image_analysis
        if extracted is None or img is None:
            return None

        try:
            prompt = contradiction_prompt(
                claim_object=context.claim.claim_object,
                object_part=extracted.object_part,
                issue_type=extracted.issue_type,
                detected_object=img.per_image[0].detected_object if img.per_image else "unknown",
                visible_parts=img.merged_visible_parts,
                observed_damages=img.merged_damages,
                contains_text_instruction="text_instruction_present" in img.flags,
            )
            messages = self.mm.build_text_messages(prompt, system_prompt=SYSTEM_INVESTIGATOR)
            raw = self.mm.generate_text(messages)
            data = self.parse_json(raw, fallback={})
            return data.get("contradiction_reason", existing_reason)
        except Exception as e:
            self.warn(f"LLM contradiction check failed: {e}")
            return existing_reason
