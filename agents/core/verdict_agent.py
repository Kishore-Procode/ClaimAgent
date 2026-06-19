"""
Agent 7 — Verdict Agent
Synthesizes all prior agent outputs into the final claim decision.
Produces all 14 output.csv columns directly.

Decision logic:
  "supported"              → evidence met + damage confirmed + no contradiction
  "contradicted"           → contradiction detected OR damage type mismatch
  "not_enough_information" → evidence not met (missing parts / invalid images)
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import Verdict, InvestigationContext
from utils.prompt_builder import verdict_prompt, SYSTEM_INVESTIGATOR
import config

logger = logging.getLogger(__name__)


class VerdictAgent(BaseAgent):
    """Agent 7: Generate the final verdict across all 14 output fields."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        claim        = context.claim
        extracted    = context.extracted
        coverage     = context.coverage
        image_anal   = context.image_analysis
        history      = context.history
        contradiction = context.contradiction

        self.log(f"Generating verdict for user={claim.user_id}")

        # ── Gather risk flags from all agents ─────────────────────────────────
        all_risk_flags: list[str] = []

        if image_anal:
            for flag in image_anal.flags:
                if flag in config.VALID_RISK_FLAGS and flag != "none":
                    all_risk_flags.append(flag)

        if contradiction:
            for flag in contradiction.contradiction_flags:
                if flag in config.VALID_RISK_FLAGS and flag != "none":
                    all_risk_flags.append(flag)

        if history:
            for flag in history.risk_flags:
                if flag in config.VALID_RISK_FLAGS and flag != "none":
                    all_risk_flags.append(flag)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_flags = []
        for f in all_risk_flags:
            if f not in seen:
                seen.add(f)
                unique_flags.append(f)

        # Add manual_review_required if high risk + low authenticity
        if history and history.risk_level == "high":
            if "manual_review_required" not in unique_flags:
                unique_flags.append("manual_review_required")

        # ── Determine claim_status ─────────────────────────────────────────────
        claim_status = self._determine_status(coverage, image_anal, contradiction)

        # ── Determine severity ────────────────────────────────────────────────
        severity = self._determine_severity(
            claim_status, image_anal, extracted
        )

        # ── Get issue_type and object_part ────────────────────────────────────
        # Try to refine from actual observed damages, fall back to extracted claim
        issue_type  = self._determine_issue_type(image_anal, extracted)
        object_part = self._determine_object_part(coverage, extracted)

        # ── Build justification ───────────────────────────────────────────────
        justification = self._build_justification(
            claim_status, coverage, image_anal, contradiction, extracted, claim
        )

        # ── For borderline cases, use LLM to refine justification ─────────────
        if self._needs_llm_refinement(coverage, contradiction):
            refined = self._llm_verdict(context, claim_status, severity, unique_flags)
            if refined:
                justification = refined.get("claim_status_justification", justification)
                # Only accept LLM status if it matches our rule-based decision
                # (prevents the LLM from overriding our core logic)

        # ── Build supporting image IDs ────────────────────────────────────────
        if coverage and coverage.supporting_image_ids:
            supporting_ids_str = ";".join(coverage.supporting_image_ids)
        else:
            supporting_ids_str = "none"

        # ── Assemble final verdict ─────────────────────────────────────────────
        risk_flags_str = ";".join(unique_flags) if unique_flags else "none"
        valid_image_str = "true" if (image_anal and image_anal.valid_image) else "false"
        evidence_met_str = "true" if (coverage and coverage.evidence_standard_met) else "false"
        evidence_reason = (
            coverage.evidence_standard_met_reason if coverage
            else "Evidence assessment unavailable."
        )

        try:
            verdict = Verdict(
                user_id=claim.user_id,
                image_paths=claim.raw_image_paths,
                user_claim=claim.user_claim,
                claim_object=claim.claim_object,
                evidence_standard_met=evidence_met_str,
                evidence_standard_met_reason=evidence_reason,
                risk_flags=risk_flags_str,
                issue_type=issue_type,
                object_part=object_part,
                claim_status=claim_status,
                claim_status_justification=justification,
                supporting_image_ids=supporting_ids_str,
                valid_image=valid_image_str,
                severity=severity,
            )
            context.verdict = verdict
            self.log(
                f"Verdict: status={claim_status}, severity={severity}, "
                f"risk_flags={risk_flags_str}"
            )
        except Exception as e:
            self.warn(f"Verdict assembly failed: {e}")
            context.errors.append(f"VerdictAgent: {e}")
            context.verdict = self._fallback_verdict(claim, str(e))

        return context

    # ── Status determination ───────────────────────────────────────────────────

    def _determine_status(self, coverage, image_anal, contradiction) -> str:
        """Core decision logic — rule-based, image-first."""

        # If images are not valid / missing → not_enough_information
        if not image_anal or not image_anal.valid_image:
            return "not_enough_information"

        # If required evidence not met → not_enough_information
        if not coverage or not coverage.evidence_standard_met:
            return "not_enough_information"

        # If strong contradiction detected by ContradictionAgent → contradicted
        if contradiction and contradiction.has_contradiction:
            if "claim_mismatch" in contradiction.contradiction_flags:
                return "contradicted"

        # ── Negative-evidence fallback (defence-in-depth) ──────────────────────
        # Even if the ContradictionAgent didn't fire, catch the case where the
        # claimed part IS visible but zero damage was detected.  This is active
        # contradicting evidence, not "not_enough_information".
        no_damage_detected = not image_anal.merged_damages
        if no_damage_detected and coverage and coverage.evidence_standard_met:
            # The required parts are present (coverage met) but nothing looks damaged
            return "contradicted"

        # Evidence is met and no contradiction → supported
        return "supported"

    # ── Field refinements ─────────────────────────────────────────────────────

    def _determine_severity(self, claim_status, image_anal, extracted) -> str:
        if claim_status == "not_enough_information":
            return "unknown"
        if not image_anal:
            return "unknown"

        # Use observed severity from image analysis
        observed = image_anal.overall_severity
        if observed in config.VALID_SEVERITIES:
            return observed
        return "unknown"

    def _determine_issue_type(self, image_anal, extracted) -> str:
        """Prefer observed damage type from image; fall back to claim."""
        if image_anal and image_anal.merged_damages:
            types = [d.get("type", "") for d in image_anal.merged_damages if d.get("type")]
            if types:
                return types[0]
        if extracted:
            return extracted.issue_type
        return "unknown"

    def _determine_object_part(self, coverage, extracted) -> str:
        """Return the claimed part name."""
        if extracted:
            return extracted.object_part
        return "unknown"

    # ── Justification building ─────────────────────────────────────────────────

    def _build_justification(
        self, status, coverage, image_anal, contradiction, extracted, claim
    ) -> str:
        if status == "not_enough_information":
            if image_anal and not image_anal.valid_image:
                return "The submitted image does not show the claimed area, so the damage cannot be verified."
            if coverage and coverage.missing_evidence:
                missing_str = ", ".join(coverage.missing_evidence)
                return (
                    f"The submitted image does not show the {missing_str}, "
                    f"so the claimed damage cannot be verified."
                )
            return "The submitted images do not provide sufficient evidence to assess this claim."

        if status == "contradicted":
            # Prefer the detailed reason from the ContradictionAgent
            if contradiction and contradiction.contradiction_reason:
                return contradiction.contradiction_reason
            # Negative-evidence case: coverage met but no damage at all
            if extracted and image_anal and not image_anal.merged_damages:
                return (
                    f"The {extracted.object_part} is visible in the submitted image "
                    f"but shows no signs of {extracted.issue_type}. "
                    f"The absence of any detectable damage directly contradicts the claim."
                )
            if extracted and image_anal:
                observed = [d.get("type", "") for d in image_anal.merged_damages]
                return (
                    f"The images show {', '.join(observed) or 'different damage'} "
                    f"rather than the claimed {extracted.issue_type}."
                )
            return "The visual evidence contradicts the submitted claim."

        # supported
        if coverage and coverage.supporting_image_ids and extracted:
            img_str = ", ".join(coverage.supporting_image_ids)
            return (
                f"The {img_str} image(s) directly support the claim by showing "
                f"{extracted.issue_type} on the {extracted.object_part}."
            )
        return "The submitted images support the claimed damage."

    def _needs_llm_refinement(self, coverage, contradiction) -> bool:
        """Use LLM only for borderline cases."""
        if coverage and 0.4 <= coverage.coverage_score <= 0.7:
            return True
        if contradiction and 0.4 <= contradiction.alignment_score <= 0.6:
            return True
        return False

    def _llm_verdict(
        self, context, claim_status, severity, risk_flags
    ) -> dict | None:
        """Use LLM to refine justification for borderline cases."""
        extracted  = context.extracted
        coverage   = context.coverage
        image_anal = context.image_analysis
        contradiction = context.contradiction

        if not all([extracted, coverage, image_anal]):
            return None

        try:
            prompt = verdict_prompt(
                claim_object=context.claim.claim_object,
                object_part=extracted.object_part,
                issue_type=extracted.issue_type,
                evidence_standard_met=coverage.evidence_standard_met,
                evidence_standard_met_reason=coverage.evidence_standard_met_reason,
                observed_damages=image_anal.merged_damages,
                coverage_score=coverage.coverage_score,
                has_contradiction=contradiction.has_contradiction if contradiction else False,
                contradiction_reason=contradiction.contradiction_reason if contradiction else "",
                risk_flags=risk_flags,
                overall_severity=image_anal.overall_severity,
                valid_image=image_anal.valid_image,
            )
            messages = self.mm.build_text_messages(prompt, system_prompt=SYSTEM_INVESTIGATOR)
            raw = self.mm.generate_text(messages)
            return self.parse_json(raw, fallback=None)
        except Exception as e:
            self.warn(f"LLM verdict refinement failed: {e}")
            return None

    def _fallback_verdict(self, claim, error: str) -> Verdict:
        return Verdict(
            user_id=claim.user_id,
            image_paths=claim.raw_image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met="false",
            evidence_standard_met_reason="Pipeline error — manual review required.",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification=f"Pipeline error: {error}",
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
        )
