"""
Bonus Agent 10 — Self-Critique Agent ⭐
Acts as a second investigator who challenges the primary verdict.
Asks: "Why might the current decision be wrong?"
May trigger a verdict revision for borderline cases.
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import CritiqueResult, InvestigationContext
from utils.prompt_builder import critique_prompt, SYSTEM_INVESTIGATOR

logger = logging.getLogger(__name__)


class CritiqueAgent(BaseAgent):
    """Bonus Agent 10: Self-critique pass on the primary verdict."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        verdict   = context.verdict
        extracted = context.extracted
        coverage  = context.coverage

        if verdict is None:
            self.warn("No verdict to critique — skipping.")
            context.critique = CritiqueResult(
                challenges=[],
                should_revise=False,
                revised_decision=None,
                revised_justification=None,
            )
            return context

        self.log(f"Critiquing verdict: {verdict.claim_status}")

        prompt = critique_prompt(
            claim_status=verdict.claim_status,
            justification=verdict.claim_status_justification,
            claim_object=context.claim.claim_object,
            object_part=extracted.object_part if extracted else "unknown",
            issue_type=extracted.issue_type if extracted else "unknown",
            coverage_score=coverage.coverage_score if coverage else 0.0,
            risk_flags=verdict.risk_flags.split(";") if verdict.risk_flags != "none" else [],
        )

        try:
            messages = self.mm.build_text_messages(prompt, system_prompt=SYSTEM_INVESTIGATOR)
            raw = self.mm.generate_text(messages)
            data = self.parse_json(raw, fallback={})

            should_revise = self.safe_bool(data.get("should_revise"), False)
            revised_decision = data.get("revised_decision") if should_revise else None
            revised_justification = data.get("revised_justification") if should_revise else None

            # Validate revised decision
            valid_statuses = {"supported", "contradicted", "not_enough_information"}
            if revised_decision and revised_decision not in valid_statuses:
                revised_decision = None
                should_revise = False

            context.critique = CritiqueResult(
                challenges=self.safe_list(data.get("challenges")),
                should_revise=should_revise,
                revised_decision=revised_decision,
                revised_justification=revised_justification,
            )

            # If critique recommends revision, update the verdict
            if should_revise and revised_decision and context.verdict:
                self.log(
                    f"Critique suggests revision: "
                    f"{verdict.claim_status} → {revised_decision}"
                )
                context.verdict.claim_status = revised_decision
                if revised_justification:
                    context.verdict.claim_status_justification = (
                        f"[Revised after self-critique] {revised_justification}"
                    )
                # Recalculate severity if status was revised
                if revised_decision == "not_enough_information":
                    context.verdict.severity = "unknown"
                elif context.image_analysis:
                    observed = context.image_analysis.overall_severity
                    import config
                    if observed in config.VALID_SEVERITIES:
                        context.verdict.severity = observed
                    else:
                        context.verdict.severity = "unknown"

            self.log(
                f"Critique: should_revise={should_revise}, "
                f"challenges={context.critique.challenges}"
            )

        except Exception as e:
            self.warn(f"Critique failed: {e}")
            context.errors.append(f"CritiqueAgent: {e}")
            context.critique = CritiqueResult(
                challenges=[],
                should_revise=False,
                revised_decision=None,
                revised_justification=None,
            )

        return context
