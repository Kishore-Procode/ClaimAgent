"""
Pipeline Orchestrator — InvestigationPipeline
Runs all agents in order for each claim, managing the InvestigationContext.
Bonus agents are optional and can be individually toggled in config.py.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

from models.model_manager import ModelManager
from models.schemas import ClaimRecord, InvestigationContext, Verdict

from agents.core.claim_extractor    import ClaimExtractorAgent
from agents.core.evidence_agent     import EvidenceRequirementAgent
from agents.core.image_analyzer     import ImageAnalyzerAgent
from agents.core.coverage_agent     import CoverageAnalyzerAgent
from agents.core.history_agent      import HistoryRiskAgent
from agents.core.contradiction_agent import ContradictionAgent
from agents.core.verdict_agent      import VerdictAgent

import config

logger = logging.getLogger(__name__)


class InvestigationPipeline:
    """
    Orchestrates all agents for claims investigation.

    Usage:
        pipeline = InvestigationPipeline(model_manager, history_map)
        pipeline.setup()
        verdict = pipeline.investigate(claim_record)
        pipeline.teardown()
    """

    def __init__(
        self,
        model_manager: ModelManager,
        history_map: dict[str, list[ClaimRecord]],
        enable_authenticity: bool = config.ENABLE_AUTHENTICITY_AGENT,
        enable_duplicate: bool    = config.ENABLE_DUPLICATE_AGENT,
        enable_critique: bool     = config.ENABLE_CRITIQUE_AGENT,
    ):
        self.mm = model_manager
        self.history_map = history_map
        self.enable_authenticity = enable_authenticity
        self.enable_duplicate    = enable_duplicate
        self.enable_critique     = enable_critique

        # Core agents (always run)
        self.claim_extractor    = ClaimExtractorAgent(model_manager)
        self.evidence_agent     = EvidenceRequirementAgent(model_manager)
        self.image_analyzer     = ImageAnalyzerAgent(model_manager)
        self.coverage_agent     = CoverageAnalyzerAgent(model_manager)
        self.history_agent      = HistoryRiskAgent(model_manager, history_map)
        self.contradiction_agent = ContradictionAgent(model_manager)
        self.verdict_agent      = VerdictAgent(model_manager)

        # Bonus agents (optional)
        self._authenticity_agent = None
        self._duplicate_agent    = None
        self._critique_agent     = None

        # Shared hash store for duplicate detection (persists across claims)
        self._hash_store: dict[str, str] = {}

    def setup(self) -> None:
        """Initialize bonus agents and load the AI model."""
        if self.enable_authenticity:
            from agents.bonus.authenticity_agent import AuthenticityAgent
            self._authenticity_agent = AuthenticityAgent(self.mm)

        if self.enable_duplicate:
            from agents.bonus.duplicate_agent import DuplicateAgent
            self._duplicate_agent = DuplicateAgent(self.mm, self._hash_store)

        if self.enable_critique:
            from agents.bonus.critique_agent import CritiqueAgent
            self._critique_agent = CritiqueAgent(self.mm)

        logger.info(
            f"Pipeline setup: authenticity={self.enable_authenticity}, "
            f"duplicate={self.enable_duplicate}, critique={self.enable_critique}"
        )

    def teardown(self) -> None:
        """Unload the model and free resources."""
        self.mm.unload()

    def investigate(self, claim: ClaimRecord) -> Verdict:
        """
        Run the full investigation pipeline for one claim.

        Returns:
            Verdict with all 14 output fields populated.
        """
        start_time = time.time()
        logger.info(
            f"━━━ Investigating claim: user={claim.user_id}, "
            f"object={claim.claim_object} ━━━"
        )

        context = InvestigationContext(claim=claim)

        # ── Stage 1: Understand the claim ────────────────────────────────────
        context = self._run_agent(self.claim_extractor, context, "ClaimExtractor")
        context = self._run_agent(self.evidence_agent, context, "EvidenceRequirement")

        # ── Stage 2: Visual investigation ────────────────────────────────────
        context = self._run_agent(self.image_analyzer, context, "ImageAnalyzer")

        # ── Bonus: Authenticity (Python-only, no model needed) ────────────────
        if self._authenticity_agent:
            context = self._run_agent(self._authenticity_agent, context, "Authenticity")

        # ── Bonus: Duplicate (Python-only, no model needed) ───────────────────
        if self._duplicate_agent:
            context = self._run_agent(self._duplicate_agent, context, "Duplicate")

        # ── Stage 3: Coverage verification ───────────────────────────────────
        context = self._run_agent(self.coverage_agent, context, "CoverageAnalyzer")

        # ── Stage 4: Risk and contradiction ──────────────────────────────────
        context = self._run_agent(self.history_agent, context, "HistoryRisk")
        context = self._run_agent(self.contradiction_agent, context, "Contradiction")

        # ── Stage 5: Verdict ─────────────────────────────────────────────────
        context = self._run_agent(self.verdict_agent, context, "Verdict")

        # ── Bonus: Self-Critique (optional — may revise verdict) ──────────────
        if self._critique_agent:
            context = self._run_agent(self._critique_agent, context, "SelfCritique")

        elapsed = time.time() - start_time
        logger.info(
            f"━━━ Done: user={claim.user_id} → "
            f"status={context.verdict.claim_status if context.verdict else 'ERROR'} "
            f"({elapsed:.1f}s) ━━━"
        )

        if context.errors:
            logger.warning(f"Non-fatal errors for {claim.user_id}: {context.errors}")

        # Return final verdict (or a fallback if something went very wrong)
        if context.verdict is None:
            return self._emergency_fallback(claim)

        return context.verdict

    def investigate_stream(self, claim: ClaimRecord):
        """
        Run the full pipeline for one claim, yielding status events for the WebSocket dashboard.
        Each agent_completed event includes a duration_s field (seconds, 2 dp).
        A final pipeline_complete event carries accumulated token usage.
        """
        context = InvestigationContext(claim=claim)
        self.mm.reset_accumulated_usage()

        def to_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, "dict"):
                return obj.dict()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj

        def run_timed(agent, name):
            t0 = time.time()
            ctx = self._run_agent(agent, context, name)
            return ctx, round(time.time() - t0, 2)

        # --- ClaimExtractor ---
        yield {"event": "agent_started", "agent": "ClaimExtractor"}
        context, dur = run_timed(self.claim_extractor, "ClaimExtractor")
        yield {"event": "agent_completed", "agent": "ClaimExtractor",
               "data": to_dict(context.extracted), "duration_s": dur}

        # --- EvidenceRequirement ---
        yield {"event": "agent_started", "agent": "EvidenceRequirement"}
        context, dur = run_timed(self.evidence_agent, "EvidenceRequirement")
        yield {"event": "agent_completed", "agent": "EvidenceRequirement",
               "data": to_dict(context.requirements), "duration_s": dur}

        # --- ImageAnalyzer ---
        yield {"event": "agent_started", "agent": "ImageAnalyzer"}
        context, dur = run_timed(self.image_analyzer, "ImageAnalyzer")
        yield {"event": "agent_completed", "agent": "ImageAnalyzer",
               "data": to_dict(context.image_analysis), "duration_s": dur}

        # --- Authenticity (run silently) ---
        if self._authenticity_agent:
            context = self._run_agent(self._authenticity_agent, context, "Authenticity")

        # --- Duplicate (run silently) ---
        if self._duplicate_agent:
            context = self._run_agent(self._duplicate_agent, context, "Duplicate")

        # --- CoverageAnalyzer ---
        yield {"event": "agent_started", "agent": "CoverageAnalyzer"}
        context, dur = run_timed(self.coverage_agent, "CoverageAnalyzer")
        yield {"event": "agent_completed", "agent": "CoverageAnalyzer",
               "data": to_dict(context.coverage), "duration_s": dur}

        # --- HistoryRisk ---
        yield {"event": "agent_started", "agent": "HistoryRisk"}
        context, dur = run_timed(self.history_agent, "HistoryRisk")
        yield {"event": "agent_completed", "agent": "HistoryRisk",
               "data": to_dict(context.history), "duration_s": dur}

        # --- Contradiction (run silently) ---
        if self.contradiction_agent:
            context = self._run_agent(self.contradiction_agent, context, "Contradiction")

        # --- Verdict ---
        yield {"event": "agent_started", "agent": "Verdict"}
        context, dur = run_timed(self.verdict_agent, "Verdict")

        # --- FAST DEBUG LOGS FOR USER ---
        logger.info("=== FAST DIAGNOSTIC LOGS ===")
        logger.info(f"VISIBLE={context.image_analysis.merged_visible_parts if context.image_analysis else 'None'}")
        logger.info(f"DAMAGES={context.image_analysis.merged_damages if context.image_analysis else 'None'}")
        logger.info(f"VALID={context.image_analysis.valid_image if context.image_analysis else 'None'}")
        logger.info(f"COVERAGE={context.coverage.evidence_standard_met if context.coverage else 'None'}")
        logger.info(f"CONTRADICTION={context.contradiction.has_contradiction if context.contradiction else 'None'}")
        logger.info(f"STATUS_PRE_CRITIQUE={context.verdict.claim_status if context.verdict else 'None'}")

        # --- SelfCritique (may revise verdict, part of Verdict stage timing) ---
        if self._critique_agent:
            t0 = time.time()
            context = self._run_agent(self._critique_agent, context, "SelfCritique")
            dur = round(dur + (time.time() - t0), 2)

        logger.info(f"STATUS_POST_CRITIQUE={context.verdict.claim_status if context.verdict else 'None'}")
        logger.info("============================")

        yield {"event": "agent_completed", "agent": "Verdict",
               "data": to_dict(context.verdict), "duration_s": dur}

        # --- Final pipeline summary with token usage ---
        usage = dict(self.mm.accumulated_usage)
        used = usage.get("total_tokens", 0)
        budget = 128000  # conservative context window for qwen2.5-vl-72b
        pct = round((used / budget) * 100, 1) if budget > 0 else 0.0
        yield {
            "event": "pipeline_complete",
            "token_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": used,
                "budget_tokens": budget,
                "usage_pct": pct,
            }
        }

    def _run_agent(
        self, agent, context: InvestigationContext, name: str
    ) -> InvestigationContext:
        """Run a single agent with error isolation."""
        try:
            return agent.run(context)
        except Exception as e:
            logger.error(f"Agent {name} crashed: {e}", exc_info=True)
            context.errors.append(f"{name}: {e}")
            return context

    def _emergency_fallback(self, claim: ClaimRecord) -> Verdict:
        """Absolute last-resort fallback if the verdict is None."""
        return Verdict(
            user_id=claim.user_id,
            image_paths=claim.raw_image_paths,
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            evidence_standard_met="false",
            evidence_standard_met_reason="Pipeline failure — manual review required.",
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="The pipeline encountered an error. Manual review required.",
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
        )
