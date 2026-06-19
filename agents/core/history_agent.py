"""
Agent 5 — History Risk Agent
Cross-references the current user's claim history (within the dataset)
to detect high-frequency or suspicious filing patterns.

IMPORTANT: History only adds risk context.
It NEVER overrides image evidence or changes claim_status directly.
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import HistoryRisk, InvestigationContext, ClaimRecord
import config

logger = logging.getLogger(__name__)


class HistoryRiskAgent(BaseAgent):
    """Agent 5: Assess user history risk."""

    def __init__(self, model_manager, history_map: dict[str, list[ClaimRecord]]):
        """
        Args:
            model_manager: The shared model manager.
            history_map:   user_id → list of all ClaimRecords in the dataset.
                           Built by csv_handler.load_claims_with_history().
        """
        super().__init__(model_manager)
        self.history_map = history_map

    def run(self, context: InvestigationContext) -> InvestigationContext:
        user_id = context.claim.user_id
        all_user_claims = self.history_map.get(user_id, [])
        claim_count = len(all_user_claims)

        self.log(f"User {user_id} has {claim_count} claim(s) in dataset.")

        risk_flags: list[str] = []
        risk_level = "low"

        # Rule 1: Repeated claims
        if claim_count >= config.USER_HISTORY_RISK_COUNT:
            risk_flags.append("user_history_risk")
            risk_level = "medium"

        # Rule 2: Many claims → high risk
        if claim_count >= 4:
            risk_level = "high"
            if "manual_review_required" not in risk_flags:
                risk_flags.append("manual_review_required")

        # Rule 3: Same object claimed multiple times
        objects_claimed = [c.claim_object for c in all_user_claims]
        if len(objects_claimed) > 1 and len(set(objects_claimed)) == 1:
            # All claims for same object type
            if "user_history_risk" not in risk_flags:
                risk_flags.append("user_history_risk")
            if risk_level == "low":
                risk_level = "medium"

        if not risk_flags:
            risk_flags = ["none"]

        context.history = HistoryRisk(
            user_claim_count=claim_count,
            risk_level=risk_level,
            risk_flags=risk_flags,
        )

        self.log(f"Risk: level={risk_level}, flags={risk_flags}")
        return context
