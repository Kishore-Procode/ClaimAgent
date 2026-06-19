"""
Agent 1 — Claim Extractor
Parses the customer-agent conversation to extract structured claim info.
Injection-resistant: ignores embedded instructions in the conversation text.
Multilingual: Qwen2.5-VL handles Hindi, Spanish, Chinese, etc.
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import ExtractedClaim, InvestigationContext
from utils.prompt_builder import claim_extractor_prompt, SYSTEM_INVESTIGATOR

logger = logging.getLogger(__name__)


class ClaimExtractorAgent(BaseAgent):
    """Agent 1: Extract structured claim from conversation."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        claim = context.claim
        self.log(f"Extracting claim for user={claim.user_id}, object={claim.claim_object}")

        prompt = claim_extractor_prompt(claim.user_claim, claim.claim_object)
        messages = self.mm.build_text_messages(prompt, system_prompt=SYSTEM_INVESTIGATOR)

        try:
            raw = self.mm.generate_text(messages)
            data = self.parse_json(raw, fallback={})

            extracted = ExtractedClaim(
                object_part=self.safe_str(data.get("object_part"), "unknown"),
                issue_type=self.safe_str(data.get("issue_type"), "unknown"),
                incident_summary=self.safe_str(data.get("incident_summary"), ""),
                is_multi_part=self.safe_bool(data.get("is_multi_part"), False),
                all_parts=self.safe_list(data.get("all_parts")),
                all_issue_types=self.safe_list(data.get("all_issue_types")),
            )

            # Ensure all_parts always contains at least the primary part
            if not extracted.all_parts:
                extracted.all_parts = [extracted.object_part]
            if not extracted.all_issue_types:
                extracted.all_issue_types = [extracted.issue_type]

            self.log(
                f"Extracted: part={extracted.object_part}, "
                f"type={extracted.issue_type}, multi={extracted.is_multi_part}"
            )
            context.extracted = extracted

        except Exception as e:
            self.warn(f"Claim extraction failed: {e}")
            context.errors.append(f"ClaimExtractor: {e}")
            # Fallback — minimal extracted claim
            context.extracted = ExtractedClaim(
                object_part="unknown",
                issue_type="unknown",
                incident_summary="Extraction failed.",
            )

        return context
