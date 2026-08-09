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


PART_KEYWORDS: dict[str, set[str]] = {
    "car": {
        "front_bumper", "rear_bumper", "bumper", "windshield", "hood", "bonnet",
        "door", "door_panel", "side_mirror", "mirror", "headlight", "taillight",
        "fender", "wheel", "tire", "tyre", "roof", "trunk", "boot", "panel",
    },
    "laptop": {
        "screen", "display", "hinge", "keyboard", "keys", "trackpad", "touchpad",
        "palm rest", "body", "lid", "corner", "bezel", "frame", "laptop",
    },
    "package": {
        "package", "parcel", "box", "box corner", "seal", "label", "shipping label",
        "contents", "item", "product", "charging case", "earbuds", "earbud", "phone",
        "laptop", "headphones", "device",
    },
}

ISSUE_KEYWORDS: dict[str, set[str]] = {
    "crack": {"crack", "cracked", "fracture", "split", "broken screen"},
    "broken_part": {"broken", "broke", "broken part", "not working", "wobble", "loose", "detached", "missing", "damage"},
    "dent": {"dent", "dented", "deform", "deformed", "bend", "bent", "ding"},
    "scratch": {"scratch", "scratched", "scrape", "scraped", "scuff", "scuffed", "mark"},
    "stain": {"stain", "stained", "spill", "spilled", "dirty", "smudge"},
    "torn_packaging": {"tear", "torn", "ripped", "open", "opened", "split packaging", "damaged packaging"},
    "crushed_packaging": {"crushed", "squashed", "flattened", "crumpled"},
    "water_damage": {"water", "wet", "liquid", "soaked", "damp", "moisture", "leak"},
    "missing_contents": {"missing", "not inside", "empty", "gone", "lost"},
    "shatter": {"shatter", "shattered", "shattering"},
}


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

            extracted = self._normalize_extracted_claim(extracted, claim.user_claim, claim.claim_object)

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
            context.extracted = self._normalize_extracted_claim(
                context.extracted, claim.user_claim, claim.claim_object
            )

        return context

    def _normalize_extracted_claim(
        self,
        extracted: ExtractedClaim,
        user_claim: str,
        claim_object: str,
    ) -> ExtractedClaim:
        """Apply deterministic cleanup so obvious parts/damage types don't stay unknown."""
        claim_text = user_claim.lower()
        claim_object_lower = claim_object.lower()

        object_part = extracted.object_part.lower().strip().replace(" ", "_")
        issue_type = extracted.issue_type.lower().strip().replace(" ", "_")

        if object_part in {"unknown", "", "none"}:
            object_part = self._infer_part_from_text(claim_text, claim_object_lower) or object_part
        else:
            object_part = self._canonical_part(object_part, claim_object_lower)

        if issue_type in {"unknown", "", "none"}:
            issue_type = self._infer_issue_from_text(claim_text) or issue_type
        else:
            issue_type = self._canonical_issue(issue_type)

        if claim_object_lower == "package" and object_part in {"package", "parcel", "box"}:
            if any(term in claim_text for term in ("contents", "inside", "item", "product", "earbud", "case", "device", "phone", "laptop")):
                object_part = "contents"

        extracted.object_part = object_part or "unknown"
        extracted.issue_type = issue_type or "unknown"

        if not extracted.all_parts:
            extracted.all_parts = [extracted.object_part]
        else:
            extracted.all_parts = [self._canonical_part(p.lower().strip().replace(" ", "_"), claim_object_lower) for p in extracted.all_parts]

        if not extracted.all_issue_types:
            extracted.all_issue_types = [extracted.issue_type]
        else:
            extracted.all_issue_types = [self._canonical_issue(t.lower().strip().replace(" ", "_")) for t in extracted.all_issue_types]

        return extracted

    def _infer_part_from_text(self, claim_text: str, claim_object: str) -> str | None:
        keywords = PART_KEYWORDS.get(claim_object, set())
        for keyword in keywords:
            normalized = keyword.replace("_", " ")
            if normalized in claim_text:
                return keyword.replace(" ", "_")
        return None

    def _infer_issue_from_text(self, claim_text: str) -> str | None:
        for issue, keywords in ISSUE_KEYWORDS.items():
            if any(keyword in claim_text for keyword in keywords):
                return issue
        return None

    def _canonical_part(self, part: str, claim_object: str) -> str:
        aliases = {
            "screen": "screen",
            "display": "screen",
            "monitor": "screen",
            "touchpad": "trackpad",
            "palm_rest": "trackpad",
            "keys": "keyboard",
            "keypad": "keyboard",
            "bonnet": "hood",
            "boot": "trunk",
            "tyre": "tire",
            "rear": "rear_bumper",
            "front": "front_bumper",
            "package": "contents" if claim_object == "package" else part,
            "parcel": "contents" if claim_object == "package" else part,
            "box": "contents" if claim_object == "package" else part,
            "item": "contents" if claim_object == "package" else part,
            "product": "contents" if claim_object == "package" else part,
            "charging_case": "contents" if claim_object == "package" else part,
            "earbud": "contents" if claim_object == "package" else part,
            "earbuds": "contents" if claim_object == "package" else part,
        }
        return aliases.get(part, part)

    def _canonical_issue(self, issue: str) -> str:
        aliases = {
            "cracked": "crack",
            "fracture": "crack",
            "split": "crack",
            "scratched": "scratch",
            "scrape": "scratch",
            "scraped": "scratch",
            "deform": "dent",
            "dented": "dent",
            "bent": "dent",
            "broke": "broken_part",
            "broken": "broken_part",
            "shattered": "shatter",
            "wet": "water_damage",
            "damp": "water_damage",
            "leak": "water_damage",
            "torn": "torn_packaging",
            "ripped": "torn_packaging",
            "crushed": "crushed_packaging",
            "squashed": "crushed_packaging",
            "missing": "missing_contents",
            "gone": "missing_contents",
        }
        return aliases.get(issue, issue)
