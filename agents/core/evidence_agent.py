"""
Agent 2 — Evidence Requirement Agent
Determines what parts and damage must be visible to verify the claim.
Uses a rule-based table first, falls back to Qwen2.5-VL for edge cases.
"""
from __future__ import annotations
import logging

from agents.base_agent import BaseAgent
from models.schemas import EvidenceRequirement, InvestigationContext
from utils.prompt_builder import evidence_requirement_prompt, SYSTEM_INVESTIGATOR

logger = logging.getLogger(__name__)

# Rule-based part → required visual evidence map
# Reduces unnecessary LLM calls for common cases
PART_EVIDENCE_RULES: dict[str, dict] = {
    # Car parts
    "rear_bumper":   {"required_parts": ["rear_bumper"], "required_damage_types": ["dent", "crack", "scratch"]},
    "front_bumper":  {"required_parts": ["front_bumper"], "required_damage_types": ["dent", "crack", "scratch"]},
    "windshield":    {"required_parts": ["windshield"], "required_damage_types": ["crack", "shatter"]},
    "hood":          {"required_parts": ["hood"], "required_damage_types": ["dent", "scratch", "hail_damage"]},
    "door":          {"required_parts": ["door_panel"], "required_damage_types": ["dent", "scratch"]},
    "side_mirror":   {"required_parts": ["side_mirror"], "required_damage_types": ["broken_part", "missing_part"]},
    "headlight":     {"required_parts": ["headlight"], "required_damage_types": ["crack", "broken_part"]},
    "taillight":     {"required_parts": ["taillight"], "required_damage_types": ["crack", "broken_part"]},
    # Laptop parts
    "screen":        {"required_parts": ["screen", "display"], "required_damage_types": ["crack", "shatter", "stain"]},
    "hinge":         {"required_parts": ["hinge", "hinge_area"], "required_damage_types": ["broken_part", "misalignment"]},
    "keyboard":      {"required_parts": ["keyboard"], "required_damage_types": ["stain", "missing_keys", "broken_keys"]},
    "trackpad":      {"required_parts": ["trackpad", "palm_rest"], "required_damage_types": ["crack", "broken_part"]},
    "corner":        {"required_parts": ["laptop_body", "corner"], "required_damage_types": ["dent", "crack"]},
    "body":          {"required_parts": ["laptop_body"], "required_damage_types": ["crack", "dent"]},
    "lid":           {"required_parts": ["laptop_lid", "outer_lid"], "required_damage_types": ["crack", "dent"]},
    # Package parts
    "package":        {"required_parts": ["package"], "required_damage_types": ["crushed_packaging", "water_damage", "torn_packaging", "dent", "stain"]},
    "box":            {"required_parts": ["package"], "required_damage_types": ["crushed_packaging", "water_damage", "torn_packaging", "dent", "stain"]},
    "parcel":         {"required_parts": ["package"], "required_damage_types": ["crushed_packaging", "water_damage", "torn_packaging", "dent", "stain"]},
    "package_corner": {"required_parts": ["package_corner", "box_corner"], "required_damage_types": ["crushed_packaging"]},
    "seal":           {"required_parts": ["package_seal", "box_opening"], "required_damage_types": ["torn_packaging"]},
    "contents":       {"required_parts": ["package_interior", "contents_area"], "required_damage_types": ["missing_contents"]},
    "box_part":       {"required_parts": ["outer_box"], "required_damage_types": ["crushed_packaging", "water_damage"]},
    "label":          {"required_parts": ["shipping_label"], "required_damage_types": ["water_damage", "torn_packaging"]},
    "package_side":   {"required_parts": ["package_surface"], "required_damage_types": ["water_damage", "stain"]},
}


class EvidenceRequirementAgent(BaseAgent):
    """Agent 2: Determine what evidence is needed for this claim."""

    def run(self, context: InvestigationContext) -> InvestigationContext:
        if context.extracted is None:
            self.warn("No extracted claim available — using defaults.")
            context.requirements = EvidenceRequirement(
                required_parts=["unknown"],
                required_damage_types=["unknown"],
            )
            return context

        extracted = context.extracted
        self.log(f"Determining evidence requirements for {extracted.object_part} / {extracted.issue_type}")

        # Override for package claims
        if context.claim.claim_object == "package":
            box_parts = {"package", "box", "parcel", "package_corner", "box_corner", "seal", "box_opening", "label", "shipping_label", "package_side", "package_surface"}
            part_lower = extracted.object_part.lower()
            
            # If the claimed part is one of the box/packaging parts
            if any(bp in part_lower for bp in box_parts):
                required_parts = ["package"]
            else:
                required_parts = ["contents"]
                
            rule = PART_EVIDENCE_RULES.get(part_lower)
            required_types = rule["required_damage_types"] if rule else [extracted.issue_type]
            
            context.requirements = EvidenceRequirement(
                required_parts=required_parts,
                required_damage_types=required_types,
                minimum_images_needed=1,
            )
            self.log(f"Package evidence override: required_parts={required_parts}, damage_types={required_types}")
            return context

        # Try rule-based lookup first
        rule = PART_EVIDENCE_RULES.get(extracted.object_part.lower())

        # For multi-part claims, merge rules for each part
        if extracted.is_multi_part and extracted.all_parts:
            merged_parts: set[str] = set()
            merged_types: set[str] = set()
            for part in extracted.all_parts:
                r = PART_EVIDENCE_RULES.get(part.lower())
                if r:
                    merged_parts.update(r["required_parts"])
                    merged_types.update(r["required_damage_types"])
            if merged_parts:
                context.requirements = EvidenceRequirement(
                    required_parts=list(merged_parts),
                    required_damage_types=list(merged_types),
                    minimum_images_needed=min(len(extracted.all_parts), 3),
                )
                self.log(f"Rule-based (multi-part): {context.requirements.required_parts}")
                return context

        if rule:
            context.requirements = EvidenceRequirement(
                required_parts=rule["required_parts"],
                required_damage_types=rule["required_damage_types"],
                minimum_images_needed=1,
            )
            self.log(f"Rule-based: {context.requirements.required_parts}")
            return context

        # Fall back to LLM for unknown parts
        self.log("No rule match — falling back to LLM.")
        prompt = evidence_requirement_prompt(
            context.claim.claim_object,
            extracted.object_part,
            extracted.issue_type,
            extracted.all_parts,
        )
        messages = self.mm.build_text_messages(prompt, system_prompt=SYSTEM_INVESTIGATOR)

        try:
            raw = self.mm.generate_text(messages)
            data = self.parse_json(raw, fallback={})

            context.requirements = EvidenceRequirement(
                required_parts=self.safe_list(data.get("required_parts")) or [extracted.object_part],
                required_damage_types=self.safe_list(data.get("required_damage_types")) or [extracted.issue_type],
                minimum_images_needed=int(data.get("minimum_images_needed", 1)),
            )
            self.log(f"LLM-based: {context.requirements.required_parts}")

        except Exception as e:
            self.warn(f"Evidence requirement failed: {e}")
            context.errors.append(f"EvidenceAgent: {e}")
            context.requirements = EvidenceRequirement(
                required_parts=[extracted.object_part],
                required_damage_types=[extracted.issue_type],
            )

        return context
