"""
Pydantic schemas for every data structure flowing through the pipeline.
All field names mirror the confirmed output.csv column names exactly.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, field_validator


# ── INPUT ─────────────────────────────────────────────────────────────────────

class ClaimRecord(BaseModel):
    """One row from claims.csv."""
    user_id: str
    image_paths: list[str]          # already split on ";"
    user_claim: str                 # full conversation string
    claim_object: str               # "car" | "laptop" | "package"

    @property
    def raw_image_paths(self) -> str:
        """Reconstruct the original semicolon-joined string."""
        return ";".join(self.image_paths)


# ── AGENT 1: Claim Extractor ──────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    """Structured claim extracted from free-text conversation."""
    object_part: str                # e.g. "rear_bumper", "screen", "hinge"
    issue_type: str                 # e.g. "dent", "crack", "scratch"
    incident_summary: str           # one-sentence description
    is_multi_part: bool = False
    all_parts: list[str] = []       # populated if is_multi_part
    all_issue_types: list[str] = [] # populated if is_multi_part


# ── AGENT 2: Evidence Requirement ─────────────────────────────────────────────

class EvidenceRequirement(BaseModel):
    """What must be visible in images to verify the claim."""
    required_parts: list[str]
    required_damage_types: list[str]
    minimum_images_needed: int = 1


# ── AGENT 3: Image Analyzer ───────────────────────────────────────────────────

class SingleImageAnalysis(BaseModel):
    """Analysis result for one image."""
    image_id: str                   # "img_1", "img_2", etc.
    image_path: str                 # resolved full path
    detected_object: str
    object_matches_claim: bool
    visible_parts: list[str]
    damages: list[dict]             # [{"type": "crack", "part": "screen", "severity": "medium"}]
    overall_severity: str           # "low"|"medium"|"high"|"critical"|"unknown"
    image_quality: str              # "good"|"blurry"|"partial"|"wrong_angle"
    valid_image: bool
    contains_text_instruction: bool = False  # prompt injection in image
    raw_response: str = ""

class ImageAnalysis(BaseModel):
    """Merged analysis across all images for a claim."""
    per_image: list[SingleImageAnalysis]
    merged_visible_parts: list[str]
    merged_damages: list[dict]
    best_image_id: Optional[str]
    overall_severity: str
    valid_image: bool               # True if ANY image is valid
    flags: list[str]                # blurry_image, wrong_angle, text_instruction_present, etc.


# ── AGENT 4: Coverage Analyzer ───────────────────────────────────────────────

class CoverageAnalysis(BaseModel):
    """Whether image evidence covers what the claim requires."""
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    coverage_map: dict[str, bool]   # {"screen": True, "hinge": False}
    coverage_score: float           # 0.0 – 1.0
    missing_evidence: list[str]
    supporting_image_ids: list[str] # ["img_1", "img_2"]
    next_best_evidence: list[str]


# ── AGENT 5: History Risk ─────────────────────────────────────────────────────

class HistoryRisk(BaseModel):
    """Risk assessment from user claim history."""
    user_claim_count: int
    risk_level: str                 # "low"|"medium"|"high"
    risk_flags: list[str]           # ["user_history_risk", "manual_review_required"]


# ── AGENT 6: Contradiction ────────────────────────────────────────────────────

class ContradictionResult(BaseModel):
    """Whether claim and image evidence contradict each other."""
    has_contradiction: bool
    contradiction_flags: list[str]  # ["claim_mismatch", "text_instruction_present"]
    alignment_score: float          # 0.0 – 1.0
    contradiction_reason: str       # human-readable explanation


# ── BONUS AGENT 8: Authenticity ───────────────────────────────────────────────

class AuthenticityResult(BaseModel):
    """Basic image authenticity from EXIF metadata."""
    score: float                    # 0–100
    exif_present: bool
    timestamp: Optional[str]
    gps_present: bool
    camera_model: Optional[str]
    flags: list[str]                # ["possible_manipulation"]


# ── BONUS AGENT 9: Duplicate ──────────────────────────────────────────────────

class DuplicateResult(BaseModel):
    """Cross-claim image reuse detection."""
    duplicate_risk: str             # "low"|"medium"|"high"
    similar_claim_ids: list[str]


# ── BONUS AGENT 10: Self-Critique ─────────────────────────────────────────────

class CritiqueResult(BaseModel):
    """Second-pass challenge of the primary verdict."""
    challenges: list[str]
    should_revise: bool
    revised_decision: Optional[str]
    revised_justification: Optional[str]


# ── AGENT 7: Verdict (maps 1:1 to output.csv) ─────────────────────────────────

class Verdict(BaseModel):
    """Final output row — mirrors output.csv exactly."""
    user_id: str
    image_paths: str                # original semicolon-joined string
    user_claim: str
    claim_object: str
    evidence_standard_met: str      # "true" | "false"
    evidence_standard_met_reason: str
    risk_flags: str                 # semicolon-joined or "none"
    issue_type: str
    object_part: str
    claim_status: str               # "supported"|"contradicted"|"not_enough_information"
    claim_status_justification: str
    supporting_image_ids: str       # semicolon-joined or "none"
    valid_image: str                # "true" | "false"
    severity: str                   # "low"|"medium"|"high"|"critical"|"unknown"|"none"

    @field_validator("claim_status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"supported", "contradicted", "not_enough_information"}
        if v not in valid:
            raise ValueError(f"claim_status must be one of {valid}, got '{v}'")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = {"low", "medium", "high", "critical", "unknown", "none"}
        if v not in valid:
            raise ValueError(f"severity must be one of {valid}, got '{v}'")
        return v

    def to_csv_row(self) -> dict:
        """Return ordered dict matching output.csv column order."""
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": self.evidence_standard_met,
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": self.valid_image,
            "severity": self.severity,
        }


# ── FULL INVESTIGATION REPORT (internal audit trail) ─────────────────────────

class InvestigationContext(BaseModel):
    """Accumulated context passed between agents during one claim investigation."""
    claim: ClaimRecord
    extracted: Optional[ExtractedClaim] = None
    requirements: Optional[EvidenceRequirement] = None
    image_analysis: Optional[ImageAnalysis] = None
    coverage: Optional[CoverageAnalysis] = None
    history: Optional[HistoryRisk] = None
    contradiction: Optional[ContradictionResult] = None
    authenticity: Optional[AuthenticityResult] = None
    duplicate: Optional[DuplicateResult] = None
    critique: Optional[CritiqueResult] = None
    verdict: Optional[Verdict] = None
    errors: list[str] = []          # non-fatal errors from individual agents

    class Config:
        arbitrary_types_allowed = True
