"""
Central configuration for VisionClaim Investigator.
Uses OpenRouter API (OpenAI-compatible) — no local GPU required.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file (safe no-op if file doesn't exist)
load_dotenv(Path(__file__).parent / ".env")

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── OpenRouter API ────────────────────────────────────────────────────────────
# Set OPENROUTER_API_KEY in your .env file (see .env.example).
# Never hardcode secrets here — they can be accidentally committed.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file (see .env.example) or set it in your environment."
    )
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model selection ───────────────────────────────────────────────────────────
# Vision model: handles images + text (used for all agents)
# Free options on OpenRouter (check https://openrouter.ai/models?q=free):
#   "qwen/qwen2.5-vl-72b-instruct:free"   ← best accuracy, free
#   "qwen/qwen2.5-vl-7b-instruct:free"    ← lighter, free
#   "google/gemma-3-27b-it:free"           ← fallback
#   "meta-llama/llama-4-scout:free"        ← vision capable, free
VISION_MODEL_ID   = "qwen/qwen2.5-vl-72b-instruct"
FALLBACK_MODEL_ID = "qwen/qwen3-vl-8b-instruct"

# ── Generation settings ───────────────────────────────────────────────────────
MAX_TOKENS      = 1024
TEMPERATURE     = 0.1     # Low temperature = more deterministic outputs
REQUEST_TIMEOUT = 120     # Seconds before giving up on an API call
MAX_RETRIES     = 3       # Retry on rate limit / transient errors
RETRY_DELAY     = 5       # Seconds to wait between retries

# ── Data paths ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = ROOT_DIR / "data"
DEFAULT_INPUT_CSV  = DEFAULT_DATA_DIR / "claims.csv"
DEFAULT_SAMPLE_CSV = DEFAULT_DATA_DIR / "sample_claims.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "output.csv"

# ── Thresholds ────────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD        = 0.5   # evidence_standard_met requires ≥50% coverage
AUTHENTICITY_THRESHOLD    = 60.0  # below this → possible_manipulation flag
USER_HISTORY_RISK_COUNT   = 2     # ≥2 claims by same user → user_history_risk
DUPLICATE_HASH_THRESHOLD  = 8     # Hamming distance for image duplicate detection

# ── Risk flag vocabulary (strict — used for validation) ───────────────────────
VALID_RISK_FLAGS = {
    "none",
    "claim_mismatch",
    "user_history_risk",
    "manual_review_required",
    "blurry_image",
    "wrong_angle",
    "damage_not_visible",
    "cropped_or_obstructed",
    "possible_manipulation",
    "text_instruction_present",
}

# ── Output field vocabulary ───────────────────────────────────────────────────
VALID_CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
VALID_SEVERITIES     = {"low", "medium", "high", "critical", "unknown", "none"}
VALID_CLAIM_OBJECTS  = {"car", "laptop", "package"}

# ── Bonus agents toggle ───────────────────────────────────────────────────────
ENABLE_AUTHENTICITY_AGENT = True
ENABLE_DUPLICATE_AGENT    = True
ENABLE_CRITIQUE_AGENT     = True
