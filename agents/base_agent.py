"""
Base agent class — all agents inherit from this.
Provides JSON parsing, error handling, and logging.
"""
from __future__ import annotations
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from models.model_manager import ModelManager
from models.schemas import InvestigationContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all pipeline agents."""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, context: InvestigationContext) -> InvestigationContext:
        """
        Execute this agent's logic. Receives the full context (with all
        prior agents' outputs) and returns it updated with this agent's result.
        """
        ...

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def parse_json(self, raw: str, fallback: Optional[dict] = None) -> dict:
        """
        Robustly parse JSON from a model response.
        Handles markdown code fences, trailing commas, and other common artifacts.

        Returns fallback dict on parse failure.
        """
        text = raw.strip()

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Extract first JSON object if extra text surrounds it
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try removing trailing commas before } or ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{self.name}] JSON parse failed: {e}\nRaw response:\n{raw[:500]}"
                )
                return fallback or {}

    def safe_str(self, val: Any, default: str = "unknown") -> str:
        """Safely convert a value to a non-empty string."""
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    def safe_list(self, val: Any) -> list:
        """Safely coerce a value to a list."""
        if isinstance(val, list):
            return val
        if val is None:
            return []
        return [val]

    def safe_float(self, val: Any, default: float = 0.0) -> float:
        """Safely coerce a value to float."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def safe_bool(self, val: Any, default: bool = False) -> bool:
        """Safely coerce a value to bool."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        try:
            return bool(val)
        except Exception:
            return default

    def log(self, msg: str) -> None:
        logger.info(f"[{self.name}] {msg}")

    def warn(self, msg: str) -> None:
        logger.warning(f"[{self.name}] {msg}")
