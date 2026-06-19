"""
Model Manager — OpenRouter API (OpenAI-compatible).
No local GPU required. Sends requests to OpenRouter's cloud inference.

All agents call:
  - generate_text(messages)   → text-only inference
  - generate_vision(messages) → vision inference (images encoded as base64)

Images are base64-encoded locally and sent as data URLs in the API request.
"""
from __future__ import annotations
import base64
import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI, RateLimitError, APIError, APITimeoutError

import config

logger = logging.getLogger(__name__)


def _encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    Read a local image file and return (base64_string, mime_type).
    Supports JPEG, PNG, WEBP, GIF.
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded, mime_type


def _image_to_data_url(image_path: str | Path) -> str:
    """Convert a local image file to a base64 data URL for the API."""
    b64, mime = _encode_image(image_path)
    return f"data:{mime};base64,{b64}"


class ModelManager:
    """
    API-based model manager using OpenRouter (OpenAI-compatible endpoint).

    Usage:
        mm = ModelManager()
        mm.load()   # validates API key, no heavy download
        response = mm.generate_vision(messages)
        mm.unload() # no-op for API mode, included for interface compatibility
    """

    def __init__(
        self,
        model_id: str = config.VISION_MODEL_ID,
        api_key: str = config.OPENROUTER_API_KEY,
        base_url: str = config.OPENROUTER_BASE_URL,
    ):
        self.model_id  = model_id
        self.api_key   = api_key
        self.base_url  = base_url
        self._client: Optional[OpenAI] = None
        self._loaded   = False
        # Per-investigation token counters (reset each run)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Session-level counters (survive page reloads / multiple investigations)
        self.session_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._api_call_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Initialize the OpenAI client pointing at OpenRouter. Validates API key."""
        if self._loaded:
            return

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Run: $env:OPENROUTER_API_KEY = 'sk-or-...'"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=0,  # We handle retries manually for better control
        )
        self._loaded = True
        logger.info(f"OpenRouter client ready. Model: {self.model_id}")

    def unload(self) -> None:
        """No-op for API mode — included for pipeline interface compatibility."""
        self._loaded = False
        self._client = None
        logger.info("ModelManager unloaded (API mode — no resources to free).")

    # ── Text-only inference ───────────────────────────────────────────────────

    def generate_text(
        self,
        messages: list[dict],
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """
        Run text-only inference via OpenRouter.

        Args:
            messages:    OpenAI-style chat messages (no image content).
            max_tokens:  Token limit for the response.
            temperature: Sampling temperature.

        Returns:
            Model response string (stripped).
        """
        self._assert_loaded()
        return self._call_api(messages, max_tokens, temperature)

    # ── Vision inference ──────────────────────────────────────────────────────

    def generate_vision(
        self,
        messages: list[dict],
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """
        Run vision inference via OpenRouter.

        Accepts messages already built by build_vision_messages().
        Images are expected as local file paths under the "image" key —
        this method encodes them to base64 data URLs before sending.

        Returns:
            Model response string (stripped).
        """
        self._assert_loaded()
        encoded_messages = self._encode_images_in_messages(messages)
        return self._call_api(encoded_messages, max_tokens, temperature)

    # ── Message builders (same interface as before) ───────────────────────────

    def build_vision_messages(
        self,
        image_paths: list[str],
        prompt: str,
        system_prompt: str = (
            "You are an expert insurance claims investigator. "
            "Be precise, objective, and base all conclusions strictly on visual evidence. "
            "Always respond with valid JSON only — no markdown, no extra text."
        ),
    ) -> list[dict]:
        """
        Build messages list with images. Images are stored as local paths here
        and encoded to base64 just before the API call in generate_vision().
        """
        content = []
        for path in image_paths:
            # Store as a special dict; _encode_images_in_messages will convert
            content.append({
                "type": "image_url",
                "image_url": {"url": f"__LOCAL_PATH__:{path}"},
            })
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def build_text_messages(
        self,
        prompt: str,
        system_prompt: str = (
            "You are an expert insurance claims investigator. "
            "Be precise and objective. "
            "Always respond with valid JSON only — no markdown, no extra text."
        ),
    ) -> list[dict]:
        """Build text-only messages list."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_images_in_messages(self, messages: list[dict]) -> list[dict]:
        """
        Walk through messages and replace __LOCAL_PATH__ markers
        with actual base64 data URLs.
        """
        encoded = []
        for msg in messages:
            if not isinstance(msg.get("content"), list):
                encoded.append(msg)
                continue

            new_content = []
            for part in msg["content"]:
                if (
                    part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                    and str(part["image_url"].get("url", "")).startswith("__LOCAL_PATH__:")
                ):
                    local_path = part["image_url"]["url"][len("__LOCAL_PATH__:"):]
                    path = Path(local_path)

                    if not path.exists():
                        logger.warning(f"Image not found, skipping: {local_path}")
                        continue

                    try:
                        data_url = _image_to_data_url(local_path)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        })
                        logger.debug(f"Encoded image: {path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to encode image {local_path}: {e}")
                        continue
                else:
                    new_content.append(part)

            encoded.append({**msg, "content": new_content})
        return encoded

    def _call_api(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Make the API call with retry logic for rate limits and transient errors.
        """
        last_error: Exception | None = None

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                
                # Extract and accumulate token usage details
                if hasattr(response, "usage") and response.usage:
                    pt = response.usage.prompt_tokens or 0
                    ct = response.usage.completion_tokens or 0
                    tt = response.usage.total_tokens or 0
                    self.last_usage = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                    }
                    # Per-investigation accumulator (reset per run)
                    self.accumulated_usage["prompt_tokens"] += pt
                    self.accumulated_usage["completion_tokens"] += ct
                    self.accumulated_usage["total_tokens"] += tt
                    # Session accumulator (never reset — survives page reloads)
                    self.session_usage["prompt_tokens"] += pt
                    self.session_usage["completion_tokens"] += ct
                    self.session_usage["total_tokens"] += tt
                self._api_call_count += 1

                return content.strip()

            except RateLimitError as e:
                wait = config.RETRY_DELAY * attempt
                logger.warning(
                    f"Rate limited (attempt {attempt}/{config.MAX_RETRIES}). "
                    f"Waiting {wait}s... Error: {e}"
                )
                time.sleep(wait)
                last_error = e

            except APITimeoutError as e:
                logger.warning(f"API timeout (attempt {attempt}/{config.MAX_RETRIES}): {e}")
                time.sleep(config.RETRY_DELAY)
                last_error = e

            except APIError as e:
                # Try fallback model on 4xx model errors
                if e.status_code in (400, 404, 422) and self.model_id != config.FALLBACK_MODEL_ID:
                    logger.warning(
                        f"Model error {e.status_code} with {self.model_id}. "
                        f"Switching to fallback: {config.FALLBACK_MODEL_ID}"
                    )
                    self.model_id = config.FALLBACK_MODEL_ID
                    last_error = e
                else:
                    logger.error(f"API error (attempt {attempt}): {e}")
                    time.sleep(config.RETRY_DELAY)
                    last_error = e

            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt}): {e}")
                time.sleep(config.RETRY_DELAY)
                last_error = e

        raise RuntimeError(
            f"API call failed after {config.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def reset_accumulated_usage(self) -> None:
        """Reset the per-investigation token counters (session_usage is preserved)."""
        self.accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_session_usage_summary(self, budget_tokens: int = 128_000) -> dict:
        """
        Return a serialisable dict of session-level token usage suitable for the
        /api/token-usage REST endpoint.  The budget is a soft cap used only for the
        percentage bar in the UI — it does NOT enforce any hard limit.
        """
        used = self.session_usage.get("total_tokens", 0)
        pct  = round((used / budget_tokens) * 100, 1) if budget_tokens > 0 else 0.0
        return {
            "prompt_tokens":     self.session_usage.get("prompt_tokens", 0),
            "completion_tokens": self.session_usage.get("completion_tokens", 0),
            "total_tokens":      used,
            "budget_tokens":     budget_tokens,
            "usage_pct":         pct,
            "api_calls":         self._api_call_count,
        }

    def _assert_loaded(self) -> None:
        if not self._loaded or self._client is None:
            raise RuntimeError(
                "ModelManager.load() must be called before inference."
            )

    @property
    def device(self) -> str:
        return "api"  # No local device — all inference is remote
