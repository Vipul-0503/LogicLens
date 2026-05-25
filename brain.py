"""
LogicLens Brain Module — Production Version
===================================================
Intelligence layer for LogicLens. Accepts timestamped transcripts produced
by engine.py and uses the Gemini 2.5 Flash model to extract, anchor, and
deeply explain complex technical methods with precision timestamps.

Dependencies:
    pip install google-genai python-dotenv
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Bootstrap environment & logging
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LogicLens.Brain")

# Execution and Error Management Constants
_MODEL_ID = "gemini-2.5-flash"
_TEMPERATURE = 0.2
_MAX_OUTPUT_TOKENS = 8192

_MAX_RETRIES = 3
_INITIAL_RETRY_DELAY = 2.0  # seconds; doubles on each consecutive failure

# HTTP status codes that represent backend transient bottlenecks
_TRANSIENT_ERROR_CODES = frozenset({429, 503})

# HTTP status codes indicating permanent configuration breakdown
_FATAL_ERROR_CODES = frozenset({401, 403})

# System Instruction Prompt Template
_SYSTEM_INSTRUCTION_TEMPLATE = """
You are a Distinguished University Professor and Technical Researcher
specializing in {focus_topic}. Your sole objective is to produce a
structured, academically rigorous technical breakdown of the lecture
transcript provided below.

════════════════════════════════════════════════════════════
MISSION
════════════════════════════════════════════════════════════
Scan the entire timestamped transcript and identify EVERY complex
engineering method, algorithmic mechanic, architectural pattern, or
mathematical concept that is discussed or explained.

Ignore all motivational commentary, anecdotes, biographical tangents,
audience interaction, and any content that is not directly technical or
conceptual in nature.

════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT
════════════════════════════════════════════════════════════
For EACH technical method or concept identified, produce a block using
EXACTLY this structure:

──────────────────────────────────────────────────────────
METHOD: <Precise Technical Name>
TIMESTAMP: [HH:MM:SS]  ← exact timestamp where this explanation BEGINS
──────────────────────────────────────────────────────────
HOW IT WORKS:
  Provide a deep, step-by-step mechanistic explanation. Do NOT
  summarize superficially. Explain the internal mechanics, data flow,
  mathematical operations, or algorithmic steps that constitute this
  method. Use numbered steps where a sequence exists.

WHY IT IS STRUCTURALLY IMPORTANT:
  Explain the structural or systemic role this method plays within
  the broader domain of {focus_topic}. Why does this technique exist?
  What failure modes or limitations does it address? What would break
  without it?

PREREQUISITE CONCEPTS:
  List (as a bullet list) any foundational concepts a learner must
  already understand before this method makes sense.

LECTURER'S KEY INSIGHT (if present):
  Quote or closely paraphrase the specific insight or framing the
  lecturer offers that goes BEYOND textbook definitions. If no unique
  insight is present, write "N/A".
──────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════════
1. Every METHOD block MUST have a TIMESTAMP anchored to the transcript.
   Do not fabricate or estimate timestamps — use only what is present.
2. Order the METHOD blocks chronologically by TIMESTAMP.
3. If fewer than 2 clear technical methods are present, state exactly:
   "ANALYSIS RESULT: No sufficient technical content detected in the
   provided transcript segment for the focus topic: {focus_topic}."
4. Do NOT include any preamble, introduction, or closing remarks in
   your response. Output ONLY the METHOD blocks (or the statement in
   Rule 3).
5. Technical precision is paramount. Prefer exact terminology over
   plain-language paraphrasing when describing mechanisms.
""".strip()


# Helper utilities
def _extract_http_status(exc: Exception) -> Optional[int]:
    """
    Best-effort extraction of an HTTP status code from a google-genai exception.
    Ensures structural tracking without internal failures across SDK versions.
    """
    for attr in ("code", "status_code", "http_status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code

    error_dict = getattr(exc, "error", None)
    if isinstance(error_dict, dict):
        code = error_dict.get("code")
        if isinstance(code, int):
            return code

    msg = str(exc)
    for token in msg.split():
        token_stripped = token.rstrip(":")
        if token_stripped.isdigit():
            candidate = int(token_stripped)
            if 100 <= candidate < 600:
                return candidate

    return None


# LogicBrain
class LogicBrain:
    """
    Intelligence layer for LogicLens.

    Wraps the Gemini 2.5 Flash model with strict stable production gateways,
    deterministic configuration constants, and fault-tolerant retry mechanics
    to perform technical analysis of timestamped transcripts.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._validate_environment(api_key)
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        
        logger.info("Initialising LogicBrain → model=%s endpoint=v1", _MODEL_ID)
        try:
            # Force SDK to utilize stable production v1 gateway routing
            self._client = genai.Client(
                api_key=resolved_key,
                http_options=types.HttpOptions(api_version="v1")
            )
            logger.info("LogicBrain core setup successfully configured.")
        except Exception as exc:
            logger.critical("Failed to initialise Gemini client architecture: %s", exc)
            raise RuntimeError(f"Could not create Gemini client wrapper context: {exc}") from exc

    # Public API
    def analyze_lecture(
        self,
        transcript: list[dict],
        focus_topic: str = "AI/ML",
    ) -> str:
        """
        Analyse a timestamped transcript and return a structured breakdown
        of every complex technical method found within it.
        """
        if not transcript:
            raise ValueError("transcript must be a non-empty list of dicts.")

        logger.info(
            "Starting analysis: %d segments | topic: '%s'.",
            len(transcript),
            focus_topic,
        )

        timeline_text = self._build_timeline(transcript)
        logger.debug("Timeline built (%d characters).", len(timeline_text))

        system_prompt = _SYSTEM_INSTRUCTION_TEMPLATE.format(
            focus_topic=focus_topic
        )
        user_message = (
            f"TRANSCRIPT TIMELINE (focus topic: {focus_topic}):\n\n"
            f"{timeline_text}"
        )

        response_text = self._call_gemini(system_prompt, user_message)
        logger.info("Analysis complete. Output length: %d chars.", len(response_text))
        return response_text

    # Internal Network Call & Formatting Engine
    @staticmethod
    def _build_timeline(transcript: list[dict]) -> str:
        """Convert transcript array into a serialized plain text timeline stream."""
        lines: list[str] = []
        for i, segment in enumerate(transcript):
            try:
                start = segment["start"]
                text = segment["text"].strip()
                if text:
                    lines.append(f"[{start}] {text}")
            except (KeyError, AttributeError) as exc:
                logger.warning(
                    "Skipping malformed segment at index %d: %s", i, exc
                )
        if not lines:
            raise ValueError(
                "No valid segments found after parsing the transcript list."
            )
        return "\n".join(lines)

    def _call_gemini(
        self,
        system_instruction: str,
        user_message: str,
    ) -> str:
        """
        Send a single-turn payload request to Gemini with fault-tolerant retry logic.
        Bypasses SDK configuration naming bugs by injecting structural instructions 
        directly into the contents payload stream.
        """
        retry_delay = _INITIAL_RETRY_DELAY

        # Merge system instructions dynamically into the text content block
        # to ensure zero dependencies on buggy config camelCase serializers.
        unified_content = (
            f"=== SYSTEM INSTRUCTIONS ===\n"
            f"{system_instruction}\n"
            f"===========================\n\n"
            f"{user_message}"
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            logger.info(
                "Calling Gemini (attempt %d/%d) model=%s",
                attempt,
                _MAX_RETRIES,
                _MODEL_ID,
            )
            try:
                response = self._client.models.generate_content(
                    model=_MODEL_ID,
                    contents=unified_content,
                    config=types.GenerateContentConfig(
                        temperature=_TEMPERATURE,
                        max_output_tokens=_MAX_OUTPUT_TOKENS,
                    ),
                )

                if hasattr(response, "text") and response.text:
                    logger.info("Gemini responded successfully.")
                    return response.text.strip()

                candidates = getattr(response, "candidates", [])
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    if content:
                        parts = getattr(content, "parts", [])
                        if parts:
                            text = getattr(parts[0], "text", "")
                            if text:
                                logger.info("Gemini responded (via manual traversal).")
                                return text.strip()

                logger.warning("Attempt %d: Gemini returned an empty response.", attempt)

            except Exception as exc:
                status_code = _extract_http_status(exc)

                # Fatal API Errors (401/403) — Abort instantly
                if status_code in _FATAL_ERROR_CODES:
                    logger.error("Fatal API error (HTTP %s): %s — aborting.", status_code, exc)
                    raise RuntimeError(f"Unrecoverable Gemini API error (HTTP {status_code}): {exc}") from exc

                # Non-transient errors (like a 400 Syntax anomaly) — Do not waste retry loops
                if status_code not in _TRANSIENT_ERROR_CODES and status_code is not None:
                    logger.error("Non-retryable API error (HTTP %s): %s", status_code, exc)
                    raise RuntimeError(f"Non-retryable Gemini API error (HTTP {status_code}): {exc}") from exc

                # Transient infrastructure limits (429/503) — Validate loop bounds before throttling
                if attempt == _MAX_RETRIES:
                    logger.critical("All %d retry thresholds exhausted. Execution failing.", _MAX_RETRIES)
                    raise RuntimeError(f"Gemini call failed after {_MAX_RETRIES} attempts due to server/quota load: {exc}") from exc
                
                logger.warning("Attempt %d/%d failed (HTTP %s): %s. Backing off...", attempt, _MAX_RETRIES, status_code or "unknown", exc)

            # Throttled backoff progression logic
            if attempt < _MAX_RETRIES:
                logger.info("Throttling runtime trace. Sleeping %.1fs...", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2


    @staticmethod
    def _validate_environment(explicit_key: Optional[str]) -> None:
        """Warn early if key is absent from the target context environment."""
        if explicit_key:
            return
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            logger.warning(
                "GEMINI_API_KEY is not set in the environment. "
                "The Gemini client will likely fail on the first API call. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        else:
            logger.info("GEMINI_API_KEY found in environment.")


# Smoke-test entry point
if __name__ == "__main__":
    dummy_transcript: list[dict] = [
        {"start": "00:00:05", "text": "Welcome to today's lecture on transformer architectures."},
        {"start": "00:00:12", "text": "We'll start by revisiting the self-attention mechanism."},
        {"start": "00:00:20", "text": "Self-attention computes a weighted sum of value vectors."},
        {"start": "00:00:28", "text": "The weight for each value is the dot product of its key with the query vector, scaled by the square root of the key dimension."},
        {"start": "00:00:45", "text": "This scaling prevents the dot products from growing too large in high dimensions, which would push the softmax into regions of near-zero gradient."},
        {"start": "00:01:02", "text": "After attention we apply layer normalization followed by a position-wise feed-forward network."},
        {"start": "00:01:15", "text": "The feed-forward block uses two linear transformations with a ReLU activation in between."},
        {"start": "00:01:30", "text": "Positional encodings are added to token embeddings to inject sequence order, since the attention operation itself is permutation invariant."},
        {"start": "00:01:50", "text": "We use sinusoidal positional encodings: sine for even dimensions and cosine for odd dimensions at each position."},
        {"start": "00:02:10", "text": "This allows the model to generalise to sequence lengths longer than those seen during training."},
    ]

    brain = LogicBrain()

    try:
        result = brain.analyze_lecture(
            transcript=dummy_transcript,
            focus_topic="AI/ML and Transformer Architecture",
        )
        print("\n" + "=" * 70)
        print("LOGICBRAIN ANALYSIS OUTPUT")
        print("=" * 70)
        print(result)
        print("=" * 70 + "\n")
    except (ValueError, RuntimeError) as e:
        logger.error("Smoke-test failed: %s", e)