"""
Claude-powered Banglish transcript interpreter.

Takes a raw Whisper transcript and its low-confidence words, sends a
structured prompt to Claude claude-opus-4-5, and returns a cleaned/reinterpreted
version.  Maintains conversation history so the user can ask follow-up
questions about the same transcript.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

import config
from transcriber import LowConfidenceWord
from banglish_hints import CLAUDE_REFERENCE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 16384

SYSTEM_PROMPT = f"""\
You are an expert Banglish interpreter. "Banglish" is Bengali (Bangla) \
written in Latin/English letters, frequently mixed with real English words \
and phrases in the same sentence.

You will receive:
1. A raw transcript produced by OpenAI Whisper from spoken Banglish audio.
2. A list of words Whisper was **uncertain** about (each with its confidence \
   score and timestamp).

Your job:
- **Assess** the transcript: identify likely Whisper errors, especially where \
  Bengali words were misheard as English words or vice-versa.
- **Reinterpret** uncertain words using Banglish context (e.g. "call" might \
  really be "kol", "key" might be "ki", "bah" might be "bhai").
- For every word you change, suggest **alternatives** the speaker might have \
  said and explain why you chose one.
- Produce a **clean version** of the full transcript with your corrections \
  applied.

Always reply with ONLY a JSON object (no markdown fences, no commentary \
outside the JSON). Use this exact schema:

{{
  "assessment": "<1-3 sentence overall quality assessment of the transcript>",
  "clean_version": "<full corrected transcript>",
  "alternatives": [
    {{
      "original": "<word Whisper produced>",
      "replacement": "<your chosen correction>",
      "candidates": ["<option1>", "<option2>", "..."],
      "reason": "<why you chose this replacement>"
    }}
  ],
  "changes_made": "<short, human-readable summary of all changes>"
}}

If the transcript looks correct and no changes are needed, return the same \
text in "clean_version" and an empty "alternatives" list.

---

Below is a reference of known Whisper mishearings, common Banglish \
code-switching patterns, and high-frequency vocabulary. Use this to inform \
your corrections — if Whisper produced an English word that appears in the \
mishearing map, strongly consider the Banglish alternative, especially when \
the surrounding context is Bengali.

{CLAUDE_REFERENCE}\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    raw_text: str,
    low_confidence_words: list[LowConfidenceWord],
) -> str:
    """Format the first user turn that delivers the transcript to Claude."""
    lines = [
        "## Raw Whisper transcript",
        "",
        raw_text,
        "",
    ]

    if low_confidence_words:
        # Cap to avoid blowing up the context window when Whisper hallucinates.
        # Keep the LEAST confident words first (most likely to be wrong).
        MAX_LOW_CONF = 200
        sorted_words = sorted(low_confidence_words, key=lambda w: w.confidence)
        shown = sorted_words[:MAX_LOW_CONF]
        truncated = len(low_confidence_words) - len(shown)
        header = "## Low-confidence words (probability < 0.65)"
        if truncated > 0:
            header += f" — showing {len(shown)} lowest of {len(low_confidence_words)} total"
        lines.append(header)
        lines.append("")
        for w in shown:
            lines.append(
                f"- \"{w.word}\"  —  confidence {w.confidence:.2f}  "
                f"[{w.start:.2f}s → {w.end:.2f}s]"
            )
        lines.append("")
    else:
        lines.append("Whisper reported **no** low-confidence words.\n")

    lines.append(
        "Please assess this Banglish transcript, fix errors, and "
        "return the JSON response described in your instructions."
    )

    return "\n".join(lines)


def _parse_response(text: str) -> dict[str, Any]:
    """
    Extract the JSON object from Claude's reply.

    Claude should return raw JSON, but we handle the case where it wraps
    the response in markdown code fences just in case.
    """
    cleaned = text.strip()

    # Strip optional ```json ... ``` fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: try to find the first { … } block.
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        # If all parsing fails, return a wrapper so callers always get a dict.
        return {
            "assessment": "Could not parse Claude's response as JSON.",
            "clean_version": text,
            "alternatives": [],
            "changes_made": "none (parse error)",
            "_raw_response": text,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BanglishInterpreter:
    """
    Stateful interpreter that talks to Claude claude-opus-4-5 about a Banglish
    transcript and keeps conversation history for follow-ups.
    """

    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(
            api_key=api_key or config.ANTHROPIC_API_KEY,
        )
        self.conversation_history: list[dict[str, str]] = []
        self._last_result: dict[str, Any] | None = None

    # ----- primary entry point -----

    def interpret(
        self,
        raw_text: str,
        low_confidence_words: list[LowConfidenceWord],
    ) -> dict[str, Any]:
        """
        Send the transcript + uncertain words to Claude and return a
        structured interpretation.

        Parameters
        ----------
        raw_text : str
            The full raw transcript string from Whisper.
        low_confidence_words : list[LowConfidenceWord]
            Words whose probability fell below the confidence threshold.

        Returns
        -------
        dict
            Keys: ``assessment``, ``clean_version``, ``alternatives``,
            ``changes_made``.
        """
        user_msg = _build_user_message(raw_text, low_confidence_words)

        # Reset history for a fresh transcript.
        self.conversation_history = [
            {"role": "user", "content": user_msg},
        ]

        assistant_text = self._send()
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text},
        )

        self._last_result = _parse_response(assistant_text)
        return self._last_result

    # ----- follow-up questions -----

    def follow_up(self, question: str) -> dict[str, Any]:
        """
        Ask a follow-up question about the current transcript.

        The full conversation history is included so Claude has context.

        Parameters
        ----------
        question : str
            Free-form follow-up (e.g. "What did the speaker mean by X?").

        Returns
        -------
        dict
            Same schema as ``interpret`` if Claude returns JSON, otherwise
            a dict with ``"response"`` containing the raw text.
        """
        if not self.conversation_history:
            raise RuntimeError(
                "No active transcript. Call interpret() first."
            )

        self.conversation_history.append(
            {"role": "user", "content": question},
        )

        assistant_text = self._send()
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text},
        )

        # Follow-ups may or may not be JSON — try to parse, fall back to raw.
        try:
            return _parse_response(assistant_text)
        except Exception:
            return {"response": assistant_text}

    # ----- internals -----

    def _send(self) -> str:
        """Call the Anthropic API with the current conversation history."""
        message = self._client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history,
        )
        return message.content[0].text

    # ----- convenience -----

    @property
    def last_result(self) -> dict[str, Any] | None:
        """The most recent parsed interpretation, or *None*."""
        return self._last_result

    def reset(self) -> None:
        """Clear conversation history (ready for a new transcript)."""
        self.conversation_history.clear()
        self._last_result = None


# ---------------------------------------------------------------------------
# Quick CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a transcript with a few shaky words.
    demo_text = "ami today office jabo but traffic onek bad ache"
    demo_uncertain = [
        LowConfidenceWord(word="jabo",  confidence=0.42, start=1.20, end=1.55),
        LowConfidenceWord(word="onek",  confidence=0.38, start=3.10, end=3.40),
        LowConfidenceWord(word="ache",  confidence=0.61, start=4.50, end=4.80),
    ]

    print("Sending demo transcript to Claude …\n")
    interp = BanglishInterpreter()
    result = interp.interpret(demo_text, demo_uncertain)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example follow-up
    print("\n--- follow-up ---\n")
    fu = interp.follow_up("Can you translate the clean version to English?")
    print(json.dumps(fu, indent=2, ensure_ascii=False))
