"""
Claude-powered Banglish transcript interpreter.

Takes a raw Whisper transcript and its low-confidence words, sends a
structured prompt to Claude (current Sonnet 4.6 by default; see MODEL), and
returns a cleaned/reinterpreted version.  Maintains conversation history so the
user can ask follow-up questions about the same transcript.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic

import config
from transcriber import LowConfidenceWord, DiarizedSegment
from banglish_hints import CLAUDE_REFERENCE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Model id is env-overridable; defaults to current Sonnet 4.6. The previous
# default (claude-sonnet-4-20250514) was retired by Anthropic on 2026-06-15.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 16384

SYSTEM_PROMPT = f"""\
You are an expert transcript editor for code-switched multilingual speech. \
Your primary use case is Banglish — Bengali (Bangla) written in Latin/English \
letters, frequently mixed with real English words and phrases in the same \
sentence. That pattern and its specific vocabulary appear below as the \
**primary** examples, and Banglish is what you will almost always be editing.

The same correction principle extends to other forms of cross-language ASR \
drift: speech-recognition decoders can mishear sounds from one language as \
words from a different language. Apply the correction principle to whatever \
cross-language drift you detect — including Hindi/Urdu-shaped vocabulary that \
has been substituted for Bengali audio (examples in the secondary block \
below). In all cases the target is the same: recover what the Banglish speaker \
actually said.

You will receive:
1. A raw transcript produced by OpenAI Whisper from spoken Banglish audio, \
   broken into timestamped speaker-labelled segments of the form \
   `[<start>s-<end>s] SPEAKER_XX: <text>`. (Older callers may pass flat \
   text instead — see below.)
2. A list of words Whisper was **uncertain** about (each with its confidence \
   score and timestamp).

Your job is to **return ONLY the corrections needed**, anchored to specific \
input segments. **Do NOT echo back the corrected transcript** — that gets \
reconstructed in code by applying your corrections to the raw segments. \
Returning the full transcript would balloon the response unnecessarily and \
truncates on long meetings.

For each correction:
- **Anchor to a specific source segment** by its `start` time in seconds \
  (the `<start>` in the `[<start>s-<end>s]` prefix). Also include the \
  same segment's `end` time.
- Copy the `original` substring **verbatim** from that segment's text — \
  whatever Whisper produced for those words. The matcher uses an exact \
  substring lookup with a case-insensitive fallback, so spelling matters.
- Provide the `replacement`. Use empty string `""` if the segment is \
  fully hallucinated or empty filler and should be deleted entirely.
- Provide up to a few `candidates` (alternative possible Banglish readings) \
  and a one-sentence `reason`.

Always reply with ONLY a JSON object (no markdown fences, no commentary \
outside the JSON). Use this exact schema:

{{
  "assessment": "<1-3 sentence overall quality assessment of the transcript>",
  "alternatives": [
    {{
      "segment_start": <float seconds — copied verbatim from the input segment's start>,
      "segment_end":   <float seconds — copied verbatim from the input segment's end>,
      "original":      "<exact substring from that segment's text to be replaced>",
      "replacement":   "<corrected text; use \\"\\" to delete the original substring>",
      "candidates":    ["<option1>", "<option2>", "..."],
      "reason":        "<why this replacement>"
    }}
  ],
  "changes_made": "<short, human-readable summary of all changes>"
}}

Notes:
- Do **not** include `clean_version` or `clean_segments` in your response — \
  those are reconstructed in code.
- If the transcript looks correct and no changes are needed, return an empty \
  `alternatives` list.
- If the input arrives as flat text (no `[<start>s-<end>s]` segment prefixes), \
  return your corrections with `segment_start` / `segment_end` set to `0.0`. \
  The caller knows that path is best-effort.
- Preserve speaker attribution implicitly by anchoring corrections to \
  segments; the speaker label is carried by the segment, not by your output.

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
    diarized_segments: list[DiarizedSegment] | None = None,
) -> str:
    """
    Format the first user turn that delivers the transcript to Claude.

    If `diarized_segments` is provided and non-empty, the transcript is sent
    as one timestamped speaker-labelled line per segment:

        [12.30s-15.80s] SPEAKER_00: hello kemon acho
        [15.80s-17.20s] SPEAKER_01: bhalo achi

    Otherwise the existing flat-text path is used (raw_text as one block).
    """
    lines = [
        "## Raw Whisper transcript",
        "",
    ]
    if diarized_segments:
        for seg in diarized_segments:
            txt = (seg.text or "").strip()
            if not txt:
                continue
            lines.append(
                f"[{seg.start:.2f}s-{seg.end:.2f}s] {seg.speaker}: {txt}"
            )
        lines.append("")
    else:
        lines.append(raw_text)
        lines.append("")

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

    Under the diff-only Stage 3 schema (v2.0+), Claude returns
    ``assessment``, ``alternatives``, ``changes_made`` only — no
    ``clean_version`` or ``clean_segments``; those are reconstructed in
    code by :func:`_reconstruct_clean`. The parser fills in defensive
    defaults so callers never have to KeyError-check.
    """
    cleaned = text.strip()

    # Strip optional ```json ... ``` fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: try to find the first { … } block.
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        # If all parsing fails, return a wrapper so callers always get a dict.
        return {
            "assessment": "Could not parse Claude's response as JSON.",
            "alternatives": [],
            "changes_made": "none (parse error)",
            "_raw_response": text,
        }

    # Normalise: ensure the v2.0 schema keys are always present.
    parsed.setdefault("assessment", "")
    parsed.setdefault("alternatives", [])
    parsed.setdefault("changes_made", "")
    return parsed


# ---------------------------------------------------------------------------
# Reconstruction — apply Claude's corrections to raw segments
# ---------------------------------------------------------------------------

# Tolerance for matching an `alternatives` entry's segment_start to a
# DiarizedSegment's start time. Diarization stamps drift slightly between
# runs so an exact match is too brittle, but ±0.5s is far smaller than any
# real segment length.
_SEGMENT_MATCH_TOLERANCE_S = 0.5


def _reconstruct_clean(
    diarized_segments: list[DiarizedSegment],
    alternatives: list[dict],
) -> tuple[list[dict], str, list[str]]:
    """
    Apply Claude's diff-only ``alternatives`` to the raw diarized
    segments and reconstruct the cleaned transcript in code.

    Parameters
    ----------
    diarized_segments
        The Stage 2 output — one ``DiarizedSegment`` per Whisper segment,
        each carrying ``speaker``, ``start``, ``end``, ``text``.
    alternatives
        Claude's corrections list. Each entry is expected to carry
        ``segment_start``, ``segment_end``, ``original``, ``replacement``
        (and optionally ``candidates``, ``reason``). Anchors that miss
        their target segment or whose ``original`` is not a substring of
        the matched segment are skipped with a warning.

    Returns
    -------
    clean_segments
        List of ``{speaker, start, end, text}`` dicts in the same order as
        ``diarized_segments`` but with corrections applied. Segments whose
        corrected text is empty/whitespace-only are dropped.
    clean_version
        ``"[<start>s-<end>s] SPEAKER_XX: <text>\\n"`` per surviving
        segment, joined.
    warnings
        Human-readable warning strings for diagnostics (segment-match
        misses, substring not found, malformed entry). Empty list when
        everything matched cleanly.
    """
    warnings: list[str] = []

    # Group corrections by matching segment index. Skip malformed entries
    # and anchors that fall outside ±_SEGMENT_MATCH_TOLERANCE_S of any
    # segment's start.
    by_segment_idx: dict[int, list[dict]] = {}
    for alt in alternatives or []:
        if not isinstance(alt, dict):
            warnings.append(f"[reconstruct] alternative is not a dict: {alt!r}")
            continue
        if "original" not in alt or "replacement" not in alt:
            warnings.append(
                f"[reconstruct] alternative missing required fields: {alt!r}"
            )
            continue
        try:
            anchor = float(alt.get("segment_start", 0.0))
        except (TypeError, ValueError):
            warnings.append(
                f"[reconstruct] segment_start not a number: {alt!r}"
            )
            continue

        # Linear scan is fine — segments per meeting are O(hundreds).
        best_idx = -1
        best_delta = float("inf")
        for i, seg in enumerate(diarized_segments):
            delta = abs(seg.start - anchor)
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        if best_idx == -1 or best_delta > _SEGMENT_MATCH_TOLERANCE_S:
            warnings.append(
                f"[reconstruct] no segment within "
                f"{_SEGMENT_MATCH_TOLERANCE_S}s of "
                f"segment_start={anchor:.2f}s "
                f"(closest delta={best_delta:.2f}s)"
            )
            continue
        by_segment_idx.setdefault(best_idx, []).append(alt)

    clean_segments: list[dict] = []
    out_lines: list[str] = []
    for i, seg in enumerate(diarized_segments):
        text = seg.text or ""
        edits = by_segment_idx.get(i, [])
        # Longest-original-first so a longer match doesn't get eaten by a
        # shorter overlapping one.
        edits.sort(key=lambda a: -len(str(a.get("original") or "")))
        for alt in edits:
            original = str(alt.get("original") or "")
            replacement = str(alt.get("replacement") or "")
            if not original:
                warnings.append(
                    f"[reconstruct] empty 'original' on segment "
                    f"start={seg.start:.2f}s"
                )
                continue
            if original in text:
                text = text.replace(original, replacement, 1)
            else:
                # Case-insensitive fallback so 'Gemini' vs 'gemini' etc. match.
                lowered = text.lower()
                idx = lowered.find(original.lower())
                if idx >= 0:
                    text = text[:idx] + replacement + text[idx + len(original):]
                else:
                    warnings.append(
                        f"[reconstruct] 'original'={original!r} not found in "
                        f"segment start={seg.start:.2f}s "
                        f"text={text!r}"
                    )

        text = text.strip()
        if not text:
            # Fully deleted / hallucinated segment — drop entirely.
            continue
        clean_segments.append(
            {
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "text": text,
            }
        )
        out_lines.append(f"[{seg.start:.2f}s-{seg.end:.2f}s] {seg.speaker}: {text}")

    clean_version = "\n".join(out_lines)
    return clean_segments, clean_version, warnings


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BanglishInterpreter:
    """
    Stateful interpreter that talks to Claude (see MODEL) about a Banglish
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
        diarized_segments: list[DiarizedSegment] | None = None,
    ) -> dict[str, Any]:
        """
        Send the transcript + uncertain words to Claude and return a
        structured interpretation.

        v2.0 schema: Claude now returns corrections-only via
        ``alternatives`` anchored to specific input segments. This method
        reconstructs ``clean_segments`` and ``clean_version`` in code by
        applying those corrections to ``diarized_segments`` — the output
        dict shape seen by callers is unchanged.

        Parameters
        ----------
        raw_text : str
            The full raw transcript string from Whisper.
        low_confidence_words : list[LowConfidenceWord]
            Words whose probability fell below the confidence threshold.
        diarized_segments : list[DiarizedSegment], optional
            Per-segment diarization output. When provided (and non-empty)
            the transcript is delivered to Claude as timestamped
            speaker-labelled lines AND used as the source of truth for
            reconstructing the corrected output. When omitted, the
            flat-text path is used (backward compat for
            ``run_interpret_only.py``); reconstruction is skipped and
            ``clean_version``/``clean_segments`` come back empty.

        Returns
        -------
        dict
            Keys: ``assessment``, ``clean_version``, ``clean_segments``,
            ``alternatives``, ``changes_made``. Also surfaces
            ``reconstruction_warnings`` (list of strings) when the
            reconstruction path runs.
        """
        user_msg = _build_user_message(
            raw_text, low_confidence_words, diarized_segments
        )

        # Reset history for a fresh transcript.
        self.conversation_history = [
            {"role": "user", "content": user_msg},
        ]

        assistant_text = self._send()
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text},
        )

        parsed = _parse_response(assistant_text)

        # Reconstruct clean_segments + clean_version in code from Claude's
        # diff-only output. Skip when no segments are available (the
        # run_interpret_only.py path); that fallback is a known-limited
        # mode and surfaces empty clean_* fields by design.
        if diarized_segments:
            clean_segments, clean_version, warnings = _reconstruct_clean(
                diarized_segments,
                parsed.get("alternatives") or [],
            )
            parsed["clean_segments"] = clean_segments
            parsed["clean_version"] = clean_version
            parsed["reconstruction_warnings"] = warnings
            if warnings:
                # Surface a short diagnostic line per warning. Keep stdout
                # noise low if the list is huge.
                for w in warnings[:20]:
                    print(w)
                if len(warnings) > 20:
                    print(f"... and {len(warnings) - 20} more reconstruction warnings.")
        else:
            parsed.setdefault("clean_segments", [])
            parsed.setdefault("clean_version", "")
            parsed.setdefault("reconstruction_warnings", [])

        self._last_result = parsed
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
# Stage 3.5 — polish (smoothing) pass
# ---------------------------------------------------------------------------
# A second Claude call AFTER the B3 corrections-only stage. It smooths
# grammar, punctuation, sentence flow and removes hallucinated repetition,
# WITHOUT inventing content. Non-destructive: any validation failure returns
# the input segments unchanged so we never ship worse than B3.

POLISH_MODEL = MODEL
# Polish must echo the WHOLE transcript back, so it needs far more output
# headroom than the diff-only B3 call. Bengali-heavy clips run ~300+ segments.
POLISH_MAX_TOKENS = 32000

POLISH_SYSTEM_PROMPT = """\
You are an expert transcript editor for code-switched multilingual speech \
(primarily Banglish — Bengali-Latin code-switched with English). You are \
given a list of speaker-labeled, timestamped segments that have already \
been cleaned for word-level corrections. Produce a polished, \
human-readable version that smooths grammar, punctuation, sentence flow, \
and removes repetitive filler — WITHOUT inventing content or changing \
meaning.

Rules:
- Preserve every segment's start, end, and speaker exactly.
- Preserve the actual content. Do NOT add information.
- Preserve code-switched Banglish naturally — do not translate Bengali \
words to English or vice versa.
- DO smooth: punctuation, capitalization, sentence boundaries within a \
segment, repetitive "let's say, let's say, let's say" -> "let's say".
- DO remove transcription stutters and obvious repetition loops that \
Stage 1 hallucinated.

Output ONLY a JSON array of segments with the same schema as input:
[{"start": float, "end": float, "speaker": str, "text": str}]
No prose around the JSON; no markdown fences. The array MUST have exactly \
the same number of elements as the input, in the same order.
"""


def _parse_polish_array(text: str) -> list[dict] | None:
    """Extract a JSON array of segments from Claude's polish reply."""
    cleaned = (text or "").strip()
    # Strip optional ```json ... ``` fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: grab the first [ ... ] block.
        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not arr_match:
            return None
        try:
            parsed = json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    return parsed


def polish_segments(
    segments: list[dict],
    anthropic_client: "anthropic.Anthropic",
    model: str = POLISH_MODEL,
) -> list[dict]:
    """
    Take B3-cleaned segments and return a polished, human-readable version
    with the same schema and smoothed text.

    Non-destructive contract: on ANY problem (API error, parse failure,
    segment-count mismatch) this returns the INPUT segments unchanged, so a
    polish failure can never ship something worse than the B3 output.

    start/end/speaker are always taken from the INPUT (never trusted from
    Claude), so they are preserved exactly by construction; only ``text`` is
    adopted from the polished response.
    """
    if not segments:
        return segments

    payload = [
        {
            "start": s.get("start"),
            "end": s.get("end"),
            "speaker": s.get("speaker"),
            "text": s.get("text", ""),
        }
        for s in segments
    ]
    user_msg = (
        "Polish the following already-corrected segments. Return ONLY the "
        "JSON array, same length and order.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        # Stream the call: a non-streaming request with this large a
        # max_tokens is rejected by the SDK ("Streaming is required for
        # operations that may take longer than 10 minutes"). Streaming
        # satisfies that requirement; we still collect the full final message.
        with anthropic_client.messages.stream(
            model=model,
            max_tokens=POLISH_MAX_TOKENS,
            system=POLISH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            final = stream.get_final_message()
        out_text = final.content[0].text
    except Exception as e:
        print(f"[polish] API call failed ({e}); returning B3 segments unchanged.")
        return segments

    polished = _parse_polish_array(out_text)
    if polished is None:
        print("[polish] could not parse JSON array; returning B3 segments unchanged.")
        return segments
    if len(polished) != len(segments):
        print(
            f"[polish] segment-count mismatch (got {len(polished)}, "
            f"expected {len(segments)}); returning B3 segments unchanged."
        )
        return segments

    out: list[dict] = []
    for inp, pol in zip(segments, polished):
        text = pol.get("text") if isinstance(pol, dict) else None
        if not isinstance(text, str) or not text.strip():
            text = inp.get("text", "")
        out.append(
            {
                "start": inp.get("start"),
                "end": inp.get("end"),
                "speaker": inp.get("speaker"),
                "text": text.strip(),
            }
        )
    print(f"[polish] polished {len(out)} segments.")
    return out


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
