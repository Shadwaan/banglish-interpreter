"""
Load pipeline artefacts (raw Whisper + Claude clean_version) and provide a
normalisation helper for WER scoring.

The shapes are documented in docs/CODEBASE.md §6 and §9:

  - *_raw.txt       one line per segment:  "[SPEAKER_00]: text"
  - *_output.json   { audio_file, raw_transcript, diarized_transcript,
                      speakers[], low_confidence_count,
                      interpretation: { assessment, clean_version,
                                         alternatives[], changes_made } }

When the JSON is broken (truncated / malformed), fall back to *_output.txt and
extract the block after the `CLEAN VERSION` heading — same trick as
extract_clean.py.
"""

from __future__ import annotations

import json
import os
import re


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(raw_path: str) -> str:
    """
    Read a *_raw.txt file and return the transcript text only — strip the
    `[SPEAKER_xx]: ` prefix from each line and join the segments with spaces.
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop "[SPEAKER_00]: " or "[Speaker 1]: " style prefixes
        stripped = re.sub(r"^\[[^\]]+\]:\s*", "", line)
        parts.append(stripped)
    return " ".join(parts).strip()


def load_clean(output_json_path: str) -> tuple[str, dict]:
    """
    Read *_output.json and return (clean_version_text, meta).

    meta carries useful context for the scoreboard:
      { audio_file, speakers, low_confidence_count, interpretation_assessment }

    If the JSON is unreadable, fall back to the sibling *_output.txt file and
    extract the clean version after the `CLEAN VERSION` heading.
    """
    meta: dict = {}
    try:
        with open(output_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        interp = data.get("interpretation", {}) or {}
        clean = interp.get("clean_version") or data.get("raw_transcript") or ""
        meta = {
            "audio_file": data.get("audio_file"),
            "speakers": data.get("speakers", []),
            "low_confidence_count": data.get("low_confidence_count"),
            "interpretation_assessment": interp.get("assessment"),
            "source": "json",
        }
        if clean:
            return clean.strip(), meta
        # If the JSON had no clean_version, fall through to the txt fallback
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        pass

    # Fallback: parse *_output.txt via the same approach as extract_clean.py
    txt_path = output_json_path.replace("_output.json", "_output.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"Could not load clean version: neither {output_json_path} nor "
            f"{txt_path} are usable."
        )
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    clean = _extract_clean_from_txt(content)
    meta["source"] = "txt-fallback"
    return clean.strip(), meta


def normalise_for_wer(text: str) -> str:
    """
    Standard preprocessing so reference and hypothesis are scored apples-to-
    apples. Same rules applied on BOTH sides.

      - strip `[<start>s-<end>s] SPEAKER_XX: ` v1.5 segment prefix
      - lowercase
      - strip `[SPEAKER_xx]: ` markers (just in case)
      - strip `**Name:**` markers (just in case)
      - strip the `>` blockquote prefix
      - drop most punctuation
      - collapse whitespace
    """
    s = text or ""
    # Strip the v1.5 segment prefix BEFORE lowercase / other strips, so the
    # timestamp + speaker label do not leak into the WER token stream.
    # Example match: "[0.49s-4.05s] SPEAKER_01: " → " "
    s = re.sub(
        r"\[\s*\d+\.?\d*s?\s*-\s*\d+\.?\d*s?\s*\]\s*SPEAKER_\d+:\s*",
        " ",
        s,
    )
    s = s.lower()
    # Strip speaker markers if any survived earlier passes
    s = re.sub(r"\[[^\]]+\]:\s*", " ", s)
    s = re.sub(r"\*\*[^*]+:\*\*\s*", " ", s)
    s = re.sub(r"^>\s*", " ", s, flags=re.MULTILINE)
    # Replace dashes and slashes with spaces (don't merge separate words)
    s = re.sub(r"[\-–—/]", " ", s)
    # Drop other punctuation
    s = re.sub(r"[.,!?;:'\"`()\[\]{}<>…]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_CLEAN_VERSION_BLOCK = re.compile(
    r"CLEAN VERSION\s*\n[-]+\s*\n(.*?)$", re.DOTALL
)


def _extract_clean_from_txt(content: str) -> str:
    """Mirror of extract_clean.py's logic, kept self-contained here."""
    parts = content.split("CLEAN VERSION", 1)
    if len(parts) <= 1:
        return content.strip()
    inner = parts[1]
    # Drop the underline of dashes immediately after "CLEAN VERSION"
    if "-" * 10 in inner.splitlines()[0:3]:  # rough check
        inner = inner.split("\n", 2)
        inner = inner[2] if len(inner) >= 3 else "".join(inner)
    else:
        # Best-effort: split on the long dash underline if present
        bits = re.split(r"\n[-]{10,}\n", inner, maxsplit=1)
        inner = bits[-1]

    # If the content was a JSON-wrapped clean_version (legacy bug), pull
    # the inner string out.
    if inner.strip().startswith("{"):
        cv_match = re.search(
            r'"clean_version"\s*:\s*"([\s\S]*?)"\s*,\s*"alternatives"',
            inner,
        )
        if not cv_match:
            cv_match = re.search(
                r'"clean_version"\s*:\s*"([\s\S]*)$',
                inner,
            )
        if cv_match:
            inner = cv_match.group(1)
            # Decode the most common JSON escapes
            inner = (
                inner.replace("\\n", "\n")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
    return inner.strip()


# ---------------------------------------------------------------------------
# CLI smoke test:  python eval/pipeline_io.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(eval_dir)

    with open(os.path.join(eval_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipe_dir = os.path.join(repo_dir, manifest["pipeline_dir"])

    for item in manifest["items"]:
        raw_path = os.path.join(pipe_dir, item["raw"])
        out_path = os.path.join(pipe_dir, item["output_json"])
        raw_text = load_raw(raw_path)
        clean_text, meta = load_clean(out_path)

        bar = "=" * 78
        print(f"\n{bar}")
        print(f"  ITEM       : {item['id']}")
        print(f"  RAW file   : {item['raw']}")
        print(f"  RAW chars  : {len(raw_text)}  ({len(raw_text.split())} words)")
        print(f"  CLEAN file : {item['output_json']}  [{meta.get('source')}]")
        print(f"  CLEAN chars: {len(clean_text)}  ({len(clean_text.split())} words)")
        print(f"  SPEAKERS   : {meta.get('speakers')}")
        print(f"  LOW-CONF # : {meta.get('low_confidence_count')}")
        print(f"{bar}")
        print(f"  RAW preview   : {raw_text[:180]}...")
        print(f"  CLEAN preview : {clean_text[:180]}...")
