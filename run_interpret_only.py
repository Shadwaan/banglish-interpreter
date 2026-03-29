#!/usr/bin/env python3
"""
Run ONLY the Claude interpretation stage on a pre-existing transcript file.
Skips Whisper entirely.

Usage:
    python run_interpret_only.py "C:\path\to\transcript.txt"
"""

import io
import json
import sys
import time
import re

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from interpreter import BanglishInterpreter
from transcriber import LowConfidenceWord

# -- ANSI colors --
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BAR    = "=" * 70


def header(title):
    print(f"\n{CYAN}{BAR}{RESET}")
    print(f"{CYAN}  {title}{RESET}")
    print(f"{CYAN}{BAR}{RESET}\n")


def subheader(title):
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'-' * 60}\n")


def extract_transcript(file_path):
    """Read a transcript file. Handles both plain text and structured formats."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try structured format first (TRANSCRIPTION: header with Speaker N: lines)
    if "TRANSCRIPTION:" in content:
        lines = []
        in_transcript = False
        for line in content.splitlines():
            if line.strip() == "TRANSCRIPTION:":
                in_transcript = True
                continue
            if line.strip().startswith("---") and in_transcript:
                break
            if in_transcript and line.strip():
                cleaned = re.sub(r"^Speaker \d+:\s*", "", line.strip())
                if cleaned:
                    lines.append(cleaned)
        return " ".join(lines)

    # Plain text — just return as-is
    return content.strip()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <transcript_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\n{BOLD}  BANGLISH INTERPRETER -- CLAUDE ONLY{RESET}")
    print(f"  {'-' * 40}\n")

    # -- Read transcript --
    header("READING SAVED TRANSCRIPT")
    raw_text = extract_transcript(file_path)
    print(f"  Source: {file_path}")
    print(f"  Length: {len(raw_text)} chars, ~{len(raw_text.split())} words\n")
    print(f"  {DIM}First 300 chars:{RESET}")
    print(f"  {raw_text[:300]}...\n")

    # -- Identify likely Banglish words that Whisper may have mangled --
    # Since we don't have word-level confidence from the saved transcript,
    # we flag words that match known Whisper mishearings from our hints.
    from banglish_hints import WHISPER_MISHEARINGS
    flagged = []
    words_seen = set()
    for word in raw_text.split():
        clean_word = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean_word in WHISPER_MISHEARINGS and clean_word not in words_seen:
            words_seen.add(clean_word)
            flagged.append(
                LowConfidenceWord(
                    word=clean_word,
                    confidence=0.40,  # synthetic low confidence
                    start=0.0,
                    end=0.0,
                )
            )

    header("FLAGGED WORDS (from mishearing map)")
    if flagged:
        for w in flagged:
            alternatives = WHISPER_MISHEARINGS.get(w.word, [])
            print(f"  {YELLOW}\"{w.word}\"{RESET} -> possibly: {', '.join(alternatives)}")
        print(f"\n  {DIM}({len(flagged)} words flagged for Claude to review){RESET}\n")
    else:
        print(f"  {GREEN}No known mishearing patterns found.{RESET}\n")

    # -- Send to Claude --
    header("STAGE 2 -- CLAUDE INTERPRETATION")
    print("  Sending to Claude claude-opus-4-5 ...\n")

    interp = BanglishInterpreter()
    t0 = time.time()
    data = interp.interpret(raw_text=raw_text, low_confidence_words=flagged)
    elapsed = time.time() - t0

    # -- Assessment --
    subheader("Assessment")
    print(f"  {data.get('assessment', '-')}\n")

    # -- Alternatives --
    alternatives = data.get("alternatives", [])
    subheader(f"Word-level corrections ({len(alternatives)})")
    if alternatives:
        for i, alt in enumerate(alternatives, 1):
            orig = alt.get("original", "?")
            repl = alt.get("replacement", "?")
            print(f"  {YELLOW}{i}. \"{orig}\" -> \"{repl}\"{RESET}")
            if alt.get("candidates"):
                print(f"     Candidates : {', '.join(alt['candidates'])}")
            if alt.get("reason"):
                print(f"     Reason     : {alt['reason']}")
            print()
    else:
        print(f"  {GREEN}[OK] No corrections needed.{RESET}\n")

    # -- Changes summary --
    subheader("Changes summary")
    print(f"  {data.get('changes_made', '-')}\n")

    # -- Clean version --
    subheader("Clean version")
    clean = data.get("clean_version", "-")
    # Print in readable chunks
    for line in clean.split(". "):
        print(f"  {GREEN}{line.strip()}.{RESET}")
    print()

    print(f"  {DIM}(interpreted in {elapsed:.1f}s){RESET}\n")

    # -- Before/After --
    header("STAGE 3 -- BEFORE / AFTER COMPARISON")
    print(f"  {RED}BEFORE (first 500 chars):{RESET}")
    print(f"  {raw_text[:500]}...\n")
    print(f"  {GREEN}AFTER (first 500 chars):{RESET}")
    print(f"  {clean[:500]}...\n")

    # -- Full JSON --
    header("FULL JSON RESPONSE")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    print(f"{CYAN}{BAR}{RESET}")
    print(f"{CYAN}  DONE{RESET}")
    print(f"{CYAN}{BAR}{RESET}\n")


if __name__ == "__main__":
    main()
