#!/usr/bin/env python3
"""
Verbose end-to-end test script for the Banglish Interpreter pipeline.

Usage:
    python test_audio.py <audio_file>

Runs the full pipeline and prints each stage clearly:
  1. Raw Whisper output
  2. Low-confidence words highlighted inside the transcript
  3. Claude's interpretation (assessment, alternatives, changes)
  4. Claude's clean version
"""

import io
import json
import os
import sys
import time
import re

# Force UTF-8 output on Windows so box-drawing / Unicode chars work.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

from transcriber import transcribe_audio, LowConfidenceWord, TranscriptionResult
from interpreter import BanglishInterpreter


# -- Display helpers -------------------------------------------------------

BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BAR    = "=" * 70


def header(title: str) -> None:
    print(f"\n{CYAN}{BAR}{RESET}")
    print(f"{CYAN}  {title}{RESET}")
    print(f"{CYAN}{BAR}{RESET}\n")


def subheader(title: str) -> None:
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'-' * 60}\n")


def highlight_low_confidence(
    text: str,
    words: list[LowConfidenceWord],
) -> str:
    """
    Return the transcript with low-confidence words wrapped in
    [RED markers] so they stand out visually.
    """
    if not words:
        return text

    highlighted = text
    # Sort longest-first to avoid partial replacements.
    for w in sorted(words, key=lambda w: -len(w.word)):
        pattern = re.compile(re.escape(w.word), re.IGNORECASE)
        replacement = f"{RED}{BOLD}[{w.word} ({w.confidence:.0%})]{RESET}"
        highlighted = pattern.sub(replacement, highlighted, count=1)
    return highlighted


# -- Pipeline stages -------------------------------------------------------

def stage_transcribe(audio_path: str) -> TranscriptionResult:
    header("STAGE 1 -- WHISPER TRANSCRIPTION")
    print(f"  Audio file: {audio_path}")
    print(f"  Loading model + transcribing ...\n")

    t0 = time.time()
    result = transcribe_audio(audio_path)
    elapsed = time.time() - t0

    # 1a. Raw transcript
    subheader("Raw transcript")
    print(f"  {result.raw_text}\n")
    print(f"  {DIM}(transcribed in {elapsed:.1f}s){RESET}\n")

    # 1b. Low-confidence words table
    subheader(f"Low-confidence words ({len(result.low_confidence_words)})")
    if result.low_confidence_words:
        print(f"  {'WORD':<20} {'CONF':>6}  {'TIME RANGE':<18} {'SPEAKER':<12}")
        print(f"  {'-'*20} {'-'*6}  {'-'*18} {'-'*12}")
        for w in result.low_confidence_words:
            color = RED if w.confidence < 0.40 else YELLOW
            filled = "#" * int(w.confidence * 20)
            empty  = "." * (20 - int(w.confidence * 20))
            bar = filled + empty
            spk = getattr(w, 'speaker', '') or ''
            print(
                f"  {color}{w.word:<20}{RESET} "
                f"{color}{w.confidence:>5.0%}{RESET}  "
                f"{w.start:.2f}s - {w.end:.2f}s  "
                f"{CYAN}{spk:<12}{RESET}"
                f"{DIM}[{bar}]{RESET}"
            )
        print()
    else:
        print(f"  {GREEN}[OK] Whisper was confident about every word.{RESET}\n")

    # 1c. Diarized transcript (speaker-labeled)
    if result.has_diarization:
        subheader(f"Speaker-diarized transcript ({len(set(s.speaker for s in result.diarized_segments))} speakers)")
        for seg in result.diarized_segments:
            print(f"  {CYAN}[{seg.speaker}]{RESET} {seg.text}")
        print()
    else:
        subheader("Transcript (no diarization)")
        highlighted = highlight_low_confidence(result.raw_text, result.low_confidence_words)
        print(f"  {highlighted}\n")

    return result


def stage_interpret(
    result: TranscriptionResult,
    interp: BanglishInterpreter,
) -> dict:
    header("STAGE 2 -- CLAUDE INTERPRETATION")
    print("  Sending to Claude claude-sonnet-4 ...\n")

    t0 = time.time()
    data = interp.interpret(
        raw_text=result.raw_text,
        low_confidence_words=result.low_confidence_words,
    )
    elapsed = time.time() - t0

    # 2a. Assessment
    subheader("Assessment")
    print(f"  {data.get('assessment', '-')}\n")

    # 2b. Alternatives
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

    # 2c. Changes summary
    subheader("Changes summary")
    print(f"  {data.get('changes_made', '-')}\n")

    # 2d. Clean version
    subheader("Clean version")
    print(f"  {GREEN}{BOLD}{data.get('clean_version', '-')}{RESET}\n")

    print(f"  {DIM}(interpreted in {elapsed:.1f}s){RESET}\n")

    return data


def stage_comparison(raw: str, clean: str) -> None:
    header("STAGE 3 -- BEFORE / AFTER COMPARISON")
    print(f"  {RED}BEFORE (first 500 chars):{RESET}")
    print(f"  {raw[:500]}{'...' if len(raw) > 500 else ''}\n")
    print(f"  {GREEN}AFTER (first 500 chars):{RESET}")
    print(f"  {clean[:500]}{'...' if len(clean) > 500 else ''}\n")


# -- Main ------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"\n{BOLD}  BANGLISH INTERPRETER -- VERBOSE TEST{RESET}")
    print(f"  {'-' * 40}\n")

    # Stage 1: Whisper
    transcription = stage_transcribe(audio_path)

    # -- Save raw transcript IMMEDIATELY so it can't be lost ----------------
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(output_dir, f"{audio_basename}_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        if transcription.has_diarization:
            for seg in transcription.diarized_segments:
                f.write(f"[{seg.speaker}]: {seg.text}\n")
        else:
            f.write(transcription.raw_text + "\n")
    print(f"  {GREEN}Raw transcript saved to: {raw_path}{RESET}\n")

    # Stage 2: Claude (gracefully handle failures so raw output is still kept)
    interp = BanglishInterpreter()
    try:
        interpretation = stage_interpret(transcription, interp)
    except Exception as e:
        print(f"\n  {RED}[!] Claude interpretation failed: {e}{RESET}")
        print(f"  {YELLOW}Raw transcript is still saved at: {raw_path}{RESET}\n")
        interpretation = {
            "assessment": f"Interpretation skipped: {e}",
            "alternatives": [],
            "changes_made": "",
            "clean_version": transcription.raw_text,
        }

    # Stage 3: Side-by-side
    stage_comparison(
        transcription.raw_text,
        interpretation.get("clean_version", transcription.raw_text),
    )

    # Dump full JSON for inspection
    header("FULL JSON RESPONSE")
    print(json.dumps(interpretation, indent=2, ensure_ascii=False))
    print()

    # -- Save output to files --------------------------------------------------
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Human-readable .txt
    txt_path = os.path.join(output_dir, f"{audio_basename}_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"BANGLISH INTERPRETER OUTPUT\n")
        f.write(f"Audio: {audio_path}\n")
        f.write(f"{'=' * 70}\n\n")

        # Diarized transcript
        f.write("SPEAKER-LABELED TRANSCRIPT\n")
        f.write(f"{'-' * 70}\n\n")
        if transcription.has_diarization:
            for seg in transcription.diarized_segments:
                f.write(f"[{seg.speaker}]: {seg.text}\n")
        else:
            f.write(transcription.raw_text + "\n")

        f.write(f"\n\n{'=' * 70}\n")
        f.write("CLAUDE ASSESSMENT\n")
        f.write(f"{'-' * 70}\n\n")
        f.write(interpretation.get("assessment", "-") + "\n")

        f.write(f"\n\nWORD CORRECTIONS ({len(interpretation.get('alternatives', []))})\n")
        f.write(f"{'-' * 70}\n\n")
        for i, alt in enumerate(interpretation.get("alternatives", []), 1):
            f.write(f'{i}. "{alt.get("original", "?")}" -> "{alt.get("replacement", "?")}"\n')
            if alt.get("reason"):
                f.write(f'   Reason: {alt["reason"]}\n')
            f.write("\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("CLEAN VERSION\n")
        f.write(f"{'-' * 70}\n\n")
        f.write(interpretation.get("clean_version", transcription.raw_text) + "\n")

    # 2. JSON for programmatic use
    json_path = os.path.join(output_dir, f"{audio_basename}_output.json")
    output_data = {
        "audio_file": audio_path,
        "raw_transcript": transcription.raw_text,
        "diarized_transcript": transcription.diarized_text(),
        "speakers": list(set(s.speaker for s in transcription.diarized_segments)),
        "low_confidence_count": len(transcription.low_confidence_words),
        "interpretation": interpretation,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  {GREEN}Output saved to:{RESET}")
    print(f"    {GREEN}Readable : {txt_path}{RESET}")
    print(f"    {GREEN}JSON     : {json_path}{RESET}\n")

    print(f"{CYAN}{BAR}{RESET}")
    print(f"{CYAN}  DONE{RESET}")
    print(f"{CYAN}{BAR}{RESET}\n")


if __name__ == "__main__":
    main()
