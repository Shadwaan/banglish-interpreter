"""
Banglish Interpreter — main entry point.

Wires together recorder, transcriber, and interpreter into an interactive
loop:  record/import → transcribe → interpret → follow-up → repeat.
"""

import json
import sys

import config
from recorder import get_audio
from transcriber import transcribe_audio
from interpreter import BanglishInterpreter


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60


def _print_header(text: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {text}")
    print(DIVIDER)


def _print_transcription(result) -> None:
    """Pretty-print Whisper's raw output and flagged words."""
    _print_header("📝  WHISPER TRANSCRIPTION")
    print(f"\n  {result.raw_text}\n")

    if result.has_uncertain_words:
        print(f"  ⚠  {len(result.low_confidence_words)} low-confidence word(s):\n")
        for w in result.low_confidence_words:
            bar = "█" * int(w.confidence * 20) + "░" * (20 - int(w.confidence * 20))
            print(
                f'    "{w.word}"  '
                f"[{bar}] {w.confidence:.0%}  "
                f"({w.start:.1f}s – {w.end:.1f}s)"
            )
        print()
    else:
        print("  ✅  Whisper was confident about every word.\n")


def _print_interpretation(data: dict) -> None:
    """Pretty-print Claude's structured interpretation."""
    _print_header("🤖  CLAUDE INTERPRETATION")

    print(f"\n  Assessment:    {data.get('assessment', '—')}")
    print(f"  Clean version: {data.get('clean_version', '—')}")
    print(f"  Changes made:  {data.get('changes_made', '—')}")

    alternatives = data.get("alternatives", [])
    if alternatives:
        print(f"\n  Alternatives ({len(alternatives)}):\n")
        for i, alt in enumerate(alternatives, 1):
            print(f"    {i}. \"{alt.get('original')}\" → \"{alt.get('replacement')}\"")
            if alt.get("candidates"):
                print(f"       Candidates: {', '.join(alt['candidates'])}")
            if alt.get("reason"):
                print(f"       Reason:     {alt['reason']}")
        print()
    else:
        print("\n  No word-level changes suggested.\n")


def _prompt(message: str, default: str = "") -> str:
    """Prompt the user, returning stripped input or *default*."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {message}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def _yes_no(message: str, default_yes: bool = True) -> bool:
    """Ask a yes/no question. Returns bool."""
    hint = "Y/n" if default_yes else "y/N"
    answer = _prompt(f"{message} ({hint})")
    if not answer:
        return default_yes
    return answer.lower().startswith("y")


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_get_audio() -> str:
    """Ask the user whether to record or supply a file, then return the path."""
    _print_header("🎧  AUDIO INPUT")
    print()
    print("    [1] Record from microphone")
    print("    [2] Provide an audio file path")
    print()

    choice = _prompt("Choose 1 or 2", default="1")

    if choice == "2":
        path = _prompt("Enter the audio file path")
        if not path:
            print("  ⚠  No path entered — falling back to mic recording.")
            return _record_with_duration()
        return get_audio(file_path=path)

    return _record_with_duration()


def _record_with_duration() -> str:
    """Ask for duration and record."""
    dur = _prompt("Recording duration in seconds", default="5")
    try:
        dur = max(1, int(dur))
    except ValueError:
        dur = 5
    return get_audio(duration=dur)


def step_transcribe(audio_path: str):
    """Run Whisper and display the result. Returns a TranscriptionResult."""
    _print_header("⏳  TRANSCRIBING WITH WHISPER …")
    result = transcribe_audio(audio_path)
    _print_transcription(result)
    return result


def step_interpret(interp: BanglishInterpreter, transcription_result) -> dict:
    """Send to Claude and display the structured interpretation."""
    _print_header("⏳  INTERPRETING WITH CLAUDE …")
    data = interp.interpret(
        raw_text=transcription_result.raw_text,
        low_confidence_words=transcription_result.low_confidence_words,
    )
    _print_interpretation(data)
    return data


def step_follow_ups(interp: BanglishInterpreter) -> None:
    """Multi-turn follow-up conversation about the current transcript."""
    while True:
        if not _yes_no("Ask a follow-up about this transcript?", default_yes=False):
            break

        question = _prompt("Your question")
        if not question:
            continue

        print(f"\n  ⏳  Asking Claude …\n")
        response = interp.follow_up(question)

        # If Claude returned the standard JSON schema, pretty-print it;
        # otherwise just print the raw text response.
        if "clean_version" in response:
            _print_interpretation(response)
        elif "response" in response:
            print(f"  Claude: {response['response']}\n")
        else:
            print(f"  Claude: {json.dumps(response, indent=2, ensure_ascii=False)}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        BANGLISH  INTERPRETER  v1.0           ║")
    print("  ║  Whisper transcription → Claude correction   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print("  Press Ctrl+C at any time to exit.\n")

    # Pre-load the Whisper model once before entering the loop.
    _print_header("⏳  LOADING WHISPER MODEL …")
    from transcriber import _get_model
    _get_model()

    interp = BanglishInterpreter()

    try:
        while True:
            # 1 — Get audio
            audio_path = step_get_audio()

            # 2 — Transcribe
            transcription = step_transcribe(audio_path)

            # 3 — Interpret with Claude
            step_interpret(interp, transcription)

            # 4 — Follow-up conversation
            step_follow_ups(interp)

            # 5 — Another round?
            print(DIVIDER)
            if not _yes_no("Process another audio clip?", default_yes=True):
                break

            # Reset interpreter history for the next transcript.
            interp.reset()

    except KeyboardInterrupt:
        pass

    print("\n  👋  Goodbye!\n")


if __name__ == "__main__":
    main()
