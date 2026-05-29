"""
Local invoker for the deployed Modal transcription function.

Reads an audio file off disk, sends the bytes to the Modal app
`banglish-cloud-transcriber`, and writes the result into the local
outputs/ folder in exactly the same shape test_audio.py writes — so the
existing eval harness and any downstream tooling work unchanged.

Usage:
    python cloud/transcribe_cloud.py "<path to .m4a / .wav / etc.>"

Also supports back-filling a human-readable .txt from an existing JSON
(no remote call) — useful for retrofitting earlier cloud runs:

    python cloud/transcribe_cloud.py --write-txt "<path to *_output.json>"
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout on Windows so any non-ASCII printed by the pipeline
# does not crash the console (same trick test_audio.py uses).
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import modal


APP_NAME = "banglish-cloud-transcriber"
FN_NAME = "transcribe"


# ---------------------------------------------------------------------------
# Raw transcript writer — mirrors what test_audio.py produces
# ---------------------------------------------------------------------------

def write_raw_txt(out_path: Path, data: dict) -> None:
    """
    Write a ``*_raw.txt`` with one segment per line in the same format
    test_audio.py uses:  ``[SPEAKER_XX]: text``

    Sources from ``data["diarized_segments"]``. No timestamps in this
    file — it's a flat speaker-labelled transcript intended to be read
    by humans and by the eval harness (``eval/pipeline_io.load_raw`` strips
    the speaker prefix when computing WER).

    If diarization is absent, falls back to dumping ``raw_transcript``.
    """
    segs = data.get("diarized_segments") or []
    with open(out_path, "w", encoding="utf-8") as f:
        if segs:
            for seg in segs:
                f.write(
                    f"[{seg.get('speaker', 'Unknown')}]: {seg.get('text', '')}\n"
                )
        else:
            f.write((data.get("raw_transcript") or "") + "\n")


# ---------------------------------------------------------------------------
# Human-readable .txt formatter
# ---------------------------------------------------------------------------

def write_output_txt(
    out_path: Path,
    audio_path: str,
    data: dict,
) -> None:
    """
    Write the human-readable .txt that mirrors test_audio.py's format.

    Difference from test_audio.py: the CLEAN VERSION section emits one
    line per ``interpretation.clean_segments`` item with timestamps and
    speaker, reflecting the v1.5 segment-aware schema:

        [12.30s-15.80s] SPEAKER_00: corrected text

    If ``clean_segments`` is empty (older runs or flat-text Stage 3) the
    section falls back to the flat ``clean_version`` string.

    NOTE: this is duplicated from the inline writer in ``test_audio.py``
    by design — refactoring both to share a helper is a separate cleanup
    (see v1.5 BUILD-LOG note on three-entry-point duplication).
    """
    interp = data.get("interpretation") or {}
    diarized_segs = data.get("diarized_segments") or []
    alternatives = interp.get("alternatives") or []
    clean_segments = interp.get("clean_segments") or []

    bar = "=" * 70
    sub = "-" * 70

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("BANGLISH INTERPRETER OUTPUT\n")
        f.write(f"Audio: {audio_path}\n")
        f.write(f"{bar}\n\n")

        f.write("SPEAKER-LABELED TRANSCRIPT\n")
        f.write(f"{sub}\n\n")
        if diarized_segs:
            for seg in diarized_segs:
                f.write(
                    f"[{seg.get('speaker', 'Unknown')}]: {seg.get('text', '')}\n"
                )
        else:
            f.write((data.get("raw_transcript") or "") + "\n")

        f.write(f"\n\n{bar}\n")
        f.write("CLAUDE ASSESSMENT\n")
        f.write(f"{sub}\n\n")
        f.write((interp.get("assessment") or "-") + "\n")

        f.write(f"\n\nWORD CORRECTIONS ({len(alternatives)})\n")
        f.write(f"{sub}\n\n")
        for i, alt in enumerate(alternatives, 1):
            orig = alt.get("original", "?")
            repl = alt.get("replacement", "?")
            f.write(f'{i}. "{orig}" -> "{repl}"\n')
            if alt.get("reason"):
                f.write(f'   Reason: {alt["reason"]}\n')
            f.write("\n")

        f.write(f"\n{bar}\n")
        f.write("CLEAN VERSION\n")
        f.write(f"{sub}\n\n")
        if clean_segments:
            for seg in clean_segments:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                spk = seg.get("speaker", "Unknown")
                text = seg.get("text", "")
                f.write(f"[{start:.2f}s-{end:.2f}s] {spk}: {text}\n")
        else:
            f.write(
                (interp.get("clean_version") or data.get("raw_transcript") or "")
                + "\n"
            )


def _backfill_txt_from_json(json_path: Path) -> tuple[Path, Path]:
    """
    Read ``*_output.json`` and write the sibling ``*_output.txt`` AND the
    sibling ``*_raw.txt`` (in the same outputs/ directory, named after the
    audio: ``<basename>_raw.txt``).

    Intentional overwrite — early cloud runs predate the .txt and _raw.txt
    writers, so the on-disk siblings may be stale.
    """
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_path = data.get("audio_file") or json_path.stem.replace("_output", "")

    # *_output.txt — same path as the JSON with the suffix swapped.
    txt_path = json_path.with_suffix(".txt")
    write_output_txt(txt_path, audio_path, data)

    # *_raw.txt — derived from the audio basename (strip the "_output" suffix).
    raw_path = json_path.parent / (json_path.stem.replace("_output", "") + "_raw.txt")
    write_raw_txt(raw_path, data)

    return txt_path, raw_path


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage:")
        print(f"  python {sys.argv[0]} <audio_file>")
        print(f"  python {sys.argv[0]} --write-txt <existing *_output.json>")
        sys.exit(1)

    # Back-fill mode: reproduce the .txt for an existing JSON without
    # touching Modal at all. Used after adding the writer to retrofit
    # earlier cloud runs.
    if sys.argv[1] == "--write-txt":
        if len(sys.argv) < 3:
            print(f"Usage: python {sys.argv[0]} --write-txt <json_path>")
            sys.exit(1)
        json_path = Path(sys.argv[2])
        txt_path, raw_path = _backfill_txt_from_json(json_path)
        print(f"  Wrote: {txt_path}")
        print(f"  Wrote: {raw_path}")
        return

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}")
        sys.exit(2)

    # Where to drop the result locally. Mirror test_audio.py's layout.
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}_output.json"

    audio_bytes = audio_path.read_bytes()
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"  Audio: {audio_path}")
    print(f"  Size:  {size_mb:.2f} MB")
    print(f"  Calling Modal app {APP_NAME!r} fn {FN_NAME!r} ...")

    fn = modal.Function.from_name(APP_NAME, FN_NAME)

    t0 = time.time()
    result = fn.remote(audio_bytes, audio_path.name)
    elapsed = time.time() - t0
    print(f"  Round-trip time: {elapsed:.1f}s")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Wrote: {out_path}")

    # Also write the human-readable .txt next to the JSON.
    txt_path = out_path.with_suffix(".txt")
    write_output_txt(txt_path, str(audio_path), result)
    print(f"  Wrote: {txt_path}")

    # And a *_raw.txt for downstream tooling (eval harness reads from here).
    raw_path = output_dir / f"{audio_path.stem}_raw.txt"
    write_raw_txt(raw_path, result)
    print(f"  Wrote: {raw_path}")

    # Quick sanity peek
    interp = result.get("interpretation", {}) or {}
    print()
    print(f"  Speakers       : {result.get('speakers')}")
    print(f"  Segments       : {len(result.get('diarized_segments', []))}")
    print(f"  Clean segments : {len(interp.get('clean_segments', []))}")
    print(f"  Low-conf count : {result.get('low_confidence_count')}")
    cv = interp.get("clean_version", "")
    preview = cv[:240].replace("\n", " ⏎ ")
    print(f"  Clean preview  : {preview}{'...' if len(cv) > 240 else ''}")


if __name__ == "__main__":
    main()
