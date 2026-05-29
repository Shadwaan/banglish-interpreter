"""
Local invoker for the deployed Modal transcription function.

Reads an audio file off disk, sends the bytes to the Modal app
`banglish-cloud-transcriber`, and writes the result into the local
outputs/ folder in exactly the same shape test_audio.py writes — so the
existing eval harness and any downstream tooling work unchanged.

Usage:
    python cloud/transcribe_cloud.py "<path to .m4a / .wav / etc.>"
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


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_file>")
        sys.exit(1)

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
