"""
Microphone recording module for banglish-interpreter.

Records 16 kHz mono WAV audio from the default input device, or accepts an
existing file path as an alternative.  Provides a live countdown during
recording and handles Ctrl+C gracefully.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import config

# Supported extensions we'll accept when the user provides a file path.
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _countdown_callback(duration: int) -> None:
    """Print a live countdown to stderr while recording."""
    for remaining in range(duration, 0, -1):
        print(f"\r  ⏺  Recording … {remaining}s remaining ", end="", flush=True)
        time.sleep(1)
    print("\r  ✓  Recording complete.              ")


def _temp_wav_path() -> str:
    """Return a path inside the OS temp directory for a new WAV file."""
    tmp_dir = os.path.join(tempfile.gettempdir(), "banglish-interpreter")
    os.makedirs(tmp_dir, exist_ok=True)
    # Use a timestamp so files don't collide across runs.
    name = f"recording_{int(time.time())}.wav"
    return os.path.join(tmp_dir, name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_audio(duration: int = 5) -> str:
    """
    Record audio from the default microphone.

    Parameters
    ----------
    duration : int
        Length of recording in seconds (default 5).

    Returns
    -------
    str
        Absolute path to the saved 16 kHz mono WAV file.

    Raises
    ------
    KeyboardInterrupt
        Re-raised after cleanup so callers can handle it.
    """
    sample_rate = config.SAMPLE_RATE
    channels = config.CHANNELS
    dtype = config.DTYPE

    out_path = _temp_wav_path()

    print(f"\n  🎤  Get ready — recording {duration}s of audio …")
    time.sleep(0.4)  # tiny pause so the user can prepare

    try:
        # Start a non-blocking recording so we can print the countdown
        # on the main thread at the same time.
        audio_data = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
        )

        # Live countdown (blocks for `duration` seconds).
        _countdown_callback(duration)

        # Make sure the stream is fully flushed.
        sd.wait()

    except KeyboardInterrupt:
        # Stop the stream immediately, save whatever we captured.
        sd.stop()
        print("\n  ⚠  Recording interrupted by user.")
        # audio_data may be partially filled — still save it so the
        # caller can decide whether to use it.
        if audio_data is not None and audio_data.any():
            wav_write(out_path, sample_rate, audio_data)
            print(f"  ℹ  Partial recording saved → {out_path}")
            return out_path
        raise

    wav_write(out_path, sample_rate, audio_data)
    print(f"  📁  Saved → {out_path}")
    return out_path


def use_existing_file(file_path: str) -> str:
    """
    Validate and stage an existing audio file.

    If the file is already a WAV we simply return its absolute path.
    For any other supported format we copy it to the temp directory so
    downstream code has a consistent location to work with (Whisper
    accepts most formats via ffmpeg, so no conversion is needed).

    Parameters
    ----------
    file_path : str
        Path to an existing audio file.

    Returns
    -------
    str
        Absolute path to the (possibly copied) audio file.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the extension is not in the supported set.
    """
    src = Path(file_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    if src.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{src.suffix}'. "
            f"Accepted: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )

    # Copy non-WAV files (or WAVs from outside tmp) into our temp dir
    # so every path we hand back lives in a predictable place.
    tmp_dir = os.path.join(tempfile.gettempdir(), "banglish-interpreter")
    os.makedirs(tmp_dir, exist_ok=True)
    dest = os.path.join(tmp_dir, src.name)

    if str(src) != dest:
        shutil.copy2(src, dest)
        print(f"  📁  Staged → {dest}")

    return dest


def get_audio(file_path: str | None = None, duration: int = 5) -> str:
    """
    One-stop helper: record from mic **or** accept an existing file.

    Parameters
    ----------
    file_path : str | None
        If provided (and non-empty), use this file instead of recording.
    duration : int
        Recording length when *file_path* is None.

    Returns
    -------
    str
        Absolute path to the audio file ready for transcription.
    """
    if file_path and file_path.strip():
        return use_existing_file(file_path.strip())
    return record_audio(duration=duration)


# ---------------------------------------------------------------------------
# Quick CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            # Accept an existing file from the command line.
            path = get_audio(file_path=sys.argv[1])
        else:
            secs = 5
            if len(sys.argv) > 1 and sys.argv[1].isdigit():
                secs = int(sys.argv[1])
            path = get_audio(duration=secs)
        print(f"\nReady for transcription: {path}")
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
