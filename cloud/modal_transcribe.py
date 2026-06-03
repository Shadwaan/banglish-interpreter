"""
Modal deployment of the Banglish transcription pipeline.

Wraps the existing local pipeline (transcriber.py + interpreter.py) and runs
it on a cloud GPU with Whisper large-v3-turbo instead of the local "small"
model. Stage 2 (diarization) and Stage 3 (Claude cleanup) are unchanged —
this is a model swap + a hosting change, not a redesign.

I/O contract matches what test_audio.py writes locally:

    {
        "audio_file": str,
        "raw_transcript": str,
        "diarized_transcript": str,
        "diarized_segments": [ {speaker, start, end, text}, ... ],
        "speakers": [str, ...],
        "low_confidence_count": int,
        "interpretation": { assessment, clean_version, clean_segments,
                            alternatives, changes_made }
    }

Deploy:
    modal deploy cloud/modal_transcribe.py

Call (see cloud/transcribe_cloud.py):
    fn = modal.Function.from_name("banglish-cloud-transcriber", "transcribe")
    result = fn.remote(audio_bytes, "Foo.m4a")
"""

from __future__ import annotations

import modal

APP_NAME = "banglish-cloud-transcriber"

# ---------------------------------------------------------------------------
# Image — debian + ffmpeg + the same Python deps as the local pipeline,
# plus the local source modules (config, banglish_hints, transcriber,
# interpreter) shipped via add_local_python_source so the function can
# import them exactly as the CLI does.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "whisperx",
        "pyannote.audio",
        "anthropic",
        "numpy",
        "scipy",
        "python-dotenv",
        "torch",
        # Stage 1b two-pass: deterministic language classifier to flag
        # non-English/non-Banglish segments for bn re-transcription, plus a
        # ready-made English vocabulary set for the word-density signal.
        "lingua-language-detector",
        "english-words",
    )
    # WhisperX 3.8.x alignment uses NLTK's punkt/punkt_tab for sentence
    # splitting and downloads them from the network at runtime — flaky on
    # Modal (seen as `urlopen error [Errno 104]` → alignment fails → pipeline
    # degrades to coarse segments with zero low-confidence words). Bake them
    # into the image so alignment never needs the network. NLTK_DATA is
    # pointed at this baked path in the function body.
    .run_commands(
        "python -m nltk.downloader -d /usr/local/nltk_data punkt punkt_tab"
    )
    .add_local_python_source(
        "config",
        "banglish_hints",
        "transcriber",
        "interpreter",
    )
)

# Persist model downloads across function invocations — first call grabs
# large-v3-turbo (~1.5 GB) and the pyannote diarization model, every
# subsequent call reuses them. Mounted at /cache so it lands on a fresh
# empty path (the image's pip install populates /root/.cache, so we cannot
# mount there). Cache env vars are set inside the function body at runtime,
# NOT in the image — setting them in the image would cause pip's transitive
# imports during build to write into /cache and make /cache non-empty,
# which would then fail the runtime mount.
volume = modal.Volume.from_name("whisper-models", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    # 16 GB system RAM — hedge against the worker OOM that caused a
    # silent Modal retry on the 25 MB Zayan update clip (Stage 2
    # diarization on long Bengali-heavy audio can spike well past the
    # 2 GB default).
    memory=16384,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    volumes={"/cache": volume},
    timeout=3600,
)
def transcribe(
    audio_bytes: bytes | None = None,
    filename: str | None = None,
    *,
    media_url: str | None = None,
) -> dict:
    """
    Run the full pipeline (Stage 1 Whisper large-v3-turbo + Stage 2
    diarization + Stage 3 Claude cleanup) and return the same JSON dict shape
    that test_audio.py writes locally.

    Input is supplied EITHER as raw bytes OR as a URL — exactly one:

      - ``audio_bytes`` (+ ``filename``): the original path, unchanged.
        Positional callers — ``fn.remote(media_bytes, filename)`` — behave
        exactly as before.
      - ``media_url`` (keyword-only) with ``audio_bytes=None``: the media is
        stream-downloaded from the URL into ``/tmp/{filename}`` using stdlib
        urllib (no image-dependency change).

    ``filename`` is required at runtime (it drives the /tmp path + extension);
    it only carries a ``None`` default so it can follow the now-optional
    ``audio_bytes`` positionally. ``ValueError`` is raised if ``filename`` is
    missing, if neither input is given, or if both are given.
    """
    import os

    # 0. Redirect model/asset caches into the mounted /cache volume so
    #    HuggingFace, Torch and pyannote downloads persist across calls.
    #    Set BEFORE any heavy import (transcriber pulls in WhisperX which
    #    reads HF_HOME at import time).
    os.environ["XDG_CACHE_HOME"] = "/cache"
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/cache/huggingface/hub"
    os.environ["TORCH_HOME"] = "/cache/torch"
    os.environ["PYANNOTE_CACHE"] = "/cache/pyannote"
    # NLTK data baked into the image (see Image.run_commands above) so
    # WhisperX alignment's punkt/punkt_tab lookup never hits the network.
    os.environ["NLTK_DATA"] = "/usr/local/nltk_data"
    for sub in (
        "/cache",
        "/cache/huggingface",
        "/cache/huggingface/hub",
        "/cache/torch",
        "/cache/pyannote",
    ):
        os.makedirs(sub, exist_ok=True)

    # 1. Validate inputs and materialize the media to a predictable local
    #    path. Exactly one of (audio_bytes, media_url) must be supplied;
    #    filename is always required (it drives the tmp path + extension).
    if not filename:
        raise ValueError("filename is required (drives the /tmp path and extension).")
    if audio_bytes is None and media_url is None:
        raise ValueError("Provide either audio_bytes or media_url.")
    if audio_bytes is not None and media_url is not None:
        raise ValueError("Provide only ONE of audio_bytes or media_url, not both.")

    tmp_path = f"/tmp/{filename}"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    if media_url is not None:
        # Stream-download with stdlib urllib — no image-dependency change.
        import shutil
        import urllib.request

        req = urllib.request.Request(media_url, headers={"User-Agent": "banglish-cloud-transcriber"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    else:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

    # 2. Whisper model size. v2 used large-v3-turbo, but its larger
    #    multilingual vocabulary is what introduced the Hindi-drift regression
    #    (it substitutes romanized Hindi/Urdu for Bengali audio). Step 10
    #    reverts to "small" — the model v1.5 used — to test whether the model
    #    upgrade (not the cloud move) was the regression source. config.py
    #    already hard-codes "small", so we simply DON'T override it here; the
    #    cloud GPU still gives the v2 speed win on the smaller model.
    import config
    config.WHISPER_MODEL = "medium"

    from transcriber import transcribe_audio
    from interpreter import BanglishInterpreter, polish_segments

    # 3. Stage 1 + 2 — Whisper + diarization.
    transcription = transcribe_audio(tmp_path)

    # 4. Stage 3 — Claude cleanup, with v1.5 segment-aware path.
    interp = BanglishInterpreter()
    interpretation = interp.interpret(
        raw_text=transcription.raw_text,
        low_confidence_words=transcription.low_confidence_words,
        diarized_segments=transcription.diarized_segments,
    )

    # 4b. Stage 3.5 — polish (smoothing) pass over the B3-cleaned segments.
    #     Non-destructive: returns the B3 segments unchanged on any failure.
    #     clean_segments is preserved as-is (the eval scores against it).
    interpretation["polished_segments"] = polish_segments(
        segments=interpretation.get("clean_segments") or [],
        anthropic_client=interp._client,
    )

    # 5. Assemble the same output_data dict test_audio.py writes locally.
    output_data = {
        "audio_file": filename,
        "raw_transcript": transcription.raw_text,
        "diarized_transcript": transcription.diarized_text(),
        "diarized_segments": [
            seg.to_dict() for seg in transcription.diarized_segments
        ],
        "speakers": list(
            {seg.speaker for seg in transcription.diarized_segments}
        ),
        "low_confidence_count": len(transcription.low_confidence_words),
        "interpretation": interpretation,
    }
    return output_data
