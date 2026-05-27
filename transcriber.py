"""
WhisperX transcription + speaker diarization module for Banglish.

Uses WhisperX (faster-whisper / CTranslate2) for transcription with
word-level timestamps, then pyannote.audio for speaker diarization.

Includes hallucination detection to remove repeated phrases and gibberish
that Whisper produces on silence / low-quality audio sections.
"""

from __future__ import annotations

import re
import unicodedata
import torch
import whisperx
import config
from banglish_hints import WHISPER_PROMPT as BANGLISH_PROMPT

# Confidence threshold — words below this are flagged as uncertain.
LOW_CONFIDENCE_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

def _remove_repeated_phrases(text: str, max_repeats: int = 2) -> str:
    """
    Remove phrases (1-6 words) that repeat more than `max_repeats` times
    consecutively. Whisper hallucinates by looping the same phrase over and
    over when processing silence or low-quality audio.
    """
    # Single-token repeats (e.g. "CAAS, CAAS, CAAS, CAAS, ...")
    text = re.sub(
        r'(\b\S+\b)(?:\s*[,.]?\s*\1){' + str(max_repeats) + r',}',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )

    # Multi-word phrase repeats (2-15 word phrases)
    for ngram_size in range(15, 1, -1):
        pattern = (
            r'((?:\S+\s+){' + str(ngram_size - 1) + r'}\S+)'
            r'(?:\s*[.,;]?\s*\1){' + str(max_repeats) + r',}'
        )
        text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

    return text


def _remove_gibberish(text: str) -> str:
    """
    Remove lines/segments containing non-Latin gibberish characters that
    Whisper hallucinates (Devanagari, extended Latin diacritics, etc.).
    Keeps standard ASCII, common punctuation, and a few expected chars.
    """
    cleaned_parts = []
    for segment in text.split('.'):
        segment = segment.strip()
        if not segment:
            continue

        # Count characters that are "gibberish" for Banglish
        total_alpha = 0
        gibberish = 0
        for ch in segment:
            if ch.isalpha():
                total_alpha += 1
                cat = unicodedata.category(ch)
                # Allow basic Latin (ASCII letters) and keep going
                if ord(ch) > 0x024F:  # Beyond Latin Extended-B
                    gibberish += 1

        # If more than 40% of alpha chars are non-Latin, skip the segment
        if total_alpha > 0 and (gibberish / total_alpha) > 0.40:
            continue

        cleaned_parts.append(segment)

    return '. '.join(cleaned_parts)


def _clean_hallucinations(text: str) -> str:
    """Full hallucination cleanup pipeline."""
    text = _remove_repeated_phrases(text)
    text = _remove_gibberish(text)
    # Collapse multiple spaces / clean up punctuation artifacts
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s*([,.])\s*([,.])+', r'\1', text)  # remove double punctuation
    text = text.strip()
    return text


def _clean_segment_text(text: str) -> str:
    """Clean a single segment's text of hallucinations."""
    text = _remove_repeated_phrases(text)
    text = _remove_gibberish(text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Return device string for WhisperX: 'cuda' or 'cpu'."""
    if torch.cuda.is_available():
        print("[transcriber] Using CUDA GPU.")
        return "cuda"
    print("[transcriber] Using CPU.")
    return "cpu"


def _get_compute_type(device: str) -> str:
    """Pick the best compute type for the device."""
    if device == "cuda":
        return "float16"
    return "int8"


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_model = None
_device = None


def _get_model():
    """Load (or return cached) WhisperX model."""
    global _model, _device
    if _model is None:
        _device = _get_device()
        compute_type = _get_compute_type(_device)
        print(
            f"[transcriber] Loading WhisperX '{config.WHISPER_MODEL}' model "
            f"on {_device} ({compute_type}) ..."
        )
        _model = whisperx.load_model(
            config.WHISPER_MODEL,
            device=_device,
            compute_type=compute_type,
            asr_options={"initial_prompt": BANGLISH_PROMPT},
            language=getattr(config, "LANGUAGE", None),
        )
        print("[transcriber] Model loaded.")
    return _model, _device


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LowConfidenceWord:
    """A single word that fell below the confidence threshold."""

    __slots__ = ("word", "confidence", "start", "end", "speaker")

    def __init__(
        self,
        word: str,
        confidence: float,
        start: float,
        end: float,
        speaker: str = "",
    ):
        self.word = word
        self.confidence = confidence
        self.start = start
        self.end = end
        self.speaker = speaker

    def __repr__(self) -> str:
        spk = f", speaker={self.speaker!r}" if self.speaker else ""
        return (
            f"LowConfidenceWord(word={self.word!r}, "
            f"confidence={self.confidence:.2f}, "
            f"start={self.start:.2f}s, end={self.end:.2f}s{spk})"
        )

    def to_dict(self) -> dict:
        d = {
            "word": self.word,
            "confidence": round(self.confidence, 4),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }
        if self.speaker:
            d["speaker"] = self.speaker
        return d


class DiarizedSegment:
    """A segment of text attributed to a speaker."""

    __slots__ = ("speaker", "start", "end", "text")

    def __init__(self, speaker: str, start: float, end: float, text: str):
        self.speaker = speaker
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self) -> str:
        return f"[{self.speaker}] ({self.start:.1f}s-{self.end:.1f}s): {self.text}"

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


class TranscriptionResult:
    """Structured result from ``transcribe_audio``."""

    def __init__(
        self,
        raw_text: str,
        low_confidence_words: list[LowConfidenceWord],
        diarized_segments: list[DiarizedSegment] | None = None,
    ):
        self.raw_text = raw_text
        self.low_confidence_words = low_confidence_words
        self.diarized_segments = diarized_segments or []

    @property
    def has_uncertain_words(self) -> bool:
        return len(self.low_confidence_words) > 0

    @property
    def has_diarization(self) -> bool:
        return len(self.diarized_segments) > 0

    def diarized_text(self) -> str:
        """Return speaker-labeled transcript."""
        if not self.diarized_segments:
            return self.raw_text
        lines = []
        for seg in self.diarized_segments:
            lines.append(f"[{seg.speaker}]: {seg.text}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"Transcript : {self.raw_text[:200]}{'...' if len(self.raw_text) > 200 else ''}",
            f"Uncertain  : {len(self.low_confidence_words)} word(s)",
            f"Speakers   : {len(set(s.speaker for s in self.diarized_segments))}",
        ]
        for w in self.low_confidence_words[:20]:
            spk = f" [{w.speaker}]" if w.speaker else ""
            lines.append(
                f"  * '{w.word}' -- conf {w.confidence:.2f}  "
                f"[{w.start:.2f}s -> {w.end:.2f}s]{spk}"
            )
        if len(self.low_confidence_words) > 20:
            lines.append(f"  ... and {len(self.low_confidence_words) - 20} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core transcription + diarization
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    language: str | None = None,
    diarize: bool = True,
) -> TranscriptionResult:
    """
    Transcribe an audio file with WhisperX and optionally run speaker
    diarization via pyannote.audio.

    Parameters
    ----------
    audio_path : str
        Path to any audio file ffmpeg can decode.
    language : str | None
        ISO-639-1 code hint. Falls back to ``config.LANGUAGE``.
    diarize : bool
        Whether to run speaker diarization (requires HF_TOKEN).

    Returns
    -------
    TranscriptionResult
        Raw text, low-confidence words, and speaker-labeled segments.
    """
    model, device = _get_model()
    lang = language or getattr(config, "LANGUAGE", None)

    # ── Step 1: Transcribe ────────────────────────────────────────────
    print("[transcriber] Transcribing audio ...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        language=lang,
        batch_size=16,
    )

    raw_text: str = " ".join(
        seg.get("text", "").strip() for seg in result.get("segments", [])
    ).strip()

    # ── Step 1b: Clean hallucinations ─────────────────────────────────
    original_len = len(raw_text)
    raw_text = _clean_hallucinations(raw_text)
    cleaned_len = len(raw_text)
    if cleaned_len < original_len:
        removed = original_len - cleaned_len
        print(
            f"[transcriber] Hallucination cleanup: removed {removed} chars "
            f"({removed * 100 // original_len}% of transcript)"
        )

    # ── Step 2: Skip alignment for Bengali ────────────────────────────
    # Bengali has no default wav2vec2 alignment model in WhisperX.
    # We skip alignment entirely and just use the raw segment timestamps.
    if lang and lang not in ("en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"):
        print(
            f"[transcriber] Skipping alignment (no wav2vec2 model for '{lang}'). "
            f"Using raw segment timestamps."
        )
    else:
        print("[transcriber] Aligning word timestamps ...")
        try:
            align_model, align_meta = whisperx.load_align_model(
                language_code=lang or "en",
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_meta,
                audio,
                device,
                return_char_alignments=False,
            )
        except Exception as e:
            print(f"[transcriber] Alignment failed ({e}), using raw timestamps.")

    # ── Step 3: Diarization ───────────────────────────────────────────
    diarized_segments: list[DiarizedSegment] = []

    if diarize and config.HF_TOKEN:
        print("[transcriber] Running speaker diarization ...")
        try:
            # FIX: Use direct import instead of whisperx.DiarizationPipeline
            # which fails due to lazy import in WhisperX 3.8.2
            from whisperx.diarize import DiarizationPipeline

            diarize_model = DiarizationPipeline(
                token=config.HF_TOKEN,
                device=device,
            )
            diarize_result = diarize_model(
                audio,
                min_speakers=getattr(config, "MIN_SPEAKERS", 2),
                max_speakers=getattr(config, "MAX_SPEAKERS", 4),
            )
            result = whisperx.assign_word_speakers(diarize_result, result)
            print("[transcriber] Diarization complete.")
        except Exception as e:
            print(f"[transcriber] Diarization failed: {e}")
            import traceback
            traceback.print_exc()
            print("[transcriber] Continuing without speaker labels.")
    elif diarize and not config.HF_TOKEN:
        print("[transcriber] HF_TOKEN not set, skipping diarization.")

    # ── Step 4: Build output ──────────────────────────────────────────
    low_conf: list[LowConfidenceWord] = []

    for segment in result.get("segments", []):
        seg_speaker = segment.get("speaker", "Unknown")
        seg_text = segment.get("text", "").strip()
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", 0.0)

        # Clean each segment individually too
        seg_text = _clean_segment_text(seg_text)

        if seg_text:
            diarized_segments.append(
                DiarizedSegment(
                    speaker=seg_speaker,
                    start=seg_start,
                    end=seg_end,
                    text=seg_text,
                )
            )

        for word_info in segment.get("words", []):
            score = word_info.get("score", 1.0)
            if score < LOW_CONFIDENCE_THRESHOLD:
                low_conf.append(
                    LowConfidenceWord(
                        word=word_info.get("word", "").strip(),
                        confidence=score,
                        start=word_info.get("start", 0.0),
                        end=word_info.get("end", 0.0),
                        speaker=word_info.get("speaker", seg_speaker),
                    )
                )

    # ── Step 5: Deduplicate repeated segments ────────────────────────
    # Whisper hallucination creates many identical/near-identical segments.
    # Uses fuzzy matching: skips if new segment is a substring of a recent
    # one (or vice versa), or if 80%+ word overlap with last 3 segments.
    def _seg_similar(a: str, b: str) -> bool:
        a, b = a.strip().lower(), b.strip().lower()
        if a == b or a in b or b in a:
            return True
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return False
        return len(wa & wb) / max(len(wa), len(wb)) > 0.80

    deduped_segments: list[DiarizedSegment] = []
    for seg in diarized_segments:
        txt = seg.text.strip()
        if not txt:
            continue
        # Check against last 5 kept segments
        is_dup = False
        for prev in deduped_segments[-5:]:
            if _seg_similar(txt, prev.text):
                is_dup = True
                break
        if not is_dup:
            deduped_segments.append(seg)

    if len(deduped_segments) < len(diarized_segments):
        removed = len(diarized_segments) - len(deduped_segments)
        print(
            f"[transcriber] Deduplication: removed {removed} repeated segments "
            f"({removed * 100 // len(diarized_segments)}%)"
        )

    return TranscriptionResult(
        raw_text=raw_text,
        low_confidence_words=low_conf,
        diarized_segments=deduped_segments,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Transcribing: {path}\n")
    result = transcribe_audio(path)
    print(result.summary())
    print("\n--- Diarized transcript ---\n")
    print(result.diarized_text())
