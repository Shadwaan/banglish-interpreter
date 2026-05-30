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
# Stage 1b — heuristic two-pass language flagging (step 7)
# ---------------------------------------------------------------------------
# After the English first pass, a deterministic language classifier flags
# segments that look non-English / non-Banglish (i.e. Indic-family drift),
# and those segments are re-transcribed in bn mode and stitched back in.
#
# The lingua + english_words imports are deferred behind a lazy builder so
# that importing this module on a machine without those deps (the local
# venv) does not hard-fail; on Modal the image installs both.

_LANG_DETECTOR = None
_ENGLISH_VOCAB = None
_INDIC_DRIFT_LANGUAGES = None


def _get_lang_detector():
    """Build (or return cached) the lingua language detector + helpers."""
    global _LANG_DETECTOR, _ENGLISH_VOCAB, _INDIC_DRIFT_LANGUAGES
    if _LANG_DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder
        from english_words import get_english_words_set

        # English, Bengali, plus the Indic neighbours we want to flag.
        # NOTE: lingua (2.1.x) does not include Nepali in its supported
        # language set, so it is omitted; Hindi/Urdu/Marathi/Punjabi cover
        # the romanized-Indic drift we actually see on Bengali audio.
        _LANG_DETECTOR = (
            LanguageDetectorBuilder
            .from_languages(
                Language.ENGLISH,
                Language.BENGALI,
                Language.HINDI,
                Language.URDU,
                Language.MARATHI,
                Language.PUNJABI,
            )
            .build()
        )
        _ENGLISH_VOCAB = get_english_words_set(["web2"], lower=True, alpha=True)
        _INDIC_DRIFT_LANGUAGES = {
            Language.HINDI, Language.URDU, Language.MARATHI,
            Language.PUNJABI,
        }
    return _LANG_DETECTOR, _ENGLISH_VOCAB, _INDIC_DRIFT_LANGUAGES


def _english_word_density(text: str) -> float:
    """Fraction of tokens (lowercased, alpha-only) recognised as English."""
    _, vocab, _ = _get_lang_detector()
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in vocab) / len(tokens)


def _non_latin_alpha_fraction(text: str) -> float:
    """
    Fraction of alphabetic characters that are NOT basic Latin. Used to
    detect when a bn-mode re-transcription came back as Bengali *script*
    (which this romanized pipeline cannot use — see
    ``_retranscribe_segment_in_bn``).
    """
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    non_latin = sum(1 for ch in alpha if ord(ch) > 0x024F)  # beyond Latin Ext-B
    return non_latin / len(alpha)


def _should_flag_for_retranscribe(text: str) -> bool:
    """
    Aggressive flagger: returns True if this segment should be
    re-transcribed in ``bn`` mode. Per the build-log decision, we err
    toward flagging on ambiguous cases.

    Rules:
    - Empty or very short (<3 tokens) text: don't flag (classifier is
      unreliable, and the segment is too small for the cost to matter).
    - Any Indic-family language detected (Hindi/Urdu/Marathi/Nepali/
      Punjabi): flag.
    - Bengali detected (lingua sometimes classifies clean Banglish as
      Bengali): don't flag, it's already Bengali-shaped.
    - English detected with high confidence (>=0.7) AND English-word
      density >= 0.5: don't flag, it's clean English or English-heavy
      Banglish.
    - All other cases (English-but-low-confidence, low English density,
      classifier returning None): FLAG. Aggressive default.
    """
    detector, _, indic = _get_lang_detector()
    from lingua import Language

    text = (text or "").strip()
    if not text:
        return False
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 3:
        return False

    detected = detector.detect_language_of(text)

    if detected in indic:
        return True
    if detected == Language.BENGALI:
        return False
    if detected == Language.ENGLISH:
        eng_conf = detector.compute_language_confidence(text, Language.ENGLISH)
        eng_density = _english_word_density(text)
        if eng_conf >= 0.7 and eng_density >= 0.5:
            return False
        return True  # aggressive: ambiguous English → flag
    # detected is None or some unexpected language
    return True


def _extract_audio_slice(audio, start_sec: float, end_sec: float,
                         sample_rate: int = 16000):
    """Numpy-slice the loaded audio array by time range."""
    start_idx = max(0, int(start_sec * sample_rate))
    end_idx = min(len(audio), int(end_sec * sample_rate))
    return audio[start_idx:end_idx]


def _retranscribe_segment_in_bn(model, audio_slice) -> str:
    """
    Re-transcribe a single audio slice with ``language='bn'``. Returns the
    concatenated text of whatever segments Whisper produces for the slice.

    IMPORTANT: forced bn mode tends to emit Bengali *script*, which this
    romanized pipeline's downstream gibberish-stripper would delete. We
    therefore REJECT (return "") any output that is predominantly non-Latin,
    so the caller keeps the original en-pass text instead of losing the
    segment entirely. The non-Latin fraction is logged either way so the
    experiment can see exactly what bn mode produced.
    """
    if len(audio_slice) < 1600:  # less than 0.1s of audio at 16kHz
        return ""
    try:
        # Clear the cached tokenizer before each differently-languaged call.
        # WhisperX's transcribe() does `task = task or self.tokenizer.task`
        # when a tokenizer already exists; faster-whisper's `.task` returns
        # the task TOKEN-ID (e.g. 50360), not the string "transcribe", which
        # then raises "'50360' is not a valid task". Resetting to None forces
        # the clean `tokenizer is None` path (task defaults to "transcribe").
        # Same fix Path B used (see BUILD-LOG 2026-05-30 Path B entry).
        try:
            model.tokenizer = None
        except Exception:
            pass
        result = model.transcribe(audio_slice, language="bn", batch_size=16)
        segs = result.get("segments", [])
        text = " ".join(s.get("text", "").strip() for s in segs).strip()
        if not text:
            return ""
        nl = _non_latin_alpha_fraction(text)
        if nl > 0.5:
            print(
                f"[transcriber] bn output rejected (non-Latin frac={nl:.2f}, "
                f"Bengali script): {text[:60]!r}"
            )
            return ""
        if nl > 0.0:
            print(f"[transcriber] bn output kept (non-Latin frac={nl:.2f})")
        return text
    except Exception as e:
        print(f"[transcriber] bn re-transcribe failed for slice: {e}")
        return ""


# Lightweight unit check for the flagger logic. Run on Modal (where the
# deps exist) with:  python -c "import transcriber; transcriber._selftest_flagger()"
def _selftest_flagger() -> None:
    tests = [
        ("I think we can start with the MVP next week", False),   # clean English
        ("ami today office jabo, traffic onek bad ache", False),  # clean Banglish
        ("pa meela hai, wadah madat cha, ai hai", True),          # Hindi-drift
        ("hi", False),                                            # too short
    ]
    for text, expected in tests:
        got = _should_flag_for_retranscribe(text)
        density = _english_word_density(text)
        status = "OK" if got == expected else "FAIL"
        print(f'{status}: "{text[:40]}" -> flag={got} '
              f'(expected={expected}, density={density:.2f})')

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


# ---------------------------------------------------------------------------
# Holistic Whisper hallucination / prompt-echo filtering (step 12)
# ---------------------------------------------------------------------------
# Two well-documented, model-agnostic Whisper failure modes produce whole
# junk SEGMENTS (not repeated phrases, so _remove_repeated_phrases misses
# them):
#   (A) YouTube-tail hallucinations on silent / low-content audio
#   (B) initial_prompt echo — the decoder regurgitating the prompt as text
# We drop a segment only on a WHOLE-segment (normalized) match, never a
# substring, so real sentences that merely contain "thanks" are untouched.

WHISPER_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "thank you very much for watching",
    "please subscribe",
    "please subscribe to the channel",
    "don't forget to like and subscribe",
    "like and subscribe",
    "subscribe to the channel",
    "thanks for watching and see you next time",
}

# Known initial_prompt echoes — the current fragment prompt plus the
# historical full-sentence Banglish prompts that leaked before step 12.
_KNOWN_PROMPT_ECHOES = {
    "the speaker will freely mix bengali and english",
    "the speaker will freely mix bengali words in english",
    "transcribe bengali words in their roman/latin transliteration, not in bengali script",
    "transcribe bengali words in english",
    "transcribe bengali words in bengali script",
    "banglish: bengali in latin/roman script, code-switched with english",
    "this is a conversation in banglish",
}


def _normalize_for_match(text: str) -> str:
    """Lowercase, strip surrounding whitespace and trailing punctuation."""
    return (text or "").strip().lower().rstrip(".!?,;:").strip()


def _is_hallucination_phrase(text: str) -> bool:
    return _normalize_for_match(text) in WHISPER_HALLUCINATION_PHRASES


def _is_prompt_echo(text: str) -> bool:
    """
    True if the whole segment matches an initial_prompt echo — either a known
    historical/current echo string, or a complete short sentence within the
    current BANGLISH_PROMPT. The huge comma-joined vocabulary sentence is
    skipped (len guard) so we never partial-match real content.
    """
    norm = _normalize_for_match(text)
    if not norm:
        return False
    if norm in _KNOWN_PROMPT_ECHOES:
        return True
    prompt = BANGLISH_PROMPT or ""
    for sent in re.split(r"[.\n]", prompt):
        s = _normalize_for_match(sent)
        if s and len(s) < 120 and s == norm:
            return True
    return False


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
    # Drop whole-segment YouTube-tail hallucinations and initial_prompt
    # echoes — return "" so the caller's `if seg_text:` skips the segment.
    if _is_hallucination_phrase(text) or _is_prompt_echo(text):
        print(f"[transcriber] dropped hallucination/prompt-echo segment: {text[:70]!r}")
        return ""
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
    # Reset any tokenizer left over from a previous clip's bn re-transcribe
    # pass (the module-level model is cached across invocations). A leftover
    # bn tokenizer would force a language-change rebuild here and trip the
    # faster-whisper task-token-id bug; starting from None rebuilds en cleanly.
    try:
        model.tokenizer = None
    except Exception:
        pass
    result = model.transcribe(
        audio,
        language=lang,
        batch_size=16,
    )

    # ── Step 1b: Heuristic two-pass for non-English drift ─────────────
    # Flag segments that look non-English / non-Banglish (Indic-family
    # drift) with a deterministic classifier, then re-transcribe just
    # those slices in bn mode and stitch the text back in. Runs on the
    # raw first-pass segments, before hallucination cleanup / alignment /
    # diarization (all of which then operate on the stitched text).
    flagged_count = 0
    retranscribed_count = 0
    rejected_script_count = 0
    first_pass_segments = result.get("segments", [])
    for seg in first_pass_segments:
        if _should_flag_for_retranscribe(seg.get("text", "")):
            flagged_count += 1
            new_text = _retranscribe_segment_in_bn(
                model,
                _extract_audio_slice(
                    audio, seg.get("start", 0.0), seg.get("end", 0.0)
                ),
            )
            if new_text and new_text != seg.get("text", "").strip():
                seg["text"] = new_text
                retranscribed_count += 1
            elif not new_text:
                # _retranscribe_segment_in_bn returns "" both on failure and
                # when it rejected Bengali-script output; the per-slice log
                # distinguishes them. Track the rejection bucket loosely.
                rejected_script_count += 1
    print(
        f"[transcriber] two-pass: flagged {flagged_count} of "
        f"{len(first_pass_segments)} segments; {retranscribed_count} replaced "
        f"with bn-mode output; {rejected_script_count} flagged-but-unreplaced "
        f"(bn empty/failed/script-rejected)"
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
