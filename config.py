import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Ensure ffmpeg is on PATH (winget installs it outside the default PATH on
# some Windows setups).
# ---------------------------------------------------------------------------
_FFMPEG_WINGET = os.path.expanduser(
    r"~\AppData\Local\Microsoft\WinGet\Links"
)
if sys.platform == "win32" and os.path.isdir(_FFMPEG_WINGET):
    os.environ["PATH"] = _FFMPEG_WINGET + os.pathsep + os.environ.get("PATH", "")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Whisper settings
WHISPER_MODEL = "small"
# Use "en" — the audio is English-dominant Banglish. Using "bn" causes
# massive hallucination/repetition because Whisper tries to force Bengali
# phonemes onto English-heavy speech.  The Banglish words are close enough
# to English that the "en" model captures them, and Claude fixes the rest.
LANGUAGE = "en"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
