# Banglish Interpreter

Audio-to-text pipeline that transcribes Banglish (Bengali written in Latin letters, often mixed with English) and produces a clean, chatbot-ready transcript.

**Pipeline:**
1. **Whisper** (via WhisperX) transcribes the audio
2. **Speaker diarization** (via pyannote) labels who said what
3. **Claude** cleans up Whisper's mistakes using Banglish-aware corrections

---

## Prerequisites

- **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- **ffmpeg** — required for audio decoding
  - **Windows:** download from [ffmpeg.org](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
  - **Mac:** `brew install ffmpeg`
  - **Linux:** `sudo apt install ffmpeg`
- **API keys** — Anthropic + Hugging Face (see [Secrets](#secrets) below)

---

## Setup on a new machine

```bash
# 1. Clone the repo
git clone https://github.com/Shadwaan/banglish-interpreter.git
cd banglish-interpreter

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Add your API keys (see "Secrets" below)
#    Create a file called .env in this folder with:
#    ANTHROPIC_API_KEY=sk-ant-...
#    HF_TOKEN=hf_...
```

---

## Usage

### Transcribe an audio file

```bash
python test_audio.py "path/to/audio.m4a"
```

This produces three files inside the `outputs/` folder (auto-created):

| File | What it is |
|---|---|
| `outputs/<name>_raw.txt` | Raw Whisper transcript (saved immediately, survives crashes) |
| `outputs/<name>_output.txt` | Full output: assessment + corrections + clean version |
| `outputs/<name>_output.json` | Same as above, machine-readable JSON |

Feed `outputs/<name>_output.txt` into any chatbot (Claude, ChatGPT, etc.) for context.

> The `outputs/` folder is gitignored — your transcripts stay local.

### Re-run only the Claude step

If the Claude cleanup failed but Whisper succeeded (raw file already exists):

```bash
python run_interpret_only.py "<name>_raw.txt"
```

Bare filenames are auto-resolved against the `outputs/` folder.

### Recover a transcript from a malformed output

If Claude's JSON got truncated or wrapped weirdly:

```bash
python extract_clean.py "<name>_output.txt" "<name>_chatready.txt"
```

Same auto-resolve applies — both args are looked up in `outputs/` if not found directly.

---

## Secrets

The repo intentionally **does not** include the `.env` file. Two API keys are required:

### 1. `ANTHROPIC_API_KEY` (paid)

For Claude — the model that cleans up the transcript.

Get one from [console.anthropic.com](https://console.anthropic.com). Claude API usage is metered (a few cents per long transcript).

### 2. `HF_TOKEN` (free, but requires extra setup)

For Hugging Face — used to download the **pyannote speaker diarization** model.

**One-time setup steps:**

1. Create a free Hugging Face account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token with `Read` access
3. Accept the pyannote model terms (one-click "Agree" on each page — required before the model will download):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Without this step the first run will fail with a 401 error when trying to load the diarization model.

### Adding the keys

Create a file called `.env` in the project root with:

```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

---

## Performance notes

Whisper transcription speed depends on hardware:

| Hardware | Speed |
|---|---|
| **NVIDIA GPU with CUDA** | Fastest (~5x faster than CPU) |
| **CPU only** (Intel or AMD) | ~30 min per audio file |
| **Apple Silicon (M1/M2/M3)** | Works on CPU; partial MPS support |

There's **no AMD-specific dependency** in this project. It runs on any x86/x64 or ARM machine.

---

## Project structure

```
banglish-interpreter/
├── app.py                  # Flask web app (planned)
├── main.py                 # CLI entry point
├── test_audio.py           # Verbose pipeline runner (recommended)
├── run_interpret_only.py   # Claude-only re-run on existing raw transcript
├── extract_clean.py        # Recover clean text from broken outputs
├── transcriber.py          # WhisperX + diarization wrapper
├── interpreter.py          # Claude API client
├── recorder.py             # Live audio recording helper
├── banglish_hints.py       # Reference vocab + mishearing map for Claude
├── config.py               # Loads env vars
└── requirements.txt
```
