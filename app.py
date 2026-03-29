"""
Banglish Audio Transcriber — Web Interface

Upload an audio file, get back a cleaned Banglish transcript as a .txt download.

    python app.py          # starts on http://0.0.0.0:5000
"""

from flask import Flask, request, send_file, render_template_string
import tempfile, os, io

from transcriber import transcribe_audio, _get_model
from interpreter import BanglishInterpreter

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ── HTML template ────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banglish Transcriber</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5; color: #1a1a1a;
  }
  .card {
    background: #fff; border-radius: 12px; padding: 2.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    max-width: 480px; width: 90%;
  }
  h1 { margin: 0 0 .25rem; font-size: 1.5rem; }
  .sub { color: #666; font-size: .9rem; margin-bottom: 1.5rem; }
  label { display: block; font-weight: 600; margin-bottom: .4rem; }
  input[type=file] {
    width: 100%; padding: .6rem; border: 2px dashed #ccc; border-radius: 8px;
    background: #fafafa; cursor: pointer; margin-bottom: 1rem;
  }
  input[type=file]:hover { border-color: #888; }
  .checkbox-row {
    display: flex; align-items: center; gap: .5rem;
    margin-bottom: 1.5rem; font-size: .9rem;
  }
  button {
    width: 100%; padding: .75rem; border: none; border-radius: 8px;
    background: #2563eb; color: #fff; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background .15s;
  }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #93b4f5; cursor: wait; }
  .note { margin-top: 1rem; font-size: .8rem; color: #888; text-align: center; }
  .spinner { display: none; margin: 1rem auto 0; text-align: center; color: #2563eb; }
  .spinner.show { display: block; }
  .error { background: #fef2f2; color: #b91c1c; padding: .75rem; border-radius: 8px;
           margin-bottom: 1rem; font-size: .9rem; }
</style>
</head>
<body>
<div class="card">
  <h1>Banglish Transcriber</h1>
  <p class="sub">Upload an audio file to get a cleaned Banglish transcript.</p>

  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}

  <form method="POST" action="/transcribe" enctype="multipart/form-data" id="form">
    <label for="audio">Audio file</label>
    <input type="file" name="audio" id="audio"
           accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.opus" required>

    <div class="checkbox-row">
      <input type="checkbox" name="diarize" id="diarize" checked>
      <label for="diarize" style="font-weight:normal;margin:0">
        Speaker diarization (identify who is speaking)
      </label>
    </div>

    <button type="submit" id="btn">Transcribe</button>
  </form>

  <div class="spinner" id="spinner">Processing&hellip; this usually takes 1-3 minutes.</div>
  <p class="note">Supports WAV, MP3, M4A, OGG, FLAC, and more. Max 100 MB.</p>
</div>

<script>
document.getElementById('form').addEventListener('submit', function() {
  var btn = document.getElementById('btn');
  btn.disabled = true;
  btn.textContent = 'Transcribing\u2026';
  document.getElementById('spinner').classList.add('show');
});
</script>
</body>
</html>
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_txt(result, interpretation, filename):
    """Build the human-readable .txt output."""
    lines = []
    lines.append("BANGLISH INTERPRETER OUTPUT")
    lines.append(f"Audio: {filename}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("SPEAKER-LABELED TRANSCRIPT")
    lines.append("-" * 70)
    lines.append("")
    if result.has_diarization:
        for seg in result.diarized_segments:
            lines.append(f"[{seg.speaker}]: {seg.text}")
    else:
        lines.append(result.raw_text)

    lines.append("")
    lines.append("=" * 70)
    lines.append("CLAUDE ASSESSMENT")
    lines.append("-" * 70)
    lines.append("")
    lines.append(interpretation.get("assessment", "-"))

    alts = interpretation.get("alternatives", [])
    lines.append("")
    lines.append(f"WORD CORRECTIONS ({len(alts)})")
    lines.append("-" * 70)
    lines.append("")
    for i, alt in enumerate(alts, 1):
        lines.append(f'{i}. "{alt.get("original", "?")}" -> "{alt.get("replacement", "?")}"')
        if alt.get("reason"):
            lines.append(f'   Reason: {alt["reason"]}')
        lines.append("")

    lines.append("=" * 70)
    lines.append("CLEAN VERSION")
    lines.append("-" * 70)
    lines.append("")
    lines.append(interpretation.get("clean_version", result.raw_text))
    return "\n".join(lines)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, error=None)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files.get("audio")
    if not file or file.filename == "":
        return render_template_string(HTML, error="No file selected."), 400

    diarize = "diarize" in request.form

    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, file.filename)
    file.save(audio_path)

    try:
        # Transcribe with WhisperX
        result = transcribe_audio(audio_path, diarize=diarize)

        # Interpret with Claude
        interp = BanglishInterpreter()
        interpretation = interp.interpret(
            raw_text=result.raw_text,
            low_confidence_words=result.low_confidence_words,
        )

        # Build downloadable .txt
        txt = build_txt(result, interpretation, file.filename)
        basename = os.path.splitext(file.filename)[0]

        buf = io.BytesIO(txt.encode("utf-8"))
        buf.seek(0)
        return send_file(
            buf,
            mimetype="text/plain",
            as_attachment=True,
            download_name=f"{basename}_transcript.txt",
        )
    except Exception as e:
        return render_template_string(HTML, error=f"Transcription failed: {e}"), 500
    finally:
        try:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[app] Pre-loading Whisper model...")
    _get_model()
    print("[app] Model loaded. Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
