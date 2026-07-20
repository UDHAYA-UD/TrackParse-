# Signal — music analysis web app

A Flask web app version of the notebook: upload a track and get genre (Discogs-400),
Demucs-separated vocals transcribed by Whisper, language ID (IndicLID + fastText), and a
song-ID lookup (AcoustID for audio, Genius for pasted lyrics).

## What changed vs. the notebook

- **Models load once**, at app startup, instead of once per notebook cell run.
- **API keys are environment variables**, not hardcoded strings — see `.env.example`.
  Never commit your real `.env` file.
- **Every pipeline function keeps working if another one fails to load.** If, say, the
  genre model files can't download, the app still starts; that route just returns a
  clear error instead of the whole process crashing. Check `/api/health` to see status.
- Uploaded files are deleted after each request so disk usage doesn't grow unbounded.
  Demucs' separated stems are cached under `separated/` (see "Production notes" below).

## Project layout

```
music_flask_app/
├── app.py              # Flask routes
├── models.py            # the analysis pipeline (genre, Demucs, Whisper, lang ID, song ID)
├── config.py             # env-based settings
├── requirements.txt
├── .env.example
├── templates/index.html
├── static/style.css
├── static/script.js
├── uploads/              # temp storage for incoming files (auto-created, gitignored)
├── separated/            # Demucs output cache (auto-created, gitignored)
└── model_files/          # downloaded model weights, cached after first run (gitignored)
```

## Setup

1. **System dependency** — Chromaprint (needed for AcoustID fingerprinting):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y libchromaprint-tools
   # macOS
   brew install chromaprint
   ```

2. **Python environment** (Python 3.10–3.11 recommended for Essentia/Torch compatibility):
   ```bash
   python -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **API keys** — copy `.env.example` to `.env` and fill in your free keys:
   ```bash
   cp .env.example .env
   # then edit .env with your ACOUSTID_API_KEY and GENIUS_ACCESS_TOKEN
   ```
   - AcoustID: https://acoustid.org/new-application
   - Genius: https://genius.com/api-clients

4. **Run it**:
   ```bash
   python app.py
   ```
   First run will download the genre model files, Whisper weights, and IndicLID models —
   this can take several minutes and needs several GB of disk. Subsequent starts are fast
   since everything is cached locally. Open http://localhost:5000.

## Timing expectations (carried over from the notebook's own honesty check)

Vocal separation adds real processing time — a track that used to go straight to Whisper now
runs through Demucs first. On CPU this is noticeably slower than on GPU. Uncheck "Separate
vocals before transcribing" in the UI (or send `separate_vocals=false` to the API) to skip
straight to transcribing the full mix if that latency isn't acceptable for your use case.

## Production notes (not done here, worth doing before real deployment)

This is a **working, synchronous** app — good for local use, demos, or low-traffic internal
tools. A few things to add before putting it in front of real traffic:

- **Job queue** (Celery/RQ + Redis): audio requests can take 30s–several minutes. A
  synchronous Flask request will time out under a real load balancer or reverse proxy. Swap
  the audio route to enqueue a job and poll/return a job ID instead of blocking.
  `models.analyze_audio_upload()` is already a plain function with no Flask dependency, so it
  drops into a worker task unchanged.
- **A production WSGI server** (gunicorn/uwsgi) instead of Flask's dev server (`debug=True`
  should never run in production).
- **Clearing `separated/` periodically** — it's used as a cache but will grow unbounded.
- **Rate limiting / auth** if this is public-facing, since each request is compute-heavy.
- **HTTPS + a real `SECRET_KEY`** if you add sessions or CSRF protection later.

## Licensing reminder (from the notebook)

Genre model and AcoustID are non-commercial licenses; check Genius's API terms before
commercial use. Demucs (MIT) and Whisper (MIT) have no such restriction.
