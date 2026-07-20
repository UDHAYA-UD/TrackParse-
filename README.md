<div align="center">
  <h1>🎶 Signal</h1>
  <p><b>A Powerful Music Analysis Web App</b></p>
  <p>
    <i>Upload a track or provide a URL to get genre prediction, separated stems, transcriptions, language ID, and more!</i>
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Audio-ee4c2c?logo=pytorch)](https://pytorch.org/)
</div>

---

## ✨ Features

- 🎸 **Genre Prediction**: Classifies tracks into 400 micro-genres using the Essentia Discogs-400 model.
- ✂️ **Vocal Separation**: Uses [Demucs](https://github.com/facebookresearch/demucs) to extract isolated vocals and instrumentals.
- 📝 **Transcription**: Transcribes vocals using OpenAI's [Whisper](https://github.com/openai/whisper).
- 🌍 **Language ID**: Detects language via IndicLID + fastText.
- 🔍 **Song Identification**: Matches audio with [AcoustID](https://acoustid.org/) and fetches lyrics via [Genius](https://genius.com/).
- 🔗 **URL Support**: Download and analyze audio directly from YouTube and other platforms via `yt-dlp`.

---

## ⚡ What changed vs. the original notebook?

- **Models load once**, at app startup, instead of once per notebook cell run.
- **API keys are environment variables**, not hardcoded strings — see `.env.example`. Never commit your real `.env` file!
- **Resilient Pipeline**: Every pipeline function keeps working if another one fails to load. If the genre model files can't download, the app still starts and that route just returns a clear error. Check `/api/health` to see status.
- **Auto-Cleanup**: Uploaded files are deleted after each request so disk usage doesn't grow unbounded. Demucs' separated stems are cached under `separated/`.

---

## 🚀 Setup & Installation

### 1️⃣ System Dependencies
You need **Chromaprint** for AcoustID fingerprinting.
```bash
# Ubuntu/Debian
sudo apt-get install -y libchromaprint-tools

# macOS
brew install chromaprint
```

### 2️⃣ Python Environment
Python 3.10–3.11 is recommended for Essentia/Torch compatibility.
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ API Keys
Copy `.env.example` to `.env` and fill in your free keys:
```bash
cp .env.example .env
```
- **AcoustID**: Get your key [here](https://acoustid.org/new-application)
- **Genius**: Get your token [here](https://genius.com/api-clients)

### 4️⃣ Run the App
```bash
python app.py
```
> **Note:** The first run will download the genre model files, Whisper weights, and IndicLID models. This can take several minutes and requires a few GB of disk space. Subsequent starts are lightning fast! ⚡️

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🏗 Project Layout

```text
music_flask_app/
├── app.py              # 🚀 Flask routes
├── models.py           # 🧠 The analysis pipeline (genre, Demucs, Whisper, etc.)
├── config.py           # ⚙️ Env-based settings
├── requirements.txt    # 📦 Dependencies
├── .env.example        # 🔑 Template for API keys
├── templates/          # 🖥️ HTML templates
├── static/             # 🎨 CSS and JS assets
├── uploads/            # 📁 Temp storage for incoming files (auto-created)
├── separated/          # ✂️ Demucs output cache (auto-created)
└── model_files/        # 📥 Downloaded model weights (cached)
```

---

## ⏱️ Timing Expectations

Vocal separation adds real processing time — a track that used to go straight to Whisper now runs through Demucs first. On a CPU, this is noticeably slower than on a GPU. 

> **Tip:** Uncheck "Separate vocals before transcribing" in the UI (or send `separate_vocals=false` to the API) to skip straight to transcribing the full mix if that latency isn't acceptable for your use case.

---

## 🛠️ Production Notes

This is a **working, synchronous** app — perfect for local use, demos, or low-traffic internal tools. If you plan to deploy it in front of real traffic, consider adding:

- **Job queue (Celery/RQ + Redis)**: Audio requests can take 30s–several minutes. A synchronous Flask request will time out under a real load balancer or reverse proxy. Swap the audio route to enqueue a job and poll/return a job ID instead of blocking. `models.analyze_audio_upload()` is already a plain function with no Flask dependency, so it drops into a worker task unchanged.
- **Production WSGI server**: Use `gunicorn` or `uwsgi` instead of Flask's dev server (`debug=True` should never run in production).
- **Cache Management**: The `separated/` folder will grow unbounded. Implement a cron job to clear it periodically.
- **Rate Limiting / Auth**: Protect your endpoints since each request is compute-heavy.
- **HTTPS & Secure Cookies**: Use HTTPS + a real `SECRET_KEY` if you add sessions or CSRF protection later.

---

## 📜 Licensing Reminder

- **Non-Commercial**: Genre model and AcoustID. Check Genius's API terms before commercial use.
- **MIT License**: Demucs and Whisper have no such restriction.
