"""
Configuration for the music analysis Flask app.

API keys are read from environment variables instead of being hardcoded, so you
can rotate them without touching code and never commit secrets to source control.

Set them before running the app, e.g. on Linux/Mac:
    export ACOUSTID_API_KEY="your_key_here"
    export GENIUS_ACCESS_TOKEN="your_token_here"

Or put them in a `.env` file (see .env.example) and load it with python-dotenv
(already wired up below).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads a .env file in the project root if present, no-op otherwise

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
SEPARATED_FOLDER = BASE_DIR / "separated"

UPLOAD_FOLDER.mkdir(exist_ok=True)
SEPARATED_FOLDER.mkdir(exist_ok=True)

ACOUSTID_API_KEY = os.environ.get("ACOUSTID_API_KEY", "")
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN", "")

ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "webm"}
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "50"))

# Whisper / Demucs model choice — same "start fast, upgrade later" knobs as the notebook.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-base")
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
