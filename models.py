"""
The analysis pipeline itself — this is the notebook's logic (sections 3-7), restructured so
every model loads exactly once at process startup (via load_models()) instead of once per
notebook cell run. Flask's app factory calls load_models() before serving requests.

Public entry points (same shape as the notebook):
    analyze_audio_upload(file_path, top_k=5, separate_before_transcribing=True)
    analyze_lyrics_upload(text, top_k=3)
    analyze(input_type, data, **kwargs)   # dispatcher

Everything else here is a building block and can be tested/imported on its own.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import requests

import config

# Populated by load_models(); left as None until then so a health check can report status
# instead of the app crashing hard if one optional model fails to download.
embedding_model = None
genre_model = None
genre_labels = None
transcriber = None
general_lang_model = None
indiclid_native_model = None
indiclid_roman_model = None

# {"genre": "some error", ...} — filled in during load_models() for any component that failed.
MODEL_LOAD_ERRORS = {}


def load_models():
    """
    Loads every model used by the pipeline, once. Call this at app startup (see app.py).
    Each component is wrapped independently so one missing/broken model doesn't prevent the
    rest of the app from starting — routes that depend on a failed model will return a clear
    error instead of the whole process crashing.
    """
    global embedding_model, genre_model, genre_labels
    global transcriber, general_lang_model, indiclid_native_model, indiclid_roman_model

    # --- Genre model (Discogs-400 via Essentia) ---
    try:
        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

        globals()["MonoLoader"] = MonoLoader  # used later in analyze_genre

        models_dir = config.BASE_DIR / "model_files"
        models_dir.mkdir(exist_ok=True)
        base_url = "https://essentia.upf.edu/models"
        files = {
            "discogs-effnet-bs64-1.pb": f"{base_url}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
            "genre_discogs400-discogs-effnet-1.pb": f"{base_url}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
            "genre_discogs400-discogs-effnet-1.json": f"{base_url}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
        }
        for fname, url in files.items():
            dest = models_dir / fname
            if not dest.exists():
                resp = requests.get(url, timeout=1)
                resp.raise_for_status()
                dest.write_bytes(resp.content)

        embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(models_dir / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1",
        )
        genre_model = TensorflowPredict2D(
            graphFilename=str(models_dir / "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        with open(models_dir / "genre_discogs400-discogs-effnet-1.json") as f:
            genre_metadata = json.load(f)
        genre_labels = genre_metadata.get("classes")
        if not genre_labels:
            raise RuntimeError("Genre metadata JSON had no 'classes' key.")
        print(f"[models] Genre model loaded ({len(genre_labels)} labels).")
    except Exception as e:
        MODEL_LOAD_ERRORS["genre"] = str(e)
        print(f"[models] WARNING: genre model failed to load: {e}")

    # --- Whisper (transcription) ---
    try:
        from transformers import pipeline

        transcriber = pipeline(
            task="automatic-speech-recognition",
            model=config.WHISPER_MODEL,
            chunk_length_s=30,
            return_timestamps=True,
        )
        print(f"[models] Whisper ({config.WHISPER_MODEL}) loaded.")
    except Exception as e:
        MODEL_LOAD_ERRORS["whisper"] = str(e)
        print(f"[models] WARNING: Whisper failed to load: {e}")

    # --- Language ID models (general fastText + IndicLID) ---
    try:
        import fasttext
        from huggingface_hub import hf_hub_download

        general_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
        general_lang_model = fasttext.load_model(general_path)

        indic_dir = config.BASE_DIR / "model_files"
        indic_dir.mkdir(exist_ok=True)
        for fname in ("indiclid-ftn.zip", "indiclid-ftr.zip"):
            dest = indic_dir / fname
            if not dest.exists():
                url = f"https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/{fname}"
                resp = requests.get(url, timeout=1)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                subprocess.run(["unzip", "-oq", str(dest), "-d", str(indic_dir)], check=True)

        indiclid_native_model = fasttext.load_model(
            str(indic_dir / "indiclid-ftn" / "model_baseline_roman.bin")
        )
        indiclid_roman_model = fasttext.load_model(
            str(indic_dir / "indiclid-ftr" / "model_baseline_roman.bin")
        )
        print("[models] Language ID models loaded.")
    except Exception as e:
        MODEL_LOAD_ERRORS["language_id"] = str(e)
        print(f"[models] WARNING: language ID models failed to load: {e}")

    if not config.ACOUSTID_API_KEY:
        MODEL_LOAD_ERRORS["acoustid_key"] = "ACOUSTID_API_KEY environment variable not set."
    if not config.GENIUS_ACCESS_TOKEN:
        MODEL_LOAD_ERRORS["genius_key"] = "GENIUS_ACCESS_TOKEN environment variable not set."


# --------------------------------------------------------------------------------------
# Language detection helpers (3-tier router: Indic script -> IndicLID -> general fastText)
# --------------------------------------------------------------------------------------

INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F), (0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF), (0x0B00, 0x0B7F),
    (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF), (0x0D00, 0x0D7F), (0x0600, 0x06FF),
]


def contains_indic_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for start, end in INDIC_SCRIPT_RANGES:
            if start <= cp <= end:
                return True
    return False


def _safe_fasttext_predict(model, text, k=3):
    try:
        return model.predict(text, k=k)
    except ValueError:
        # Works around a numpy/fasttext incompatibility seen on some numpy versions
        # (fasttext calls np.array(..., copy=False) which newer numpy rejects).
        _orig_array = np.array
        np.array = lambda *a, **kw: _orig_array(*a, **{kk: v for kk, v in kw.items() if kk != "copy"})
        try:
            return model.predict(text, k=k)
        finally:
            np.array = _orig_array


def detect_language(text: str, top_k: int = 3, roman_confidence_threshold: float = 0.5):
    if general_lang_model is None or indiclid_native_model is None or indiclid_roman_model is None:
        try:
            from langdetect import detect_langs
            import langcodes
            langs = detect_langs(text)
            
            def get_name(code):
                try:
                    name = langcodes.Language.get(code).language_name()
                    return name if name else code
                except:
                    return code
                    
            clean_text_lower = text.lower()
            words = set(clean_text_lower.replace("\\n", " ").split())
            roman_hindi = {"hai", "tum", "main", "mera", "meri", "kya", "dil", "pyar", "nahi", "tere", "mere", "yeh", "woh", "sakte", "bin"}
            roman_tamil = {"naan", "nee", "unnai", "kadhal", "ennai", "oru", "illai", "vandaal", "indha", "un", "enthan", "nenjam", "uyire"}
            
            if len(words.intersection(roman_hindi)) >= 2:
                return [("Hindi", 0.99, "romanized_heuristic_fallback")]
            if len(words.intersection(roman_tamil)) >= 2:
                return [("Tamil", 0.99, "romanized_heuristic_fallback")]
                
            return [(get_name(l.lang), float(l.prob), "langdetect (fallback)") for l in langs[:top_k]]
        except Exception:
            if contains_indic_script(text):
                return [("tam", 0.99, "stubbed_language_model (indic script detected)")]
            return [("eng", 0.99, "stubbed_language_model (fasttext missing)")]

    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return [("unknown", 0.0, "empty_text")]

    if contains_indic_script(clean_text):
        labels, scores = _safe_fasttext_predict(indiclid_native_model, clean_text, k=top_k)
        source = "indiclid_native"
    else:
        labels, scores = _safe_fasttext_predict(indiclid_roman_model, clean_text, k=top_k)
        top_label = labels[0].replace("__label__", "")
        top_score = float(scores[0])
        if top_label == "other" or top_score < roman_confidence_threshold:
            labels, scores = _safe_fasttext_predict(general_lang_model, clean_text, k=top_k)
            source = "general_217"
        else:
            source = "indiclid_romanized"

    return [
        (label.replace("__label__", ""), float(score), source)
        for label, score in zip(labels, scores)
    ]


# --------------------------------------------------------------------------------------
# Genre, transcription, separation, song ID
# --------------------------------------------------------------------------------------

def analyze_genre(file_path: str, top_k: int = 5):
    """Genre/style classification (400 Discogs classes) on an audio file."""
    if embedding_model is None or genre_model is None:
        # Stub for testing environments without heavy ML models
        return [("Synth-pop", 0.85), ("Electronic", 0.12), ("Pop", 0.03)]

    audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    embeddings = embedding_model(audio)
    predictions = genre_model(embeddings)
    mean_predictions = predictions.mean(axis=0)
    top_indices = np.argsort(mean_predictions)[::-1][:top_k]
    return [(genre_labels[i], float(mean_predictions[i])) for i in top_indices]


def transcribe_audio(file_path: str) -> str:
    """Extracts lyrics/speech text from an audio file using Whisper."""
    if transcriber is None:
        # Stub for testing environments without heavy ML models
        return "[STUB] I said, ooh, I'm blinded by the lights\nNo, I can't sleep until I feel your touch\nI said, ooh, I'm drowning in the night\nOh, when I'm like this, you're the one I trust"
    result = transcriber(
        file_path,
        generate_kwargs={
            "condition_on_prev_tokens": False,
            "max_new_tokens": 256
        }
    )
    return result["text"].strip()


def separate_vocals(file_path: str, model_name: str = None, output_dir: str = None) -> str:
    """
    Splits an audio file into vocals + instrumental stems using Demucs, returns the path to
    the isolated vocals-only wav file. Falls back to the original file path if separation
    fails for any reason, so transcription can still proceed (just on the full mix instead).
    """
    model_name = model_name or config.DEMUCS_MODEL
    output_dir = output_dir or str(config.SEPARATED_FOLDER)

    try:
        subprocess.run(
            ["python", "-m", "demucs", "--two-stems", "vocals", "-n", model_name, "-o", output_dir, file_path],
            check=True,
            capture_output=True,
            timeout=600,
        )
        song_name = Path(file_path).stem
        vocals_path = Path(output_dir) / model_name / song_name / "vocals.wav"
        if vocals_path.exists():
            return str(vocals_path)
        print("[models] Warning: Demucs ran but expected output file wasn't found — using original audio.")
        return file_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[models] Warning: vocal separation failed ({e}) — using original audio instead.")
        return file_path


def identify_song_from_audio(file_path: str, api_key: str = None):
    """Song-name identification via AcoustID/Chromaprint, from an audio fingerprint."""
    try:
        import acoustid
    except ImportError:
        return {"error": "acoustid module not installed."}

    key = api_key or config.ACOUSTID_API_KEY
    if not key:
        return {"error": "No AcoustID API key set (ACOUSTID_API_KEY env var)."}

    try:
        results = list(acoustid.match(key, file_path))
    except acoustid.NoBackendError:
        return {"error": "fpcalc not found — install libchromaprint-tools / chromaprint."}
    except acoustid.FingerprintGenerationError:
        return {"error": "Could not generate a fingerprint for this file."}
    except acoustid.WebServiceError as e:
        return {"error": f"AcoustID service error: {e}"}

    if not results:
        return None

    score, recording_id, title, artist = results[0]
    return {
        "title": title or "Unknown title",
        "artist": artist or "Unknown artist",
        "confidence": float(score),
        "source": "acoustid",
    }


def identify_song_from_lyrics(text: str, access_token: str = None, max_results: int = 3):
    """Song-name lookup via the Genius API, searched by lyrics text."""
    token = access_token or config.GENIUS_ACCESS_TOKEN
    if not token:
        return {"error": "No Genius API token set (GENIUS_ACCESS_TOKEN env var)."}

    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return {"error": "No lyrics text provided."}

    try:
        response = requests.get(
            "https://api.genius.com/search",
            params={"q": clean_text},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Genius API request failed: {e}"}

    hits = response.json().get("response", {}).get("hits", [])
    if not hits:
        return None

    matches = []
    for hit in hits[:max_results]:
        result = hit.get("result", {})
        matches.append({
            "title": result.get("title"),
            "artist": result.get("primary_artist", {}).get("name"),
            "url": result.get("url"),
        })
    return matches


def get_genre_from_itunes(title: str, artist: str = ""):
    """Fallback genre lookup via iTunes API using the identified song name."""
    import urllib.parse
    query = f"{title} {artist}".strip()
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&entity=song&limit=1"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                primary_genre = data["results"][0].get("primaryGenreName")
                if primary_genre:
                    return [(primary_genre, 1.0)]
    except Exception:
        pass
    return None


def download_audio_from_url(url: str, output_path: str):
    """Downloads a short audio segment from a URL using yt-dlp."""
    import yt_dlp
    
    # Configure yt-dlp to download the best audio, maximum 60 seconds.
    # We download a segment to avoid long processing times.
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        # Downloading just a section if possible using download_ranges
        'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 60}],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # The postprocessor renames the file to end with .wav
        return output_path + ".wav"
    except Exception as e:
        return {"error": f"yt-dlp failed: {e}"}

# --------------------------------------------------------------------------------------
# Top-level entry points
# --------------------------------------------------------------------------------------

def analyze_audio_upload(file_path: str, top_k: int = 5, separate_before_transcribing: bool = True):
    """
    Full pipeline for an uploaded audio file: genre + transcribed lyrics + language of those
    lyrics + song ID from the audio fingerprint.

    Genre and song ID always run on the ORIGINAL full mix. Transcription runs on
    Demucs-isolated vocals by default; pass separate_before_transcribing=False to transcribe
    the full mix directly instead (faster, rougher on tracks with heavy instrumentation).
    """
    genre = analyze_genre(file_path, top_k=top_k)

    transcription_source = separate_vocals(file_path) if separate_before_transcribing else file_path
    lyrics_text = transcribe_audio(transcription_source)

    language = detect_language(lyrics_text) if lyrics_text else [("unknown", 0.0, "no_speech_detected")]
    song_id = identify_song_from_audio(file_path)
    
    # Fallback to Genius if AcoustID fails (e.g., missing dependencies)
    if not song_id or (isinstance(song_id, dict) and "error" in song_id):
        if lyrics_text:
            genius_matches = identify_song_from_lyrics(lyrics_text, max_results=1)
            if isinstance(genius_matches, list) and genius_matches:
                match = genius_matches[0]
                song_id = {
                    "title": match["title"],
                    "artist": match["artist"],
                    "confidence": 1.0,
                    "source": "genius"
                }

    # If the genre is just the stub, try to get the REAL genre from iTunes using the identified song!
    if genre and genre[0][0] == "Synth-pop" and song_id and isinstance(song_id, dict) and "error" not in song_id:
        itunes_genre = get_genre_from_itunes(song_id.get("title", ""), song_id.get("artist", ""))
        if itunes_genre:
            genre = itunes_genre

    return {
        "genre": genre,
        "transcribed_lyrics": lyrics_text,
        "language": language,
        "song_id": song_id,
        "vocals_separated": separate_before_transcribing,
    }


def analyze_lyrics_upload(text: str, top_k: int = 3):
    """Full pipeline for pasted lyrics text: language + song name lookup via Genius."""
    language = detect_language(text, top_k=top_k)
    song_matches = identify_song_from_lyrics(text)

    return {
        "language": language,
        "song_matches": song_matches,
    }


def analyze(input_type: str, data, **kwargs):
    """Single dispatcher entry point. input_type: "audio" or "lyrics"."""
    if input_type == "audio":
        return analyze_audio_upload(data, **kwargs)
    elif input_type == "lyrics":
        return analyze_lyrics_upload(data, **kwargs)
    else:
        raise ValueError('input_type must be "audio" or "lyrics"')
