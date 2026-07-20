"""
Flask web app for the music analysis pipeline.

Routes:
    GET  /                        -> the upload page
    POST /api/analyze/audio       -> upload an audio file, get genre + lyrics + language + song ID
    POST /api/analyze/lyrics      -> submit pasted lyrics, get language + Genius song matches
    GET  /api/health              -> which models loaded OK / which didn't, and API key status

Run with:
    python app.py
"""

import time
import uuid

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import config
import models


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024

    # Load every model once, at process startup, instead of once per request.
    # This can take a while the first time (model downloads) -- subsequent
    # starts are fast since files are cached under model_files/.
    print("Loading models... this can take a few minutes on first run.")
    models.load_models()
    print("Model loading complete. MODEL_LOAD_ERRORS:", models.MODEL_LOAD_ERRORS or "none")

    def allowed_audio_file(filename):
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_AUDIO_EXTENSIONS
        )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok" if not models.MODEL_LOAD_ERRORS else "degraded",
            "model_errors": models.MODEL_LOAD_ERRORS,
        })

    @app.route("/api/analyze/audio", methods=["POST"])
    def analyze_audio_route():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file included in the request."}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not allowed_audio_file(file.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_AUDIO_EXTENSIONS))}"
            }), 400

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = config.UPLOAD_FOLDER / unique_name
        file.save(save_path)

        # Form field "separate_vocals" defaults to true; pass "false" to skip Demucs
        # and transcribe the full mix directly (faster, per the notebook's escape hatch).
        separate = request.form.get("separate_vocals", "true").lower() != "false"

        try:
            start = time.time()
            result = models.analyze_audio_upload(
                str(save_path), separate_before_transcribing=separate
            )
            result["processing_seconds"] = round(time.time() - start, 1)
            return jsonify(result)
        except Exception as e:
            app.logger.exception("Audio analysis failed")
            return jsonify({"error": f"Analysis failed: {e}"}), 500
        finally:
            # Clean up the uploaded file so disk usage doesn't grow unbounded.
            # (Separated stems in separated/ are left in place as a cache; see README
            # for a note on periodically clearing that folder in production.)
            try:
                save_path.unlink(missing_ok=True)
            except OSError:
                pass

    @app.route("/api/analyze/url", methods=["POST"])
    def analyze_url_route():
        data = request.get_json(silent=True) or request.form
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "No URL provided."}), 400

        unique_name = f"{uuid.uuid4().hex}"
        save_path = config.UPLOAD_FOLDER / unique_name

        try:
            start = time.time()
            # Download audio using yt-dlp
            download_result = models.download_audio_from_url(url, str(save_path))
            
            if isinstance(download_result, dict) and "error" in download_result:
                return jsonify(download_result), 400
                
            downloaded_file = download_result

            # Run analysis
            separate = data.get("separate_vocals", "true").lower() != "false"
            result = models.analyze_audio_upload(
                downloaded_file, separate_before_transcribing=separate
            )
            result["processing_seconds"] = round(time.time() - start, 1)
            return jsonify(result)
        except Exception as e:
            app.logger.exception("URL analysis failed")
            return jsonify({"error": f"Analysis failed: {e}"}), 500
        finally:
            # Clean up the downloaded file
            try:
                import os
                if 'downloaded_file' in locals() and os.path.exists(downloaded_file):
                    os.remove(downloaded_file)
            except OSError:
                pass

    @app.route("/api/analyze/lyrics", methods=["POST"])
    def analyze_lyrics_route():
        data = request.get_json(silent=True) or request.form
        text = (data.get("lyrics") or "").strip()
        if not text:
            return jsonify({"error": "No lyrics text provided."}), 400

        try:
            result = models.analyze_lyrics_upload(text)
            return jsonify(result)
        except Exception as e:
            app.logger.exception("Lyrics analysis failed")
            return jsonify({"error": f"Analysis failed: {e}"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    # debug=True is convenient locally but should be off in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
