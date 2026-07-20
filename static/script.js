// --- Tab switching ---
const tabs = document.querySelectorAll(".tab");
const panels = { audio: document.getElementById("panel-audio"), lyrics: document.getElementById("panel-lyrics") };

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    Object.values(panels).forEach((p) => p.classList.remove("active"));
    panels[tab.dataset.tab].classList.add("active");
  });
});

// --- Health check ---
fetch("/api/health")
  .then((r) => r.json())
  .then((data) => {
    const el = document.getElementById("health-status");
    if (data.status === "ok") {
      el.textContent = "all models ready";
      el.classList.add("ok");
    } else {
      const issues = Object.keys(data.model_errors).join(", ");
      el.textContent = `degraded: ${issues}`;
      el.classList.add("degraded");
    }
  })
  .catch(() => {
    document.getElementById("health-status").textContent = "unable to reach server";
  });

// --- Audio upload flow ---
const dropzone = document.getElementById("dropzone");
const audioInput = document.getElementById("audio-input");
const dropzoneLabel = document.getElementById("dropzone-label");
const waveform = dropzone.querySelector(".waveform");
const audioSubmit = document.getElementById("audio-submit");
const audioForm = document.getElementById("audio-form");

["dragover", "dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.toggle("drag-over", evt === "dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    audioInput.files = e.dataTransfer.files;
    handleFileChosen(file);
  }
});
audioInput.addEventListener("change", () => {
  if (audioInput.files[0]) handleFileChosen(audioInput.files[0]);
});

function handleFileChosen(file) {
  dropzoneLabel.textContent = file.name;
  audioSubmit.disabled = false;
}

audioForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = audioInput.files[0];
  if (!file) return;

  const separate = document.getElementById("separate-vocals").checked;
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("separate_vocals", separate ? "true" : "false");

  const loadingEl = document.getElementById("audio-loading");
  const loadingText = document.getElementById("audio-loading-text");
  const resultsEl = document.getElementById("audio-results");
  const errorEl = document.getElementById("audio-error");

  loadingText.textContent = separate
    ? "Running genre, vocal separation, transcription, language, song ID…"
    : "Running genre, transcription, language, song ID…";

  audioSubmit.disabled = true;
  waveform.classList.add("active");
  loadingEl.hidden = false;
  resultsEl.hidden = true;
  errorEl.hidden = true;

  try {
    const resp = await fetch("/api/analyze/audio", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Analysis failed.");
    renderAudioResults(data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    loadingEl.hidden = true;
    waveform.classList.remove("active");
    audioSubmit.disabled = false;
  }
});

function renderAudioResults(data) {
  const el = document.getElementById("audio-results");
  let html = "";

  html += `<div class="meta-line">processed in ${data.processing_seconds}s ${data.vocals_separated ? "· vocals separated" : "· full mix transcribed"}</div>`;

  html += `<div class="result-block"><div class="result-label">Genre</div>`;
  if (Array.isArray(data.genre)) {
    data.genre.forEach(([label, score]) => {
      const pct = (score * 100).toFixed(1);
      html += `<div class="bar-row"><span class="name">${escapeHtml(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span class="bar-pct">${pct}%</span></div>`;
    });
  } else {
    html += `<div class="meta-line">${escapeHtml(data.genre.error || "unavailable")}</div>`;
  }
  html += `</div>`;

  html += `<div class="result-block"><div class="result-label">Transcribed lyrics</div>`;
  html += `<div class="lyrics-box">${data.transcribed_lyrics ? escapeHtml(data.transcribed_lyrics) : "(no speech detected)"}</div></div>`;

  html += `<div class="result-block"><div class="result-label">Language</div>`;
  data.language.forEach(([lang, score, source]) => {
    html += `<div class="bar-row"><span class="name">${escapeHtml(lang)}</span><span class="bar-pct">${(score * 100).toFixed(1)}%</span><span class="meta-line" style="margin:0">via ${escapeHtml(source)}</span></div>`;
  });
  html += `</div>`;

  html += `<div class="result-block"><div class="result-label">Song ID</div>`;
  const song = data.song_id;
  if (!song) {
    html += `<div class="meta-line">No match found in AcoustID's database.</div>`;
  } else if (song.error) {
    html += `<div class="meta-line">${escapeHtml(song.error)}</div>`;
  } else {
    html += `<div class="song-card"><div class="title">${escapeHtml(song.title)}</div><div class="artist">${escapeHtml(song.artist)} — ${(song.confidence * 100).toFixed(1)}% confidence</div></div>`;
  }
  html += `</div>`;

  el.innerHTML = html;
  el.hidden = false;
}

// --- Lyrics-only flow ---
const lyricsForm = document.getElementById("lyrics-form");
lyricsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = document.getElementById("lyrics-input").value.trim();
  if (!text) return;

  const loadingEl = document.getElementById("lyrics-loading");
  const resultsEl = document.getElementById("lyrics-results");
  const errorEl = document.getElementById("lyrics-error");

  loadingEl.hidden = false;
  resultsEl.hidden = true;
  errorEl.hidden = true;

  try {
    const resp = await fetch("/api/analyze/lyrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lyrics: text }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Analysis failed.");
    renderLyricsResults(data);
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    loadingEl.hidden = true;
  }
});

function renderLyricsResults(data) {
  const el = document.getElementById("lyrics-results");
  let html = "";

  html += `<div class="result-block"><div class="result-label">Language</div>`;
  data.language.forEach(([lang, score, source]) => {
    html += `<div class="bar-row"><span class="name">${escapeHtml(lang)}</span><span class="bar-pct">${(score * 100).toFixed(1)}%</span><span class="meta-line" style="margin:0">via ${escapeHtml(source)}</span></div>`;
  });
  html += `</div>`;

  html += `<div class="result-block"><div class="result-label">Song matches (Genius)</div>`;
  const matches = data.song_matches;
  if (!matches) {
    html += `<div class="meta-line">No matches found — try a more distinctive lyric fragment.</div>`;
  } else if (matches.error) {
    html += `<div class="meta-line">${escapeHtml(matches.error)}</div>`;
  } else {
    matches.forEach((m) => {
      html += `<div class="song-card"><div class="title">${escapeHtml(m.title)}</div><div class="artist">${escapeHtml(m.artist)}</div><a href="${m.url}" target="_blank" rel="noopener">view on genius →</a></div>`;
    });
  }
  html += `</div>`;

  el.innerHTML = html;
  el.hidden = false;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
