"use strict";

const $ = (id) => document.getElementById(id);

const INPUTS = {
  prompt: $("prompt"),
  negative: $("negative"),
  steps: $("steps"),
  guidance: $("guidance"),
  width: $("width"),
  height: $("height"),
  seed: $("seed"),
  count: $("count"),
  scheduler: $("scheduler"),
  preset: $("preset"),
};

const statusEl = $("status");
const infoEl = $("result-info");
const progressEl = $("progress");
const errorEl = $("error");
const previewEl = $("preview");
const historyEl = $("history");
const historyCountEl = $("history-count");
const historyEmptyEl = $("history-empty");
const outputsEl = $("outputs");
const outputsCountEl = $("outputs-count");
const outputsEmptyEl = $("outputs-empty");
const outputsLoading = $("outputs-loading");
const outputsError = $("outputs-error");
const button = $("generate");
const warningEl = document.createElement("div");
warningEl.className = "warning";
warningEl.hidden = true;
previewEl.parentElement.insertBefore(warningEl, previewEl.nextSibling);

const BUTTON_LABEL = "Generate";
const PROGRESS_TIMEOUT_MS = 650000;
let busy = false;              // single in-flight request guard
const history = [];            // session-only, newest first

const PRESET_VALUES = {
  quality: { scheduler: "pndm", steps: 25 },
  balanced: { scheduler: "pndm", steps: 8 },
  fast: { scheduler: "lcm", steps: 8 },
};

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status " + kind;
}

function showError(message) {
  // Server messages are already sanitized (no paths/tracebacks); the display
  // still uses textContent so nothing can be injected into the page.
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.textContent = "";
  errorEl.hidden = true;
}

function showWarning(message) {
  warningEl.textContent = message;
  warningEl.hidden = false;
}

function clearWarning() {
  warningEl.textContent = "";
  warningEl.hidden = true;
}

function readPayload() {
  const seedText = INPUTS.seed.value.trim();
  return {
    prompt: INPUTS.prompt.value.trim(),
    negative_prompt: INPUTS.negative.value.trim(),
    steps: parseInt(INPUTS.steps.value, 10),
    guidance: parseFloat(INPUTS.guidance.value),
    width: parseInt(INPUTS.width.value, 10),
    height: parseInt(INPUTS.height.value, 10),
    seed: seedText === "" ? -1 : parseInt(seedText, 10),
    count: parseInt(INPUTS.count.value, 10),
    scheduler: INPUTS.scheduler.value,
    preset: INPUTS.preset.value || null,
  };
}

function setBusy(value) {
  busy = value;
  button.disabled = value;
  button.textContent = value ? "Generating..." : BUTTON_LABEL;
  button.classList.toggle("working", value);
}

const PROGRESS_POLL_INTERVAL = 500;
let _progressTimer = null;

function startProgressPolling() {
  stopProgressPolling();
  progressEl.hidden = false;
  _pollProgress();
  _progressTimer = setInterval(_pollProgress, PROGRESS_POLL_INTERVAL);
}

function stopProgressPolling() {
  if (_progressTimer) {
    clearInterval(_progressTimer);
    _progressTimer = null;
  }
  progressEl.hidden = true;
  progressEl.textContent = "";
}

async function _pollProgress() {
  try {
    const response = await fetch("/api/progress");
    if (!response.ok) return;
    const data = await response.json();
    if (!data || !data.active) {
      progressEl.hidden = true;
      return;
    }
    const current = Math.min(data.current_step, data.total_steps);
    const total = data.total_steps || 1;
    const pct = Math.min(100, Math.round((current / total) * 100));
    progressEl.innerHTML =
      "Generating... Step " + current + "/" + total +
      ' <span class="progress-bar"><span class="progress-fill" style="width:'
      + pct + '%"></span></span>' + pct + "%";
  } catch {
    // Progress polling best-effort.
  }
}

function showTimeoutError() {
  setStatus("Error", "error");
  showError("Generation timed out. The operation is still running on the server. Try again later.");
  button.focus();
}

function showShutdownError() {
  setStatus("Error", "error");
  showError("Server is shutting down. Try again shortly.");
  button.focus();
}

function metaChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "chip";
  const k = document.createElement("span");
  k.className = "chip-key";
  k.textContent = label;
  const v = document.createElement("span");
  v.className = "chip-val";
  v.textContent = value;
  chip.append(k, v);
  return chip;
}

// Show one image prominently, with its compact metadata block.
function renderPreview(image, settings, totalTime) {
  previewEl.innerHTML = "";
  previewEl.classList.remove("empty");

  const img = document.createElement("img");
  img.src = image.url;
  img.alt = "Generated image (seed " + image.seed + ")";
  img.className = "preview-img";

  const link = document.createElement("a");
  link.href = image.url;
  link.download = image.filename;
  link.className = "preview-link";
  link.textContent = "Open / save PNG";

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.append(
    metaChip("seed", String(image.seed)),
    metaChip("time", totalTime + "s"),
    metaChip("size", settings.width + "x" + settings.height),
    metaChip("steps", String(settings.steps)),
    metaChip("guidance", String(settings.guidance)),
    metaChip("scheduler", String(settings.scheduler)),
    metaChip("preset", settings.preset || "custom"),
  );

  previewEl.append(img, meta, link);
}

function renderHistory() {
  historyCountEl.textContent = String(history.length);
  historyEmptyEl.hidden = history.length > 0;

  for (const item of history) {
    if (item.el) continue;
    const figure = document.createElement("figure");
    figure.className = "history-item";

    const img = document.createElement("img");
    img.src = item.url;
    img.alt = "History image (seed " + item.seed + ")";
    img.loading = "lazy";

    const info = document.createElement("div");
    info.className = "history-info";

    const promptEl = document.createElement("div");
    promptEl.className = "history-prompt";
    promptEl.textContent = item.prompt;

    const chips = document.createElement("div");
    chips.className = "history-chips";
    chips.append(
      metaChip("seed", String(item.seed)),
      metaChip("scheduler", String(item.scheduler)),
      metaChip("steps", String(item.steps)),
    );
    if (item.preset) {
      chips.append(metaChip("preset", item.preset));
    }
    if (item.guidance !== undefined) {
      chips.append(metaChip("guidance", String(item.guidance)));
    }
    if (item.width !== undefined && item.height !== undefined) {
      chips.append(metaChip("size", item.width + "x" + item.height));
    }
    if (item.device !== undefined) {
      chips.append(metaChip("device", item.device));
    }
    if (item.generation_time !== undefined) {
      chips.append(metaChip("time", item.generation_time + "s"));
    }

    const timeEl = document.createElement("div");
    timeEl.className = "history-time";
    timeEl.textContent = formatTime(item.timestamp);

    const reuseBtn = document.createElement("button");
    reuseBtn.type = "button";
    reuseBtn.className = "reuse-btn";
    reuseBtn.textContent = "Reuse Settings";
    reuseBtn.addEventListener("click", () => reuseSettings(item));

    info.append(promptEl, chips, timeEl, reuseBtn);
    figure.append(img, info);
    item.el = figure;
  }

  historyEl.innerHTML = "";
  historyEl.append(historyEmptyEl);
  for (const item of history) {
    historyEl.append(item.el);
  }
}

function formatTime(ts) {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${mo}-${day} ${hh}:${mm}:${ss}`;
}

async function loadHistory() {
  try {
    const response = await fetch("/api/history");
    if (!response.ok) return;
    const items = await response.json();
    history.length = 0;
    for (const item of items) {
      item.el = null;
      history.push(item);
    }
    renderHistory();
  } catch {
    // History is best-effort; a failure must not affect the UI.
  }
}

function reuseSettings(item) {
  INPUTS.prompt.value = item.prompt || "";
  INPUTS.negative.value = item.negative_prompt || "";
  INPUTS.seed.value = String(item.seed);
  INPUTS.scheduler.value = item.scheduler || "pndm";
  INPUTS.steps.value = String(item.steps);
  if (item.preset && PRESET_VALUES[item.preset]) {
    INPUTS.preset.value = item.preset;
  } else {
    INPUTS.preset.value = "";
  }
}

async function loadOutputs() {
  outputsLoading.hidden = false;
  outputsError.hidden = true;
  try {
    const response = await fetch("/api/outputs");
    if (!response.ok) throw new Error("Failed to load outputs");
    const items = await response.json();
    outputsLoading.hidden = true;
    renderOutputs(items);
  } catch {
    outputsLoading.hidden = true;
    outputsError.textContent = "Failed to load outputs.";
    outputsError.hidden = false;
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function renderOutputs(items) {
  outputsCountEl.textContent = String(items.length);
  outputsEmptyEl.hidden = items.length > 0;

  for (const item of items) {
    if (item.el) continue;
    const card = document.createElement("div");
    card.className = "output-card";

    const img = document.createElement("img");
    img.src = item.url;
    img.alt = "Output " + item.filename;
    img.className = "output-thumb";
    img.loading = "lazy";

    const info = document.createElement("div");
    info.className = "output-info";

    const nameEl = document.createElement("div");
    nameEl.className = "output-name";
    nameEl.textContent = item.filename;

    const metaEl = document.createElement("div");
    metaEl.className = "output-meta";
    metaEl.textContent = formatFileSize(item.size || 0);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "delete-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.setAttribute("aria-label", "Delete " + item.filename);
    deleteBtn.addEventListener("click", () => deleteOutput(item.filename));

    info.append(nameEl, metaEl, deleteBtn);
    card.append(img, info);
    item.el = card;
  }

  outputsEl.innerHTML = "";
  outputsEl.append(outputsEmptyEl);
  for (const item of items) {
    outputsEl.append(item.el);
  }
}

async function deleteOutput(filename) {
  const confirmed = confirm("Delete " + filename + "?");
  if (!confirmed) return;

  try {
    const response = await fetch("/api/outputs/" + encodeURIComponent(filename), {
      method: "DELETE",
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Failed to delete");
    }
    await loadOutputs();
  } catch (err) {
    outputsError.textContent = err && err.message ? err.message : "Failed to delete output.";
    outputsError.hidden = false;
  }
}

async function generate() {
  if (busy) return;                 // hard guard against double submits

  const payload = readPayload();
  if (!payload.prompt) {
    showError("Please enter a prompt.");
    setStatus("Ready", "ready");
    return;
  }

  clearError();
  clearWarning();
  infoEl.textContent = "";

  if (payload.scheduler === "lcm" && payload.steps < 8) {
    showWarning("LCM with fewer than 8 steps may reduce image quality. 8 steps is recommended.");
  }

  setBusy(true);
  setStatus("Generating...", "working");
  startProgressPolling();

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), PROGRESS_TIMEOUT_MS);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    let data = null;
    try {
      data = await response.json();
    } catch (parseErr) {
      data = null;
    }

    if (response.status === 504) {
      showTimeoutError();
      return;
    }
    if (response.status === 503) {
      showShutdownError();
      return;
    }
    if (!response.ok || !data || data.success !== true) {
      const message = (data && data.error) || ("Request failed (HTTP " + response.status + ").");
      throw new Error(message);
    }

    const first = data.images[0];
    renderPreview(first, payload, data.total_time);
    loadHistory();
    loadOutputs();

    const seeds = data.images.map((im) => im.seed).join(", ");
    infoEl.textContent =
      data.images.length + " image(s) in " + data.total_time + "s" +
      " · device: " + data.device +
      " · seed(s): " + seeds;
    setStatus("Completed", "done");
  } catch (err) {
    if (err && err.name === "AbortError") {
      showTimeoutError();
    } else {
      setStatus("Error", "error");
      showError(err && err.message ? err.message : "Something went wrong. Please try again.");
    }
    button.focus();
  } finally {
    stopProgressPolling();
    setBusy(false);
  }
}

INPUTS.preset.addEventListener("change", () => {
  const preset = INPUTS.preset.value;
  if (preset && PRESET_VALUES[preset]) {
    INPUTS.scheduler.value = PRESET_VALUES[preset].scheduler;
    INPUTS.steps.value = PRESET_VALUES[preset].steps;
    clearWarning();
  }
});

INPUTS.scheduler.addEventListener("change", () => {
  INPUTS.preset.value = "";
  clearWarning();
});

INPUTS.steps.addEventListener("input", () => {
  INPUTS.preset.value = "";
  clearWarning();
});

button.addEventListener("click", generate);

loadHistory();
loadOutputs();