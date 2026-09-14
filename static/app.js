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
};

const statusEl = $("status");
const infoEl = $("result-info");
const errorEl = $("error");
const previewEl = $("preview");
const historyEl = $("history");
const historyCountEl = $("history-count");
const historyEmptyEl = $("history-empty");
const button = $("generate");
const warningEl = document.createElement("div");
warningEl.className = "warning";
warningEl.hidden = true;
previewEl.parentElement.insertBefore(warningEl, previewEl.nextSibling);

const BUTTON_LABEL = "Generate";
let busy = false;              // single in-flight request guard
const history = [];            // session-only, newest first

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
    negative: INPUTS.negative.value.trim(),
    steps: parseInt(INPUTS.steps.value, 10),
    guidance: parseFloat(INPUTS.guidance.value),
    width: parseInt(INPUTS.width.value, 10),
    height: parseInt(INPUTS.height.value, 10),
    seed: seedText === "" ? null : parseInt(seedText, 10),
    count: parseInt(INPUTS.count.value, 10),
    scheduler: INPUTS.scheduler.value,
  };
}

function setBusy(value) {
  busy = value;
  button.disabled = value;
  button.textContent = value ? "Generating..." : BUTTON_LABEL;
  button.classList.toggle("working", value);
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
  );

  previewEl.append(img, meta, link);
}

function renderHistory() {
  historyCountEl.textContent = String(history.length);
  historyEmptyEl.hidden = history.length > 0;

  // Build each entry once: plain <img> + seed label (never innerHTML input).
  for (const item of history) {
    if (item.el) continue;
    const figure = document.createElement("figure");
    figure.className = "history-item";

    const img = document.createElement("img");
    img.src = item.url;
    img.alt = "History image (seed " + item.seed + ")";
    img.loading = "lazy";

    const caption = document.createElement("figcaption");
    caption.textContent = "seed " + item.seed;

    figure.append(img, caption);
    item.el = figure;
  }

  historyEl.innerHTML = "";
  historyEl.append(historyEmptyEl);
  for (const item of history) {
    historyEl.append(item.el);
  }
}

function addHistory(images) {
  // Newest first; the batch itself is reversed so count=2 reads 42 then 43.
  for (let i = images.length - 1; i >= 0; i--) {
    history.unshift({ url: images[i].url, seed: images[i].seed, el: null });
  }
  renderHistory();
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

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data = null;
    try {
      data = await response.json();
    } catch (parseErr) {
      data = null;
    }

    if (!response.ok || !data || data.success !== true) {
      const message = (data && data.error) || ("Request failed (HTTP " + response.status + ").");
      throw new Error(message);
    }

    const first = data.images[0];
    renderPreview(first, payload, data.total_time);
    addHistory(data.images);

    const seeds = data.images.map((im) => im.seed).join(", ");
    infoEl.textContent =
      data.images.length + " image(s) in " + data.total_time + "s" +
      " · device: " + data.device +
      " · seed(s): " + seeds;
    setStatus("Completed", "done");
  } catch (err) {
    setStatus("Error", "error");
    showError(err && err.message ? err.message : "Something went wrong. Please try again.");
  } finally {
    setBusy(false);
  }
}

button.addEventListener("click", generate);