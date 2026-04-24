function isLikelyChatModel(modelId) {
  const lower = String(modelId || "").toLowerCase();
  const deny = ["embedding", "rerank", "reranker", "whisper", "asr", "tts", "speech"];
  return !deny.some((k) => lower.includes(k));
}

async function fetchJson(url, init = undefined) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function postJson(url, payload) {
  return fetchJson(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function debounce(fn, waitMs) {
  let timer = null;
  return (...args) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  };
}

function emptyJobCard(title, detail) {
  return `
    <div class="job-empty">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(detail)}</span>
    </div>
  `;
}

function buildModelOptions({ include = () => true, loadedLabel = "loaded", installedLabel = "installed" } = {}) {
  const options = [];
  const seen = new Set();
  for (const modelId of state.runtimeLoadedModels) {
    if (!modelId || seen.has(modelId) || !include(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, src: "loaded", label: `${modelId} (${loadedLabel})` });
  }
  for (const item of state.installed) {
    const modelId = item?.model_id;
    if (!modelId || seen.has(modelId) || !include(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, src: "installed", label: `${modelId} (${installedLabel})` });
  }
  return options;
}

function renderSelectOptions(select, options, selectedValue, placeholder) {
  const selected = options.some((opt) => opt.id === selectedValue) ? selectedValue : "";
  const placeholderHtml = `<option value="" ${selected ? "" : "selected"}>${placeholder}</option>`;
  const optionHtml = options
    .map((opt) => `<option value="${opt.id}" ${opt.id === selected ? "selected" : ""}>${opt.label || opt.id}</option>`)
    .join("");
  select.innerHTML = placeholderHtml + optionHtml;
  return selected;
}

function renderTrainProgress() {
  const wrap = el("train-progress-wrap");
  const fill = el("train-progress-fill");
  const text = el("train-progress-text");
  const progress = Math.max(0, Math.min(100, Math.round(state.trainProgress || 0)));
  const visible = state.trainRunning || progress > 0;

  if (wrap) wrap.style.display = visible ? "flex" : "none";
  if (fill) fill.style.width = `${progress}%`;
  if (text) text.textContent = `${progress}%`;
}

function stopTrainProgressTimer() {
  if (trainProgressTimer) {
    clearInterval(trainProgressTimer);
    trainProgressTimer = null;
  }
}

function beginTrainProgress() {
  stopTrainProgressTimer();
  state.trainProgress = Math.max(1, state.trainProgress || 0);
  renderTrainProgress();
  trainProgressTimer = setInterval(() => {
    if (!state.trainRunning) return;
    const step = Math.max(1, Math.floor(Math.random() * 4));
    state.trainProgress = Math.min(92, state.trainProgress + step);
    renderTrainProgress();
  }, 220);
}

function completeTrainProgress() {
  stopTrainProgressTimer();
  state.trainProgress = 100;
  renderTrainProgress();
  setTimeout(() => {
    if (state.trainRunning) return;
    state.trainProgress = 0;
    renderTrainProgress();
  }, 700);
}

function setTrainGuideModalOpen(open) {
  const modal = el("train-guide-modal");
  if (!(modal instanceof HTMLDivElement)) return;
  modal.hidden = !open;
}

function setTrainSaveModalOpen(open) {
  const modal = el("train-save-modal");
  if (!(modal instanceof HTMLDivElement)) return;
  if (!open) {
    state.saveTrainTaskId = "";
    state.saveTrainProfile = "";
    state.saveTrainStatus = "";
    state.saveTrainRunning = false;
  }
  modal.hidden = !open;
  renderTrainSaveModal();
}

function renderTrainSaveModal() {
  const status = el("train-save-status");
  const confirm = el("train-save-confirm");
  const input = el("train-save-path");
  if (status) {
    status.textContent = state.saveTrainStatus || "";
    status.style.display = state.saveTrainStatus ? "inline-flex" : "none";
  }
  if (confirm instanceof HTMLButtonElement) {
    confirm.disabled = state.saveTrainRunning;
    confirm.textContent = state.saveTrainRunning ? "Saving..." : "Save";
  }
  if (input instanceof HTMLInputElement) {
    input.disabled = state.saveTrainRunning;
  }
}

function defaultSavePathForTask(task) {
  const profile = String(task?.profile || state.selectedTrainProfile || "profile");
  const slug = String(profile || "profile")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
  return `./exports/${slug || "profile"}-merged`;
}

function openTrainSaveModal(taskId, profileHint = "") {
  const task = state.trainingModels.find((item) => item.task_id === taskId);
  if (!task || task.status !== "completed") return;
  state.saveTrainTaskId = taskId;
  state.saveTrainProfile = String(task.profile || profileHint || "");
  state.saveTrainStatus = "";
  state.saveTrainRunning = false;
  const subtitle = el("train-save-subtitle");
  const input = el("train-save-path");
  if (subtitle) subtitle.textContent = `Save merged model for profile ${state.saveTrainProfile || "profile"}`;
  if (input instanceof HTMLInputElement) {
    input.value = defaultSavePathForTask(task);
  }
  setTrainSaveModalOpen(true);
}

