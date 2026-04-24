const state = {
  pulse: [],
  lastProcessed: 0,
  catalog: [],
  downloads: [],
  installed: [],
  chat: [],
  modelQuery: "",
  modelLoading: false,
  modelSearchError: "",
  modelSystem: null,
  activeTab: "metrics",
  modelPage: 1,
  modelPerPage: 4,
  modelTotal: 0,
  configuredModel: "",
  runtimeLoadedModels: [],
  trainingModels: [],
  selectedChatModel: "",
  selectedTrainModel: "",
  uploadedTrainSamples: [],
  uploadedTrainFileName: "",
  chatModelLoading: false,
  chatModelLoadingId: "",
  trainRunning: false,
  trainStatus: "",
  trainProgress: 0,
  gpuLimitPercent: 100,
  gpuLimitSaving: false,
  gpuLimitAdapter: null,
};

let trainProgressTimer = null;

const el = (id) => document.getElementById(id);

function setText(id, value) {
  const node = el(id);
  if (node) node.textContent = String(value);
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function compactSample(value) {
  const text = String(value || "").trim();
  return text ? text : "";
}

function extractSamplesFromItem(item) {
  if (typeof item === "string") {
    const sample = compactSample(item);
    return sample ? [sample] : [];
  }

  if (Array.isArray(item)) {
    return item.flatMap((entry) => extractSamplesFromItem(entry));
  }

  if (!item || typeof item !== "object") return [];

  const obj = item;
  const directKeys = ["text", "content", "sample", "value"];
  for (const key of directKeys) {
    if (typeof obj[key] === "string") {
      const sample = compactSample(obj[key]);
      if (sample) return [sample];
    }
  }

  if (typeof obj.prompt === "string" || typeof obj.response === "string") {
    const sample = compactSample(`${obj.prompt || ""}\n${obj.response || ""}`);
    return sample ? [sample] : [];
  }
  if (typeof obj.question === "string" || typeof obj.answer === "string") {
    const sample = compactSample(`Q: ${obj.question || ""}\nA: ${obj.answer || ""}`);
    return sample ? [sample] : [];
  }
  if (
    typeof obj.instruction === "string" ||
    typeof obj.input === "string" ||
    typeof obj.output === "string"
  ) {
    const sample = compactSample(
      `Instruction: ${obj.instruction || ""}\nInput: ${obj.input || ""}\nOutput: ${obj.output || ""}`,
    );
    return sample ? [sample] : [];
  }

  if (Array.isArray(obj.messages)) {
    const lines = obj.messages
      .map((m) => {
        if (!m || typeof m !== "object") return "";
        const role = compactSample(m.role || m.from || "user").toUpperCase();
        const content = compactSample(m.content || m.value || "");
        return role && content ? `${role}: ${content}` : "";
      })
      .filter(Boolean);
    const sample = compactSample(lines.join("\n"));
    return sample ? [sample] : [];
  }
  if (Array.isArray(obj.conversations)) {
    const lines = obj.conversations
      .map((m) => {
        if (!m || typeof m !== "object") return "";
        const role = compactSample(m.role || m.from || "user").toUpperCase();
        const content = compactSample(m.content || m.value || "");
        return role && content ? `${role}: ${content}` : "";
      })
      .filter(Boolean);
    const sample = compactSample(lines.join("\n"));
    return sample ? [sample] : [];
  }

  const nestedKeys = ["samples", "data", "records", "items", "examples"];
  for (const key of nestedKeys) {
    if (Array.isArray(obj[key])) {
      return obj[key].flatMap((entry) => extractSamplesFromItem(entry));
    }
  }

  return [];
}

function parseTrainJson(text) {
  const parsed = JSON.parse(text);
  const samples = extractSamplesFromItem(parsed).filter(Boolean);
  return [...new Set(samples)];
}

function parseTrainJsonl(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const out = [];
  for (const line of lines) {
    const parsed = JSON.parse(line);
    out.push(...extractSamplesFromItem(parsed));
  }
  return [...new Set(out.filter(Boolean))];
}

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

function setActiveTab(tab) {
  state.activeTab =
    tab === "models" ? "models" : tab === "chat" ? "chat" : tab === "training" ? "training" : "metrics";
  const metricsPanel = el("panel-metrics");
  const chatPanel = el("panel-chat");
  const modelsPanel = el("panel-models");
  const trainingPanel = el("panel-training");
  const metricsTab = el("tab-metrics");
  const chatTab = el("tab-chat");
  const modelsTab = el("tab-models");
  const trainingTab = el("tab-training");
  if (metricsPanel) metricsPanel.classList.toggle("hidden", state.activeTab !== "metrics");
  if (chatPanel) chatPanel.classList.toggle("hidden", state.activeTab !== "chat");
  if (modelsPanel) modelsPanel.classList.toggle("hidden", state.activeTab !== "models");
  if (trainingPanel) trainingPanel.classList.toggle("hidden", state.activeTab !== "training");
  if (metricsTab) metricsTab.classList.toggle("active", state.activeTab === "metrics");
  if (chatTab) chatTab.classList.toggle("active", state.activeTab === "chat");
  if (modelsTab) modelsTab.classList.toggle("active", state.activeTab === "models");
  if (trainingTab) trainingTab.classList.toggle("active", state.activeTab === "training");
  if (metricsTab) metricsTab.setAttribute("aria-selected", state.activeTab === "metrics" ? "true" : "false");
  if (chatTab) chatTab.setAttribute("aria-selected", state.activeTab === "chat" ? "true" : "false");
  if (modelsTab) modelsTab.setAttribute("aria-selected", state.activeTab === "models" ? "true" : "false");
  if (trainingTab) trainingTab.setAttribute("aria-selected", state.activeTab === "training" ? "true" : "false");
}

function drawPulse() {
  const canvas = el("pulse");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const data = state.pulse;
  if (data.length < 2) return;

  const max = Math.max(...data, 1);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#6ec5ff";
  ctx.beginPath();

  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * (w - 16) + 8;
    const y = h - 12 - (v / max) * (h - 28);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function updateStats(stats) {
  const totalHits = stats.memory_hits + stats.disk_hits + stats.block_hits;
  setText("cache-hit-total", totalHits);
  setText("memory-hits", stats.memory_hits);
  setText("disk-hits", stats.disk_hits);
  setText("block-hits", stats.block_hits);
  setText("misses", stats.misses);
  setText("misses-small", stats.misses);
  setText("block-writes", stats.block_writes);
  setText("batch-runs", stats.scheduler_batch_runs);
  setText("scheduler-enqueued", stats.scheduler_enqueued);
  setText("scheduler-processed", stats.scheduler_processed);
  setText("scheduler-items", stats.scheduler_total_batch_items);

  const deltaProcessed = Math.max(0, stats.scheduler_processed - state.lastProcessed);
  state.lastProcessed = stats.scheduler_processed;
  state.pulse.push(deltaProcessed);
  if (state.pulse.length > 46) state.pulse.shift();
  drawPulse();
}

function renderGpuLimit() {
  const range = el("gpu-limit-range");
  const value = el("gpu-limit-value");
  if (range instanceof HTMLInputElement) {
    range.value = String(state.gpuLimitPercent || 100);
    range.disabled = state.gpuLimitSaving;
  }
  if (value) {
    const saving = state.gpuLimitSaving ? " • saving..." : "";
    const hard = state.gpuLimitAdapter?.supported ? " • hard" : " • soft";
    value.textContent = `${state.gpuLimitPercent}%${hard}${saving}`;
  }
}

async function setGpuLimitPercent(percent) {
  const next = Math.max(20, Math.min(100, Number(percent) || 100));
  state.gpuLimitPercent = next;
  state.gpuLimitSaving = true;
  renderGpuLimit();
  try {
    const body = await fetchJson("/v1/runtime/power", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ gpu_limit_percent: next }),
    });
    state.gpuLimitPercent = Number(body?.gpu_limit_percent || next);
    state.gpuLimitAdapter = body?.gpu_limit_adapter || state.gpuLimitAdapter;
  } catch {
    // Keep last selected value in UI when server update fails.
  } finally {
    state.gpuLimitSaving = false;
    renderGpuLimit();
    renderCatalog();
  }
}

function renderCatalog() {
  const node = el("catalog");
  if (!node) return;
  if (state.modelLoading) {
    node.innerHTML = "<div class='catalog-loading'><span>Searching models...</span></div>";
    return;
  }
  if (state.modelSearchError) {
    node.innerHTML = `<div class='job'><span>${escapeHtml(state.modelSearchError)}</span></div>`;
    return;
  }
  if (!state.catalog.length) {
    node.innerHTML = "<div class='catalog-empty'><span>No models found for this search.</span></div>";
    return;
  }

  node.innerHTML = state.catalog
    .map((item) => {
      const active = state.downloads.find((d) => d.model_id === item.id && (d.status === "queued" || d.status === "downloading"));
      const installed = state.installed.find((m) => m.model_id === item.id);
      const loaded = state.runtimeLoadedModels.includes(item.id);
      const fit = item.compatibility?.fit || "tight";
      const summary = item.compatibility?.summary || "Compatibility unknown";
      const notSuitable = fit === "no";
      const disabled = active || installed || notSuitable ? "disabled" : "";
      const button = installed ? "Installed" : active ? `${active.status}...` : "Download";
      const phase = active
        ? `downloading ${active.progress}% • ${active.message}`
        : installed && loaded
          ? "ready • downloaded + loaded in memory"
          : installed
            ? "downloaded • loads into memory on first request"
            : "not downloaded";
      const progress = active ? Math.max(0, Math.min(100, Number(active.progress || 0))) : null;
      return `
        <article class="model-card">
          <h3>${item.label}</h3>
          <div class="meta">${item.id}</div>
          <div class="meta">${item.size} • ${item.tags}</div>
          <div class="meta model-phase">${phase}</div>
          ${
            progress !== null
              ? `<div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div>`
              : ""
          }
          <div class="fit fit-${fit}">${summary}</div>
          <button data-model-id="${item.id}" ${disabled}>${button}</button>
        </article>
      `;
    })
    .join("");
}

function renderPagination() {
  const pageInfo = el("model-page-info");
  const prevBtn = el("model-prev");
  const nextBtn = el("model-next");
  const totalPages = Math.max(1, Math.ceil((state.modelTotal || 0) / state.modelPerPage));
  if (pageInfo) pageInfo.textContent = `Page ${state.modelPage} / ${totalPages} (${state.modelTotal} results)`;
  if (prevBtn) prevBtn.disabled = state.modelPage <= 1;
  if (nextBtn) nextBtn.disabled = state.modelPage >= totalPages;
}

function renderDownloads() {
  const node = el("downloads");
  if (!node) return;
  if (!state.downloads.length) {
    node.innerHTML = emptyJobCard("No downloads yet", "Downloaded models and progress tasks will appear here.");
    return;
  }

  node.innerHTML = state.downloads
    .slice(0, 10)
    .map(
      (task) => `
        <div class="job">
          <strong>${task.model_id}</strong>
          <span>${task.status} • ${task.progress}% • ${task.message}</span>
          <div class="progress-track"><div class="progress-fill" style="width:${Math.max(0, Math.min(100, Number(task.progress || 0)))}%"></div></div>
          ${task.error ? `<span>${task.error}</span>` : ""}
        </div>
      `,
    )
    .join("");
}

function renderInstalled() {
  const node = el("installed");
  if (!node) return;
  if (!state.installed.length) {
    node.innerHTML = emptyJobCard("No installed models", "Downloaded model folders will be listed here for cleanup.");
    return;
  }

  node.innerHTML = state.installed
    .map(
      (item) => `
        <div class="job">
          <strong>${item.model_id}</strong>
          <span>${item.path}</span>
          <div class="job-actions">
            <button type="button" class="ghost mini" data-action="delete-installed" data-model-id="${item.model_id}">Delete</button>
          </div>
        </div>
      `,
    )
    .join("");
}

function renderLoadedModels() {
  const node = el("loaded-models");
  if (!node) return;
  if (!state.runtimeLoadedModels.length) {
    node.innerHTML = emptyJobCard("No loaded models", "Preload a model to run chat and training instantly.");
    return;
  }

  node.innerHTML = state.runtimeLoadedModels
    .map(
      (modelId) => `
        <div class="job">
          <strong>${modelId}</strong>
          <span>loaded in memory</span>
          <div class="job-actions">
            <button type="button" class="ghost mini" data-action="unload-model" data-model-id="${modelId}">Unload</button>
          </div>
        </div>
      `,
    )
    .join("");
}

function renderTrainModelSelect() {
  const select = el("train-model-select");
  const button = el("train-submit");
  const uploadBtn = el("train-upload-btn");
  const epochsSelect = el("train-epochs-select");
  const datasetInput = el("train-dataset");
  const fileMeta = el("train-file-meta");
  const status = el("train-status");
  if (!(select instanceof HTMLSelectElement)) return;

  const options = [];
  const seen = new Set();
  for (const modelId of state.runtimeLoadedModels) {
    if (!modelId || seen.has(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, src: "loaded" });
  }
  for (const item of state.installed) {
    const modelId = item?.model_id;
    if (!modelId || seen.has(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, src: "installed" });
  }

  if (!options.length) {
    state.selectedTrainModel = "";
    select.innerHTML = `<option value="">No loaded/installed model</option>`;
    select.disabled = true;
    if (button instanceof HTMLButtonElement) button.disabled = true;
    if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = true;
    if (epochsSelect instanceof HTMLSelectElement) epochsSelect.disabled = true;
    if (datasetInput instanceof HTMLTextAreaElement) datasetInput.disabled = true;
  } else {
    if (!options.some((x) => x.id === state.selectedTrainModel)) {
      state.selectedTrainModel = options[0].id;
    }
    select.disabled = state.trainRunning;
    select.innerHTML = options
      .map(
        (opt) =>
          `<option value="${opt.id}" ${opt.id === state.selectedTrainModel ? "selected" : ""}>${opt.id} (${opt.src})</option>`,
      )
      .join("");
    if (button instanceof HTMLButtonElement) button.disabled = state.trainRunning;
    if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = state.trainRunning;
    if (epochsSelect instanceof HTMLSelectElement) epochsSelect.disabled = state.trainRunning;
    if (datasetInput instanceof HTMLTextAreaElement) datasetInput.disabled = state.trainRunning;
  }

  if (status) {
    const msg = state.trainRunning ? "training..." : state.trainStatus || "";
    status.textContent = msg;
    status.style.display = msg ? "inline-flex" : "none";
  }
  if (fileMeta) {
    if (!state.uploadedTrainFileName) {
      fileMeta.textContent = "No file selected.";
    } else {
      fileMeta.textContent = `${state.uploadedTrainFileName} • ${state.uploadedTrainSamples.length} samples`;
    }
  }
  renderTrainProgress();
}

function renderTrainingModels() {
  const node = el("training-models");
  if (!node) return;
  if (!state.trainingModels.length) {
    node.innerHTML = emptyJobCard("No training history", "After training, summaries for each model will show up here.");
    return;
  }

  node.innerHTML = state.trainingModels
    .map((item) => {
      const updated = new Date(Number(item.updated_at || item.finished_at || 0) * 1000);
      const dateLabel = Number.isFinite(updated.getTime()) ? updated.toLocaleString() : "unknown";
      const progress = Math.max(0, Math.min(100, Number(item.progress || 0)));
      const stateText = `${item.status || "unknown"} • ${progress}%`;
      return `
        <div class="job">
          <strong>${item.model_id || item.effective_model}</strong>
          <span>${stateText} • type: ${item.fine_tune_type || "qlora"} • samples: ${item.train_samples || 0}</span>
          <div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div>
          <span>${item.message || ""}${item.error ? ` • ${item.error}` : ""} • updated: ${dateLabel}</span>
        </div>
      `;
    })
    .join("");
}

function renderChatModelSelect() {
  const select = el("chat-model-select");
  const source = el("chat-model-source");
  const sendBtn = el("chat-send");
  if (!(select instanceof HTMLSelectElement)) return;

  const options = [];
  const seen = new Set();
  for (const modelId of state.runtimeLoadedModels) {
    if (!isLikelyChatModel(modelId)) continue;
    if (seen.has(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, label: `${modelId} (loaded)`, src: "loaded" });
  }
  for (const item of state.installed) {
    if (!isLikelyChatModel(item?.model_id)) continue;
    if (!item?.model_id || seen.has(item.model_id)) continue;
    seen.add(item.model_id);
    options.push({ id: item.model_id, label: `${item.model_id} (installed)`, src: "installed" });
  }

  if (!options.length) {
    state.selectedChatModel = "";
    select.innerHTML = `<option value="">No chat-capable loaded/installed model</option>`;
    select.disabled = true;
    if (sendBtn instanceof HTMLButtonElement) sendBtn.disabled = true;
    if (source) source.textContent = "source: choose an instruct/chat model";
    return;
  }

  if (!options.some((x) => x.id === state.selectedChatModel)) {
    state.selectedChatModel = "";
  }

  select.disabled = state.chatModelLoading;
  if (sendBtn instanceof HTMLButtonElement) sendBtn.disabled = state.chatModelLoading;
  const placeholder = `<option value="" ${state.selectedChatModel ? "" : "selected"}>Select model...</option>`;
  const optionHtml = options
    .map((opt) => `<option value="${opt.id}" ${opt.id === state.selectedChatModel ? "selected" : ""}>${opt.label}</option>`)
    .join("");
  select.innerHTML = placeholder + optionHtml;

  const selected = options.find((x) => x.id === state.selectedChatModel);
  if (source) {
    if (state.chatModelLoading && state.chatModelLoadingId) {
      source.textContent = `source: loading ${state.chatModelLoadingId} into memory...`;
    } else {
      source.textContent = selected ? `source: ${selected.src}` : "source: select model";
    }
  }
}

function ensureChatModelSelection() {
  return state.selectedChatModel || "";
}

function setChatModelLoading(loading, modelId = "") {
  state.chatModelLoading = Boolean(loading);
  state.chatModelLoadingId = loading ? modelId : "";
  renderChatModelSelect();
}

async function preloadModel(modelId) {
  if (!modelId) return false;
  try {
    await fetchJson("/v1/models/preload", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    await refresh();
    return true;
  } catch {
    return false;
  }
}

async function refreshModels() {
  try {
    const [downloadsRes, installedRes, trainingRes] = await Promise.all([
      fetchJson("/v1/models/downloads"),
      fetchJson("/v1/models/installed"),
      fetchJson("/v1/models/training"),
    ]);
    state.downloads = downloadsRes.tasks || [];
    state.installed = installedRes.models || [];
    state.trainingModels = trainingRes.tasks || trainingRes.models || [];
    renderCatalog();
    renderDownloads();
    renderInstalled();
    renderChatModelSelect();
    renderTrainModelSelect();
    renderTrainingModels();
  } catch {
    // Silent fail to keep dashboard responsive even when model APIs are unavailable.
  }
}

async function fetchCatalogOnlineOrCurated() {
  state.modelLoading = true;
  state.modelSearchError = "";
  renderCatalog();
  try {
    const q = state.modelQuery.trim();
    const pageParams = `page=${state.modelPage}&per_page=${state.modelPerPage}`;
    const endpoint = q
      ? `/v1/models/search?q=${encodeURIComponent(q)}&${pageParams}`
      : `/v1/models/catalog?${pageParams}`;
    const body = await fetchJson(endpoint);
    state.catalog = body.models || [];
    state.modelSystem = body.system || null;
    state.modelTotal = body.pagination?.total || 0;
    state.modelSearchError = "";
    renderModelSystem();
    renderCatalog();
    renderPagination();
  } catch {
    state.modelSearchError = "Search unavailable right now.";
  } finally {
    state.modelLoading = false;
    renderCatalog();
  }
}

function renderModelSystem() {
  const pill = el("model-system");
  if (!pill || !state.modelSystem) return;
  const sys = state.modelSystem;
  const silicon = sys.apple_silicon ? "Apple Silicon" : "Non-Apple Silicon";
  pill.textContent = `${silicon} • RAM ${sys.ram_gb} GB • Free ${sys.free_disk_gb} GB`;
}

async function refresh() {
  try {
    const [health, stats, runtime] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/v1/cache/stats"),
      fetchJson("/v1/runtime"),
    ]);

    const healthEl = el("runtime-health");
    if (healthEl) {
      healthEl.textContent = `status: ${health.status}`;
      healthEl.classList.remove("ok", "fail");
      healthEl.classList.add(health.status === "ok" ? "ok" : "fail");
    }

    state.configuredModel = runtime.configured_model || "";
    state.runtimeLoadedModels = runtime.loaded_models || [];
    state.gpuLimitPercent = Number(runtime.gpu_limit_percent || state.gpuLimitPercent || 100);
    state.gpuLimitAdapter = runtime.gpu_limit_adapter || null;
    updateStats(stats);
    renderGpuLimit();
    renderCatalog();
    renderLoadedModels();
    renderChatModelSelect();
    renderTrainModelSelect();
  } catch {
    const healthEl = el("runtime-health");
    if (healthEl) {
      healthEl.textContent = "status: disconnected";
      healthEl.classList.remove("ok");
      healthEl.classList.add("fail");
    }
    state.runtimeLoadedModels = [];
    state.gpuLimitSaving = false;
    state.gpuLimitAdapter = null;
    renderGpuLimit();
    renderLoadedModels();
    renderChatModelSelect();
    renderTrainModelSelect();
  }
}

async function runPrompt() {
  const input = el("chat-input");
  const prompt = input?.value?.trim() || "";
  if (!prompt) return;

  const sendBtn = el("chat-send");
  const clearBtn = el("chat-clear");
  const model = ensureChatModelSelection();
  if (!model) {
    state.chat.push({
      role: "assistant",
      content: "Select a model first, then send chat.",
    });
    renderChat();
    return;
  }
  if (!state.runtimeLoadedModels.includes(model)) {
    await preloadModel(model);
  }

  if (input) input.value = "";
  state.chat.push({ role: "user", content: prompt });
  const pendingIndex = state.chat.length;
  state.chat.push({ role: "assistant", content: "AI is thinking...", pending: true });
  renderChat();
  if (sendBtn) sendBtn.disabled = true;
  if (clearBtn) clearBtn.disabled = true;

  try {
    const payload = {
      model,
      messages: state.chat,
      max_tokens: 256,
      temperature: 0.2,
    };

    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        if (err?.detail) detail = String(err.detail);
      } catch {
        // ignore parse error
      }
      throw new Error(detail);
    }
    const body = await res.json();
    const text = body.choices?.[0]?.message?.content || "(empty)";
    state.chat[pendingIndex] = { role: "assistant", content: text, pending: false };
  } catch (err) {
    const msg = `Request failed (${model}): ${err}`;
    state.chat[pendingIndex] = { role: "assistant", content: msg, pending: false };
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    if (clearBtn) clearBtn.disabled = false;
  }

  renderChat();
  await refresh();
}

function renderChat() {
  const node = el("chat-log");
  if (!node) return;
  if (!state.chat.length) {
    node.innerHTML = "";
    return;
  }

  node.innerHTML = state.chat
    .filter((msg) => msg.role === "user" || msg.role === "assistant")
    .map(
      (msg) => `
      <div class="chat-msg ${msg.role}">
        <div class="bubble ${msg.pending ? "pending" : ""}">${escapeHtml(msg.content)}</div>
      </div>
    `,
    )
    .join("");
  node.scrollTop = node.scrollHeight;
}

function clearChat() {
  state.chat = [];
  renderChat();
}

async function requestDownload(modelId) {
  try {
    await fetch("/v1/models/download", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    await refreshModels();
  } catch {
    // Ignore UI-level errors; detailed reason is available in task error when created.
  }
}

async function requestUnload(modelId) {
  try {
    await fetch("/v1/models/unload", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    await refresh();
    await refreshModels();
  } catch {
    // no-op
  }
}

async function handleTrainFileUpload(file) {
  if (!(file instanceof File)) return;
  const name = String(file.name || "dataset");
  const lower = name.toLowerCase();
  try {
    const text = await file.text();
    let samples = [];
    if (lower.endsWith(".jsonl")) {
      samples = parseTrainJsonl(text);
    } else if (lower.endsWith(".json")) {
      samples = parseTrainJson(text);
    } else {
      try {
        samples = parseTrainJsonl(text);
      } catch {
        samples = parseTrainJson(text);
      }
    }
    if (!samples.length) {
      state.uploadedTrainFileName = name;
      state.uploadedTrainSamples = [];
      state.trainStatus = "file parsed but no samples found";
      renderTrainModelSelect();
      return;
    }
    state.uploadedTrainFileName = name;
    state.uploadedTrainSamples = samples;
    state.trainStatus = `parsed ${samples.length} samples from ${name}`;
    renderTrainModelSelect();
    await requestTrain({ samples });
  } catch (err) {
    state.uploadedTrainFileName = name;
    state.uploadedTrainSamples = [];
    state.trainStatus = `invalid file: ${err}`;
    renderTrainModelSelect();
  }
}

async function requestTrain(options = {}) {
  const modelId = state.selectedTrainModel || "";
  const input = el("train-dataset");
  const epochSelect = el("train-epochs-select");
  const dataset = input instanceof HTMLTextAreaElement ? input.value : "";
  const epochs =
    epochSelect instanceof HTMLSelectElement
      ? Math.max(1, Math.min(10, Number(epochSelect.value) || 1))
      : 1;
  const providedSamples = Array.isArray(options.samples) ? options.samples.filter(Boolean) : [];
  const datasetLines = dataset
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const samples = [...new Set([...providedSamples, ...datasetLines])];

  if (!modelId) {
    state.trainStatus = "select a model first";
    renderTrainModelSelect();
    return;
  }
  if (!samples.length) {
    state.trainStatus = "training data is empty";
    renderTrainModelSelect();
    return;
  }

  state.trainRunning = true;
  state.trainProgress = 1;
  state.trainStatus = `training ${modelId}...`;
  renderTrainModelSelect();
  beginTrainProgress();
  try {
    const created = await fetchJson("/v1/models/train", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model_id: modelId, samples, epochs, fine_tune_type: "qlora" }),
    });
    const taskId = created?.task_id;
    if (!taskId) {
      throw new Error("missing train task id");
    }
    state.trainStatus = `queued ${taskId}`;
    renderTrainModelSelect();

    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 900));
      const body = await fetchJson("/v1/models/training");
      const tasks = body?.tasks || [];
      const task = tasks.find((x) => x.task_id === taskId);
      if (!task) continue;

      state.trainProgress = Math.max(state.trainProgress, Number(task.progress || 0));
      state.trainStatus = task.message || task.status || state.trainStatus;
      renderTrainProgress();
      renderTrainModelSelect();

      if (task.status === "completed") {
        state.trainStatus = "fine-tune completed";
        if (input instanceof HTMLTextAreaElement) input.value = "";
        state.uploadedTrainSamples = [];
        state.uploadedTrainFileName = "";
        const fileInput = el("train-file-input");
        if (fileInput instanceof HTMLInputElement) fileInput.value = "";
        break;
      }
      if (task.status === "failed") {
        throw new Error(task.error || task.message || "fine-tune failed");
      }
    }
    await refreshModels();
  } catch (err) {
    state.trainStatus = `failed: ${err}`;
  } finally {
    state.trainRunning = false;
    completeTrainProgress();
    renderTrainModelSelect();
  }
}

async function clearCache() {
  const btn = el("cache-clear");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  try {
    await fetch("/v1/cache", { method: "DELETE" });
    await refresh();
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
  }
}

async function requestDeleteInstalled(modelId) {
  try {
    await fetch("/v1/models/delete", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    await refresh();
    await refreshModels();
    await fetchCatalogOnlineOrCurated();
  } catch {
    // no-op
  }
}

function init() {
  el("tab-metrics")?.addEventListener("click", () => {
    setActiveTab("metrics");
  });
  el("tab-chat")?.addEventListener("click", () => {
    setActiveTab("chat");
  });
  el("tab-models")?.addEventListener("click", () => {
    setActiveTab("models");
  });
  el("tab-training")?.addEventListener("click", () => {
    setActiveTab("training");
  });

  el("cache-clear")?.addEventListener("click", () => {
    void clearCache();
  });
  el("chat-send")?.addEventListener("click", () => {
    void runPrompt();
  });
  el("chat-clear")?.addEventListener("click", () => {
    clearChat();
  });
  el("chat-input")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void runPrompt();
    }
  });
  el("chat-model-select")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.selectedChatModel = target.value || state.configuredModel;
    renderChatModelSelect();
    if (!state.selectedChatModel) return;
    if (state.runtimeLoadedModels.includes(state.selectedChatModel)) return;
    void (async () => {
      const modelId = state.selectedChatModel;
      setChatModelLoading(true, modelId);
      const ok = await preloadModel(modelId);
      setChatModelLoading(false);
      if (!ok) {
        state.chat.push({
          role: "assistant",
          content: `Failed to preload model: ${modelId}`,
        });
        renderChat();
      }
    })();
  });
  el("train-model-select")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.selectedTrainModel = target.value || "";
    renderTrainModelSelect();
  });
  el("train-submit")?.addEventListener("click", () => {
    void requestTrain();
  });
  el("train-guide-open")?.addEventListener("click", () => {
    setTrainGuideModalOpen(true);
  });
  el("train-guide-close")?.addEventListener("click", () => {
    setTrainGuideModalOpen(false);
  });
  el("train-guide-modal")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.id !== "train-guide-modal") return;
    setTrainGuideModalOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const modal = el("train-guide-modal");
    if (!(modal instanceof HTMLDivElement)) return;
    if (modal.hidden) return;
    setTrainGuideModalOpen(false);
  });
  el("train-upload-btn")?.addEventListener("click", () => {
    const fileInput = el("train-file-input");
    if (!(fileInput instanceof HTMLInputElement)) return;
    fileInput.click();
  });
  el("train-file-input")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const file = target.files?.[0];
    if (!file) return;
    void handleTrainFileUpload(file);
  });

  el("catalog")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestDownload(modelId);
  });
  el("loaded-models")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "unload-model") return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestUnload(modelId);
  });
  el("installed")?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "delete-installed") return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestDeleteInstalled(modelId);
  });
  el("model-search")?.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    state.modelQuery = target.value || "";
    state.modelPage = 1;
    debouncedFetchCatalog();
  });
  el("model-prev")?.addEventListener("click", () => {
    if (state.modelPage <= 1) return;
    state.modelPage -= 1;
    void fetchCatalogOnlineOrCurated();
  });
  el("model-next")?.addEventListener("click", () => {
    const totalPages = Math.max(1, Math.ceil((state.modelTotal || 0) / state.modelPerPage));
    if (state.modelPage >= totalPages) return;
    state.modelPage += 1;
    void fetchCatalogOnlineOrCurated();
  });

  el("gpu-limit-range")?.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    state.gpuLimitPercent = Math.max(20, Math.min(100, Number(target.value) || 100));
    renderGpuLimit();
    debouncedGpuLimitSave();
  });

  void refresh();
  void fetchCatalogOnlineOrCurated();
  void refreshModels();
  setActiveTab(state.activeTab);
  renderGpuLimit();
  renderChat();
  renderTrainModelSelect();
  renderTrainingModels();
  setInterval(() => {
    void refresh();
    void refreshModels();
  }, 2000);
}

const debouncedFetchCatalog = debounce(() => {
  void fetchCatalogOnlineOrCurated();
}, 350);

const debouncedGpuLimitSave = debounce(() => {
  void setGpuLimitPercent(state.gpuLimitPercent);
}, 220);

init();
