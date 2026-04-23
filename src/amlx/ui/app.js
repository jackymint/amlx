const state = {
  pulse: [],
  lastProcessed: 0,
  catalog: [],
  downloads: [],
  installed: [],
  chat: [],
  modelQuery: "",
  modelSystem: null,
  activeTab: "metrics",
  modelPage: 1,
  modelPerPage: 5,
  modelTotal: 0,
  configuredModel: "",
  runtimeLoadedModels: [],
  selectedChatModel: "",
  chatModelLoading: false,
  chatModelLoadingId: "",
  gpuLimitPercent: 100,
  gpuLimitSaving: false,
  gpuLimitAdapter: null,
  maxModelB: 0,
};

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

function isLikelyChatModel(modelId) {
  const lower = String(modelId || "").toLowerCase();
  const deny = ["embedding", "rerank", "reranker", "whisper", "asr", "tts", "speech"];
  return !deny.some((k) => lower.includes(k));
}

function estimateModelB(modelId) {
  const match = String(modelId || "")
    .toLowerCase()
    .match(/(\d+(?:\.\d+)?)b/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function isModelBlockedByCap(modelId) {
  if (!state.maxModelB || state.maxModelB <= 0) return false;
  const modelB = estimateModelB(modelId);
  if (modelB == null) return false;
  return modelB > state.maxModelB;
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

function setActiveTab(tab) {
  state.activeTab = tab === "models" ? "models" : tab === "chat" ? "chat" : "metrics";
  const metricsPanel = el("panel-metrics");
  const chatPanel = el("panel-chat");
  const modelsPanel = el("panel-models");
  const metricsTab = el("tab-metrics");
  const chatTab = el("tab-chat");
  const modelsTab = el("tab-models");
  if (metricsPanel) metricsPanel.classList.toggle("hidden", state.activeTab !== "metrics");
  if (chatPanel) chatPanel.classList.toggle("hidden", state.activeTab !== "chat");
  if (modelsPanel) modelsPanel.classList.toggle("hidden", state.activeTab !== "models");
  if (metricsTab) metricsTab.classList.toggle("active", state.activeTab === "metrics");
  if (chatTab) chatTab.classList.toggle("active", state.activeTab === "chat");
  if (modelsTab) modelsTab.classList.toggle("active", state.activeTab === "models");
  if (metricsTab) metricsTab.setAttribute("aria-selected", state.activeTab === "metrics" ? "true" : "false");
  if (chatTab) chatTab.setAttribute("aria-selected", state.activeTab === "chat" ? "true" : "false");
  if (modelsTab) modelsTab.setAttribute("aria-selected", state.activeTab === "models" ? "true" : "false");
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

function renderModelCap() {
  const select = el("model-cap-select");
  if (!(select instanceof HTMLSelectElement)) return;
  select.disabled = state.gpuLimitSaving;
  const value = state.maxModelB > 0 ? String(state.maxModelB) : "0";
  if ([...select.options].some((opt) => opt.value === value)) {
    select.value = value;
  } else {
    select.value = "0";
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
      body: JSON.stringify({ gpu_limit_percent: next, max_model_b: state.maxModelB }),
    });
    state.gpuLimitPercent = Number(body?.gpu_limit_percent || next);
    state.gpuLimitAdapter = body?.gpu_limit_adapter || state.gpuLimitAdapter;
    state.maxModelB = Number(body?.max_model_b || state.maxModelB || 0);
  } catch {
    // Keep last selected value in UI when server update fails.
  } finally {
    state.gpuLimitSaving = false;
    renderGpuLimit();
    renderModelCap();
    renderCatalog();
  }
}

async function setModelCap(maxModelB) {
  const next = Math.max(0, Number(maxModelB) || 0);
  state.maxModelB = next;
  state.gpuLimitSaving = true;
  renderModelCap();
  try {
    const body = await fetchJson("/v1/runtime/power", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ gpu_limit_percent: state.gpuLimitPercent, max_model_b: next }),
    });
    state.gpuLimitPercent = Number(body?.gpu_limit_percent || state.gpuLimitPercent);
    state.gpuLimitAdapter = body?.gpu_limit_adapter || state.gpuLimitAdapter;
    state.maxModelB = Number(body?.max_model_b || next);
  } catch {
    // Keep last selected value in UI when server update fails.
  } finally {
    state.gpuLimitSaving = false;
    renderGpuLimit();
    renderModelCap();
    renderCatalog();
  }
}

function renderCatalog() {
  const node = el("catalog");
  if (!node) return;
  if (!state.catalog.length) {
    node.innerHTML = "<div class='job'><span>No models found for this search.</span></div>";
    return;
  }

  node.innerHTML = state.catalog
    .map((item) => {
      const active = state.downloads.find((d) => d.model_id === item.id && (d.status === "queued" || d.status === "downloading"));
      const installed = state.installed.find((m) => m.model_id === item.id);
      const loaded = state.runtimeLoadedModels.includes(item.id);
      const fit = item.compatibility?.fit || "tight";
      const summary = item.compatibility?.summary || "Compatibility unknown";
      const blockedByCap = isModelBlockedByCap(item.id);
      const notSuitable = fit === "no";
      const disabled = active || installed || notSuitable || blockedByCap ? "disabled" : "";
      const button = installed ? "Installed" : active ? `${active.status}...` : "Download";
      const phase = active
        ? `downloading ${active.progress}% • ${active.message}`
        : installed && loaded
          ? "ready • downloaded + loaded in memory"
          : installed
            ? "downloaded • loads into memory on first request"
            : "not downloaded";
      const capNote = blockedByCap ? `<div class="fit fit-no">Blocked by model cap (${state.maxModelB}B)</div>` : "";
      const progress = active ? Math.max(0, Math.min(100, Number(active.progress || 0))) : null;
      return `
        <article class="model-card">
          <h3>${item.label}</h3>
          <div class="meta">${item.id}</div>
          <div class="meta">${item.size} • ${item.tags}</div>
          <div class="meta">source: ${item.source || "curated"}</div>
          <div class="meta model-phase">${phase}</div>
          ${
            progress !== null
              ? `<div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div>`
              : ""
          }
          ${capNote}
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
    node.textContent = "No downloads yet.";
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
    node.textContent = "No installed models.";
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
    node.textContent = "No loaded models.";
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

function renderChatModelSelect() {
  const select = el("chat-model-select");
  const source = el("chat-model-source");
  const sendBtn = el("chat-send");
  if (!(select instanceof HTMLSelectElement)) return;

  const options = [];
  const seen = new Set();
  for (const modelId of state.runtimeLoadedModels) {
    if (!isLikelyChatModel(modelId)) continue;
    if (isModelBlockedByCap(modelId)) continue;
    if (seen.has(modelId)) continue;
    seen.add(modelId);
    options.push({ id: modelId, label: `${modelId} (loaded)`, src: "loaded" });
  }
  for (const item of state.installed) {
    if (!isLikelyChatModel(item?.model_id)) continue;
    if (isModelBlockedByCap(item?.model_id)) continue;
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
    const [downloadsRes, installedRes] = await Promise.all([
      fetchJson("/v1/models/downloads"),
      fetchJson("/v1/models/installed"),
    ]);
    state.downloads = downloadsRes.tasks || [];
    state.installed = installedRes.models || [];
    renderCatalog();
    renderDownloads();
    renderInstalled();
    renderChatModelSelect();
  } catch {
    // Silent fail to keep dashboard responsive even when model APIs are unavailable.
  }
}

async function fetchCatalogOnlineOrCurated() {
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
    renderModelSystem();
    renderCatalog();
    renderPagination();
  } catch {
    const node = el("catalog");
    if (node) node.innerHTML = "<div class='job'><span>Search unavailable right now.</span></div>";
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

    setText("runtime-engine", `engine: ${runtime.engine}`);
    state.configuredModel = runtime.configured_model || "";
    state.runtimeLoadedModels = runtime.loaded_models || [];
    state.gpuLimitPercent = Number(runtime.gpu_limit_percent || state.gpuLimitPercent || 100);
    state.gpuLimitAdapter = runtime.gpu_limit_adapter || null;
    state.maxModelB = Number(runtime.max_model_b || state.maxModelB || 0);
    updateStats(stats);
    renderGpuLimit();
    renderModelCap();
    renderCatalog();
    renderLoadedModels();
    renderChatModelSelect();
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
    renderModelCap();
    renderLoadedModels();
    renderChatModelSelect();
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
  if (isModelBlockedByCap(model)) {
    state.chat.push({
      role: "assistant",
      content: `Model is blocked by cap (${state.maxModelB}B). Choose a smaller model.`,
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
  el("model-cap-select")?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.maxModelB = Math.max(0, Number(target.value) || 0);
    renderModelCap();
    debouncedModelCapSave();
  });

  void refresh();
  void fetchCatalogOnlineOrCurated();
  void refreshModels();
  setActiveTab(state.activeTab);
  renderGpuLimit();
  renderModelCap();
  renderChat();
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

const debouncedModelCapSave = debounce(() => {
  void setModelCap(state.maxModelB);
}, 220);

init();
