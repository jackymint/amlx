function renderChatModelSelect() {
  const select = el("chat-model-select");
  const source = el("chat-model-source");
  const sendBtn = el("chat-send");
  if (!(select instanceof HTMLSelectElement)) return;
  const options = buildModelOptions({
    include: (modelId) => isLikelyChatModel(modelId),
    loadedLabel: "loaded",
    installedLabel: "installed",
  });
  if (!options.length) {
    state.selectedChatModel = "";
    select.innerHTML = `<option value="">No chat-capable loaded/installed model</option>`;
    select.disabled = true;
    if (sendBtn instanceof HTMLButtonElement) sendBtn.disabled = true;
    if (source) source.textContent = "source: choose an instruct/chat model";
    return;
  }
  select.disabled = state.chatModelLoading;
  if (sendBtn instanceof HTMLButtonElement) sendBtn.disabled = state.chatModelLoading;
  state.selectedChatModel = renderSelectOptions(select, options, state.selectedChatModel, "Select model...");
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
    await postJson("/v1/models/preload", { model_id: modelId });
    await refresh();
    return true;
  } catch {
    return false;
  }
}
function renderQuantModelSelect() {
  const select = el("quant-model-select");
  if (!(select instanceof HTMLSelectElement)) return;
  const options = state.installed.map((item) => ({ id: item.model_id })).filter((item) => item.id);
  if (!options.length) {
    select.innerHTML = `<option value="">No installed models</option>`;
    select.disabled = true;
    state.selectedQuantModel = "";
  } else {
    select.disabled = false;
    state.selectedQuantModel = renderSelectOptions(select, options, state.selectedQuantModel, "Select model...");
  }
  const btn = el("quant-submit");
  if (btn instanceof HTMLButtonElement) btn.disabled = !state.selectedQuantModel || state.quantRunning;
  autoFillQuantOutput();
}
function autoFillQuantOutput() {
  const input = el("quant-output-input");
  if (!(input instanceof HTMLInputElement)) return;
  if (!state.selectedQuantModel) return;
  const bitsSelect = el("quant-bits-select");
  const bits = bitsSelect instanceof HTMLSelectElement ? bitsSelect.value : "4";
  const slug = state.selectedQuantModel.split("/").pop()?.replace(/[-_]?\d+bit$/i, "") || "model";
  if (!input.value || input.dataset.auto === "1") {
    input.value = `./exports/${slug}-${bits}bit`;
    input.dataset.auto = "1";
  }
}
function renderQuantJobs() {
  const node = el("quant-jobs");
  if (!node) return;
  const jobs = state.quantizeJobs;
  if (!jobs.length) {
    node.innerHTML = emptyJobCard("No quantization jobs", "Quantized models will appear here.");
    return;
  }
  node.innerHTML = jobs
    .slice(0, 10)
    .map((job) => {
      const isActive = job.status === "queued" || job.status === "running";
      const pct = Math.max(0, Math.min(100, Number(job.progress || 0)));
      const statusColor = job.status === "completed" ? "var(--accent)" : job.status === "failed" ? "#f66" : "";
      return `
        <div class="job">
          <strong>${escapeHtml(job.model_id)}</strong>
          <span style="color:${statusColor}">${escapeHtml(job.message || job.status)}${job.error ? " • " + escapeHtml(job.error) : ""}</span>
          <span class="meta">${job.q_bits}bit / group ${job.q_group_size} → ${escapeHtml(job.output_path)}</span>
          ${isActive ? `<div class="dl-progress-row"><span class="dl-pct">${pct}%</span><div class="progress-track" style="flex:1"><div class="progress-fill" style="width:${pct}%"></div></div><button type="button" class="ghost mini" data-action="cancel-quantize" data-task-id="${escapeHtml(job.task_id)}">Cancel</button></div>` : ""}
        </div>
      `;
    })
    .join("");
}
async function requestQuantize() {
  const model = state.selectedQuantModel;
  if (!model) return;
  const outputInput = el("quant-output-input");
  const output = outputInput instanceof HTMLInputElement ? outputInput.value.trim() : "";
  if (!output) {
    state.quantStatus = "Set an output path";
    setChip("quant-status", state.quantStatus, true);
    return;
  }
  const bitsSelect = el("quant-bits-select");
  const groupSelect = el("quant-group-select");
  const bits = bitsSelect instanceof HTMLSelectElement ? Number(bitsSelect.value) : 4;
  const group = groupSelect instanceof HTMLSelectElement ? Number(groupSelect.value) : 64;
  state.quantRunning = true;
  setChip("quant-status", "Starting...", true);
  renderQuantModelSelect();
  try {
    await postJson("/v1/models/quantize", { model_id: model, output_path: output, q_bits: bits, q_group_size: group });
    setChip("quant-status", "Queued", true);
    const outputInput2 = el("quant-output-input");
    if (outputInput2 instanceof HTMLInputElement) { outputInput2.value = ""; outputInput2.dataset.auto = ""; }
  } catch (err) {
    setChip("quant-status", String(err), true);
  } finally {
    state.quantRunning = false;
    renderQuantModelSelect();
  }
}
async function refreshModels() {
  try {
    const [downloadsRes, installedRes, trainingRes, quantizeRes] = await Promise.all([
      fetchJson("/v1/models/downloads"),
      fetchJson("/v1/models/installed"),
      fetchJson("/v1/models/training"),
      fetchJson("/v1/models/quantize"),
    ]);
    state.downloads = downloadsRes.tasks || [];
    state.installed = installedRes.models || [];
    state.trainingModels = trainingRes.tasks || trainingRes.models || [];
    state.quantizeJobs = quantizeRes.tasks || [];
    renderCatalog();
    renderDownloads();
    renderInstalled();
    renderChatModelSelect();
    renderTrainModelSelect();
    renderTrainingModels();
    renderQuantModelSelect();
    renderQuantJobs();
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
