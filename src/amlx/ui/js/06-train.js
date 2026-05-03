
const HF_PER_PAGE = 4;
let _hfResults = [];
let _hfPage = 1;

function renderHFPage() {
  const catalogEl = el("hf-catalog");
  const pager = el("hf-pager");
  const pageInfo = el("hf-page-info");
  const prevBtn = el("hf-prev");
  const nextBtn = el("hf-next");
  if (!catalogEl) return;

  const totalPages = Math.max(1, Math.ceil(_hfResults.length / HF_PER_PAGE));
  const start = (_hfPage - 1) * HF_PER_PAGE;
  const page = _hfResults.slice(start, start + HF_PER_PAGE);

  if (!page.length) {
    catalogEl.style.display = "none";
    if (pager) pager.style.display = "none";
    return;
  }

  catalogEl.style.display = "grid";
  catalogEl.innerHTML = page.map((r) => {
    const dl = (r.downloads || 0) >= 1000 ? `${Math.round((r.downloads || 0) / 1000)}k ↓` : `${r.downloads || 0} ↓`;
    const tags = (r.tags || []).slice(0, 4).map((t) => `<span class="cap no">${escapeHtml(t)}</span>`).join("");
    const desc = r.description ? `<div class="meta">${escapeHtml(r.description)}</div>` : "";
    return `
      <article class="model-card" data-dataset-id="${escapeHtml(r.id)}">
        <h3>${escapeHtml(r.id.split("/").pop())}</h3>
        <div class="meta">${escapeHtml(r.id)}</div>
        ${desc}
        <div class="cap-row">${tags}</div>
        <div class="meta">${dl}</div>
        <button type="button" data-action="select-hf-dataset" data-dataset-id="${escapeHtml(r.id)}">Use</button>
      </article>`;
  }).join("");

  if (pager) pager.style.display = _hfResults.length > HF_PER_PAGE ? "flex" : "none";
  if (pageInfo) pageInfo.textContent = `Page ${_hfPage} / ${totalPages}`;
  if (prevBtn instanceof HTMLButtonElement) prevBtn.disabled = _hfPage <= 1;
  if (nextBtn instanceof HTMLButtonElement) nextBtn.disabled = _hfPage >= totalPages;
}

function selectHFDataset(id) {
  const hidden = el("hf-dataset-input");
  const loadRow = el("hf-load-row");
  const label = el("hf-selected-label");
  const statusEl = el("hf-status");
  if (hidden instanceof HTMLInputElement) hidden.value = id;
  if (label) label.textContent = id;
  if (loadRow) loadRow.style.display = "";
  if (statusEl) { statusEl.textContent = ""; statusEl.style.display = "none"; }
}

async function searchHFDatasets() {
  const input = el("hf-search-input");
  const catalogEl = el("hf-catalog");
  const pager = el("hf-pager");
  const q = input instanceof HTMLInputElement ? input.value.trim() : "";

  if (!q) {
    _hfResults = [];
    _hfPage = 1;
    if (catalogEl) catalogEl.style.display = "none";
    if (pager) pager.style.display = "none";
    return;
  }
  if (!catalogEl) return;

  catalogEl.style.display = "grid";
  catalogEl.innerHTML = `<div class="catalog-loading"><span class="catalog-searching">Searching datasets...</span></div>`;
  if (pager) pager.style.display = "none";

  try {
    const body = await fetchJson(`/v1/datasets/search?q=${encodeURIComponent(q)}&limit=100`);
    _hfResults = body.results || [];
    _hfPage = 1;
    if (!_hfResults.length) {
      catalogEl.innerHTML = `<div class="catalog-empty"><span>No datasets found for "${escapeHtml(q)}"</span></div>`;
      return;
    }
    renderHFPage();
  } catch (err) {
    catalogEl.innerHTML = `<div class="catalog-empty"><span>Search error: ${escapeHtml(String(err))}</span></div>`;
  }
}


async function fetchHFDataset() {
  const hidden = el("hf-dataset-input");
  const splitSel = el("hf-split-select");
  const limitSel = el("hf-limit-select");
  const btn = el("hf-load-btn");
  const statusEl = el("hf-status");
  const textarea = el("train-dataset");

  const datasetId = hidden instanceof HTMLInputElement ? hidden.value.trim() : "";
  if (!datasetId) {
    if (statusEl) { statusEl.textContent = "select a dataset first"; statusEl.style.display = ""; }
    return;
  }
  const split = splitSel instanceof HTMLSelectElement ? splitSel.value : "train";
  const limit = limitSel instanceof HTMLSelectElement ? Number(limitSel.value) : 100;

  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  if (statusEl) { statusEl.textContent = "loading…"; statusEl.style.display = ""; }

  try {
    const body = await postJson("/v1/datasets/fetch", { dataset_id: datasetId, split, limit });
    const samples = body.samples || [];
    if (!samples.length) {
      if (statusEl) { statusEl.textContent = "no samples found"; statusEl.style.display = ""; }
      return;
    }
    if (textarea instanceof HTMLTextAreaElement) {
      textarea.value = samples.join("\n");
    }
    if (statusEl) { statusEl.textContent = `loaded ${samples.length} samples`; statusEl.style.display = ""; }
  } catch (err) {
    if (statusEl) { statusEl.textContent = `error: ${err}`; statusEl.style.display = ""; }
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
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
    const textarea = el("train-dataset");
    if (textarea instanceof HTMLTextAreaElement) textarea.value = samples.join("\n");
    renderTrainModelSelect();
  } catch (err) {
    state.uploadedTrainFileName = name;
    state.uploadedTrainSamples = [];
    state.trainStatus = `invalid file: ${err}`;
    renderTrainModelSelect();
  }
}

async function requestTrain(options = {}) {
  const modelId = state.selectedTrainModel || "";
  const profile = String(state.selectedTrainProfile || "").trim();
  const input = el("train-dataset");
  const epochSelect = el("train-epochs-select");
  const lrSelect = el("train-lr-select");
  const loraRankSelect = el("train-lora-rank-select");
  const loraLayersSelect = el("train-lora-layers-select");
  const seqLenSelect = el("train-seq-len-select");
  const batchSizeSelect = el("train-batch-size-select");
  const dataset = input instanceof HTMLTextAreaElement ? input.value : "";
  const epochs =
    epochSelect instanceof HTMLSelectElement
      ? Math.max(1, Math.min(20, Number(epochSelect.value) || 1))
      : 1;
  const learning_rate = lrSelect instanceof HTMLSelectElement ? Number(lrSelect.value) || 1e-5 : 1e-5;
  const lora_rank = loraRankSelect instanceof HTMLSelectElement ? Number(loraRankSelect.value) || 8 : 8;
  const lora_layers = loraLayersSelect instanceof HTMLSelectElement ? Number(loraLayersSelect.value) || 16 : 16;
  const max_seq_length = seqLenSelect instanceof HTMLSelectElement ? Number(seqLenSelect.value) || 2048 : 2048;
  const batch_size = batchSizeSelect instanceof HTMLSelectElement ? Number(batchSizeSelect.value) || 2 : 2;
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
  if (!profile) {
    state.trainStatus = "profile is required";
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
  state.trainStatus = `training profile ${profile}...`;
  renderTrainModelSelect();
  beginTrainProgress();
  try {
    const created = await postJson("/v1/models/train", {
      model_id: modelId,
      profile,
      samples,
      epochs,
      fine_tune_type: "qlora",
      learning_rate,
      lora_rank,
      lora_layers,
      max_seq_length,
      batch_size,
    });
    const taskId = created?.task_id;
    if (!taskId) {
      throw new Error("missing train task id");
    }
    state.trainStatus = `queued profile ${profile}`;
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

