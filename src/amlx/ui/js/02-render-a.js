function setActiveTab(tab) {
  state.activeTab =
    tab === "models" ? "models"
    : tab === "chat" ? "chat"
    : tab === "training" ? "training"
    : tab === "quantize" ? "quantize"
    : "metrics";
  for (const name of TAB_NAMES) {
    const panel = el(`panel-${name}`);
    const tabBtn = el(`tab-${name}`);
    const isActive = state.activeTab === name;
    if (panel) panel.classList.toggle("hidden", !isActive);
    if (tabBtn) {
      tabBtn.classList.toggle("active", isActive);
      tabBtn.setAttribute("aria-selected", isActive ? "true" : "false");
    }
  }
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
    const body = await postJson("/v1/runtime/power", { gpu_limit_percent: next });
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
    node.innerHTML = `<div class="catalog-loading"><span class="catalog-searching">Searching models...</span></div>`;
    return;
  }
  if (state.modelSearchError) {
    node.innerHTML = `<div class="catalog-empty"><span>${escapeHtml(state.modelSearchError)}</span></div>`;
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
