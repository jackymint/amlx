function init() {
  bindChatAndTrainEvents();
  bindModelQuantAndRuntimeEvents();
  let _hfSearchTimer = null;
  on("hf-search-input", "input", () => {
    if (_hfSearchTimer) clearTimeout(_hfSearchTimer);
    _hfSearchTimer = setTimeout(() => void searchHFDatasets(), 350);
  });
  on("hf-catalog", "click", (e) => {
    const btn = e.target instanceof HTMLButtonElement ? e.target : e.target?.closest?.("button");
    if (!btn || btn.dataset.action !== "select-hf-dataset") return;
    selectHFDataset(btn.dataset.datasetId || "");
  });
  on("hf-prev", "click", () => { _hfPage = Math.max(1, _hfPage - 1); renderHFPage(); });
  on("hf-next", "click", () => { _hfPage = Math.min(Math.ceil(_hfResults.length / HF_PER_PAGE), _hfPage + 1); renderHFPage(); });
  on("hf-load-btn", "click", () => void fetchHFDataset());
  on("eval-submit", "click", () => void runEval());
  on("eval-model-select", "change", (e) => { if (e.target instanceof HTMLSelectElement) state.evalModel = e.target.value; });
  on("eval-upload-btn", "click", () => el("eval-file-input")?.click());
  on("eval-file-input", "change", async (event) => {
    const file = event.target instanceof HTMLInputElement ? event.target.files?.[0] : null;
    if (!file) return;
    const text = await file.text();
    const textarea = el("eval-dataset");
    if (textarea instanceof HTMLTextAreaElement) textarea.value = text;
    const meta = el("eval-file-meta");
    if (meta) meta.textContent = file.name;
    if (event.target instanceof HTMLInputElement) event.target.value = "";
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
