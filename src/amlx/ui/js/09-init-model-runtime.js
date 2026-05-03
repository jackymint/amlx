function bindModelQuantAndRuntimeEvents() {
  on("catalog", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestDownload(modelId);
  });
  on("loaded-models", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "unload-model") return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestUnload(modelId);
  });
  on("installed", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "delete-installed") return;
    const modelId = target.dataset.modelId;
    if (!modelId) return;
    void requestDeleteInstalled(modelId);
  });
  on("training-models", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "save-merged-train") return;
    const taskId = target.dataset.taskId;
    const profile = target.dataset.profile || "";
    if (!taskId) return;
    openTrainSaveModal(taskId, profile);
  });
  on("model-search", "input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    state.modelQuery = target.value || "";
    state.modelPage = 1;
    debouncedFetchCatalog();
  });
  on("model-prev", "click", () => {
    if (state.modelPage <= 1) return;
    state.modelPage -= 1;
    void fetchCatalogOnlineOrCurated().then(() => el("catalog")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  });
  on("model-next", "click", () => {
    const totalPages = Math.max(1, Math.ceil((state.modelTotal || 0) / state.modelPerPage));
    if (state.modelPage >= totalPages) return;
    state.modelPage += 1;
    void fetchCatalogOnlineOrCurated().then(() => el("catalog")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  });

  on("quant-model-select", "change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.selectedQuantModel = target.value;
    const outputInput = el("quant-output-input");
    if (outputInput instanceof HTMLInputElement) {
      outputInput.value = "";
      outputInput.dataset.auto = "";
    }
    renderQuantModelSelect();
  });
  on("quant-bits-select", "change", () => {
    const outputInput = el("quant-output-input");
    if (outputInput instanceof HTMLInputElement) outputInput.dataset.auto = "";
    autoFillQuantOutput();
  });
  on("quant-output-input", "input", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement) target.dataset.auto = "";
  });
  on("import-submit", "click", () => void requestImportLocal());
  on("import-folder-picker", "change", (event) => {
    const files = event.target instanceof HTMLInputElement ? event.target.files : null;
    if (!files || files.length === 0) return;
    const rel = files[0].webkitRelativePath;
    const folderName = rel.split("/")[0];
    if (!folderName) return;
    const display = el("import-path-display");
    const idInput = el("import-id-input");
    const idRow = el("import-id-row");
    const actions = el("import-actions");
    state.importResolvedPath = folderName;
    state.importFolderFiles = Array.from(files);
    if (display) display.textContent = folderName;
    if (idInput instanceof HTMLInputElement) idInput.value = folderName;
    if (idRow) idRow.style.display = "";
    if (actions) actions.style.display = "";
    if (event.target instanceof HTMLInputElement) event.target.value = "";
  });
  on("quant-submit", "click", () => void requestQuantize());
  on("quant-jobs", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.dataset.action !== "cancel-quantize") return;
    const taskId = target.dataset.taskId;
    if (!taskId) return;
    target.disabled = true;
    void fetch(`/v1/models/quantize/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });
  });

  on("gpu-limit-range", "input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    state.gpuLimitPercent = Math.max(20, Math.min(100, Number(target.value) || 100));
    renderGpuLimit();
    debouncedGpuLimitSave();
  });
}
