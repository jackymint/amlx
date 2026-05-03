async function requestSaveMergedTrain() {
  const taskId = state.saveTrainTaskId;
  const profile = String(state.saveTrainProfile || state.selectedTrainProfile || "").trim();
  const input = el("train-save-path");
  const outputPath = input instanceof HTMLInputElement ? String(input.value || "").trim() : "";
  if (!profile && !taskId) {
    state.saveTrainStatus = "missing profile";
    renderTrainSaveModal();
    return;
  }
  if (!outputPath) {
    state.saveTrainStatus = "output path is required";
    renderTrainSaveModal();
    return;
  }
  state.saveTrainRunning = true;
  state.saveTrainStatus = "saving merged model...";
  renderTrainingModels();
  renderTrainSaveModal();
  try {
    const body = await postJson("/v1/models/train/save", { task_id: null, profile: profile || null, output_path: outputPath });
    state.saveTrainStatus = `saved to ${body?.merged_path || outputPath}`;
    renderTrainSaveModal();
    await refreshModels();
    setTimeout(() => {
      if (!state.saveTrainRunning) setTrainSaveModalOpen(false);
    }, 350);
  } catch (err) {
    state.saveTrainStatus = `failed: ${err}`;
    renderTrainSaveModal();
  } finally {
    state.saveTrainRunning = false;
    renderTrainingModels();
    renderTrainSaveModal();
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

async function requestImportLocal() {
  const files = state.importFolderFiles;
  const idInput = el("import-id-input");
  const submitBtn = el("import-submit");
  if (!files || files.length === 0) return;
  const modelId = idInput instanceof HTMLInputElement ? idInput.value.trim() : "";
  if (submitBtn instanceof HTMLButtonElement) submitBtn.disabled = true;

  const progressWrap = el("import-progress-wrap");
  const progressFill = el("import-progress-fill");
  const progressText = el("import-progress-text");
  const statusEl = el("import-status");
  if (progressWrap) progressWrap.style.display = "flex";
  if (progressFill) progressFill.style.width = "0%";
  if (progressText) progressText.textContent = "0%";
  if (statusEl) statusEl.style.display = "none";

  const form = new FormData();
  for (const file of files) form.append("files", file, file.webkitRelativePath);
  if (modelId) form.append("model_id", modelId);

  try {
    const body = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/v1/models/import");
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const pct = Math.round((e.loaded / e.total) * 100);
        if (progressFill) progressFill.style.width = `${pct}%`;
        if (progressText) progressText.textContent = `${pct}%`;
      };
      xhr.onload = () => {
        try {
          const data = JSON.parse(xhr.responseText);
          if (xhr.status >= 400) reject(new Error(data?.detail || `HTTP ${xhr.status}`));
          else resolve(data);
        } catch { reject(new Error(`HTTP ${xhr.status}`)); }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(form);
    });

    if (progressFill) progressFill.style.width = "100%";
    if (progressText) progressText.textContent = "100%";
    if (statusEl) { statusEl.textContent = `imported: ${body.model_id}`; statusEl.style.display = ""; }
    state.importResolvedPath = "";
    state.importFolderFiles = null;
    const display = el("import-path-display");
    if (display) display.textContent = "";
    if (idInput instanceof HTMLInputElement) idInput.value = "";
    const idRow = el("import-id-row");
    const actions = el("import-actions");
    if (idRow) idRow.style.display = "none";
    if (actions) actions.style.display = "none";
    setTimeout(() => { if (progressWrap) progressWrap.style.display = "none"; }, 1200);
    await refreshModels();
  } catch (err) {
    if (progressWrap) progressWrap.style.display = "none";
    if (statusEl) { statusEl.textContent = `failed: ${err}`; statusEl.style.display = ""; }
  } finally {
    if (submitBtn instanceof HTMLButtonElement) submitBtn.disabled = false;
  }
}

async function requestDeleteInstalled(modelId) {
  try {
    await postJson("/v1/models/delete", { model_id: modelId });
    if (modelId && modelId === state.selectedTrainModel) {
      state.selectedTrainModel = "";
    }
    await refresh();
    await refreshModels();
    await fetchCatalogOnlineOrCurated();
  } catch {
    // no-op
  }
}

