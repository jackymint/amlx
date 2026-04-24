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

