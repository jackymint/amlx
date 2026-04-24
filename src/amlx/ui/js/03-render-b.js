function renderDownloads() {
  const node = el("downloads");
  if (!node) return;
  if (!state.downloads.length) {
    node.innerHTML = emptyJobCard("No downloads yet", "Downloaded models and progress tasks will appear here.");
    return;
  }

  node.innerHTML = state.downloads
    .slice(0, 10)
    .map((task) => {
      const pct = Math.max(0, Math.min(100, Number(task.progress || 0)));
      const isActive = task.status === "queued" || task.status === "downloading";
      return `
        <div class="job">
          <strong>${escapeHtml(task.model_id)}</strong>
          <span>${task.message || task.status}${task.error ? " • " + escapeHtml(task.error) : ""}</span>
          ${isActive ? `<div class="dl-progress-row"><span class="dl-pct">${pct}%</span><div class="progress-track" style="flex:1"><div class="progress-fill" style="width:${pct}%"></div></div></div>` : ""}
        </div>
      `;
    })
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
  const profileInput = el("train-profile-input");
  const datasetInput = el("train-dataset");
  const fileMeta = el("train-file-meta");
  const status = el("train-status");
  if (!(select instanceof HTMLSelectElement)) return;

  const options = buildModelOptions({ loadedLabel: "loaded", installedLabel: "installed" });

  if (!options.length) {
    state.selectedTrainModel = "";
    select.innerHTML = `<option value="">No loaded/installed model</option>`;
    select.disabled = true;
    if (button instanceof HTMLButtonElement) button.disabled = true;
    if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = true;
    if (epochsSelect instanceof HTMLSelectElement) epochsSelect.disabled = true;
    if (profileInput instanceof HTMLInputElement) profileInput.disabled = true;
    if (datasetInput instanceof HTMLTextAreaElement) datasetInput.disabled = true;
  } else {
    state.selectedTrainModel = options.some((x) => x.id === state.selectedTrainModel) ? state.selectedTrainModel : "";
    if (state.selectedTrainModel && state.selectedTrainModel !== state.trainModelInfoFor) {
      state.trainModelInfoFor = state.selectedTrainModel;
      void fetchTrainModelInfo(state.selectedTrainModel);
    }
    select.disabled = state.trainRunning;
    state.selectedTrainModel = renderSelectOptions(select, options, state.selectedTrainModel, "Select model...");
    if (button instanceof HTMLButtonElement) button.disabled = state.trainRunning || !state.selectedTrainModel;
    if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = state.trainRunning || !state.selectedTrainModel;
    if (epochsSelect instanceof HTMLSelectElement) epochsSelect.disabled = state.trainRunning;
    if (profileInput instanceof HTMLInputElement) profileInput.disabled = state.trainRunning;
    if (datasetInput instanceof HTMLTextAreaElement) datasetInput.disabled = state.trainRunning;
    for (const id of ["train-lr-select", "train-lora-rank-select", "train-lora-layers-select", "train-seq-len-select"]) {
      const s = el(id);
      if (s instanceof HTMLSelectElement) s.disabled = state.trainRunning;
    }
  }

  if (profileInput instanceof HTMLInputElement) {
    if (!state.selectedTrainProfile && state.selectedTrainModel) {
      state.selectedTrainProfile = state.selectedTrainModel.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    }
    if (!state.selectedTrainProfile) state.selectedTrainProfile = "my-profile";
    profileInput.value = state.selectedTrainProfile;
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

  // Group by profile_slug, newest task first within each group
  const groups = new Map();
  for (const item of state.trainingModels) {
    const itemModel = String(item.model_id || item.effective_model || "");
    if (state.selectedTrainModel && itemModel && itemModel !== state.selectedTrainModel) continue;
    const key = String(item.profile_slug || item.profile || "legacy-profile");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }

  if (!groups.size) {
    node.innerHTML = emptyJobCard("No training history for this model", "Train this model to see results here.");
    return;
  }

  node.innerHTML = Array.from(groups.values())
    .map((runs) => {
      const latest = runs[0]; // already sorted newest-first from server
      const profile = latest.profile || "legacy-profile";
      const totalRounds = runs.length;
      const progress = Math.max(0, Math.min(100, Number(latest.progress || 0)));
      const stateText = `${latest.status || "unknown"} • ${progress}%`;
      const canSave = latest.status === "completed" && latest.adapter_path;
      const savingThisTask = state.saveTrainRunning && state.saveTrainTaskId === latest.task_id;
      const updated = new Date(Number(latest.updated_at || latest.finished_at || 0) * 1000);
      const dateLabel = Number.isFinite(updated.getTime()) ? updated.toLocaleString() : "unknown";

      const historyRows = totalRounds > 1
        ? runs.slice(1).map((r) => {
            const d = new Date(Number(r.updated_at || r.finished_at || 0) * 1000);
            const dl = Number.isFinite(d.getTime()) ? d.toLocaleString() : "unknown";
            return `<span style="opacity:0.6;font-size:0.85em">round ${r.round || "?"} • ${r.status} • ${r.train_samples || 0} samples • ${dl}</span>`;
          }).join("")
        : "";

      return `
        <div class="job">
          <strong>${profile}</strong>
          <span>${stateText} • round: ${latest.round || 1}${totalRounds > 1 ? ` / ${totalRounds} total` : ""} • model: ${latest.model_id || latest.effective_model}</span>
          <span>type: ${latest.fine_tune_type || "qlora"} • samples: ${latest.train_samples || 0}</span>
          <div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div>
          <span>${latest.message || ""}${latest.error ? ` • ${latest.error}` : ""} • updated: ${dateLabel}</span>
          ${
            canSave
              ? `<div class="job-actions"><button type="button" class="ghost mini" data-action="save-merged-train" data-task-id="${latest.task_id}" data-profile="${profile}" ${savingThisTask ? "disabled" : ""}>${savingThisTask ? "Saving..." : "Save"}</button></div>`
              : ""
          }
          ${latest.merged_path ? `<span>saved: ${latest.merged_path}</span>` : ""}
          ${historyRows}
        </div>
      `;
    })
    .join("");
}
