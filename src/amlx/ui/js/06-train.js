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
  const profile = String(state.selectedTrainProfile || "").trim();
  const input = el("train-dataset");
  const epochSelect = el("train-epochs-select");
  const lrSelect = el("train-lr-select");
  const loraRankSelect = el("train-lora-rank-select");
  const loraLayersSelect = el("train-lora-layers-select");
  const seqLenSelect = el("train-seq-len-select");
  const dataset = input instanceof HTMLTextAreaElement ? input.value : "";
  const epochs =
    epochSelect instanceof HTMLSelectElement
      ? Math.max(1, Math.min(20, Number(epochSelect.value) || 1))
      : 1;
  const learning_rate = lrSelect instanceof HTMLSelectElement ? Number(lrSelect.value) || 1e-5 : 1e-5;
  const lora_rank = loraRankSelect instanceof HTMLSelectElement ? Number(loraRankSelect.value) || 8 : 8;
  const lora_layers = loraLayersSelect instanceof HTMLSelectElement ? Number(loraLayersSelect.value) || 16 : 16;
  const max_seq_length = seqLenSelect instanceof HTMLSelectElement ? Number(seqLenSelect.value) || 2048 : 2048;
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

