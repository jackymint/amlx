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

    state.configuredModel = runtime.configured_model || "";
    state.runtimeLoadedModels = runtime.loaded_models || [];
    state.gpuLimitPercent = Number(runtime.gpu_limit_percent || state.gpuLimitPercent || 100);
    state.gpuLimitAdapter = runtime.gpu_limit_adapter || null;
    updateStats(stats);
    renderGpuLimit();
    renderCatalog();
    renderLoadedModels();
    renderChatModelSelect();
    renderTrainModelSelect();
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
    renderLoadedModels();
    renderChatModelSelect();
    renderTrainModelSelect();
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
      messages: state.chat.filter((m) => !m.pending),
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
    await postJson("/v1/models/download", { model_id: modelId });
    await refreshModels();
  } catch {
    // Ignore UI-level errors; detailed reason is available in task error when created.
  }
}

async function fetchTrainModelInfo(modelId) {
  const node = el("train-model-info");
  if (!node) return;
  if (!modelId) { node.innerHTML = ""; return; }
  try {
    const info = await fetchJson(`/v1/models/info?model_id=${encodeURIComponent(modelId)}`);
    const layers = Number(info.num_hidden_layers) || 0;
    const hidden = Number(info.hidden_size) || 0;
    const maxPos = Number(info.max_position_embeddings) || 0;
    const modelType = String(info.model_type || "");

    if (!layers && !hidden) { node.innerHTML = ""; return; }

    // Auto-suggest params
    const suggestLoraLayers = layers ? Math.min(layers, 16) : 16;
    const suggestLoraRank = hidden >= 4096 ? 16 : hidden >= 2048 ? 8 : 4;
    const suggestSeqLen = maxPos >= 4096 ? 2048 : maxPos >= 2048 ? 2048 : 1024;

    const setSelect = (id, val) => {
      const s = el(id);
      if (!(s instanceof HTMLSelectElement)) return;
      const opt = Array.from(s.options).find((o) => Number(o.value) === val);
      if (opt) s.value = String(val);
    };
    setSelect("train-lora-layers-select", suggestLoraLayers);
    setSelect("train-lora-rank-select", suggestLoraRank);
    setSelect("train-seq-len-select", suggestSeqLen);

    const chips = [
      modelType ? `<span class="train-info-chip">${modelType}</span>` : "",
      layers ? `<span class="train-info-chip">${layers} layers</span>` : "",
      hidden ? `<span class="train-info-chip">hidden ${hidden}</span>` : "",
      maxPos ? `<span class="train-info-chip">ctx ${maxPos.toLocaleString()}</span>` : "",
    ].filter(Boolean).join("");
    node.innerHTML = `<div class="train-info-bar">${chips}</div>`;
  } catch {
    node.innerHTML = "";
  }
}

async function requestUnload(modelId) {
  try {
    await postJson("/v1/models/unload", { model_id: modelId });
    if (modelId && (modelId === state.selectedChatModel || modelId === state.configuredModel)) {
      state.selectedChatModel = "";
      clearChat();
    }
    await refresh();
    await refreshModels();
  } catch {
    // no-op
  }
}

