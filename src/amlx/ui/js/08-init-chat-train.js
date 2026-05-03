function bindChatAndTrainEvents() {
  for (const tab of TAB_NAMES) {
    on(`tab-${tab}`, "click", () => {
      setActiveTab(tab);
    });
  }

  on("cache-clear", "click", () => {
    void clearCache();
  });
  on("chat-send", "click", () => {
    void runPrompt();
  });
  on("chat-clear", "click", () => {
    clearChat();
  });
  on("chat-input", "keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void runPrompt();
    }
  });
  on("chat-model-select", "change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.selectedChatModel = target.value || state.configuredModel;
    renderChatModelSelect();
    if (!state.selectedChatModel) return;
    if (state.runtimeLoadedModels.includes(state.selectedChatModel)) return;
    void (async () => {
      const modelId = state.selectedChatModel;
      setChatModelLoading(true, modelId);
      const ok = await preloadModel(modelId);
      setChatModelLoading(false);
      if (!ok) {
        state.chat.push({
          role: "assistant",
          content: `Failed to preload model: ${modelId}`,
        });
        renderChat();
      }
    })();
  });

  on("train-model-select", "change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) return;
    state.selectedTrainModel = target.value || "";
    state.selectedTrainProfile = "";
    state.trainModelInfoFor = "";
    renderTrainModelSelect();
    renderTrainingModels();
  });
  on("train-profile-input", "input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    state.selectedTrainProfile = target.value || "";
  });
  on("train-submit", "click", () => {
    void requestTrain();
  });
  on("train-save-cancel", "click", () => {
    if (state.saveTrainRunning) return;
    setTrainSaveModalOpen(false);
  });
  on("train-save-confirm", "click", () => {
    void requestSaveMergedTrain();
  });
  on("train-save-modal", "click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.id !== "train-save-modal") return;
    if (state.saveTrainRunning) return;
    setTrainSaveModalOpen(false);
  });
  on("train-save-path", "keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    if (state.saveTrainRunning) return;
    void requestSaveMergedTrain();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const saveModal = el("train-save-modal");
    if (!(saveModal instanceof HTMLDivElement)) return;
    if (saveModal.hidden || state.saveTrainRunning) return;
    setTrainSaveModalOpen(false);
  });

  on("train-upload-btn", "click", () => {
    const fileInput = el("train-file-input");
    if (!(fileInput instanceof HTMLInputElement)) return;
    fileInput.click();
  });
  on("train-file-input", "change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const file = target.files?.[0];
    if (!file) return;
    void handleTrainFileUpload(file);
  });
}
