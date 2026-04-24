function init() {
  bindChatAndTrainEvents();
  bindModelQuantAndRuntimeEvents();

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
