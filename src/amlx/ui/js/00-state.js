const state = {
  pulse: [],
  lastProcessed: 0,
  catalog: [],
  downloads: [],
  installed: [],
  chat: [],
  modelQuery: "",
  modelLoading: false,
  modelSearchError: "",
  modelSystem: null,
  activeTab: "metrics",
  modelPage: 1,
  modelPerPage: 4,
  modelTotal: 0,
  configuredModel: "",
  runtimeLoadedModels: [],
  trainingModels: [],
  selectedChatModel: "",
  selectedTrainModel: "",
  trainModelInfoFor: "",
  selectedTrainProfile: "customer-support-v1",
  uploadedTrainSamples: [],
  uploadedTrainFileName: "",
  chatModelLoading: false,
  chatModelLoadingId: "",
  trainRunning: false,
  trainStatus: "",
  trainProgress: 0,
  gpuLimitPercent: 100,
  gpuLimitSaving: false,
  gpuLimitAdapter: null,
  saveTrainTaskId: "",
  saveTrainProfile: "",
  saveTrainRunning: false,
  saveTrainStatus: "",
  quantizeJobs: [],
  selectedQuantModel: "",
  quantRunning: false,
  quantStatus: "",
  importResolvedPath: "",
  importFolderFiles: null,
  evalModel: "",
  evalResults: [],
  evalRunning: false,
};

let trainProgressTimer = null;
const TAB_NAMES = ["metrics", "chat", "models", "training", "quantize", "eval"];

const el = (id) => document.getElementById(id);

function setText(id, value) {
  const node = el(id);
  if (node) node.textContent = String(value);
}

function on(id, eventName, handler) {
  el(id)?.addEventListener(eventName, handler);
}

function setChip(id, message, visible = true) {
  const chip = el(id);
  if (!chip) return;
  chip.textContent = String(message || "");
  chip.style.display = visible ? "" : "none";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function compactSample(value) {
  const text = String(value || "").trim();
  return text ? text : "";
}

function extractSamplesFromItem(item) {
  if (typeof item === "string") {
    const sample = compactSample(item);
    return sample ? [sample] : [];
  }

  if (Array.isArray(item)) {
    return item.flatMap((entry) => extractSamplesFromItem(entry));
  }

  if (!item || typeof item !== "object") return [];

  const obj = item;
  const directKeys = ["text", "content", "sample", "value"];
  for (const key of directKeys) {
    if (typeof obj[key] === "string") {
      const sample = compactSample(obj[key]);
      if (sample) return [sample];
    }
  }

  if (typeof obj.prompt === "string" || typeof obj.response === "string") {
    const sample = compactSample(`${obj.prompt || ""}\n${obj.response || ""}`);
    return sample ? [sample] : [];
  }
  if (typeof obj.question === "string" || typeof obj.answer === "string") {
    const sample = compactSample(`Q: ${obj.question || ""}\nA: ${obj.answer || ""}`);
    return sample ? [sample] : [];
  }
  if (
    typeof obj.instruction === "string" ||
    typeof obj.input === "string" ||
    typeof obj.output === "string"
  ) {
    const sample = compactSample(
      `Instruction: ${obj.instruction || ""}\nInput: ${obj.input || ""}\nOutput: ${obj.output || ""}`,
    );
    return sample ? [sample] : [];
  }

  if (Array.isArray(obj.messages)) {
    const lines = obj.messages
      .map((m) => {
        if (!m || typeof m !== "object") return "";
        const role = compactSample(m.role || m.from || "user").toUpperCase();
        const content = compactSample(m.content || m.value || "");
        return role && content ? `${role}: ${content}` : "";
      })
      .filter(Boolean);
    const sample = compactSample(lines.join("\n"));
    return sample ? [sample] : [];
  }
  if (Array.isArray(obj.conversations)) {
    const lines = obj.conversations
      .map((m) => {
        if (!m || typeof m !== "object") return "";
        const role = compactSample(m.role || m.from || "user").toUpperCase();
        const content = compactSample(m.content || m.value || "");
        return role && content ? `${role}: ${content}` : "";
      })
      .filter(Boolean);
    const sample = compactSample(lines.join("\n"));
    return sample ? [sample] : [];
  }

  const nestedKeys = ["samples", "data", "records", "items", "examples"];
  for (const key of nestedKeys) {
    if (Array.isArray(obj[key])) {
      return obj[key].flatMap((entry) => extractSamplesFromItem(entry));
    }
  }

  return [];
}

function parseTrainJson(text) {
  const parsed = JSON.parse(text);
  const samples = extractSamplesFromItem(parsed).filter(Boolean);
  return [...new Set(samples)];
}

function parseTrainJsonl(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const out = [];
  for (const line of lines) {
    const parsed = JSON.parse(line);
    out.push(...extractSamplesFromItem(parsed));
  }
  return [...new Set(out.filter(Boolean))];
}

