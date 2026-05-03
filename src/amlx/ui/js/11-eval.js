function renderEvalModelSelect() {
  const select = el("eval-model-select");
  if (!(select instanceof HTMLSelectElement)) return;
  const options = buildModelOptions({ loadedLabel: "loaded", installedLabel: "installed" });
  state.evalModel = renderSelectOptions(select, options, state.evalModel, "Select model...");
}

function extractKeywords(str) {
  return str.toLowerCase().split(/[\s"'{}()\[\]:,;]+/).filter((w) => w.length > 3);
}

function scoreMatch(expected, response) {
  const resp = response.toLowerCase();
  try {
    const exp = JSON.parse(expected);
    const keywords = [];
    // tool_use format
    const toolUse = exp.tool_use || exp;
    if (toolUse.tool_name) keywords.push(String(toolUse.tool_name).toLowerCase());
    const input = toolUse.tool_input || {};
    for (const v of Object.values(input)) {
      if (typeof v === "string") extractKeywords(v).forEach((w) => keywords.push(w));
    }
    if (!keywords.length) return false;
    return keywords.every((w) => resp.includes(w));
  } catch {
    const words = extractKeywords(expected);
    if (!words.length) return false;
    return words.every((w) => resp.includes(w));
  }
}

function parseSampleObj(obj) {
  const prompt = String(obj.prompt || obj.instruction || obj.input || obj.text || "").trim();
  const expected = String(obj.response || obj.output || obj.answer || "").trim();
  return prompt ? { prompt, expected } : null;
}

function parseEvalSamples(text) {
  const trimmed = text.trim();
  // try JSON array first
  if (trimmed.startsWith("[")) {
    try {
      const arr = JSON.parse(trimmed);
      if (Array.isArray(arr)) {
        return arr.map(parseSampleObj).filter(Boolean);
      }
    } catch { /* fall through to JSONL */ }
  }
  // JSONL or plain text
  const samples = [];
  for (const line of trimmed.split("\n").map((l) => l.trim()).filter(Boolean)) {
    try {
      const obj = JSON.parse(line);
      const s = parseSampleObj(obj);
      if (s) samples.push(s);
    } catch {
      samples.push({ prompt: line, expected: "" });
    }
  }
  return samples;
}

function renderEvalResults() {
  const node = el("eval-results");
  if (!node) return;
  if (!state.evalResults.length) { node.innerHTML = ""; return; }

  node.innerHTML = state.evalResults.map((r, i) => {
    const hasPassed = r.pass === true;
    const hasFailed = !r.pending && !hasPassed && (r.error || r.expected);
    const statusClass = r.pending ? "eval-pending"
      : hasPassed ? "eval-pass"
      : hasFailed ? "eval-fail"
      : "eval-done";
    const badge = r.pending ? "…"
      : r.error ? "error"
      : hasPassed ? "pass"
      : r.expected ? "fail"
      : "done";
    const expectedHtml = r.expected
      ? `<div class="eval-expected"><span class="eval-label">expected</span><span>${escapeHtml(r.expected)}</span></div>`
      : "";
    const responseHtml = r.response != null
      ? `<div class="eval-response"><span class="eval-label">response</span><span>${escapeHtml(r.response)}</span></div>`
      : "";
    return `
      <div class="eval-card ${statusClass}">
        <div class="eval-card-head">
          <span class="eval-index">#${i + 1}</span>
          <span class="eval-prompt">${escapeHtml(r.prompt)}</span>
          <span class="eval-badge">${badge}</span>
        </div>
        ${expectedHtml}
        ${responseHtml}
        ${r.error ? `<div class="eval-error">${escapeHtml(r.error)}</div>` : ""}
      </div>`;
  }).join("");
}

async function runEval() {
  const modelSelect = el("eval-model-select");
  const textarea = el("eval-dataset");
  const submitBtn = el("eval-submit");
  const statusEl = el("eval-status");
  const progressWrap = el("eval-progress-wrap");
  const progressFill = el("eval-progress-fill");
  const progressText = el("eval-progress-text");

  const model = modelSelect instanceof HTMLSelectElement ? modelSelect.value : "";
  const text = textarea instanceof HTMLTextAreaElement ? textarea.value.trim() : "";
  if (!model) { if (statusEl) { statusEl.textContent = "select a model first"; statusEl.style.display = ""; } return; }
  if (!text) { if (statusEl) { statusEl.textContent = "paste test prompts first"; statusEl.style.display = ""; } return; }

  const samples = parseEvalSamples(text);
  if (!samples.length) { if (statusEl) { statusEl.textContent = "no samples found"; statusEl.style.display = ""; } return; }

  const maxTokens = Number(el("eval-max-tokens-select") instanceof HTMLSelectElement ? el("eval-max-tokens-select").value : 256);
  const temperature = Number(el("eval-temperature-select") instanceof HTMLSelectElement ? el("eval-temperature-select").value : 0);

  state.evalRunning = true;
  state.evalResults = samples.map((s) => ({ ...s, response: null, error: null, pending: true }));
  if (submitBtn instanceof HTMLButtonElement) submitBtn.disabled = true;
  if (statusEl) statusEl.style.display = "none";
  if (progressWrap) progressWrap.style.display = "flex";
  renderEvalResults();

  for (let i = 0; i < samples.length; i++) {
    const pct = Math.round((i / samples.length) * 100);
    if (progressFill) progressFill.style.width = `${pct}%`;
    if (progressText) progressText.textContent = `${i} / ${samples.length}`;

    try {
      const body = await postJson("/v1/chat/completions", {
        model,
        messages: [{ role: "user", content: samples[i].prompt }],
        max_tokens: maxTokens,
        temperature,
        no_cache: true,
      });
      const response = body.choices?.[0]?.message?.content || "(empty)";
      state.evalResults[i].response = response;
      state.evalResults[i].pass = state.evalResults[i].expected
        ? scoreMatch(state.evalResults[i].expected, response)
        : null;
    } catch (err) {
      state.evalResults[i].error = String(err);
      state.evalResults[i].pass = false;
    }
    state.evalResults[i].pending = false;
    renderEvalResults();
  }

  if (progressFill) progressFill.style.width = "100%";
  if (progressText) progressText.textContent = `${samples.length} / ${samples.length}`;

  const withExpected = state.evalResults.filter((r) => r.expected && r.response != null && !r.error);
  const matched = withExpected.filter((r) => scoreMatch(r.expected, r.response));

  const errors = state.evalResults.filter((r) => r.error).length;
  let summary = `done — ${samples.length} prompts`;
  if (withExpected.length > 0) {
    const pct = Math.round((matched.length / withExpected.length) * 100);
    summary += ` · ${matched.length}/${withExpected.length} matched (${pct}%)`;
    if (pct >= 80) summary += " · model เข้าใจดี";
    else if (pct >= 50) summary += " · model เข้าใจบางส่วน";
    else summary += " · model ยังไม่เข้าใจ ควร train เพิ่ม";
  }
  if (errors > 0) summary += ` · ${errors} error`;
  if (statusEl) { statusEl.textContent = summary; statusEl.style.display = ""; }
  if (submitBtn instanceof HTMLButtonElement) submitBtn.disabled = false;
  state.evalRunning = false;
  setTimeout(() => { if (progressWrap) progressWrap.style.display = "none"; }, 1000);
}
