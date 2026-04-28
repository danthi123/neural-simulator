// Neural Simulator — Research Dashboard frontend
// Phase 1 vanilla JS. No build step. ES modules in the browser.
// Phase 2 adds the World tab (2D playback) wired up via world.js.
//
// All dynamic content (filenames, markdown body, JSON values) is rendered
// via textContent or escapeHTML — never via raw template-literal innerHTML.

import { setupWorldTab, loadRunIntoWorld } from "/static/world.js";
import { makeLineChart, makeBarChart, PALETTE_EXPORT as P } from "/static/charts.js";

// Switch to the World tab and load the given run
function openInWorld(name) {
  document.querySelector('nav button[data-tab="world"]').click();
  loadRunIntoWorld(name);
}

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHTML(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v != null) {
      node.setAttribute(k, v);
    }
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}

// ─────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────
function setupTabs() {
  $$("nav button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = btn.dataset.tab;
      $$("nav button").forEach((b) => b.classList.toggle("active", b === btn));
      $$("section.tab").forEach((s) =>
        s.classList.toggle("active", s.id === `tab-${t}`),
      );
      if (t === "findings" && !window._findingsLoaded) loadFindings();
      if (t === "info" && !window._infoLoaded) loadInfo();
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────
// Runs tab
// ─────────────────────────────────────────────────────────────────────────
let _allRuns = []; // Cached runs after last fetch — re-render on filter change
const compareSet = new Set(); // run names selected for comparison (max 3)

async function loadRuns() {
  const list = $("#runs-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    $("#runs-count").textContent = `${data.count} runs`;
    if (!data.runs.length) {
      const p = el("p", { class: "muted", style: "padding:16px" },
        "No runs yet — launch one from the Launch tab.");
      list.replaceChildren(p);
      return;
    }
    data.runs.sort((a, b) => {
      if (a.sum_finalQ == null) return 1;
      if (b.sum_finalQ == null) return -1;
      return a.sum_finalQ - b.sum_finalQ;
    });
    _allRuns = data.runs;
    renderRunsList();
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function renderRunsList() {
  const list = $("#runs-list");
  const hideSmoke = $("#filter-hide-smoke")?.checked ?? true;
  const hideIncomplete = $("#filter-hide-incomplete")?.checked ?? false;
  const search = ($("#filter-search")?.value ?? "").trim().toLowerCase();

  const filtered = _allRuns.filter((r) => {
    if (hideSmoke && /smoke/i.test(r.name)) return false;
    if (hideIncomplete && r.sum_finalQ == null) return false;
    if (search && !r.name.toLowerCase().includes(search)) return false;
    return true;
  });

  $("#runs-count").textContent =
    `${filtered.length}${filtered.length !== _allRuns.length ? `/${_allRuns.length}` : ""} runs`;

  list.replaceChildren();
  for (const r of filtered) {
    const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
    const isSelected = compareSet.has(r.name);
    const item = el("div", {
      class: "list-item" + (isSelected ? " compare-selected" : ""),
      dataset: { name: r.name },
    }, [
      el("div", { class: "name" }, r.name),
      el("div", { class: "meta" }, [
        metric("sum", sumStr),
        metric("seed", r.seed ?? "—"),
        metric("phases", r.n_phases),
      ]),
    ]);
    item.addEventListener("click", (ev) => {
      if (ev.shiftKey || ev.metaKey || ev.ctrlKey) {
        toggleCompareSelection(r.name);
      } else {
        loadRunDetail(r.name, item);
      }
    });
    list.appendChild(item);
  }
}

function toggleCompareSelection(name) {
  if (compareSet.has(name)) {
    compareSet.delete(name);
  } else if (compareSet.size < 3) {
    compareSet.add(name);
  }
  $("#compare-runs").disabled = compareSet.size < 2;
  $("#compare-runs").textContent = `Compare ${compareSet.size}`;
  renderRunsList();
}

function metric(label, value) {
  return el("span", { class: "metric" }, [
    el("span", { class: "label" }, label),
    el("span", { class: "value" }, String(value)),
  ]);
}

async function loadRunDetail(name, listItem) {
  const detail = $("#run-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#runs-list .list-item").forEach((el) =>
    el.classList.toggle("active", el === listItem),
  );
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    const playBtn = el("button", { class: "play-in-world", onclick: () => openInWorld(name) }, "▶ Play in World viz");
    const distCanvas = el("canvas", { class: "chart-canvas" });
    const rewardCanvas = el("canvas", { class: "chart-canvas" });
    const motorCanvas = el("canvas", { class: "chart-canvas chart-narrow" });
    detail.replaceChildren(
      el("h2", {}, name),
      el("div", {}, [
        metric("seed", data.seed ?? "—"),
        metric("n_steps", data.n_steps ?? "—"),
        metric("grid_size", data.grid_size ?? 8),
        metric("sum_finalQ", computeSumFinalQ(data)),
      ]),
      el("div", { style: "margin: 12px 0" }, playBtn),
      el("h3", {}, "Phase stats"),
      renderPhaseStats(data.phase_stats || []),
      el("h3", {}, "Distance over time"),
      el("div", { class: "chart-row" }, distCanvas),
      el("h3", {}, "Reward over time"),
      el("div", { class: "chart-row" }, rewardCanvas),
      el("h3", {}, "Action distribution per phase"),
      el("div", { class: "chart-row chart-narrow-wrap" }, motorCanvas),
      el("h3", {}, "Raw JSON"),
      el("pre", {}, JSON.stringify(summarizeRunData(data), null, 2)),
    );
    // Charts must be rendered AFTER the canvas elements are in the DOM so
    // clientWidth/Height resolve to non-zero values for the dpr setup.
    requestAnimationFrame(() => renderRunCharts(data, distCanvas, rewardCanvas, motorCanvas));
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

function computeSumFinalQ(data) {
  const fqs = (data.phase_stats || [])
    .map((p) => p.final_quarter_mean_distance)
    .filter((v) => v != null);
  return fqs.length ? fqs.reduce((a, b) => a + b, 0).toFixed(2) : "—";
}

/** Render the three run-detail charts: distance over time, reward over time,
 *  and per-phase action distribution bars. Phase boundaries shaded. */
function renderRunCharts(data, distCanvas, rewardCanvas, motorCanvas) {
  const phases = data.phase_stats || [];
  const phaseRanges = phases.map((ps, i) => ({
    start: ps.step_start ?? 0,
    end: ps.step_end ?? (data.n_steps ?? 0),
    label: `phase ${i} → goal (${ps.goal[0]},${ps.goal[1]})`,
    color: i % 2 === 0 ? "#161922" : "#1d2230",
  }));
  const goalChangeMarkers = (data.goal_change_steps || []).map((step) => ({
    x: step,
    label: "goal change",
    color: P.warn,
  }));

  // Distance over time
  const distChart = makeLineChart(distCanvas, {
    title: "Manhattan distance to goal",
    yLabel: "distance",
    yMin: 0,
    phaseRanges,
    markers: goalChangeMarkers,
  });
  distChart.updateData([
    { values: data.distance_log || [], color: P.accent, label: "distance" },
  ]);

  // Reward over time — use a moving average over 50 steps for readability.
  const rewardLog = data.reward_log || [];
  const window = 50;
  const rewardSmooth = [];
  let runningSum = 0;
  for (let i = 0; i < rewardLog.length; i++) {
    runningSum += rewardLog[i];
    if (i >= window) runningSum -= rewardLog[i - window];
    rewardSmooth.push(i >= window - 1 ? runningSum / window : null);
  }
  const rewardChart = makeLineChart(rewardCanvas, {
    title: `Reward (50-step moving avg)`,
    yLabel: "mean reward",
    yMin: -1, yMax: 1,
    phaseRanges,
    markers: goalChangeMarkers,
  });
  rewardChart.updateData([
    { values: rewardSmooth, color: P.warn, label: "reward (avg)" },
  ]);

  // Per-phase action distribution — stacked bars (one per phase)
  const motorChart = makeBarChart(motorCanvas, {
    title: "Action counts per phase (sum across all phases)",
    labels: ["N", "E", "S", "W"],
    colors: [P.accent, P.warn, P.bad, P.blue],
  });
  const totals = [0, 0, 0, 0];
  for (const ps of phases) {
    const ac = ps.action_counts || [];
    for (let i = 0; i < 4; i++) totals[i] += ac[i] || 0;
  }
  motorChart.updateData(totals);
}

/** Open a comparison view in the right detail panel: overlays distance
 *  curves of 2-3 selected runs on one chart for visual comparison. */
async function openComparisonView() {
  if (compareSet.size < 2) return;
  const detail = $("#run-detail");
  detail.replaceChildren(el("p", { class: "muted" }, "Loading comparison…"));

  // Fetch all selected runs in parallel
  const names = Array.from(compareSet);
  let datas;
  try {
    datas = await Promise.all(names.map(async (n) => {
      const r = await fetch(`/api/runs/${encodeURIComponent(n)}`);
      if (!r.ok) throw new Error(`${n}: ${r.status}`);
      return [n, await r.json()];
    }));
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed: ${e.message}`));
    return;
  }

  // Compute summary table
  const table = el("table", { class: "markdown", style: "width:100%; max-width:780px" });
  const colors = [P.accent, P.warn, P.blue];
  const headRow = el("tr", {}, [
    el("th", {}, ""),
    el("th", {}, "name"),
    el("th", {}, "seed"),
    el("th", {}, "P0 finalQ"),
    el("th", {}, "P1 finalQ"),
    el("th", {}, "sum"),
  ]);
  const bodyRows = datas.map(([name, d], i) => {
    const ps = d.phase_stats || [];
    const fq0 = ps[0]?.final_quarter_mean_distance;
    const fq1 = ps[1]?.final_quarter_mean_distance;
    const sum = (fq0 ?? 0) + (fq1 ?? 0);
    return el("tr", {}, [
      el("td", { style: `color:${colors[i]}` }, "●"),
      el("td", {}, name),
      el("td", {}, String(d.seed ?? "—")),
      el("td", {}, fq0 != null ? fq0.toFixed(2) : "—"),
      el("td", {}, fq1 != null ? fq1.toFixed(2) : "—"),
      el("td", {}, el("strong", {}, sum.toFixed(2))),
    ]);
  });

  const distCanvas = el("canvas", { class: "chart-canvas" });
  const rewardCanvas = el("canvas", { class: "chart-canvas" });

  detail.replaceChildren(
    el("h2", {}, `Comparing ${datas.length} runs`),
    el("div", { style: "margin-bottom:16px" }, [
      el("button", {
        class: "play-in-world",
        style: "margin-right:8px",
        onclick: () => { compareSet.clear(); $("#compare-runs").disabled = true; $("#compare-runs").textContent = "Compare 0"; renderRunsList(); openComparisonView(); },
      }, "Clear selection"),
      el("span", { class: "muted" }, "Tip: shift+click runs to add/remove from comparison"),
    ]),
    el("table", { class: "markdown" }, [el("thead", {}, headRow), el("tbody", {}, bodyRows)]),
    el("h3", {}, "Distance over time"),
    el("div", { class: "chart-row" }, distCanvas),
    el("h3", {}, "Reward over time (50-step moving avg)"),
    el("div", { class: "chart-row" }, rewardCanvas),
  );

  requestAnimationFrame(() => {
    // Build phase ranges from the FIRST selected run (assume similar structure)
    const refPhases = (datas[0][1].phase_stats || []).map((ps, i) => ({
      start: ps.step_start ?? 0,
      end: ps.step_end ?? (datas[0][1].n_steps ?? 0),
      label: `phase ${i}`,
      color: i % 2 === 0 ? "#161922" : "#1d2230",
    }));
    const refMarkers = (datas[0][1].goal_change_steps || []).map((step) => ({
      x: step, label: "goal change", color: P.warn,
    }));

    const distChart = makeLineChart(distCanvas, {
      title: "Distance to goal — overlay",
      yLabel: "distance",
      yMin: 0,
      phaseRanges: refPhases,
      markers: refMarkers,
    });
    distChart.updateData(datas.map(([n, d], i) => ({
      values: d.distance_log || [],
      color: colors[i],
      label: n,
    })));

    const rewardChart = makeLineChart(rewardCanvas, {
      title: "Reward — overlay",
      yLabel: "mean reward",
      yMin: -1, yMax: 1,
      phaseRanges: refPhases,
      markers: refMarkers,
    });
    const movingAvg = (arr, w) => {
      const out = []; let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= w) sum -= arr[i - w];
        out.push(i >= w - 1 ? sum / w : null);
      }
      return out;
    };
    rewardChart.updateData(datas.map(([n, d], i) => ({
      values: movingAvg(d.reward_log || [], 50),
      color: colors[i],
      label: n.slice(0, 30),
    })));
  });
}

function summarizeRunData(data) {
  const heavy = new Set([
    "motor_counts", "distance_log", "trajectory", "spike_counts",
    "place_cell_log", "goal_cell_log", "raw_phase1_motor_counts",
  ]);
  const out = {};
  for (const [k, v] of Object.entries(data)) {
    if (heavy.has(k)) {
      out[k] = Array.isArray(v) ? `[…${v.length} entries…]` : "[…large…]";
    } else {
      out[k] = v;
    }
  }
  return out;
}

function renderPhaseStats(stats) {
  if (!stats.length) return el("p", { class: "muted" }, "No phase stats.");
  const head = el("tr", {}, ["Phase", "Goal", "Steps", "finalQ", "mean dist"]
    .map((h) => el("th", {}, h)));
  const rows = stats.map((ps, i) => {
    const fq = ps.final_quarter_mean_distance ?? ps.finalQ;
    return el("tr", {}, [
      el("td", {}, String(i + 1)),
      el("td", {}, ps.goal ? `(${ps.goal[0]},${ps.goal[1]})` : "—"),
      el("td", {}, String(ps.n_steps ?? "—")),
      el("td", {}, el("strong", {}, fq != null ? fq.toFixed(2) : "—")),
      el("td", {}, ps.mean_distance != null ? ps.mean_distance.toFixed(2) : "—"),
    ]);
  });
  return el("table", { class: "markdown" }, [
    el("thead", {}, head),
    el("tbody", {}, rows),
  ]);
}

// ─────────────────────────────────────────────────────────────────────────
// Findings tab
// ─────────────────────────────────────────────────────────────────────────
async function loadFindings() {
  window._findingsLoaded = true;
  const list = $("#findings-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/findings");
    const data = await res.json();
    $("#findings-count").textContent = `${data.count} findings`;
    list.replaceChildren();
    for (const f of data.findings) {
      const item = el("div", { class: "list-item" }, [
        el("div", { class: "name" }, f.name),
      ]);
      item.addEventListener("click", () => loadFindingDetail(f.name, item));
      list.appendChild(item);
    }
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

async function loadFindingDetail(name, listItem) {
  const detail = $("#finding-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#findings-list .list-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  try {
    const res = await fetch(`/api/findings/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const text = await res.text();
    // renderMarkdown returns sanitized HTML (escapes input first, then injects
    // markdown-derived tags). Safe to set as innerHTML on a fresh div.
    const wrapper = el("div", { class: "markdown" });
    wrapper.innerHTML = renderMarkdown(text);
    detail.replaceChildren(wrapper);
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

// Minimal markdown-ish renderer. Always escapes input first, so the only
// HTML in the output is what *this* function emits — no user-controlled
// tags. For full fidelity (footnotes, autolinks, etc.) we'd swap in
// `marked` + DOMPurify later; for Phase 1 this is enough.
function renderMarkdown(src) {
  src = escapeHTML(src);
  src = src.replace(/```(\w*)\n([\s\S]*?)```/g, (_, _lang, body) =>
    `<pre>${body}</pre>`);
  src = src.replace(/^###### (.*)$/gm, "<h6>$1</h6>");
  src = src.replace(/^##### (.*)$/gm, "<h5>$1</h5>");
  src = src.replace(/^#### (.*)$/gm, "<h4>$1</h4>");
  src = src.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  src = src.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  src = src.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  src = src.replace(/((?:^\|.*\|\n)+)/gm, (block) => {
    const rows = block.trim().split("\n");
    const isAlign = (r) => /^\|[\s:|-]+\|$/.test(r.trim());
    const cells = rows
      .filter((r) => !isAlign(r))
      .map((r) => r.trim().slice(1, -1).split("|").map((c) => c.trim()));
    if (!cells.length) return block;
    const [header, ...body] = cells;
    let html = "<table><thead><tr>" +
      header.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
    for (const row of body) {
      html += "<tr>" + row.map((c) => `<td>${c}</td>`).join("") + "</tr>";
    }
    html += "</tbody></table>";
    return html;
  });
  src = src.replace(/((?:^[-*] .*\n?)+)/gm, (block) => {
    const items = block.trim().split("\n").map((l) => l.replace(/^[-*] /, ""));
    return "<ul>" + items.map((i) => `<li>${i}</li>`).join("") + "</ul>";
  });
  src = src.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
  src = src.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  src = src.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  src = src.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  // Links: [text](url) — but the captured text/url are HTML-escaped already,
  // so ampersands etc. are safe. We restrict the URL to http(s) and relative
  // paths to defang javascript: URLs.
  src = src.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => {
    const safe = /^(https?:|\/|\.\.\/|\.\/|#)/i.test(url) ? url : "#";
    return `<a href="${safe}" target="_blank" rel="noreferrer noopener">${text}</a>`;
  });
  src = src
    .split(/\n{2,}/)
    .map((p) => /^<(h\d|ul|ol|table|pre|blockquote)/.test(p.trim())
      ? p : `<p>${p}</p>`)
    .join("\n");
  return src;
}

// ─────────────────────────────────────────────────────────────────────────
// Launcher tab
// ─────────────────────────────────────────────────────────────────────────
function setupLauncher() {
  const form = $("#launch-form");
  const out = $("#launcher-output");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    out.replaceChildren();
    appendStatus(out, "Submitting…");

    const formData = new FormData(form);
    const extraStr = String(formData.get("extra_args") || "").trim();
    const body = {
      preset: String(formData.get("preset")),
      seed: parseInt(formData.get("seed"), 10),
      extra_args: extraStr ? extraStr.split(/\s+/) : [],
    };

    try {
      const res = await fetch("/api/runs/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const launch = await res.json();
      appendStatus(out, `Launched run_id=${launch.run_id}`);
      appendStatus(out, `cmd: ${launch.cmd.join(" ")}`);
      appendStatus(out, `out: ${launch.out_path}`);
      appendStatus(out, `streaming WebSocket at ${launch.ws_url}…`);
      tailWebSocket(launch.ws_url, out);
    } catch (e) {
      appendError(out, `Launch failed: ${e.message}`);
    }
  });
}

function tailWebSocket(path, out) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}${path}`);
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "stdout") {
      const div = el("div", { class: "stdout-line" }, msg.line);
      out.appendChild(div);
      out.scrollTop = out.scrollHeight;
    } else if (msg.type === "done") {
      appendStatus(out, `Run finished. returncode=${msg.returncode}`);
      appendStatus(out, `Output saved to: ${msg.out_path}`);
      loadRuns();
    }
  };
  ws.onerror = () => appendError(out, "WebSocket error");
}

function appendStatus(out, text) {
  out.appendChild(el("div", { class: "system" }, `>>> ${text}`));
  out.scrollTop = out.scrollHeight;
}
function appendError(out, text) {
  out.appendChild(el("div", { class: "error" }, `!!! ${text}`));
  out.scrollTop = out.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────
// Info tab
// ─────────────────────────────────────────────────────────────────────────
async function loadInfo() {
  window._infoLoaded = true;
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    $("#info-output").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#info-output").textContent = `Error: ${e.message}`;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────
setupTabs();
setupLauncher();
setupWorldTab();
loadRuns();

$("#refresh-runs").addEventListener("click", loadRuns);
$("#refresh-findings").addEventListener("click", loadFindings);

$("#filter-hide-smoke")?.addEventListener("change", renderRunsList);
$("#filter-hide-incomplete")?.addEventListener("change", renderRunsList);
$("#filter-search")?.addEventListener("input", renderRunsList);
$("#compare-runs")?.addEventListener("click", openComparisonView);

// Auto-refresh runs list every 10s when the Runs tab is active. Lets the
// user see new runs land without manual refresh.
setInterval(() => {
  const runsTabActive = document.querySelector("#tab-runs")?.classList.contains("active");
  if (runsTabActive) loadRuns();
}, 10_000);
