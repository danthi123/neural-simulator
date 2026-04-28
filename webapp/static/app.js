// Neural Simulator — Research Dashboard frontend
// Phase 1 vanilla JS. No build step. ES modules in the browser.
// Phase 2 adds the World tab (2D playback) wired up via world.js.
//
// All dynamic content (filenames, markdown body, JSON values) is rendered
// via textContent or escapeHTML — never via raw template-literal innerHTML.

import { setupWorldTab, loadRunIntoWorld } from "/static/world.js";
import { makeLineChart, makeBarChart, PALETTE_EXPORT as P } from "/static/charts.js";
import {
  toast, loadState, saveState, showSkeleton,
  registerShortcut, listShortcuts,
  fmtRelTime, detectExperiment, categorizeExperiment,
  mean, stdev,
} from "/static/ui.js";

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
      saveState({ activeTab: t });
      if (t === "findings" && !window._findingsLoaded) loadFindings();
      if (t === "info" && !window._infoLoaded) loadInfo();
      if (t === "overview" && !window._overviewLoaded) loadOverview();
      if (t === "experiments" && !window._experimentsLoaded) loadExperiments();
    });
  });
}

function activateTab(tabName) {
  const btn = document.querySelector(`nav button[data-tab="${tabName}"]`);
  if (btn) btn.click();
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
    const extras = extraStr ? extraStr.split(/\s+/) : [];

    // Grid size + n_hippocampus_per_layer are exposed as separate fields
    // because they're the most-asked-for custom-world knob. Threaded into
    // extra_args (the runner's CLI). Skip when default (8 / 64) to keep the
    // command line clean.
    const gridSize = parseInt(formData.get("grid_size"), 10);
    const nHippo = parseInt(formData.get("n_hippocampus_per_layer"), 10);
    if (gridSize && gridSize !== 8) {
      extras.push("--grid-size", String(gridSize));
    }
    if (nHippo && nHippo !== 64) {
      extras.push("--n-hippocampus-per-layer", String(nHippo));
    }

    const body = {
      preset: String(formData.get("preset")),
      seed: parseInt(formData.get("seed"), 10),
      extra_args: extras,
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
      toast(`Launched ${launch.run_id} (${body.preset}, seed ${body.seed})`, { kind: "success" });
      tailWebSocket(launch.ws_url, out);
    } catch (e) {
      appendError(out, `Launch failed: ${e.message}`);
      toast(`Launch failed: ${e.message}`, { kind: "error", duration: 6000 });
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
// Overview tab — landing dashboard with KPIs, distribution, activity feeds.
// ─────────────────────────────────────────────────────────────────────────
async function loadOverview() {
  window._overviewLoaded = true;
  const kpiContainer = $("#overview-kpis");
  const activityContainer = $("#overview-activity");
  const findingsContainer = $("#overview-findings");
  showSkeleton(kpiContainer, 4, "card");
  showSkeleton(activityContainer, 8, "list");
  showSkeleton(findingsContainer, 6, "list");

  try {
    const [runsRes, findingsRes, launchesRes] = await Promise.all([
      fetch("/api/runs").then((r) => r.json()),
      fetch("/api/findings").then((r) => r.json()),
      fetch("/api/runs/launch").then((r) => r.json()),
    ]);

    renderOverviewKPIs(kpiContainer, runsRes.runs, findingsRes.findings, launchesRes.runs);
    renderOverviewDistribution(runsRes.runs);
    renderOverviewActivity(activityContainer, runsRes.runs);
    renderOverviewFindings(findingsContainer, findingsRes.findings);
  } catch (e) {
    kpiContainer.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function renderOverviewKPIs(container, runs, findings, launches) {
  // Filter out smokes for headline metrics
  const real = runs.filter((r) => !/smoke/i.test(r.name) && r.sum_finalQ != null);
  const sums = real.map((r) => r.sum_finalQ);
  const best = real.reduce((a, b) =>
    a == null || b.sum_finalQ < a.sum_finalQ ? b : a, null);

  const inFlight = (launches || []).filter((l) => l.running);
  const meanSum = mean(sums);
  const stdSum = stdev(sums);

  container.replaceChildren(
    kpiCard("Best run", best ? best.sum_finalQ.toFixed(2) : "—",
      best ? best.name : "no completed runs",
      best && best.sum_finalQ < 4.5 ? "kpi-card" : "kpi-card warn",
      best ? () => activateTab("runs") : null),
    kpiCard("Total runs", String(real.length),
      `${runs.length - real.length} smokes excluded`),
    kpiCard("Mean sum", meanSum != null ? meanSum.toFixed(2) : "—",
      stdSum != null ? `± ${stdSum.toFixed(2)} std` : ""),
    kpiCard("Findings", String(findings.length), "session-by-session"),
    kpiCard("In-flight runs", String(inFlight.length),
      inFlight.length ? "view in World tab" : "no runs running",
      inFlight.length ? "kpi-card" : "kpi-card",
      inFlight.length ? () => activateTab("world") : null),
  );
}

function kpiCard(label, value, sub = "", cls = "kpi-card", onClick = null) {
  const card = el("div", { class: cls }, [
    el("div", { class: "kpi-label" }, label),
    el("div", { class: "kpi-value" }, value),
    el("div", { class: "kpi-sub" }, sub),
  ]);
  if (onClick) {
    card.style.cursor = "pointer";
    card.addEventListener("click", onClick);
  }
  return card;
}

function renderOverviewDistribution(runs) {
  const real = runs.filter((r) => !/smoke/i.test(r.name) && r.sum_finalQ != null);
  if (!real.length) return;
  const sums = real.map((r) => r.sum_finalQ).sort((a, b) => a - b);

  // Bin into 0.5 bins
  const minB = Math.floor(Math.min(...sums));
  const maxB = Math.ceil(Math.max(...sums));
  const binSize = 0.5;
  const nBins = Math.ceil((maxB - minB) / binSize);
  const bins = new Array(nBins).fill(0);
  for (const s of sums) {
    let idx = Math.floor((s - minB) / binSize);
    if (idx >= nBins) idx = nBins - 1;
    if (idx < 0) idx = 0;
    bins[idx]++;
  }
  const labels = bins.map((_, i) =>
    (minB + i * binSize).toFixed(1));

  const canvas = $("#overview-distribution");
  // Color baseline (5.88), flagship (4.08), and current data
  const baselineBin = Math.floor((5.88 - minB) / binSize);
  const flagshipBin = Math.floor((4.08 - minB) / binSize);
  const colors = bins.map((_, i) => {
    if (i === flagshipBin) return P.accent;
    if (i === baselineBin) return P.warn;
    return "#5f6770";
  });

  const chart = makeBarChart(canvas, {
    title: `Sum_finalQ distribution across ${real.length} runs (green=flagship 4.08, yellow=baseline 5.88)`,
    labels,
    colors,
  });
  chart.updateData(bins);
}

function renderOverviewActivity(container, runs) {
  // Recent runs sorted by mtime
  const recent = [...runs].sort((a, b) => (b.modified_unix || 0) - (a.modified_unix || 0)).slice(0, 12);
  if (!recent.length) {
    container.replaceChildren(el("p", { class: "muted" }, "No runs yet."));
    return;
  }
  container.replaceChildren();
  for (const r of recent) {
    const exp = detectExperiment(r.name);
    const cat = categorizeExperiment(exp);
    const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
    const row = el("div", { class: "activity-row" }, [
      el("span", { class: "name" }, r.name),
      el("span", { class: "badge", style: `background: ${cat.color}33; color: ${cat.color}` }, cat.category),
      el("span", { class: "sum" }, sumStr),
      el("span", { class: "ts" }, fmtRelTime(r.modified_unix)),
    ]);
    row.addEventListener("click", () => {
      activateTab("runs");
      // Slight delay to let the runs tab activate, then click that row
      setTimeout(() => {
        const item = Array.from(document.querySelectorAll("#runs-list .list-item"))
          .find((i) => i.querySelector(".name")?.textContent === r.name);
        if (item) {
          item.scrollIntoView({ block: "center" });
          item.click();
        } else {
          // Maybe filter is hiding it; show toast
          toast(`Run not in current filter view: ${r.name}`, { kind: "warn" });
        }
      }, 150);
    });
    container.appendChild(row);
  }
}

function renderOverviewFindings(container, findings) {
  const recent = [...findings].slice(0, 10);
  if (!recent.length) {
    container.replaceChildren(el("p", { class: "muted" }, "No findings."));
    return;
  }
  container.replaceChildren();
  for (const f of recent) {
    const row = el("div", { class: "activity-row" }, [
      el("span", { class: "name" }, f.name),
      el("span", { class: "ts" }, fmtRelTime(f.modified_unix)),
    ]);
    row.addEventListener("click", () => {
      activateTab("findings");
      setTimeout(() => {
        const item = Array.from(document.querySelectorAll("#findings-list .list-item"))
          .find((i) => i.querySelector(".name")?.textContent === f.name);
        if (item) {
          item.scrollIntoView({ block: "center" });
          item.click();
        }
      }, 150);
    });
    container.appendChild(row);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Experiments tab — auto-group runs by filename suffix, show per-experiment
// aggregates (mean ± std, n_seeds, distribution).
// ─────────────────────────────────────────────────────────────────────────
async function loadExperiments() {
  window._experimentsLoaded = true;
  const list = $("#experiments-list");
  showSkeleton(list, 8, "list");
  try {
    const data = await fetch("/api/runs").then((r) => r.json());
    renderExperiments(list, data.runs);
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function renderExperiments(list, runs) {
  const hideSmoke = $("#exp-hide-smoke")?.checked ?? true;
  const onlyMulti = $("#exp-only-multi-seed")?.checked ?? true;

  // Group by experiment name
  const groups = new Map();
  for (const r of runs) {
    if (hideSmoke && /smoke/i.test(r.name)) continue;
    const exp = detectExperiment(r.name);
    if (!groups.has(exp)) groups.set(exp, []);
    groups.get(exp).push(r);
  }

  // Compute aggregates
  const expRows = [];
  for (const [exp, runsInExp] of groups) {
    if (onlyMulti && runsInExp.length < 2) continue;
    const sums = runsInExp.map((r) => r.sum_finalQ).filter((v) => v != null);
    const cat = categorizeExperiment(exp);
    expRows.push({
      name: exp,
      category: cat.category,
      color: cat.color,
      n_seeds: runsInExp.length,
      n_complete: sums.length,
      mean_sum: mean(sums),
      std_sum: stdev(sums),
      min_sum: sums.length ? Math.min(...sums) : null,
      max_sum: sums.length ? Math.max(...sums) : null,
      runs: runsInExp,
    });
  }

  // Sort by mean_sum ascending (best first), nulls last
  expRows.sort((a, b) => {
    if (a.mean_sum == null) return 1;
    if (b.mean_sum == null) return -1;
    return a.mean_sum - b.mean_sum;
  });

  if (!expRows.length) {
    list.replaceChildren(el("p", { class: "muted" }, "No experiments match filters."));
    return;
  }

  const head = el("tr", {}, [
    el("th", {}, "experiment"),
    el("th", {}, "category"),
    el("th", {}, "seeds"),
    el("th", {}, "mean ± std"),
    el("th", {}, "min / max"),
    el("th", {}, "vs flagship 4.08"),
  ]);
  const tbody = el("tbody");
  for (const row of expRows) {
    const meanStr = row.mean_sum != null ? row.mean_sum.toFixed(2) : "—";
    const stdStr = row.std_sum != null ? `± ${row.std_sum.toFixed(2)}` : "";
    const minMax = row.min_sum != null
      ? `${row.min_sum.toFixed(2)} / ${row.max_sum.toFixed(2)}` : "—";
    const delta = row.mean_sum != null
      ? (row.mean_sum - 4.08).toFixed(2) : "—";
    const deltaCls = row.mean_sum == null ? "" :
      row.mean_sum < 4.08 ? "good" : "bad";

    const tr = el("tr", { class: "expandable" }, [
      el("td", {}, el("strong", {}, row.name)),
      el("td", {}, el("span", {
        class: "category-pill",
        style: `background: ${row.color}33; color: ${row.color}`,
      }, row.category)),
      el("td", {}, String(row.n_seeds)),
      el("td", {}, `${meanStr} ${stdStr}`),
      el("td", {}, minMax),
      el("td", { style: deltaCls === "good" ? "color: var(--accent)" : deltaCls === "bad" ? "color: var(--accent-bad)" : "" },
        (deltaCls === "good" ? "" : "+") + delta),
    ]);
    let detail = null;
    tr.addEventListener("click", () => {
      if (detail && detail.parentNode) {
        detail.remove();
        detail = null;
        return;
      }
      detail = el("tr", {}, el("td", { colspan: "6" }, renderExperimentDetail(row)));
      tr.parentNode.insertBefore(detail, tr.nextSibling);
    });
    tbody.appendChild(tr);
  }

  list.replaceChildren(el("table", { class: "experiment-table" }, [
    el("thead", {}, head),
    tbody,
  ]));
}

function renderExperimentDetail(expRow) {
  const wrapper = el("div", { class: "experiment-detail" });
  wrapper.appendChild(el("div", { class: "muted", style: "margin-bottom:8px" },
    `${expRow.runs.length} run(s) — click a row above to collapse`));
  for (const r of expRow.runs) {
    const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
    const seedRow = el("div", { class: "seed-row" }, [
      el("span", {}, `seed ${r.seed ?? "?"}`),
      el("span", {}, r.name),
      el("strong", {}, sumStr),
    ]);
    seedRow.style.cursor = "pointer";
    seedRow.addEventListener("click", (ev) => {
      ev.stopPropagation();
      activateTab("runs");
      setTimeout(() => {
        const item = Array.from(document.querySelectorAll("#runs-list .list-item"))
          .find((i) => i.querySelector(".name")?.textContent === r.name);
        if (item) {
          item.scrollIntoView({ block: "center" });
          item.click();
        }
      }, 150);
    });
    wrapper.appendChild(seedRow);
  }
  return wrapper;
}

// ─────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────
setupTabs();
setupLauncher();
setupWorldTab();
loadRuns();
loadOverview();  // active tab on first load

// Restore persisted state
(() => {
  const state = loadState();
  if (state.hideSmoke !== undefined) {
    const cb = $("#filter-hide-smoke");
    if (cb) cb.checked = state.hideSmoke;
  }
  if (state.hideIncomplete !== undefined) {
    const cb = $("#filter-hide-incomplete");
    if (cb) cb.checked = state.hideIncomplete;
  }
  if (state.activeTab) {
    const btn = document.querySelector(`nav button[data-tab="${state.activeTab}"]`);
    if (btn) btn.click();
  }
})();

// Persist filter changes
$("#filter-hide-smoke")?.addEventListener("change",
  () => saveState({ hideSmoke: $("#filter-hide-smoke").checked }));
$("#filter-hide-incomplete")?.addEventListener("change",
  () => saveState({ hideIncomplete: $("#filter-hide-incomplete").checked }));

// Experiments tab filters
$("#exp-hide-smoke")?.addEventListener("change", () => loadExperiments());
$("#exp-only-multi-seed")?.addEventListener("change", () => loadExperiments());

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

// ─────────────────────────────────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────────────────────────────────
registerShortcut("r", () => {
  if (document.querySelector("#tab-runs")?.classList.contains("active")) {
    loadRuns();
    toast("Refreshed runs", { kind: "info", duration: 1500 });
  } else if (document.querySelector("#tab-findings")?.classList.contains("active")) {
    loadFindings();
    toast("Refreshed findings", { kind: "info", duration: 1500 });
  } else if (document.querySelector("#tab-overview")?.classList.contains("active")) {
    loadOverview();
    toast("Refreshed overview", { kind: "info", duration: 1500 });
  } else if (document.querySelector("#tab-experiments")?.classList.contains("active")) {
    loadExperiments();
    toast("Refreshed experiments", { kind: "info", duration: 1500 });
  }
}, "Refresh current tab");

registerShortcut("/", () => {
  const search = $("#filter-search");
  if (search) {
    activateTab("runs");
    setTimeout(() => search.focus(), 100);
  }
}, "Focus search box");

registerShortcut("esc", () => {
  if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
  // Also clear comparison set if any
  if (compareSet.size > 0) {
    compareSet.clear();
    $("#compare-runs").disabled = true;
    $("#compare-runs").textContent = "Compare 0";
    renderRunsList();
  }
}, "Blur input / clear comparison");

// Number-key tab navigation: 1=Overview, 2=Runs, 3=Experiments, 4=World, 5=Findings, 6=Launch
registerShortcut("1", () => activateTab("overview"), "Tab 1: Overview");
registerShortcut("2", () => activateTab("runs"), "Tab 2: Runs");
registerShortcut("3", () => activateTab("experiments"), "Tab 3: Experiments");
registerShortcut("4", () => activateTab("world"), "Tab 4: World");
registerShortcut("5", () => activateTab("findings"), "Tab 5: Findings");
registerShortcut("6", () => activateTab("launcher"), "Tab 6: Launch");

registerShortcut("?", () => {
  const lines = listShortcuts()
    .map(({ combo, description }) => `${combo.padEnd(8)} ${description}`)
    .join("\n");
  toast("Shortcuts:\n" + lines, { kind: "info", duration: 8000 });
}, "Show shortcut help");
