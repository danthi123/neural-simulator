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
// Tab registry (2026-05-02)
// ─────────────────────────────────────────────────────────────────────────
// Single source of truth for tab metadata. Adding a new tab requires:
//   1. Add an entry to this TAB_REGISTRY array
//   2. Add the matching <section id="tab-{id}" class="tab"> to index.html
//   3. (optional) Add a `<button data-tab="{id}">{label}</button>` to nav
//      OR set `autoNavButton: true` to inject it on bootstrap
//
// Each entry: { id, label, onActivate, autoNavButton, order }
//   - id: matches HTML <section id="tab-{id}"> AND nav <button data-tab="{id}">
//   - label: text shown in the nav button
//   - onActivate(): called the FIRST time the tab is activated. Set up
//                   data fetches, register live subscriptions, etc.
//                   Called only once unless tab refreshes via _loaded flag.
//   - order: sort order for nav rendering (smaller = leftmost)
//
// Adding a new visualization (e.g. "neural-3d"):
//   1. Append to TAB_REGISTRY:
//        { id: "neural-3d", label: "Brain (Live)", order: 65,
//          onActivate: () => { if (!window._neural3dLoaded) loadNeural3D(); } }
//   2. Add <section id="tab-neural-3d" class="tab">…</section> to HTML
//   3. Implement loadNeural3D() in this file or a separate module
//
// See: docs/webapp-frontend-guide.md for full architecture.
// ─────────────────────────────────────────────────────────────────────────
const TAB_REGISTRY = [
  { id: "overview",    label: "Home",        order: 10, onActivate: () => { if (!window._overviewLoaded) loadOverview(); } },
  { id: "launcher",    label: "Lab",         order: 20, onActivate: null /* setup in setupLauncher() */ },
  { id: "runs",        label: "Runs",        order: 30, onActivate: null /* loaded eagerly */ },
  { id: "experiments", label: "Experiments", order: 40, onActivate: () => { if (!window._experimentsLoaded) loadExperiments(); } },
  { id: "world",       label: "World",       order: 50, onActivate: null /* setupWorldTab() */ },
  { id: "brain",       label: "Brain",       order: 60, onActivate: null /* placeholder, no JS yet */ },
  { id: "language",    label: "Language",    order: 70, onActivate: () => { if (!window._languageLoaded) loadLanguage(); } },
  { id: "findings",    label: "Findings",    order: 80, onActivate: () => { if (!window._findingsLoaded) loadFindings(); } },
  { id: "plans",       label: "Plans",       order: 85, onActivate: () => { if (!window._plansLoaded) loadPlans(); } },
  { id: "info",        label: "About",       order: 90, onActivate: () => { if (!window._infoLoaded) loadInfo(); } },
];

const TAB_BY_ID = Object.fromEntries(TAB_REGISTRY.map((t) => [t.id, t]));

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
      // Dispatch to the tab's onActivate hook (registry-driven).
      const entry = TAB_BY_ID[t];
      if (entry?.onActivate) entry.onActivate();
      // Auto-collapse the mobile menu when a tab is picked.
      const navEl = document.getElementById("nav-tabs");
      const toggleBtn = document.getElementById("nav-mobile-toggle");
      if (navEl?.classList.contains("nav-open")) {
        navEl.classList.remove("nav-open");
        toggleBtn?.setAttribute("aria-expanded", "false");
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────
// Theme toggle (dark/light, 2026-05-02)
//
// Reads localStorage["theme"] on load. Falls back to (a) explicit
// document.documentElement.dataset.theme set by another script,
// (b) prefers-color-scheme media query handled in CSS, (c) the dark
// default in :root.
// ─────────────────────────────────────────────────────────────────────────
function setupThemeToggle() {
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const iconEl = btn.querySelector(".theme-toggle-icon");

  // Apply persisted theme before first paint.
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") {
    root.dataset.theme = saved;
  }
  updateThemeIcon();

  btn.addEventListener("click", () => {
    const cur = root.dataset.theme || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("theme", next);
    updateThemeIcon();
  });

  function updateThemeIcon() {
    const cur = root.dataset.theme || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    if (iconEl) iconEl.textContent = cur === "dark" ? "☀" : "🌙";
    btn.title = cur === "dark" ? "Switch to light theme" : "Switch to dark theme";
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Mobile nav toggle (2026-05-02)
//
// At <900px viewports, the nav collapses behind a hamburger ☰ button.
// Clicking the button reveals it; clicking a tab collapses it (handled
// in setupTabs above).
// ─────────────────────────────────────────────────────────────────────────
function setupMobileNav() {
  const btn = document.getElementById("nav-mobile-toggle");
  const navEl = document.getElementById("nav-tabs");
  if (!btn || !navEl) return;
  btn.addEventListener("click", () => {
    const open = navEl.classList.toggle("nav-open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
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
const selectionSet = new Set(); // run names selected for bulk actions (compare, trash, ...)
let _lastClickedRunIndex = -1; // for shift-click range selection
// Backwards-compatibility alias for the openComparisonView code path.
const compareSet = selectionSet;

async function loadRuns() {
  const list = $("#runs-list");
  // Only show "Loading…" on the very first load (when the list is empty
  // or still has the initial placeholder text). On periodic refreshes
  // we keep the existing rows visible and let renderRunsList swap them
  // in atomically — avoids the blank-list flicker the user reported.
  const isFirstLoad = _allRuns.length === 0;
  if (isFirstLoad) {
    list.replaceChildren(document.createTextNode("Loading…"));
  }
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    $("#runs-count").textContent = `${data.count} runs`;
    if (!data.runs.length) {
      const p = el("p", { class: "muted", style: "padding:16px" },
        "No runs yet — launch one from the Launch tab.");
      list.replaceChildren(p);
      _allRuns = [];
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
    // On refresh failure, keep the existing list visible — only show
    // an error if this was the first load.
    if (isFirstLoad) {
      list.replaceChildren(el("p", { class: "error" }, e.message));
    }
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
  filtered.forEach((r, idx) => {
    const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
    const isSelected = selectionSet.has(r.name);
    const checkbox = el("input", {
      type: "checkbox",
      class: "row-checkbox",
      "aria-label": `Select ${r.name}`,
    });
    checkbox.checked = isSelected;
    checkbox.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleSelection(r.name);
    });
    const body = el("div", { class: "row-body" }, [
      el("div", { class: "name" }, r.name),
      el("div", { class: "meta" }, [
        metric("sum", sumStr),
        metric("seed", r.seed ?? "—"),
        metric("phases", r.n_phases),
      ]),
    ]);
    const item = el("div", {
      class: "list-item" + (isSelected ? " row-selected" : ""),
      dataset: { name: r.name, idx: String(idx) },
    }, [checkbox, body]);
    body.addEventListener("click", (ev) => {
      if (ev.shiftKey) {
        // Shift-click: toggle range from last clicked index
        if (_lastClickedRunIndex >= 0) {
          const lo = Math.min(_lastClickedRunIndex, idx);
          const hi = Math.max(_lastClickedRunIndex, idx);
          for (let i = lo; i <= hi; i++) selectionSet.add(filtered[i].name);
          updateSelectionUI();
          renderRunsList();
        } else {
          toggleSelection(r.name);
        }
        return;
      }
      if (ev.metaKey || ev.ctrlKey) {
        toggleSelection(r.name);
        _lastClickedRunIndex = idx;
        return;
      }
      _lastClickedRunIndex = idx;
      loadRunDetail(r.name, item);
    });
    list.appendChild(item);
  });
  updateSelectionUI();
}

function toggleSelection(name) {
  if (selectionSet.has(name)) selectionSet.delete(name);
  else selectionSet.add(name);
  updateSelectionUI();
  renderRunsList();
}

function clearSelection() {
  selectionSet.clear();
  _lastClickedRunIndex = -1;
  updateSelectionUI();
  renderRunsList();
}

function updateSelectionUI() {
  const n = selectionSet.size;
  const bar = document.getElementById("selection-bar");
  if (bar) bar.style.display = n > 0 ? "flex" : "none";
  const cnt = document.getElementById("selection-count");
  if (cnt) cnt.textContent = String(n);
  const cmp = document.getElementById("bulk-compare-btn");
  if (cmp) cmp.disabled = n < 2 || n > 3;
  const trash = document.getElementById("bulk-trash-btn");
  if (trash) trash.disabled = n === 0;
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
    const rerunBtn = el("button", { class: "play-in-world", style: "margin-left:8px", onclick: () => rerunFromSidecar(name) }, "↻ Re-run with same config");
    const distCanvas = el("canvas", { class: "chart-canvas" });
    const rewardCanvas = el("canvas", { class: "chart-canvas" });
    const heatmapCanvas = el("canvas", { class: "chart-canvas chart-narrow" });
    const phaseMotorContainer = el("div", { class: "phase-motor-grid" });
    detail.replaceChildren(
      el("h2", {}, name),
      el("div", {}, [
        metric("seed", data.seed ?? "—"),
        metric("n_steps", data.n_steps ?? "—"),
        metric("grid_size", data.grid_size ?? 8),
        metric("sum_finalQ", computeSumFinalQ(data)),
      ]),
      el("div", { style: "margin: 12px 0" }, [playBtn, rerunBtn]),
      el("h3", {}, "Phase stats"),
      renderPhaseStats(data.phase_stats || []),
      el("h3", {}, "Distance over time"),
      el("div", { class: "chart-row" }, distCanvas),
      el("h3", {}, "Reward over time"),
      el("div", { class: "chart-row" }, rewardCanvas),
      el("h3", {}, "Agent visit heatmap"),
      el("div", { class: "chart-row chart-narrow-wrap" }, heatmapCanvas),
      el("h3", {}, "Action distribution per phase"),
      el("div", { class: "chart-row" }, phaseMotorContainer),
      el("h3", {}, "Raw JSON"),
      el("pre", {}, JSON.stringify(summarizeRunData(data), null, 2)),
    );
    // Charts must be rendered AFTER the canvas elements are in the DOM so
    // clientWidth/Height resolve to non-zero values for the dpr setup.
    requestAnimationFrame(() => renderRunCharts(data, distCanvas, rewardCanvas, heatmapCanvas, phaseMotorContainer));
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

/** Render the run-detail charts: distance over time, reward over time,
 *  agent-visit heatmap, and per-phase action distribution bars. Phase
 *  boundaries shaded. */
function renderRunCharts(data, distCanvas, rewardCanvas, heatmapCanvas, phaseMotorContainer) {
  const phases = data.phase_stats || [];
  const phaseRanges = phases.map((ps, i) => ({
    start: ps.step_start ?? 0,
    end: ps.step_end ?? (data.n_steps ?? 0),
    label: `phase ${i} → goal (${ps.goal[0]},${ps.goal[1]})`,
    // Alternate phase shading between two near-black tones, matching
    // --bg-2 and --bg-3. PALETTE values mirror the CSS vars.
    color: i % 2 === 0 ? P.bg2 : P.bg3,
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

  // Reward over time — moving average for readability.
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

  // Agent visit heatmap — count time spent in each cell across the run.
  // Reveals learned policy at a glance (orbits, direct paths, dead zones).
  renderHeatmap(heatmapCanvas, data);

  // Per-phase action distribution — one bar chart per phase.
  // Replaces the previous single-totals chart so you can SEE how the
  // action distribution shifted after each goal change.
  phaseMotorContainer.replaceChildren();
  for (let i = 0; i < phases.length; i++) {
    const ps = phases[i];
    const ac = ps.action_counts || [0, 0, 0, 0];
    const sub = document.createElement("div");
    sub.className = "phase-motor-cell";
    const canvas = document.createElement("canvas");
    canvas.className = "chart-canvas chart-narrow";
    sub.appendChild(canvas);
    phaseMotorContainer.appendChild(sub);
    const goalLabel = ps.goal ? `(${ps.goal[0]},${ps.goal[1]})` : "?";
    const chart = makeBarChart(canvas, {
      title: `phase ${i} → goal ${goalLabel} (${ps.n_steps ?? "?"} steps)`,
      labels: ["N", "E", "S", "W"],
      colors: [P.accent, P.warn, P.bad, P.blue],
    });
    chart.updateData(ac);
  }
}

/** Render a heatmap of agent visit counts on top of the gridworld layout.
 *  Each cell is colored by visit frequency (log-scaled for visual range). */
function renderHeatmap(canvas, data) {
  const trajectory = data.trajectory || [];
  const gridSize = data.grid_size || 8;
  if (!trajectory.length) {
    const ctx = canvas.getContext("2d");
    canvas.width = 1; canvas.height = 1;
    ctx.fillStyle = P.bg;
    ctx.fillRect(0, 0, 1, 1);
    return;
  }

  // Count visits per cell
  const counts = new Array(gridSize * gridSize).fill(0);
  for (const [x, y] of trajectory) {
    if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
      counts[y * gridSize + x]++;
    }
  }
  const maxC = Math.max(...counts);
  if (maxC === 0) return;

  // Render with high-DPI handling
  const dpr = window.devicePixelRatio || 1;
  const cssSize = Math.min(canvas.clientWidth || 360, 360);
  const cellPx = Math.floor((cssSize - 24) / gridSize);
  const padding = 14;
  const w = padding * 2 + cellPx * gridSize;
  const h = padding * 2 + cellPx * gridSize + 18;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = P.bg;
  ctx.fillRect(0, 0, w, h);

  // Grid cells colored by visit count (log-scaled green ramp)
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const c = counts[y * gridSize + x];
      const intensity = c === 0 ? 0 : Math.log(1 + c) / Math.log(1 + maxC);
      // y-flip so y=0 sits at the bottom (matches World tab convention)
      const px = padding + x * cellPx;
      const py = padding + (gridSize - 1 - y) * cellPx;
      // Color: dark → green for visits, faintly transparent for never visited
      ctx.fillStyle = intensity === 0
        ? P.bg2
        : `rgba(110, 231, 183, ${0.15 + intensity * 0.7})`;
      ctx.fillRect(px, py, cellPx - 1, cellPx - 1);
      // Show count if non-trivial
      if (c > 0 && cellPx > 18) {
        ctx.fillStyle = intensity > 0.6 ? P.bg : P.fg;
        ctx.font = `${Math.max(8, Math.floor(cellPx * 0.32))}px ui-monospace, Consolas, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(c), px + cellPx / 2, py + cellPx / 2);
      }
    }
  }
  // Legend / max
  ctx.fillStyle = P.fgDim;
  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(
    `Visits per cell · max=${maxC} · ${trajectory.length} total steps`,
    padding,
    padding + cellPx * gridSize + 4,
  );
}

async function rerunFromSidecar(name) {
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}/sidecar`);
    if (!res.ok) {
      if (res.status === 404) {
        toast(
          "No sidecar found — this run wasn't launched via the webapp. " +
          "Re-run is only available for runs launched from this dashboard",
          { kind: "warn", duration: 6000 }
        );
        return;
      }
      throw new Error(`${res.status}`);
    }
    const sidecar = await res.json();
    activateTab("launcher");
    await new Promise((r) => setTimeout(r, 100));
    // Prefill the form
    const form = document.querySelector("#launch-form");
    if (form) {
      const presetSel = form.querySelector('select[name="preset"]');
      if (presetSel && sidecar.preset) presetSel.value = sidecar.preset;
      const seedInput = form.querySelector('input[name="seed"]');
      if (seedInput && sidecar.seed != null) seedInput.value = sidecar.seed;
      const extrasInput = form.querySelector('input[name="extra_args"]');
      if (extrasInput) extrasInput.value = (sidecar.extra_args || []).join(" ");
    }
    toast(
      `Loaded re-run config from ${name}: preset=${sidecar.preset}, seed=${sidecar.seed}. ` +
      `Edit fields then click Launch to start`,
      { kind: "success", duration: 5000 }
    );
  } catch (e) {
    toast(`Re-run failed: ${e.message}`, { kind: "error", duration: 6000 });
  }
}

async function killLaunchedRun(runId) {
  try {
    const res = await fetch(`/api/runs/launch/${encodeURIComponent(runId)}/kill`, {
      method: "POST",
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    toast(`Run ${runId}: ${data.status} (rc=${data.returncode})`, {
      kind: data.status === "killed" ? "warn" : "info",
      duration: 4000,
    });
    return data;
  } catch (e) {
    toast(`Kill failed: ${e.message}`, { kind: "error", duration: 6000 });
    return null;
  }
}

// Expose for console debugging
window.killLaunchedRun = killLaunchedRun;

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
      color: i % 2 === 0 ? P.bg2 : P.bg3,
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
// ─────────────────────────────────────────────────────────────────────────
// Language tab (text I/O results — confusion matrices, W→A / I→W accuracy)
//
// 2026-05-02. Surfaces all text_eval_*.json files so the text I/O
// experiments aren't buried in the generic Runs tab (which is keyed off
// gridworld navigation's sum_finalQ metric and doesn't render confusion
// matrices). Powered by /api/text_io_runs.
// ─────────────────────────────────────────────────────────────────────────

const LANG_DIRS = ["north", "east", "south", "west"];
const ACTION_DIRS = ["N", "E", "S", "W"];

function fmtPercent(v, digits = 1) {
  if (v == null) return "—";
  return (100 * v).toFixed(digits) + "%";
}

function makeKpiCard(title, value, sub) {
  const card = el("div", { class: "kpi-card" });
  card.appendChild(el("div", { class: "kpi-label" }, title));
  card.appendChild(el("div", { class: "kpi-value" }, value));
  if (sub) card.appendChild(el("div", { class: "kpi-sub muted" }, sub));
  return card;
}

function renderConfusionMatrix(title, matrix, rowLabels, colLabels, chanceColor = true) {
  if (!matrix) return el("p", { class: "muted" }, `No ${title} data.`);
  const wrapper = el("div", { class: "confusion-wrapper" });
  wrapper.appendChild(el("h4", {}, title));
  const table = el("table", { class: "confusion-matrix" });
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", {}, ""),
      ...colLabels.map((c) => el("th", {}, c)),
      el("th", {}, "Σ"),
    ]),
  ]);
  table.appendChild(thead);
  const tbody = el("tbody");
  // Compute per-row total and global max for color scaling.
  let maxCell = 1;
  for (const r of rowLabels) {
    const row = matrix[r] || {};
    for (const c of colLabels) {
      const v = Number(row[c] || 0);
      if (v > maxCell) maxCell = v;
    }
  }
  for (const r of rowLabels) {
    const row = matrix[r] || {};
    let total = 0;
    const trChildren = [el("th", {}, r)];
    for (const c of colLabels) {
      const v = Number(row[c] || 0);
      total += v;
      const isDiag = (r[0]?.toLowerCase() === c[0]?.toLowerCase());
      // Bin color: diagonal gets blue scale, off-diagonal gets gray scale.
      // Lighter = higher count.
      const intensity = Math.min(1.0, v / maxCell);
      const cell = el("td", {
        class: "confusion-cell" + (isDiag ? " diag" : ""),
        style: chanceColor
          ? `background: rgba(${isDiag ? "59,130,246" : "203,213,225"}, ${intensity * 0.85}); color: ${intensity > 0.4 ? "#0a0e1a" : "#cbd5e1"}`
          : ``,
      }, String(v));
      trChildren.push(cell);
    }
    trChildren.push(el("td", { class: "confusion-total muted" }, String(total)));
    tbody.appendChild(el("tr", {}, trChildren));
  }
  table.appendChild(tbody);
  wrapper.appendChild(table);
  return wrapper;
}

let _langSortKey = "modified_unix";
let _langSortDir = -1;
let _langCache = null;

async function loadLanguage() {
  window._languageLoaded = true;
  const list = $("#language-list");
  const kpis = $("#language-kpis");
  list.replaceChildren(document.createTextNode("Loading…"));
  kpis.replaceChildren();
  try {
    const res = await fetch("/api/text_io_runs");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    _langCache = data;
    $("#language-count").textContent = `${data.count} text I/O runs`;
    renderLanguageKpis(data.aggregate, data.runs);
    renderLanguageList(data.runs);
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function renderLanguageKpis(agg, runs) {
  const kpis = $("#language-kpis");
  kpis.replaceChildren();
  if (!runs || !runs.length) {
    kpis.appendChild(el("p", { class: "muted" }, "No text I/O runs yet."));
    return;
  }
  // I→W card
  const i2wMean = agg.i2w_accuracy_mean;
  const i2wStd = agg.i2w_accuracy_std;
  const i2wN = agg.i2w_accuracy_n;
  kpis.appendChild(makeKpiCard(
    "Image → Word (I→W)",
    fmtPercent(i2wMean),
    i2wN ? `±${fmtPercent(i2wStd)} across ${i2wN} runs · chance = 25%` : "—",
  ));
  // W→A card
  const w2aMean = agg.w2a_accuracy_mean;
  const w2aStd = agg.w2a_accuracy_std;
  const w2aN = agg.w2a_accuracy_n;
  kpis.appendChild(makeKpiCard(
    "Word → Action (W→A)",
    fmtPercent(w2aMean),
    w2aN ? `±${fmtPercent(w2aStd)} across ${w2aN} runs · chance = 25%` : "—",
  ));
  // Best W→A so far
  const bestW2A = runs.reduce((b, r) =>
    (r.w2a_accuracy != null && (!b || r.w2a_accuracy > b.w2a_accuracy)) ? r : b, null);
  kpis.appendChild(makeKpiCard(
    "Best W→A",
    bestW2A ? fmtPercent(bestW2A.w2a_accuracy) : "—",
    bestW2A ? `seed ${bestW2A.seed} · ${bestW2A.name.replace(/^text_eval_/, "").slice(0, 40)}` : "—",
  ));
  // n_runs total
  kpis.appendChild(makeKpiCard(
    "Total runs",
    String(runs.length),
    `${new Set(runs.map(r => r.seed).filter(s => s != null)).size} unique seeds`,
  ));
}

function renderLanguageList(runs) {
  const list = $("#language-list");
  list.replaceChildren();
  if (!runs.length) {
    list.appendChild(el("p", { class: "muted" }, "No runs."));
    return;
  }
  // Header row with sortable columns.
  const header = el("div", { class: "list-header" });
  const cols = [
    { key: "name", label: "name", flex: 2 },
    { key: "seed", label: "seed", flex: 0.5 },
    { key: "i2w_accuracy", label: "I→W", flex: 0.7 },
    { key: "w2a_accuracy", label: "W→A", flex: 0.7 },
    { key: "correct_move_rate", label: "corr.move", flex: 0.7 },
    { key: "modified_unix", label: "mod", flex: 0.8 },
  ];
  for (const col of cols) {
    const cell = el("div", {
      class: "list-header-cell" + (_langSortKey === col.key ? " active" : ""),
      style: `flex: ${col.flex}`,
    }, [
      col.label + (_langSortKey === col.key ? (_langSortDir > 0 ? " ↑" : " ↓") : ""),
    ]);
    cell.addEventListener("click", () => {
      if (_langSortKey === col.key) _langSortDir *= -1;
      else { _langSortKey = col.key; _langSortDir = -1; }
      renderLanguageList(runs);
    });
    header.appendChild(cell);
  }
  list.appendChild(header);
  // Sorted rows.
  const sorted = [...runs].sort((a, b) => {
    let av = a[_langSortKey], bv = b[_langSortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return _langSortDir * av.localeCompare(bv);
    return _langSortDir * (av - bv);
  });
  for (const r of sorted) {
    const item = el("div", { class: "list-item lang-row" });
    item.appendChild(el("div", { class: "lang-cell name", style: "flex: 2" }, [
      el("div", { class: "name" }, r.name.replace(/\.json$/, "")),
    ]));
    item.appendChild(el("div", { class: "lang-cell", style: "flex: 0.5" }, [r.seed != null ? String(r.seed) : "—"]));
    item.appendChild(el("div", {
      class: "lang-cell" + (r.w2a_accuracy != null && r.i2w_accuracy > 0.30 ? " above-chance" : ""),
      style: "flex: 0.7",
    }, [fmtPercent(r.i2w_accuracy)]));
    item.appendChild(el("div", {
      class: "lang-cell" + (r.w2a_accuracy != null && r.w2a_accuracy > 0.30 ? " above-chance" : ""),
      style: "flex: 0.7",
    }, [fmtPercent(r.w2a_accuracy)]));
    item.appendChild(el("div", { class: "lang-cell", style: "flex: 0.7" }, [fmtPercent(r.correct_move_rate)]));
    item.appendChild(el("div", { class: "lang-cell muted", style: "flex: 0.8" }, [
      fmtRelTime(new Date(r.modified_unix * 1000)),
    ]));
    item.addEventListener("click", () => loadLanguageDetail(r.name, item));
    list.appendChild(item);
  }
}

async function loadLanguageDetail(name, listItem) {
  const detail = $("#language-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#language-list .list-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  try {
    const res = await fetch(`/api/text_io_runs/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    const wrapper = el("div", { class: "lang-detail" });
    wrapper.appendChild(el("h3", {}, name.replace(/\.json$/, "")));
    // Headline stats
    const statsRow = el("div", { class: "kpi-grid lang-detail-kpis" });
    statsRow.appendChild(makeKpiCard(
      "I→W",
      fmtPercent(data.image_to_word_eval?.accuracy),
      `${data.image_to_word_eval?.correct ?? "—"}/${data.image_to_word_eval?.n_trials ?? "—"} trials`,
    ));
    statsRow.appendChild(makeKpiCard(
      "W→A",
      fmtPercent(data.word_to_action_eval?.accuracy),
      `${data.word_to_action_eval?.correct ?? "—"}/${data.word_to_action_eval?.n_trials ?? "—"} trials`,
    ));
    statsRow.appendChild(makeKpiCard(
      "Seed",
      String(data.seed ?? "—"),
      `${data.regime ?? "—"} · ${data.n_episodes ?? "—"} episodes`,
    ));
    if (data.training_stats?.[0]?.correct_move_rate != null) {
      statsRow.appendChild(makeKpiCard(
        "Training corr.move",
        fmtPercent(data.training_stats[0].correct_move_rate),
        `${data.training_stats[0].n_correct_moves}/${data.training_stats[0].n_total_steps}`,
      ));
    }
    wrapper.appendChild(statsRow);
    // Confusion matrices side-by-side
    const matrixRow = el("div", { class: "confusion-row" });
    matrixRow.appendChild(renderConfusionMatrix(
      "Image → Word confusion",
      data.image_to_word_eval?.confusion_matrix,
      LANG_DIRS, LANG_DIRS,
    ));
    matrixRow.appendChild(renderConfusionMatrix(
      "Word → Action confusion",
      data.word_to_action_eval?.confusion_matrix,
      LANG_DIRS, ACTION_DIRS,
    ));
    wrapper.appendChild(matrixRow);
    // Raw JSON link
    const links = el("div", { class: "lang-detail-links" });
    links.appendChild(el("a", {
      href: `/api/text_io_runs/${encodeURIComponent(name)}`,
      target: "_blank",
    }, ["View raw JSON →"]));
    wrapper.appendChild(links);
    detail.replaceChildren(wrapper);
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Findings tab — chronological session findings (109+ markdown docs)
//
// 2026-05-02: added search box + auto-derived category chips because
// 100+ findings is too many to navigate as a flat list.
// ─────────────────────────────────────────────────────────────────────────

let _findingsCache = [];
let _findingsActiveTag = null;
let _findingsSearch = "";

// Category-tag classifier. Patterns are case-insensitive substring matches
// on the filename. First match wins; "uncategorized" is the fallback. Ordered
// so cluster letters resolve before generic "session" tags.
const FINDING_TAG_PATTERNS = [
  { tag: "🌟 Breakthrough", pat: /breakthrough|BREAKTHROUGH/i },
  { tag: "Cluster A", pat: /cluster-?a(-|\b)/i },
  { tag: "Cluster B", pat: /cluster-?b(-|\b)/i },
  { tag: "Cluster C", pat: /cluster-?c(-|\b)/i },
  { tag: "Cluster D", pat: /cluster-?d(-|\b)/i },
  { tag: "Cluster E", pat: /cluster-?e(-|\b)/i },
  { tag: "Cluster F", pat: /cluster-?f(-|\b)/i },
  { tag: "Cluster G", pat: /cluster-?g(-|\b)/i },
  { tag: "Cluster K", pat: /cluster-?k(-|\b)/i },
  { tag: "Text I/O", pat: /text-?io|text-?eval|word.action|i.w/i },
  { tag: "Perception arc", pat: /perception|sensed-reward|landmark|beacon|cue-reflex/i },
  { tag: "Cheat closure", pat: /cheat-?\d|cheat\d/i },
  { tag: "Phase B (BG)", pat: /phase-b|bg-acid|bg-cascade/i },
  { tag: "Plastic input", pat: /plastic-input|input-layer/i },
  { tag: "Adaptive DA", pat: /adaptive-da|asym-da|surprise-lr/i },
  { tag: "Curriculum", pat: /curriculum/i },
  { tag: "Hippocampus", pat: /hippocampus|swr|sharp-wave|trisynaptic/i },
  { tag: "G-gate", pat: /g\d+|g11|g9|g7|g6/i },
  { tag: "Negative", pat: /negative|NEGATIVE|null-|NULL/i },
];

function classifyFinding(name) {
  for (const { tag, pat } of FINDING_TAG_PATTERNS) {
    if (pat.test(name)) return tag;
  }
  return "Other";
}

async function loadFindings() {
  window._findingsLoaded = true;
  const list = $("#findings-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/findings");
    const data = await res.json();
    _findingsCache = data.findings.map((f) => ({ ...f, tag: classifyFinding(f.name) }));
    renderFindingChips();
    renderFindingsList();
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Plans tab — architecture decision records from docs/plans/
//
// 2026-05-02. Same shape as Findings but pulled from /api/plans. Plans
// are forward-looking design docs while findings are backward-looking
// experimental results. Demonstrates the TAB_REGISTRY-based extensibility
// pattern: the entire tab was added by appending one TAB_REGISTRY entry,
// one HTML section, two backend endpoints (/api/plans + /api/plans/{name}),
// and three JS functions.
// ─────────────────────────────────────────────────────────────────────────

let _plansCache = [];
let _plansSearch = "";

async function loadPlans() {
  window._plansLoaded = true;
  const list = $("#plans-list");
  list.replaceChildren(document.createTextNode("Loading…"));
  try {
    const res = await fetch("/api/plans");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    _plansCache = data.plans;
    renderPlansList();
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function renderPlansList() {
  const list = $("#plans-list");
  list.replaceChildren();
  let filtered = _plansCache;
  if (_plansSearch) {
    const needle = _plansSearch.toLowerCase();
    filtered = filtered.filter((p) => p.name.toLowerCase().includes(needle));
  }
  $("#plans-count").textContent = `${filtered.length} of ${_plansCache.length} plans`;
  if (!filtered.length) {
    list.appendChild(el("p", { class: "muted" }, "No matching plans."));
    return;
  }
  for (const p of filtered) {
    // Strip date prefix and -design.md suffix for display.
    const display = p.name
      .replace(/\.md$/, "")
      .replace(/-design$/, "")
      .replace(/-implementation$/, " (impl)")
      .replace(/^\d{4}-\d{2}-\d{2}-/, "");
    const dateStr = p.name.slice(0, 10);
    const item = el("div", { class: "list-item" }, [
      el("div", { class: "name" }, display),
      el("div", { class: "meta" }, [dateStr]),
    ]);
    item.addEventListener("click", () => loadPlanDetail(p.name, item));
    list.appendChild(item);
  }
}

async function loadPlanDetail(name, listItem) {
  const detail = $("#plan-detail");
  detail.replaceChildren(el("p", { class: "muted" }, `Loading ${name}…`));
  $$("#plans-list .list-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  try {
    const res = await fetch(`/api/plans/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const text = await res.text();
    const wrapper = el("div", { class: "markdown" });
    wrapper.innerHTML = renderMarkdown(text);
    detail.replaceChildren(wrapper);
  } catch (e) {
    detail.replaceChildren(el("p", { class: "error" }, `Failed to load: ${e.message}`));
  }
}

function renderFindingChips() {
  const row = $("#findings-chip-row");
  if (!row) return;
  row.replaceChildren();
  // Count findings per tag.
  const counts = {};
  for (const f of _findingsCache) {
    counts[f.tag] = (counts[f.tag] || 0) + 1;
  }
  // "All" chip first, then tags sorted by count descending.
  const tags = ["All", ...Object.keys(counts).sort((a, b) => counts[b] - counts[a])];
  for (const tag of tags) {
    const isAll = tag === "All";
    const isActive = (_findingsActiveTag == null && isAll) || _findingsActiveTag === tag;
    const count = isAll ? _findingsCache.length : counts[tag];
    const chip = el("button", {
      class: "filter-chip" + (isActive ? " active" : ""),
    }, [
      tag,
      el("span", { class: "filter-chip-count" }, String(count)),
    ]);
    chip.addEventListener("click", () => {
      _findingsActiveTag = isAll ? null : tag;
      renderFindingChips();
      renderFindingsList();
    });
    row.appendChild(chip);
  }
}

function renderFindingsList() {
  const list = $("#findings-list");
  list.replaceChildren();
  let filtered = _findingsCache;
  if (_findingsActiveTag != null) {
    filtered = filtered.filter((f) => f.tag === _findingsActiveTag);
  }
  if (_findingsSearch) {
    const needle = _findingsSearch.toLowerCase();
    filtered = filtered.filter((f) => f.name.toLowerCase().includes(needle));
  }
  $("#findings-count").textContent = `${filtered.length} of ${_findingsCache.length} findings`;
  if (!filtered.length) {
    list.appendChild(el("p", { class: "muted" }, "No matching findings."));
    return;
  }
  for (const f of filtered) {
    // Strip date prefix and .md suffix for shorter display name
    const display = f.name.replace(/\.md$/, "").replace(/^\d{4}-\d{2}-\d{2}-/, "");
    const item = el("div", { class: "list-item" }, [
      el("div", { class: "name" }, display),
      el("div", { class: "meta" }, [
        el("span", { class: "finding-tag" }, f.tag),
        " · ",
        el("span", {}, f.name.slice(0, 10)), // date prefix
      ]),
    ]);
    item.addEventListener("click", () => loadFindingDetail(f.name, item));
    list.appendChild(item);
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
  // Load CURRENT-STATE.md and the system info JSON in parallel.
  const [csRes, infoRes] = await Promise.allSettled([
    fetch("/api/current_state").then((r) => r.ok ? r.text() : Promise.reject(r.status)),
    fetch("/api/info").then((r) => r.json()),
  ]);

  const csEl = $("#info-current-state");
  if (csRes.status === "fulfilled") {
    const wrapper = el("div", { class: "markdown" });
    wrapper.innerHTML = renderMarkdown(csRes.value);
    csEl.replaceChildren(wrapper);
  } else {
    csEl.replaceChildren(
      el("p", { class: "error" }, `Failed to load CURRENT-STATE.md: ${csRes.reason}`),
      el("p", { class: "muted" },
        "The file may not exist on this deployment. " +
        "On a checkout, see docs/CURRENT-STATE.md."),
    );
  }

  const sysEl = $("#info-output");
  if (infoRes.status === "fulfilled") {
    sysEl.textContent = JSON.stringify(infoRes.value, null, 2);
  } else {
    sysEl.textContent = `Error: ${infoRes.reason?.message || infoRes.reason}`;
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
    // 2026-05-02: include text_io_runs so the W→A breakthrough card has data
    const [runsRes, findingsRes, launchesRes, textIoRes] = await Promise.all([
      fetch("/api/runs").then((r) => r.json()),
      fetch("/api/findings").then((r) => r.json()),
      fetch("/api/runs/launch").then((r) => r.json()),
      fetch("/api/text_io_runs").then((r) => r.json()).catch(() => ({ runs: [], aggregate: {} })),
    ]);

    renderOverviewKPIs(kpiContainer, runsRes.runs, findingsRes.findings, launchesRes.runs, textIoRes);
    renderOverviewDistribution(runsRes.runs);
    renderOverviewActivity(activityContainer, runsRes.runs);
    renderOverviewFindings(findingsContainer, findingsRes.findings);
  } catch (e) {
    kpiContainer.replaceChildren(el("p", { class: "error" }, e.message));
  }

  // 2026-05-01: in-flight detached-run monitor. Polls /api/inflight every
  // 5s and shows a progress card per active detached run. Hidden when no
  // runs are in flight (returned count == 0).
  refreshInflightPanel();
  if (!window._inflightInterval) {
    window._inflightInterval = setInterval(refreshInflightPanel, 5000);
  }
}

async function refreshInflightPanel() {
  const section = document.getElementById("overview-inflight-section");
  const container = document.getElementById("overview-inflight");
  if (!section || !container) return;
  try {
    const res = await fetch("/api/inflight").then((r) => r.json());
    const runs = res.inflight || [];
    if (runs.length === 0) {
      section.style.display = "none";
      return;
    }
    section.style.display = "";
    container.replaceChildren();
    for (const r of runs) {
      const p = r.progress || {};
      const fraction = p.fraction || 0;
      const pct = Math.round(fraction * 100);
      const stateBadge = r.alive
        ? el("span", { class: "badge", style: "background:#10b98133;color:#10b981" }, "running")
        : (r.completed
            ? el("span", { class: "badge", style: "background:#6ee7b733;color:#6ee7b7" }, "completed")
            : el("span", { class: "badge", style: "background:#fb718533;color:#fb7185" }, "stopped"));

      let progressLine;
      if (p.kind === "embodied_episode") {
        progressLine = `episode ${p.episode}/${p.episodes_total} · ` +
                       `${p.correct_moves}/${p.n_steps} correct moves (${p.correct_pct}%)`;
      } else if (p.kind === "step") {
        progressLine = `step ${p.step}/${p.total} · pos=(${p.pos.join(',')}) · goal=(${p.goal.join(',')})`;
      } else {
        progressLine = "no progress markers yet";
      }

      const card = el("div", { class: "activity-row inflight-row",
                               style: "display:grid;grid-template-columns:1fr auto auto;gap:12px;align-items:center;padding:10px 12px;border:1px solid var(--border);border-radius:6px;margin-bottom:8px;" }, [
        el("div", {}, [
          el("div", { style: "font-weight:600;font-family:ui-monospace,Consolas,monospace;" }, r.name),
          el("div", { class: "small muted" }, [
            el("span", {}, progressLine),
            el("span", { style: "margin-left:8px;color:var(--fg-muted)" },
               `pid=${r.pid} · log=${r.log_size_kb}KB`),
          ]),
          // Progress bar
          el("div", { style: "background:#2a2f3d;height:4px;border-radius:2px;margin-top:6px;overflow:hidden;" }, [
            el("div", { style: `background:#6ee7b7;height:100%;width:${pct}%;transition:width 0.5s;` }),
          ]),
        ]),
        el("div", { style: "font-family:ui-monospace,Consolas,monospace;font-size:14px;color:var(--accent);" }, `${pct}%`),
        stateBadge,
      ]);
      container.appendChild(card);
    }
  } catch (e) {
    // Silent failure — endpoint may not exist on older webapp builds
  }
}

function renderOverviewKPIs(container, runs, findings, launches, textIoRes) {
  // Filter out smokes for headline metrics
  const real = runs.filter((r) => !/smoke/i.test(r.name) && r.sum_finalQ != null);
  const sums = real.map((r) => r.sum_finalQ);
  const best = real.reduce((a, b) =>
    a == null || b.sum_finalQ < a.sum_finalQ ? b : a, null);

  const inFlight = (launches || []).filter((l) => l.running);
  const meanSum = mean(sums);
  const stdSum = stdev(sums);

  // 2026-05-02: pull text I/O W→A best from the new endpoint
  const textIoRuns = (textIoRes?.runs) || [];
  const bestW2A = textIoRuns.reduce((b, r) =>
    (r.w2a_accuracy != null && (!b || r.w2a_accuracy > b.w2a_accuracy)) ? r : b, null);
  const w2aMean = textIoRes?.aggregate?.w2a_accuracy_mean;
  const w2aN = textIoRes?.aggregate?.w2a_accuracy_n || 0;

  container.replaceChildren(
    // ── Headline navigation card ─────────────────────────────────
    kpiCard("Best navigation run",
      best ? best.sum_finalQ.toFixed(2) + " mean dist" : "—",
      best ? "click to view in Runs" : "no completed runs",
      best && best.sum_finalQ < 4.5 ? "kpi-card" : "kpi-card warn",
      best ? () => activateTab("runs") : null),
    // ── Headline language card ───────────────────────────────────
    kpiCard("Best W→A (text I/O)",
      bestW2A ? (100 * bestW2A.w2a_accuracy).toFixed(1) + "%" : "—",
      bestW2A ? `seed ${bestW2A.seed} · click to view in Language tab` : "no text I/O runs",
      bestW2A && bestW2A.w2a_accuracy > 0.27 ? "kpi-card" : "kpi-card warn",
      bestW2A ? () => activateTab("language") : null),
    kpiCard("Mean nav sum", meanSum != null ? meanSum.toFixed(2) : "—",
      stdSum != null ? `± ${stdSum.toFixed(2)} std (${real.length} runs)` : ""),
    kpiCard("Mean W→A", w2aMean != null ? (100 * w2aMean).toFixed(1) + "%" : "—",
      w2aN ? `${w2aN} text I/O runs · chance = 25%` : "no data"),
    kpiCard("Total findings", String(findings.length), "session-by-session"),
    kpiCard("In-flight runs", String(inFlight.length),
      inFlight.length ? "view in World tab" : "no runs running",
      "kpi-card",
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
    return P.fgMuted;
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
setupThemeToggle();  // 2026-05-02: applies persisted theme before first paint
setupMobileNav();    // 2026-05-02: hamburger menu for <900px viewports
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
$("#refresh-language")?.addEventListener("click", () => {
  window._languageLoaded = false;
  loadLanguage();
});

// Findings search input — debounced re-render on every keystroke
let _findingsSearchTimer = null;
$("#findings-search")?.addEventListener("input", (e) => {
  clearTimeout(_findingsSearchTimer);
  _findingsSearchTimer = setTimeout(() => {
    _findingsSearch = e.target.value.trim();
    if (_findingsCache.length > 0) renderFindingsList();
  }, 100);
});

// Plans tab: refresh + search input
$("#refresh-plans")?.addEventListener("click", () => {
  window._plansLoaded = false;
  loadPlans();
});
let _plansSearchTimer = null;
$("#plans-search")?.addEventListener("input", (e) => {
  clearTimeout(_plansSearchTimer);
  _plansSearchTimer = setTimeout(() => {
    _plansSearch = e.target.value.trim();
    if (_plansCache.length > 0) renderPlansList();
  }, 100);
});

$("#filter-hide-smoke")?.addEventListener("change", renderRunsList);
$("#filter-hide-incomplete")?.addEventListener("change", renderRunsList);
$("#filter-search")?.addEventListener("input", renderRunsList);
$("#bulk-compare-btn")?.addEventListener("click", openComparisonView);
$("#bulk-trash-btn")?.addEventListener("click", trashSelected);
$("#selection-clear-btn")?.addEventListener("click", clearSelection);
$("#trash-incomplete-btn")?.addEventListener("click", trashAllIncomplete);
$("#open-trash-btn")?.addEventListener("click", openTrashDrawer);
$("#close-trash-btn")?.addEventListener("click", closeTrashDrawer);
$("#empty-trash-btn")?.addEventListener("click", emptyTrash);
$("#restore-selected-btn")?.addEventListener("click", restoreSelectedTrashed);
$("#purge-selected-btn")?.addEventListener("click", purgeSelectedTrashed);

// ─────────────────────────────────────────────────────────────────────────
// Trash actions
// ─────────────────────────────────────────────────────────────────────────
async function trashSelected() {
  if (selectionSet.size === 0) return;
  const names = Array.from(selectionSet);
  if (!confirm(`Move ${names.length} run${names.length === 1 ? "" : "s"} to trash?`)) return;
  try {
    const res = await fetch("/api/runs/trash", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    selectionSet.clear();
    toast(`Trashed ${data.n_trashed} run${data.n_trashed === 1 ? "" : "s"}`,
      { kind: "success" });
    if (data.skipped?.length) {
      toast(`Skipped ${data.skipped.length} (already gone or invalid)`, { kind: "warn" });
    }
    await loadRuns();
    refreshTrashCount();
  } catch (e) {
    toast(`Trash failed: ${e.message}`, { kind: "error" });
  }
}

async function trashAllIncomplete() {
  if (!confirm("Move ALL incomplete runs (no phase_stats data) to trash?")) return;
  try {
    const res = await fetch("/api/runs/trash/incomplete", { method: "POST" });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    toast(`Trashed ${data.n_trashed} incomplete run${data.n_trashed === 1 ? "" : "s"}`,
      { kind: "success" });
    await loadRuns();
    refreshTrashCount();
  } catch (e) {
    toast(`Trash incomplete failed: ${e.message}`, { kind: "error" });
  }
}

async function refreshTrashCount() {
  try {
    const res = await fetch("/api/runs/trash/list");
    if (!res.ok) return;
    const data = await res.json();
    const c = document.getElementById("trash-count");
    if (c) c.textContent = String(data.count);
  } catch {}
}

// ─────────────────────────────────────────────────────────────────────────
// Trash drawer (replaces runs panel when open)
// ─────────────────────────────────────────────────────────────────────────
const trashSelection = new Set();

async function openTrashDrawer() {
  const drawer = document.getElementById("trash-drawer");
  if (!drawer) return;
  drawer.style.display = "flex";
  await loadTrashList();
}

function closeTrashDrawer() {
  const drawer = document.getElementById("trash-drawer");
  if (drawer) drawer.style.display = "none";
  trashSelection.clear();
  updateTrashSelectionUI();
}

async function loadTrashList() {
  const list = document.getElementById("trash-list");
  if (!list) return;
  showSkeleton(list, 6, "list");
  try {
    const res = await fetch("/api/runs/trash/list");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    document.getElementById("trash-list-count").textContent =
      `${data.count} trashed run${data.count === 1 ? "" : "s"}`;
    if (!data.trashed.length) {
      list.replaceChildren(el("p", { class: "muted", style: "padding:16px" }, "Trash is empty."));
      return;
    }
    list.replaceChildren();
    for (const t of data.trashed) {
      const checkbox = el("input", {
        type: "checkbox",
        class: "row-checkbox",
        "aria-label": `Select ${t.original_name}`,
      });
      checkbox.checked = trashSelection.has(t.trash_filename);
      checkbox.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (checkbox.checked) trashSelection.add(t.trash_filename);
        else trashSelection.delete(t.trash_filename);
        updateTrashSelectionUI();
        loadTrashList();
      });
      const body = el("div", { class: "row-body" }, [
        el("div", { class: "name" }, t.original_name),
        el("div", { class: "meta" }, `seed=${t.seed ?? "?"} · trashed ${formatTrashTimestamp(t.trashed_at)} · ${(t.size_bytes / 1024).toFixed(1)} KB`),
      ]);
      const restoreBtn = el("button", {
        class: "ctrl-btn",
        title: "Restore this run",
        onclick: (ev) => { ev.stopPropagation(); restoreTrashed([t.trash_filename]); },
      }, "↺ Restore");
      const purgeBtn = el("button", {
        class: "ctrl-btn bad",
        title: "Permanently delete",
        onclick: (ev) => { ev.stopPropagation(); purgeTrashed([t.trash_filename]); },
      }, "🗑 Delete");
      const actions = el("div", { class: "row-actions" }, [restoreBtn, purgeBtn]);
      const row = el("div", {
        class: "trash-row" + (trashSelection.has(t.trash_filename) ? " row-selected" : ""),
      }, [checkbox, body, actions]);
      list.appendChild(row);
    }
    updateTrashSelectionUI();
  } catch (e) {
    list.replaceChildren(el("p", { class: "error" }, e.message));
  }
}

function formatTrashTimestamp(s) {
  // s is "YYYYmmdd_HHMMSS" — make it human-readable
  if (!s || s.length < 15) return s || "?";
  const y = s.slice(0, 4), m = s.slice(4, 6), d = s.slice(6, 8);
  const hh = s.slice(9, 11), mm = s.slice(11, 13), ss = s.slice(13, 15);
  return `${y}-${m}-${d} ${hh}:${mm}:${ss}`;
}

function updateTrashSelectionUI() {
  const n = trashSelection.size;
  const restore = document.getElementById("restore-selected-btn");
  const purge = document.getElementById("purge-selected-btn");
  if (restore) {
    restore.disabled = n === 0;
    restore.textContent = `↺ Restore selected${n ? ` (${n})` : ""}`;
  }
  if (purge) {
    purge.disabled = n === 0;
    purge.textContent = `🗑 Delete forever${n ? ` (${n})` : ""}`;
  }
}

async function restoreTrashed(trashFilenames) {
  try {
    const res = await fetch("/api/runs/trash/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trash_filenames: trashFilenames }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    toast(`Restored ${data.n_restored} run${data.n_restored === 1 ? "" : "s"}`, { kind: "success" });
    if (data.skipped?.length) {
      const reason = data.skipped[0].reason;
      toast(`Skipped ${data.skipped.length}: ${reason}`, { kind: "warn" });
    }
    trashSelection.clear();
    await loadTrashList();
    await loadRuns();
    refreshTrashCount();
  } catch (e) {
    toast(`Restore failed: ${e.message}`, { kind: "error" });
  }
}

async function purgeTrashed(trashFilenames) {
  if (!confirm(`Permanently delete ${trashFilenames.length} run${trashFilenames.length === 1 ? "" : "s"}? This cannot be undone.`)) return;
  try {
    const res = await fetch("/api/runs/trash/purge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trash_filenames: trashFilenames }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    toast(`Purged ${data.n_purged} item${data.n_purged === 1 ? "" : "s"}`, { kind: "success" });
    trashSelection.clear();
    await loadTrashList();
    refreshTrashCount();
  } catch (e) {
    toast(`Purge failed: ${e.message}`, { kind: "error" });
  }
}

async function restoreSelectedTrashed() {
  if (trashSelection.size === 0) return;
  await restoreTrashed(Array.from(trashSelection));
}

async function purgeSelectedTrashed() {
  if (trashSelection.size === 0) return;
  await purgeTrashed(Array.from(trashSelection));
}

async function emptyTrash() {
  if (!confirm("Empty the entire trash? This permanently deletes all trashed runs.")) return;
  try {
    const res = await fetch("/api/runs/trash/purge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trash_filenames: null }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    toast(`Emptied trash (${data.n_purged} items)`, { kind: "success" });
    trashSelection.clear();
    await loadTrashList();
    refreshTrashCount();
  } catch (e) {
    toast(`Empty trash failed: ${e.message}`, { kind: "error" });
  }
}

// Load trash count on page load + periodically
refreshTrashCount();
setInterval(refreshTrashCount, 30_000);

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
