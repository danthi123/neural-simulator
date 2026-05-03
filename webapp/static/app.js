// Neural Simulator — Research Dashboard frontend
// Phase 1 vanilla JS. No build step. ES modules in the browser.
// Phase 2 adds the World tab (2D playback) wired up via world.js.
//
// All dynamic content (filenames, markdown body, JSON values) is rendered
// via textContent or escapeHTML — never via raw template-literal innerHTML.

import { setupWorldTab, loadRunIntoWorld } from "/static/world.js";
// 2026-05-03: 3D brain visualization. Lazy-imported on first activation
// of the Brain tab so users who never visit it don't pay the Three.js
// download cost (~600KB minified from CDN).
let _brain3dModule = null;
async function getBrain3D() {
  if (_brain3dModule) return _brain3dModule;
  _brain3dModule = await import("/static/brain3d.js");
  return _brain3dModule;
}
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
  { id: "brain",       label: "Brain",       order: 60, onActivate: () => activateBrainTab() },
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
      // 2026-05-03: update URL hash for deep linking. Format:
      //   #tab=brain&run=foo.json (run is optional; preserved from prior state)
      updateUrlHash({ tab: t });
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
// URL deep links (2026-05-03)
//
// Format: #tab=<id>&run=<filename>&step=<n>
// On load: read hash, activate tab, optionally load a specific run.
// On tab change: update hash. On run load: update hash with run name.
//
// Use cases:
//   - Share a specific 3D brain view: copy the URL
//   - Bookmark "the seed 42 SWR run in 3D" and reopen
//   - Browser back/forward navigates between recently-viewed tabs
// ─────────────────────────────────────────────────────────────────────────

function parseUrlHash() {
  const hash = window.location.hash.replace(/^#/, "");
  const out = {};
  for (const part of hash.split("&")) {
    const [k, v] = part.split("=");
    if (k && v != null) out[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return out;
}

function updateUrlHash(updates) {
  const cur = parseUrlHash();
  Object.assign(cur, updates);
  // Drop empty values for cleaner URLs
  const parts = [];
  for (const [k, v] of Object.entries(cur)) {
    if (v != null && v !== "") parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  const newHash = parts.length ? `#${parts.join("&")}` : "";
  if (newHash !== window.location.hash) {
    // history.replaceState avoids polluting browser history with every
    // tab click. Use pushState only on explicit "Open in viewer" actions.
    history.replaceState(null, "", newHash || window.location.pathname);
  }
}

// Apply a parsed URL hash by activating the right tab and (optionally)
// loading a specific run into the appropriate viewer.
function applyUrlHash() {
  const params = parseUrlHash();
  if (!params.tab) return;
  const tabBtn = document.querySelector(`nav button[data-tab="${params.tab}"]`);
  if (!tabBtn) return;
  tabBtn.click();
  if (params.run) {
    // Use the cross-tab opener — picks the right viewer for the tab+run.
    const targetViewer = params.tab === "brain" ? "brain"
                       : params.tab === "world" ? "world"
                       : params.tab === "language" ? "language"
                       : null;
    if (targetViewer) {
      // Slight delay so the tab activation completes first
      setTimeout(() => openRunInViewer(params.run, targetViewer), 200);
    }
  }
}

// Listen for back/forward navigation
window.addEventListener("hashchange", applyUrlHash);

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

// ─────────────────────────────────────────────────────────────────────────
// Browser notifications for run completion (2026-05-03)
//
// When a run that was previously alive transitions to alive=false, fire
// a Notification (if permission granted) so the user gets a desktop
// alert without having to keep the tab open.
// ─────────────────────────────────────────────────────────────────────────
const _previousLiveRunNames = new Set();
let _notificationPermission = null;

function initNotifications() {
  if (!("Notification" in window)) {
    _notificationPermission = "unsupported";
    return;
  }
  _notificationPermission = Notification.permission;
}

function requestNotificationPermissionIfNeeded() {
  if (_notificationPermission === "granted" ||
      _notificationPermission === "denied" ||
      _notificationPermission === "unsupported") return;
  Notification.requestPermission().then((p) => {
    _notificationPermission = p;
    if (p === "granted") {
      toast("Browser notifications enabled — you'll be alerted when runs finish.",
        { kind: "success" });
    }
  });
}

function notifyRunCompleted(name, completed) {
  if (_notificationPermission !== "granted") return;
  try {
    const title = completed ? `✓ Run completed: ${name}` : `■ Run stopped: ${name}`;
    const body = completed
      ? "Click to open in the dashboard."
      : "The run terminated without completion.";
    const n = new Notification(title, { body, tag: name });
    n.onclick = () => {
      window.focus();
      const isText = name.startsWith("text_eval_") || name.startsWith("text_io_");
      const fullName = name + (name.endsWith(".json") ? "" : ".json");
      openRunInViewer(fullName, isText ? "language" : "world");
      n.close();
    };
  } catch (e) {
    toast(`Run finished: ${name}`, { kind: "success" });
  }
}

function checkForCompletedRuns(inflightList) {
  const currentAliveNames = new Set();
  for (const r of inflightList) {
    if (r.alive) currentAliveNames.add(r.name);
  }
  for (const prev of _previousLiveRunNames) {
    if (!currentAliveNames.has(prev)) {
      const rec = inflightList.find((r) => r.name === prev);
      const completed = rec ? !!rec.completed : false;
      notifyRunCompleted(prev, completed);
    }
  }
  _previousLiveRunNames.clear();
  for (const n of currentAliveNames) _previousLiveRunNames.add(n);
  // 2026-05-03: update the global "live runs" badge in the nav so
  // users see active runs from any tab.
  updateActiveRunsBadge(currentAliveNames.size);
}

function updateActiveRunsBadge(n) {
  const badge = document.getElementById("active-runs-badge");
  const count = document.getElementById("active-runs-count");
  if (!badge || !count) return;
  if (n > 0) {
    count.textContent = String(n);
    badge.style.display = "";
  } else {
    badge.style.display = "none";
  }
}

// 2026-05-03 — refresh the type-filter chips' counts based on the
// current full run list. Called after each /api/runs fetch.
function refreshRunsChipCounts() {
  const all = _allRuns.filter((r) => !/_smoke/i.test(r.name));
  const nav = all.filter((r) => classifyRun(r) === "navigation");
  const lang = all.filter((r) => classifyRun(r) === "language");
  const a = $("#chip-all-count"); if (a) a.textContent = String(all.length);
  const n = $("#chip-nav-count"); if (n) n.textContent = String(nav.length);
  const l = $("#chip-lang-count"); if (l) l.textContent = String(lang.length);
}

// Live runs panel at the top of the Runs tab. Same idea as the Brain
// tab's live monitor but lives in the Runs tab where all run-related
// activity is centralized. Reuses renderBrainRunCard() (which is
// generic — name/progress/state agnostic to which tab hosts it).
async function refreshRunsLivePanel() {
  const wrap = $("#runs-live-panel-wrap");
  const container = $("#runs-live-panel");
  const counter = $("#runs-live-count");
  if (!wrap || !container) return;
  try {
    const res = await fetch("/api/inflight");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    // Diff against previously-alive set to detect completion -> notify
    checkForCompletedRuns(data.inflight || []);
    const runs = (data.inflight || []).filter((r) => r.alive);
    if (!runs.length) {
      wrap.style.display = "none";
      return;
    }
    wrap.style.display = "";
    counter.textContent = runs.length === 1 ? "1 active" : `${runs.length} active`;
    container.replaceChildren();
    for (const r of runs) {
      container.appendChild(renderBrainRunCard(r));
    }
  } catch (e) {
    container.replaceChildren(el("p", { class: "error" },
      `Failed to load live runs: ${e.message}`));
  }
}

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
    refreshRunsChipCounts();
    refreshRunsLivePanel();
  } catch (e) {
    // On refresh failure, keep the existing list visible — only show
    // an error if this was the first load.
    if (isFirstLoad) {
      list.replaceChildren(el("p", { class: "error" }, e.message));
    }
  }
}

// 2026-05-03: Run-type classifier and per-row viewer buttons.
//
// All runs are unified in the Runs tab; from each row, the user can
// open the run in any compatible viewer (Brain 3D / World 2D /
// Language confusion matrix / Stats). The buttons that appear depend
// on what data the run has — text I/O runs don't have a trajectory,
// so World/Brain replay isn't useful for them; navigation runs don't
// have confusion matrices, so Language isn't useful.

function classifyRun(r) {
  const name = r.name || "";
  if (name.startsWith("text_eval_") || name.startsWith("text_io_")) return "language";
  if (/_smoke/i.test(name)) return "smoke";
  return "navigation";
}

// Activate target tab AND load the named run into its viewer.
// Single helper called by per-row Brain / World / Language buttons.
function openRunInViewer(name, viewer) {
  switch (viewer) {
    case "world":
      activateTab("world");
      loadRunIntoWorld(name);
      updateUrlHash({ tab: "world", run: name });
      break;
    case "brain":
      activateTab("brain");
      // brain3d module is lazy-loaded; ensure it's initialized first
      initBrain3DOnce().then(async () => {
        const mod = await getBrain3D();
        mod.brain3dLoadRun(name);
      });
      updateUrlHash({ tab: "brain", run: name });
      break;
    case "language":
      activateTab("language");
      window._languageLoaded = false;  // force reload
      loadLanguage().then(() => {
        // Find the row for this run and click it
        setTimeout(() => {
          const rows = $$("#language-list .lang-row");
          const target = rows.find((row) =>
            row.querySelector(".name")?.textContent === name.replace(/\.json$/, ""));
          if (target) target.click();
        }, 200);
      });
      updateUrlHash({ tab: "language", run: name });
      break;
    case "stats":
    default:
      activateTab("runs");
      // Find row in the run list and click it to open detail
      setTimeout(() => {
        const rows = $$("#runs-list .list-item");
        const target = rows.find((row) =>
          row.querySelector(".name")?.textContent === name);
        if (target) target.querySelector(".row-body")?.click();
      }, 50);
      updateUrlHash({ tab: "runs", run: name });
      break;
  }
}

// Make the helper accessible from world.js etc. without ESM circular import.
window.openRunInViewer = openRunInViewer;

function renderRunsList() {
  const list = $("#runs-list");
  const hideSmoke = $("#filter-hide-smoke")?.checked ?? true;
  const hideIncomplete = $("#filter-hide-incomplete")?.checked ?? false;
  const search = ($("#filter-search")?.value ?? "").trim().toLowerCase();
  // 2026-05-03: type-filter chips ("All", "Navigation", "Text I/O").
  const typeFilter = window._runsTypeFilter || "all";

  const filtered = _allRuns.filter((r) => {
    if (hideSmoke && /smoke/i.test(r.name)) return false;
    if (hideIncomplete && r.sum_finalQ == null) return false;
    if (search && !r.name.toLowerCase().includes(search)) return false;
    if (typeFilter === "navigation" && classifyRun(r) !== "navigation") return false;
    if (typeFilter === "language" && classifyRun(r) !== "language") return false;
    return true;
  });

  $("#runs-count").textContent =
    `${filtered.length}${filtered.length !== _allRuns.length ? `/${_allRuns.length}` : ""} runs`;

  list.replaceChildren();
  filtered.forEach((r, idx) => {
    const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
    const isSelected = selectionSet.has(r.name);
    const runType = classifyRun(r);
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
    // Type pill (small badge to the left of the name)
    const typePill = el("span", {
      class: `run-type-pill type-${runType}`,
      title: runType === "navigation" ? "Navigation gridworld run"
            : runType === "language" ? "Text I/O language run"
            : "Smoke test (short verification)",
    }, runType === "navigation" ? "NAV" : runType === "language" ? "LANG" : "SMK");
    const body = el("div", { class: "row-body" }, [
      el("div", { class: "name" }, [typePill, " ", r.name]),
      el("div", { class: "meta" }, [
        runType === "navigation" ? metric("sum", sumStr) : metric("type", "text I/O"),
        metric("seed", r.seed ?? "—"),
        runType === "navigation" ? metric("phases", r.n_phases) : null,
      ].filter(Boolean)),
    ]);
    // Per-run viewer buttons. Clicking a button activates the target
    // tab and loads the run there. Only show buttons compatible with
    // the run's data shape.
    const viewerBtns = el("div", { class: "row-viewers" });
    if (runType === "navigation") {
      const worldBtn = el("button", {
        class: "row-viewer-btn", title: "Open in 2D World replay",
      }, "🌍");
      worldBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openRunInViewer(r.name, "world");
      });
      viewerBtns.appendChild(worldBtn);

      const brainBtn = el("button", {
        class: "row-viewer-btn", title: "Open in 3D Brain visualization",
      }, "🧠");
      brainBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openRunInViewer(r.name, "brain");
      });
      viewerBtns.appendChild(brainBtn);
    } else if (runType === "language") {
      const langBtn = el("button", {
        class: "row-viewer-btn", title: "Open Language confusion matrices",
      }, "💬");
      langBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openRunInViewer(r.name, "language");
      });
      viewerBtns.appendChild(langBtn);
    }
    const item = el("div", {
      class: "list-item" + (isSelected ? " row-selected" : ""),
      dataset: { name: r.name, idx: String(idx) },
    }, [checkbox, body, viewerBtns]);
    body.addEventListener("click", (ev) => {
      if (ev.shiftKey) {
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

// Per-direction accuracy bars — extracts the diagonal cell / row sum
// for each row label, displayed as horizontal bars. Faster to scan than
// the confusion matrix when the user just wants "which direction is
// the cascade biased toward?".
function renderPerDirectionBreakdown(title, matrix, rowLabels, colLabels) {
  const wrapper = el("div", { class: "per-dir-wrapper" });
  wrapper.appendChild(el("h4", {}, title));
  if (!matrix) {
    wrapper.appendChild(el("p", { class: "muted" }, "(no data)"));
    return wrapper;
  }
  const rows = el("div", { class: "per-dir-rows" });
  for (const rLabel of rowLabels) {
    const row = matrix[rLabel] || {};
    const total = colLabels.reduce((s, c) => s + Number(row[c] || 0), 0);
    // Find the column matching this row by first-letter prefix
    // (north→N, east→E, etc.) — works for both word→action where
    // labels differ AND image→word where they match.
    const matchCol = colLabels.find((c) =>
      c[0]?.toLowerCase() === rLabel[0]?.toLowerCase()) || rowLabels[0];
    const correct = Number(row[matchCol] || 0);
    const acc = total > 0 ? correct / total : 0;
    const pct = (acc * 100).toFixed(0);
    const aboveChance = acc > 0.30; // 25% chance + 5pp margin
    const fillBar = el("div", {
      class: "per-dir-fill" + (aboveChance ? " above" : ""),
      style: `width: ${Math.round(acc * 100)}%`,
    });
    const chanceMark = el("div", { class: "per-dir-chance", title: "25% chance" });
    rows.appendChild(el("div", { class: "per-dir-row" }, [
      el("div", { class: "per-dir-label" }, rLabel),
      el("div", { class: "per-dir-bar" }, [chanceMark, fillBar]),
      el("div", { class: "per-dir-pct" + (aboveChance ? " above" : "") },
        `${correct}/${total} = ${pct}%`),
    ]));
  }
  wrapper.appendChild(rows);
  return wrapper;
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
      fmtRelTime(r.modified_unix),
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
    // 2026-05-03: For curriculum runs (phase 1 / 2 / 3), the meaningful
    // training rate is Phase 2 (text I/O training on trained cascade).
    // Phase 1 + Phase 3 may have 0 episodes for the v2-baseline+SWR
    // configuration, which previously made this card show "0.0% 0/0".
    // Pick the phase with the most episodes — that's the active
    // training phase regardless of whether it's a curriculum run or
    // legacy text_eval_embodied run.
    const tsList = data.training_stats || [];
    const trainingPhase = tsList.reduce(
      (best, t) => (t && (t.n_total_steps || 0) > (best?.n_total_steps || 0) ? t : best),
      null,
    );
    if (trainingPhase?.correct_move_rate != null && trainingPhase.n_total_steps > 0) {
      const phaseLabel = trainingPhase.phase ? `Phase ${trainingPhase.phase}` : "Training";
      statsRow.appendChild(makeKpiCard(
        `${phaseLabel} corr.move`,
        fmtPercent(trainingPhase.correct_move_rate),
        `${trainingPhase.n_correct_moves}/${trainingPhase.n_total_steps}` +
          (trainingPhase.elapsed_seconds ? ` · ${(trainingPhase.elapsed_seconds / 60).toFixed(1)} min` : ""),
      ));
    }
    wrapper.appendChild(statsRow);
    // 2026-05-03: per-direction breakdown bars. Surfaces the cascade-N-bias
    // pattern more readably than a confusion matrix — at a glance the user
    // can see "north 80%, east 45%, south 30%, west 12%".
    if (data.image_to_word_eval?.confusion_matrix) {
      wrapper.appendChild(renderPerDirectionBreakdown(
        "Image → Word per-direction accuracy",
        data.image_to_word_eval.confusion_matrix,
        LANG_DIRS, LANG_DIRS,
      ));
    }
    if (data.word_to_action_eval?.confusion_matrix) {
      wrapper.appendChild(renderPerDirectionBreakdown(
        "Word → Action per-direction accuracy",
        data.word_to_action_eval.confusion_matrix,
        LANG_DIRS, ACTION_DIRS,
      ));
    }
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
  // Render dated findings first, then a separator, then reference docs.
  let lastWasReference = false;
  let renderedSeparator = false;
  for (const f of filtered) {
    if (f.is_reference && !lastWasReference && renderedSeparator === false) {
      // Insert section header to separate reference docs from chronological findings
      const sep = el("div", { class: "findings-section-header" },
        "📌 Reference docs (no date)");
      list.appendChild(sep);
      renderedSeparator = true;
    }
    lastWasReference = !!f.is_reference;
    // Strip date prefix and .md suffix for shorter display name
    const display = f.name.replace(/\.md$/, "").replace(/^\d{4}-\d{2}-\d{2}-/, "");
    const dateText = f.date || "no date";
    // Recent badge for findings within the last 3 days
    const recentBadge = f.is_recent
      ? el("span", { class: "finding-recent-badge", title: "modified within last 3 days" }, "RECENT")
      : null;
    const item = el("div", {
      class: "list-item" + (f.is_reference ? " is-reference" : "") + (f.is_recent ? " is-recent" : ""),
    }, [
      el("div", { class: "name" }, [display, recentBadge && " ", recentBadge].filter(Boolean)),
      el("div", { class: "meta" }, [
        el("span", { class: "finding-tag" }, f.tag),
        " · ",
        el("span", {}, dateText),
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
// 2026-05-03: parse a multi-seed input string. See setupLauncher() for
// supported syntaxes. Returns a deduped sorted-by-input-order array of
// integer seeds.
function parseSeedInput(s) {
  if (!s) return [];
  const out = [];
  for (const part of s.split(/[,\s]+/).filter(Boolean)) {
    const m = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) {
      const lo = parseInt(m[1], 10);
      const hi = parseInt(m[2], 10);
      if (!isNaN(lo) && !isNaN(hi) && lo <= hi && hi - lo < 200) {
        for (let v = lo; v <= hi; v++) out.push(v);
      }
    } else {
      const v = parseInt(part, 10);
      if (!isNaN(v)) out.push(v);
    }
  }
  // Dedup while preserving first-seen order
  const seen = new Set();
  return out.filter((v) => {
    if (seen.has(v)) return false;
    seen.add(v);
    return true;
  });
}

function setupLauncher() {
  const form = $("#launch-form");
  const out = $("#launcher-output");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    out.replaceChildren();

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

    // 2026-05-03: parse multi-seed input. Accepts:
    //   "42"            -> [42]
    //   "42,43,44"      -> [42, 43, 44]
    //   "42-47"         -> [42, 43, 44, 45, 46, 47]
    //   "42, 43, 100-102" -> [42, 43, 100, 101, 102]
    const seedStr = String(formData.get("seed") || "").trim();
    const seeds = parseSeedInput(seedStr);
    if (!seeds.length) {
      appendError(out, "Invalid seed input. Use e.g. 42 or 42,43,44 or 42-47.");
      return;
    }
    const preset = String(formData.get("preset"));

    appendStatus(out, `Submitting ${seeds.length} run${seeds.length === 1 ? "" : "s"} (seeds ${seeds.join(", ")})…`);
    // First time launching from this session — ask for notification
    // permission so the user gets desktop alerts when long runs finish.
    requestNotificationPermissionIfNeeded();

    const body0 = { preset, extra_args: extras };

    // Sequentially POST /api/runs/launch for each seed. Sequential
    // (rather than parallel) avoids overloading the backend's process
    // spawner and lets us stream stdout for each launch.
    let lastWsUrl = null;
    for (const seed of seeds) {
      try {
        const res = await fetch("/api/runs/launch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body0, seed }),
        });
        if (!res.ok) throw new Error(`${res.status}`);
        const launch = await res.json();
        appendStatus(out, `Launched seed ${seed} (run_id=${launch.run_id})`);
        appendStatus(out, `  out: ${launch.out_path}`);
        toast(`Launched ${preset} seed ${seed}`, { kind: "success" });
        lastWsUrl = launch.ws_url;
      } catch (e) {
        appendError(out, `Launch seed ${seed} failed: ${e.message}`);
      }
    }
    appendStatus(out, "All launches submitted.");
    if (lastWsUrl && seeds.length === 1) {
      appendStatus(out, `Streaming stdout for the run via WebSocket…`);
      tailWebSocket(lastWsUrl, out);
    } else if (seeds.length > 1) {
      appendStatus(out, "Browse runs (or check Brain tab Live mode picker) to follow individual progress.");
    }
    return;
    /* unreachable — original try/catch retained below for legacy path */
    try {
      // legacy single-launch fallback (no longer reached; see seed parser above)
      const res = await fetch("/api/runs/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body0, seed: seeds[0] }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const launch = await res.json();
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
// ─────────────────────────────────────────────────────────────────────────
// Brain tab — live in-flight run monitor + 3D viz placeholder
//
// 2026-05-02. The Brain tab eventually hosts the WebGL 3D visualization
// designed in docs/plans/2026-05-02-webapp-3d-visualization-design.md.
// Until that ships, the upper half of the tab is a "live runs" panel:
// polls /api/inflight every 2s, renders a card per active run with
// progress + percent + log-size growth indicator, plus a "Watch"
// button that opens a WebSocket log tail in a sliding pane.
//
// Why here vs Home: Home is the at-a-glance landing page. Brain is the
// "watch the simulator working" surface — the live monitor lives here
// because conceptually it's the closest thing to a brain visualization
// we currently have, and the 3D viz will replace it.
// ─────────────────────────────────────────────────────────────────────────

let _brainPollTimer = null;
let _brainTabActive = false;
let _brainLogWS = null;
let _brainLogTailUrl = null;

function activateBrainTab() {
  _brainTabActive = true;
  refreshBrainLive();
  if (!_brainPollTimer) {
    _brainPollTimer = setInterval(() => {
      // Only poll while the tab is currently visible — otherwise we
      // burn cycles fetching for an unwatched panel. activateBrainTab()
      // / deactivateBrainTab() flip _brainTabActive accordingly.
      if (_brainTabActive) refreshBrainLive();
    }, 2000);
  }
  // Lazy-init the 3D scene on first Brain-tab activation.
  initBrain3DOnce();
}

let _brain3dInitialized = false;
async function initBrain3DOnce() {
  if (_brain3dInitialized) return;
  _brain3dInitialized = true;
  const host = document.getElementById("brain3d-host");
  if (!host) return;
  try {
    const mod = await getBrain3D();
    await mod.initBrain3D({ canvasContainer: host });
    setupBrain3DControls(mod);
  } catch (e) {
    host.replaceChildren(el("p", { class: "error" },
      `3D viz failed to load: ${e.message}. Check browser console for details.`));
  }
}

function setupBrain3DControls(mod) {
  // Play / pause
  $("#brain3d-play")?.addEventListener("click", () => {
    const st = mod.brain3dGetState();
    if (st.replayPlaying) mod.brain3dPause();
    else mod.brain3dPlay();
  });
  // Scrubber
  $("#brain3d-scrubber")?.addEventListener("input", (e) => {
    const step = parseInt(e.target.value, 10);
    if (!isNaN(step)) mod.brain3dRenderStep(step);
  });
  // Speed
  $("#brain3d-speed")?.addEventListener("change", (e) => {
    const s = parseInt(e.target.value, 10);
    if (!isNaN(s)) mod.brain3dSetSpeed(s);
  });
  // Load run drawer
  $("#brain3d-load")?.addEventListener("click", async () => {
    const drawer = $("#brain3d-runs-drawer");
    drawer.style.display = "";
    const list = $("#brain3d-runs-list");
    list.replaceChildren(document.createTextNode("Loading…"));
    try {
      const res = await fetch("/api/runs");
      const data = await res.json();
      // Filter to navigation runs (text I/O runs don't have trajectory)
      const navRuns = (data.runs || []).filter(
        (r) => !/^text_eval_/.test(r.name) && !/^text_io_/.test(r.name) && r.sum_finalQ != null
      );
      list.replaceChildren();
      for (const r of navRuns.slice(0, 80)) {
        const item = el("div", { class: "list-item" }, [
          el("div", { class: "name" }, r.name),
          el("div", { class: "meta muted" }, [
            `seed ${r.seed} · sum=${r.sum_finalQ?.toFixed(2) ?? "—"} · phases=${r.n_phases}`,
          ]),
        ]);
        item.addEventListener("click", () => {
          mod.brain3dLoadRun(r.name);
          drawer.style.display = "none";
        });
        list.appendChild(item);
      }
    } catch (e) {
      list.replaceChildren(el("p", { class: "error" }, e.message));
    }
  });
  $("#brain3d-runs-close")?.addEventListener("click", () => {
    $("#brain3d-runs-drawer").style.display = "none";
  });
  // Live mode + live-run picker
  const livePicker = $("#brain3d-live-picker");
  $("#brain3d-live")?.addEventListener("click", () => {
    const liveLabel = $("#brain3d-live-label");
    const st = mod.brain3dGetState();
    if (st.liveMode) {
      mod.brain3dStopLive();
      if (liveLabel) liveLabel.style.display = "none";
      if (livePicker) livePicker.style.display = "none";
      $("#brain3d-live").textContent = "Live mode";
    } else {
      mod.brain3dStartLive();
      if (liveLabel) liveLabel.style.display = "";
      if (livePicker) livePicker.style.display = "";
      $("#brain3d-live").textContent = "Stop live";
    }
  });
  // When user picks a different live run from the dropdown, retarget.
  livePicker?.addEventListener("change", (e) => {
    const name = e.target.value;
    if (name) mod.brain3dSelectLiveRun(name);
  });
  // Jump to latest live sample (re-engage auto-follow)
  $("#brain3d-latest")?.addEventListener("click", () => {
    const st = mod.brain3dGetState();
    if (st.liveMode) {
      mod.brain3dJumpToLatest();
    } else if (st.replayLoaded) {
      // In replay mode, jump to last step
      mod.brain3dRenderStep(st.replayTotal - 1);
      const sl = $("#brain3d-scrubber");
      if (sl) sl.value = String(st.replayTotal - 1);
    }
  });
  // Reset camera (re-fits to full scene)
  $("#brain3d-reset-camera")?.addEventListener("click", () => {
    mod.brain3dFitCamera();
  });
  // Camera preset selector
  $("#brain3d-camera-preset")?.addEventListener("change", (e) => {
    const preset = e.target.value;
    if (preset) {
      mod.brain3dFlyToPreset(preset);
      // Reset to default option after fly so picking the same one again works
      setTimeout(() => { e.target.value = ""; }, 100);
    }
  });
  // Pathway visibility toggles
  $("#brain3d-pw-exc")?.addEventListener("change", (e) =>
    mod.brain3dSetPathwayKindVisible("exc", e.target.checked));
  $("#brain3d-pw-inh")?.addEventListener("change", (e) =>
    mod.brain3dSetPathwayKindVisible("inh", e.target.checked));
  $("#brain3d-pw-da")?.addEventListener("change", (e) =>
    mod.brain3dSetPathwayKindVisible("da", e.target.checked));
  $("#brain3d-only-flowing")?.addEventListener("change", (e) =>
    mod.brain3dSetOnlyFlowing(e.target.checked));
  // Help button
  $("#brain3d-help")?.addEventListener("click", () => {
    alert(
      "3D Brain Viz controls:\n\n" +
      "• Drag with left mouse to orbit camera\n" +
      "• Right-click drag to pan\n" +
      "• Scroll to zoom\n\n" +
      "• Load run... — pick a completed run; scrubber moves through trajectory\n" +
      "• Play / pause — auto-advance scrubber at the selected speed\n" +
      "• Live mode — animate based on currently in-flight run\n\n" +
      "Region color = functional family. Brightness = synthesized\n" +
      "activation from action+reward. Pathway lines brighten when\n" +
      "their endpoints are active."
    );
  });
}

// Stop polling when user navigates away from the Brain tab. setupTabs()
// already toggles section.active classes; we hook into that via a
// MutationObserver-free approach: just set _brainTabActive=false in the
// tab click handler below.
$$("nav button").forEach((b) => {
  b.addEventListener("click", () => {
    if (b.dataset.tab !== "brain") {
      _brainTabActive = false;
    }
  });
});

async function refreshBrainLive() {
  const container = $("#brain-live-runs");
  const counter = $("#brain-live-count");
  if (!container) return;
  try {
    const res = await fetch("/api/inflight");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    // 2026-05-03: only show actually-running runs in the "Live" panel.
    // Previously this showed all entries with a *.pid file regardless of
    // alive state, so completed runs lingered as if still in flight.
    // Completed runs are now visible via the Runs tab (with their result
    // file accessible) — that's the correct surface for them.
    const allRuns = data.inflight || [];
    const runs = allRuns.filter((r) => r.alive);
    counter.textContent = runs.length === 1 ? "1 in flight" : `${runs.length} in flight`;
    if (!runs.length) {
      container.replaceChildren(el("p", { class: "muted" },
        "No active runs. Launch one from the Lab tab to see it here."));
      return;
    }
    container.replaceChildren();
    for (const r of runs) {
      container.appendChild(renderBrainRunCard(r));
    }
  } catch (e) {
    container.replaceChildren(el("p", { class: "error" },
      `Failed to load in-flight runs: ${e.message}`));
  }
}

function renderBrainRunCard(r) {
  const p = r.progress || {};
  const fraction = p.fraction || 0;
  const pct = Math.round(fraction * 100);

  const stateBadge = r.alive
    ? el("span", { class: "badge state-running" }, "● running")
    : (r.completed
        ? el("span", { class: "badge state-completed" }, "✓ completed")
        : el("span", { class: "badge state-stopped" }, "■ stopped"));

  // Headline progress line varies by progress kind.
  let progressLines = [];
  if (p.kind === "swr_replay") {
    progressLines.push(`Phase 3 SWR replay · event ${p.ev}/${p.ev_total}`);
  } else if (p.kind === "embodied_episode") {
    const phase = p.phase_num ? `Phase ${p.phase_num} · ` : "";
    progressLines.push(`${phase}episode ${p.episode}/${p.episodes_total}`);
    progressLines.push(`${p.correct_moves}/${p.n_steps} correct moves (${p.correct_pct}%)`);
  } else if (p.kind === "step") {
    progressLines.push(`step ${p.step}/${p.total} · pos=(${p.pos.join(",")}) · goal=(${p.goal.join(",")})`);
  } else {
    progressLines.push("(no progress markers yet)");
  }

  if (p.phase_label && p.kind !== "swr_replay") {
    progressLines.push(`Phase: ${p.phase_label}`);
  }

  const card = el("div", { class: "brain-run-card" });
  // Header row: name + state
  card.appendChild(el("div", { class: "brain-run-header" }, [
    el("div", { class: "brain-run-name" }, r.name),
    stateBadge,
  ]));
  // Progress bar
  const bar = el("div", { class: "brain-progress-bar" });
  const fill = el("div", { class: "brain-progress-fill", style: `width: ${pct}%` });
  bar.appendChild(fill);
  card.appendChild(bar);
  card.appendChild(el("div", { class: "brain-progress-label" },
    [`${pct}% · `, ...progressLines.map((l, i) => i === 0 ? l : ` · ${l}`)]));
  // Meta row: PID, log size, mtime
  const ageSec = r.log_mtime ? Math.round(Date.now() / 1000 - r.log_mtime) : null;
  card.appendChild(el("div", { class: "brain-run-meta muted" }, [
    `pid ${r.pid} · log ${r.log_size_kb} KB`,
    r.log_mtime ? ` · last write ${ageSec}s ago` : "",
    r.completed ? " · result available" : "",
  ]));
  // Action buttons
  const actions = el("div", { class: "brain-run-actions" });
  if (r.log_file) {
    const watchBtn = el("button", { class: "ctrl-btn" }, "📜 Watch logs");
    watchBtn.addEventListener("click", () => openBrainLogTail(r));
    actions.appendChild(watchBtn);
  }
  if (r.completed && r.result_file) {
    const resultBtn = el("button", { class: "ctrl-btn" },
      r.result_file.startsWith("text_eval_") ? "💬 View in Language" : "🌍 View in World");
    resultBtn.addEventListener("click", () => {
      if (r.result_file.startsWith("text_eval_")) {
        activateTab("language");
        // Trigger a fresh load and click the row.
        window._languageLoaded = false;
        loadLanguage().then(() => {
          setTimeout(() => {
            const rows = $$("#language-list .lang-row");
            const target = rows.find((row) =>
              row.querySelector(".name")?.textContent?.includes(r.result_file.replace(/\.json$/, "")));
            if (target) target.click();
          }, 200);
        });
      } else {
        activateTab("world");
      }
    });
    actions.appendChild(resultBtn);
  }
  card.appendChild(actions);
  return card;
}

function openBrainLogTail(run) {
  // Close any prior tail
  if (_brainLogWS) {
    try { _brainLogWS.close(); } catch (e) { /* ignore */ }
    _brainLogWS = null;
  }
  const pane = $("#brain-log-pane");
  const out = $("#brain-log-output");
  const nameEl = $("#brain-log-name");
  if (!pane) return;
  nameEl.textContent = `Log: ${run.name}`;
  pane.style.display = "";
  out.replaceChildren();
  // Fetch the current tail (last 8KB) once for fast initial paint, then
  // open a WebSocket subscription if the run is still alive.
  fetch(`/api/runs/launch/log/${encodeURIComponent(run.log_file)}`)
    .then((r) => r.ok ? r.text() : Promise.reject(r.status))
    .then((text) => {
      const lines = text.split("\n");
      out.replaceChildren();
      for (const line of lines.slice(-200)) {
        if (line) out.appendChild(el("div", { class: "log-line" }, line));
      }
      out.scrollTop = out.scrollHeight;
    })
    .catch(() => {
      // Endpoint doesn't exist yet — fall back to "no static tail" message.
      out.replaceChildren(el("div", { class: "muted" },
        "(static tail endpoint not available; will stream new lines as they arrive)"));
    });
  // Stream new lines via the existing /ws/runs/{run_id} WebSocket if we
  // have the run_id. The detached PID-based runs don't have a run_id
  // (they were launched via Start-Process), so streaming there isn't
  // wired up — for those we just show the static tail. Foreground
  // launches via /api/runs/launch DO have run_id and stream cleanly.
}

$("#brain-refresh-now")?.addEventListener("click", refreshBrainLive);
$("#brain-log-close")?.addEventListener("click", () => {
  $("#brain-log-pane").style.display = "none";
  if (_brainLogWS) {
    try { _brainLogWS.close(); } catch (e) { /* ignore */ }
    _brainLogWS = null;
  }
});

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
    // 2026-05-02: include text_io_runs for the W→A KPI
    // 2026-05-03: include /api/inflight (detached runs) so the In-flight
    // KPI counts BOTH foreground and detached runs
    const [runsRes, findingsRes, launchesRes, textIoRes, inflightRes] = await Promise.all([
      fetch("/api/runs").then((r) => r.json()),
      fetch("/api/findings").then((r) => r.json()),
      fetch("/api/runs/launch").then((r) => r.json()),
      fetch("/api/text_io_runs").then((r) => r.json()).catch(() => ({ runs: [], aggregate: {} })),
      fetch("/api/inflight").then((r) => r.json()).catch(() => ({ inflight: [] })),
    ]);

    renderOverviewKPIs(kpiContainer, runsRes.runs, findingsRes.findings,
                       launchesRes.runs, textIoRes, inflightRes.inflight);
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
    // 2026-05-03: only show actually-running runs in the Home in-flight
    // panel. Completed runs hang around in /api/inflight (they have
    // .pid files) but don't belong in a "live" panel; they appear in
    // the regular Runs tab instead.
    const allRuns = res.inflight || [];
    // Detect completion -> notify (same as Runs-tab live panel)
    checkForCompletedRuns(allRuns);
    const runs = allRuns.filter((r) => r.alive);
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
      if (p.kind === "swr_replay") {
        const phasePrefix = p.phase_num ? `Phase ${p.phase_num} SWR · ` : "SWR · ";
        progressLine = `${phasePrefix}event ${p.ev}/${p.ev_total}`;
      } else if (p.kind === "embodied_episode") {
        const phasePrefix = p.phase_num ? `Phase ${p.phase_num} · ` : "";
        progressLine = `${phasePrefix}episode ${p.episode}/${p.episodes_total} · ` +
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

function renderOverviewKPIs(container, runs, findings, launches, textIoRes, inflight) {
  // Filter out smokes for headline metrics
  const real = runs.filter((r) => !/smoke/i.test(r.name) && r.sum_finalQ != null);
  const sums = real.map((r) => r.sum_finalQ);
  const best = real.reduce((a, b) =>
    a == null || b.sum_finalQ < a.sum_finalQ ? b : a, null);

  // 2026-05-03: combine foreground launches with detached runs.
  // /api/runs/launch only knows about webapp-launched (POST) runs.
  // /api/inflight tracks detached runs (PowerShell Start-Process).
  const fgLive = (launches || []).filter((l) => l.running);
  const detachedLive = (inflight || []).filter((r) => r.alive);
  const totalLive = fgLive.length + detachedLive.length;
  const inFlight = { length: totalLive };  // shape-compat with old code below
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
setupKeyboardShortcuts(); // 2026-05-03: arrows, space, r, t, ?
initNotifications();      // 2026-05-03: browser-notification readiness
setupTabs();
setupLauncher();
setupWorldTab();
loadRuns();
loadOverview();  // active tab on first load

// Active-runs badge click — jump to Runs tab so user sees the live panel
document.getElementById("active-runs-badge")?.addEventListener("click", () => {
  activateTab("runs");
});

// ─────────────────────────────────────────────────────────────────────────
// Keyboard shortcuts (2026-05-03)
// ─────────────────────────────────────────────────────────────────────────
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Ignore when user is typing in an input/textarea/select.
    const tag = (e.target.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) {
      return;
    }
    // Ignore when modifier keys are held (avoids breaking browser shortcuts)
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const activeTabBtn = document.querySelector("nav button.active");
    const activeTab = activeTabBtn?.dataset.tab;

    switch (e.key) {
      case "?":
        e.preventDefault();
        showKeyboardHelp();
        break;
      case "t":
      case "T":
        // Toggle theme
        e.preventDefault();
        document.getElementById("theme-toggle")?.click();
        break;
      case "r":
      case "R":
        // Refresh current view
        e.preventDefault();
        if (activeTab === "runs") loadRuns();
        else if (activeTab === "findings") loadFindings();
        else if (activeTab === "language") {
          window._languageLoaded = false; loadLanguage();
        } else if (activeTab === "plans") {
          window._plansLoaded = false; loadPlans();
        } else if (activeTab === "overview") {
          window._overviewLoaded = false; loadOverview();
        }
        break;
      case " ":
        // Play/pause in Brain or World tab
        if (activeTab === "brain") {
          e.preventDefault();
          $("#brain3d-play")?.click();
        } else if (activeTab === "world") {
          e.preventDefault();
          const btn = $("#world-play");
          if (btn && !btn.disabled) btn.click();
        }
        break;
      case "ArrowLeft":
      case "ArrowRight": {
        // Scrub backward / forward in Brain or World tab
        const delta = e.key === "ArrowRight" ? 1 : -1;
        const stepSize = e.shiftKey ? 10 : 1;
        if (activeTab === "brain") {
          const sl = $("#brain3d-scrubber");
          if (sl) {
            const cur = parseInt(sl.value, 10) || 0;
            const max = parseInt(sl.max, 10) || 0;
            const next = Math.max(0, Math.min(max, cur + delta * stepSize));
            sl.value = String(next);
            sl.dispatchEvent(new Event("input"));
            e.preventDefault();
          }
        } else if (activeTab === "world") {
          const sl = $("#world-scrubber");
          if (sl) {
            const cur = parseInt(sl.value, 10) || 0;
            const max = parseInt(sl.max, 10) || 0;
            const next = Math.max(0, Math.min(max, cur + delta * stepSize));
            sl.value = String(next);
            sl.dispatchEvent(new Event("input"));
            e.preventDefault();
          }
        }
        break;
      }
      case "1": case "2": case "3": case "4":
      case "5": case "6": case "7": case "8":
      case "9": {
        // Number keys jump to nth tab
        e.preventDefault();
        const idx = parseInt(e.key, 10) - 1;
        const navBtns = $$("nav button");
        if (navBtns[idx]) navBtns[idx].click();
        break;
      }
      case "Escape": {
        // ESC closes any open Brain panel (pinned info / log tail)
        let closed = false;
        const logPane = document.getElementById("brain-log-pane");
        if (logPane && logPane.style.display !== "none") {
          logPane.style.display = "none";
          closed = true;
        }
        const infoPanel = document.querySelector(".brain3d-info-panel");
        if (infoPanel && infoPanel.style.display !== "none") {
          infoPanel.style.display = "none";
          closed = true;
        }
        const trashDrawer = document.getElementById("trash-drawer");
        if (trashDrawer && trashDrawer.style.display !== "none") {
          trashDrawer.style.display = "none";
          closed = true;
        }
        const runDrawer = document.getElementById("brain3d-runs-drawer");
        if (runDrawer && runDrawer.style.display !== "none") {
          runDrawer.style.display = "none";
          closed = true;
        }
        if (closed) e.preventDefault();
        break;
      }
    }
  });
}

function showKeyboardHelp() {
  const msg = [
    "Keyboard shortcuts",
    "",
    "Tab navigation:",
    "  1..9       Jump to nth tab (Home / Lab / Runs / ...)",
    "",
    "Display:",
    "  t          Toggle dark / light theme",
    "  r          Refresh current view",
    "  Esc        Close open panels (info, log, drawer)",
    "  ?          Show this help",
    "",
    "Brain / World viewers:",
    "  Space      Play / pause replay",
    "  ←/→        Scrub backward / forward 1 step",
    "  Shift+←/→  Scrub 10 steps",
    "",
    "Brain 3D scene:",
    "  Drag       Orbit camera",
    "  Right-drag Pan",
    "  Scroll     Zoom",
    "  Click      Pin region info panel",
  ].join("\n");
  alert(msg);
}

// Restore persisted state. URL hash takes precedence over localStorage,
// so a shared link like #tab=brain&run=foo.json overrides the stored
// last-active-tab.
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
  // 1) URL hash wins
  const urlParams = parseUrlHash();
  if (urlParams.tab) {
    applyUrlHash();
    return;
  }
  // 2) Fall back to persisted activeTab
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

// Type-filter chips (All / Navigation / Text I/O) — selection
// persisted in localStorage so reload restores the chosen filter.
$$("#runs-type-chips .filter-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const t = chip.dataset.type;
    if (!t) return;
    window._runsTypeFilter = t;
    saveState({ runsTypeFilter: t });
    $$("#runs-type-chips .filter-chip").forEach((c) =>
      c.classList.toggle("active", c === chip));
    renderRunsList();
  });
});

// Restore the persisted type-filter chip on first paint.
(() => {
  const state = loadState();
  if (state.runsTypeFilter) {
    window._runsTypeFilter = state.runsTypeFilter;
    const chip = document.querySelector(
      `#runs-type-chips .filter-chip[data-type="${state.runsTypeFilter}"]`);
    if (chip) {
      $$("#runs-type-chips .filter-chip").forEach((c) =>
        c.classList.toggle("active", c === chip));
    }
  }
})();

// Auto-refresh the live runs panel + chip counts every 5s while the
// Runs tab is the active tab. Polls /api/inflight + /api/runs.
setInterval(() => {
  const runsTab = document.querySelector("nav button[data-tab='runs'].active");
  if (runsTab) {
    refreshRunsLivePanel();
  }
}, 5000);
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
