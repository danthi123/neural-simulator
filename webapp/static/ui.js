// UX utilities: toast notifications, loading skeletons, persistent state.
// Kept dependency-free so the dashboard remains a single-static-bundle app.

/* ─── Toast notifications ────────────────────────────────────────────
 *
 * Stack of dismissible cards in the bottom-right. Auto-dismiss after
 * `duration` ms (default 4000). Use sparingly per UX best practices —
 * reserved for "something happened" feedback (launch succeeded, run
 * completed, copy-to-clipboard, error), not constant chatter.
 *
 * Public API: toast(message, opts)
 *   message: string
 *   opts: { kind: "info" | "success" | "warn" | "error", duration: ms }
 */

let toastContainer = null;

function ensureToastContainer() {
  if (toastContainer && document.body.contains(toastContainer)) return toastContainer;
  toastContainer = document.createElement("div");
  toastContainer.id = "toast-stack";
  document.body.appendChild(toastContainer);
  return toastContainer;
}

export function toast(message, { kind = "info", duration = 4000 } = {}) {
  const c = ensureToastContainer();
  const t = document.createElement("div");
  t.className = `toast toast-${kind}`;
  t.textContent = message;
  t.addEventListener("click", () => t.remove());
  c.appendChild(t);
  // Slide in
  requestAnimationFrame(() => t.classList.add("visible"));
  if (duration > 0) {
    setTimeout(() => {
      t.classList.remove("visible");
      setTimeout(() => t.remove(), 250);
    }, duration);
  }
  return t;
}

/* ─── Persistent state (localStorage with versioning) ────────────────── */

const STATE_KEY = "neural-sim-dashboard-state-v1";

export function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STATE_KEY) || "{}");
  } catch {
    return {};
  }
}

export function saveState(updates) {
  const cur = loadState();
  Object.assign(cur, updates);
  try {
    localStorage.setItem(STATE_KEY, JSON.stringify(cur));
  } catch {
    // localStorage might be full or disabled; silently ignore.
  }
}

/* ─── Loading skeletons ──────────────────────────────────────────────── */

/**
 * Replace the children of `el` with N skeleton placeholder rows.
 * Cleaner than "Loading…" text per 2026 UX best-practice.
 */
export function showSkeleton(parent, count = 6, kind = "list") {
  parent.replaceChildren();
  for (let i = 0; i < count; i++) {
    const s = document.createElement("div");
    s.className = `skeleton skeleton-${kind}`;
    parent.appendChild(s);
  }
}

/* ─── Keyboard shortcuts ─────────────────────────────────────────────── */

const shortcuts = new Map(); // key combo string → { handler, description }

/**
 * Register a global shortcut. Combo format: "Ctrl+R", "/", "Esc", "Shift+?".
 * Excluded when an INPUT/TEXTAREA/CONTENTEDITABLE has focus (so users can
 * type freely).
 */
export function registerShortcut(combo, handler, description = "") {
  shortcuts.set(combo.toLowerCase(), { handler, description });
}

export function listShortcuts() {
  return Array.from(shortcuts.entries()).map(([combo, info]) => ({
    combo,
    description: info.description,
  }));
}

function comboKey(ev) {
  const parts = [];
  if (ev.ctrlKey || ev.metaKey) parts.push("ctrl");
  if (ev.altKey) parts.push("alt");
  if (ev.shiftKey) parts.push("shift");
  let key = ev.key.toLowerCase();
  if (key === " ") key = "space";
  if (key === "escape") key = "esc";
  if (key === "?") key = "?";
  parts.push(key);
  return parts.join("+");
}

window.addEventListener("keydown", (ev) => {
  const tag = (ev.target && ev.target.tagName) || "";
  const editable = ev.target && ev.target.isContentEditable;
  if (
    (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || editable) &&
    ev.key !== "Escape"
  ) {
    return;
  }
  const key = comboKey(ev);
  const sc = shortcuts.get(key);
  if (sc) {
    ev.preventDefault();
    sc.handler(ev);
  }
});

/* ─── Format helpers ─────────────────────────────────────────────────── */

export function fmtRelTime(unix) {
  if (!unix) return "—";
  const diff = Date.now() / 1000 - unix;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export function fmtBytes(b) {
  if (b == null) return "—";
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / 1024 / 1024).toFixed(1)}MB`;
}

/**
 * Auto-detect the experiment name from a run filename. Patterns observed:
 *   g11_seed42_v3lateral.json    → experiment "v3lateral"
 *   g11_seed42.json              → experiment "default"
 *   g11_seed100.json             → experiment "default"
 *   g11_seed42_sensedonly.json   → experiment "sensedonly" (= flagship per labeling correction)
 */
export function detectExperiment(name) {
  // Strip extension
  const stem = name.replace(/\.json$/i, "");
  // Legacy g11_seed-prefix style (e.g. g11_seed42_v3lateral)
  let m = stem.match(/^g11_seed\d+(?:_(.+))?$/);
  if (m) return m[1] || "default";
  // 2026-05-01: modern *_seed42-suffix naming (e.g. clusterG_Gfv2nmda_seed100,
  // k_v2_stress_16x16_seed42, text_eval_R6_pfc_bypass_seed42).
  m = stem.match(/^(.+?)_seed\d+(?:_[a-f0-9]{6})?$/);
  if (m) return m[1];
  // Smoke / test files (no seed) — keep their full name as the experiment
  if (stem.endsWith("_smoke") || stem.endsWith("_test")) return stem;
  // Modern files without seed suffix (e.g. text_eval_R6_pfc_bypass)
  // Recognized as their own experiment if they have a recognizable prefix.
  if (/^(text_eval|text_train|traj_train|k_v2|cluster|stress|no_heuristic)/.test(stem)) {
    return stem;
  }
  return "(other)";
}

/**
 * Categorize an experiment into a research milestone (best-effort heuristics).
 * Returns { category, color }.
 */
export function categorizeExperiment(experimentName) {
  const e = experimentName.toLowerCase();
  if (e.includes("baseline") || e === "default") return { category: "baseline", color: "#9aa3ad" };
  // 2026-05-01 NEW CATEGORIES (priority order: most-specific first)
  // Cluster K v2 visual cortex (Hubel-Wiesel + Felleman-Van Essen)
  if (e.includes("k_v2") || e.includes("kv2") || e.includes("visual_cortex") ||
      e.includes("visual-cortex") || e.includes("retina"))
    return { category: "Cluster K visual cortex", color: "#34d399" };
  // Text I/O / language regions (Wernicke/Broca-like)
  if (e.includes("text_io") || e.includes("text_eval") || e.includes("text_train") ||
      e.includes("text-io") || e.includes("language") ||
      e.includes("embodied") || e.includes("contrastive") ||
      e.includes("pfc_bypass") || e.includes("pfc-bypass") ||
      e.includes("nonzero_init") || e.includes("delta"))
    return { category: "Text I/O training", color: "#a78bfa" };
  // Cluster G NMDA flagship (Wang 2002 PFC bistability)
  if (e.includes("nmda") || e.includes("g_v25") || e.includes("gv25") ||
      e.includes("clusterg") || e.includes("cluster_g"))
    return { category: "Cluster G NMDA", color: "#10b981" };
  // Grid scaling stress tests
  if (e.includes("16x16") || e.includes("24x24") || e.includes("32x32") ||
      e.includes("stress_16") || e.includes("stress_24") ||
      e.includes("grid_") || e.includes("scaling"))
    return { category: "Grid scaling", color: "#22d3ee" };
  // No-heuristic / Tier 0 honest tests (no shortcuts allowed)
  if (e.includes("no_heuristic") || e.includes("noheuristic") || e.includes("tier0"))
    return { category: "Tier 0 honest test", color: "#f59e0b" };
  // Trajectory / imitation training
  if (e.includes("traj_train") || e.includes("trajectory") ||
      e.includes("imitation"))
    return { category: "Trajectory training", color: "#a78bfa" };
  // Cluster F cerebellum (Marr-Albus-Ito)
  if (e.includes("clusterf") || e.includes("cluster_f") || e.includes("cerebell"))
    return { category: "Cluster F cerebellum", color: "#fb7185" };
  // Cluster D hippocampus (trisynaptic + SWR)
  if (e.includes("clusterd") || e.includes("cluster_d") || e.includes("hippo_swr") ||
      e.includes("d_v2_swr"))
    return { category: "Cluster D hippocampus", color: "#f472b6" };
  // Cluster B striatal interneurons (D1/D2/PV-FSI/TANs)
  if (e.includes("clusterb") || e.includes("cluster_b") || e.includes("d1d2") ||
      e.includes("striatal_fs") || e.includes("tans"))
    return { category: "Cluster B striatum", color: "#fda4af" };
  // Cluster A (closed BG loop) / Cluster E (topography) composite
  if (e.includes("clustera") || e.includes("cluster_a") || e.includes("clustere") ||
      e.includes("cluster_e") || e.includes("clusterae") || e.includes("cluster_ae") ||
      e.includes("ae_") || e.startsWith("ae"))
    return { category: "Cluster A/E", color: "#86efac" };
  // PRE-EXISTING CATEGORIES
  if (e.includes("cheat5") || e.includes("v3lateral") || e.includes("v3.1") ||
      e.includes("v4dev"))
    return { category: "cheat #5", color: "#fbbf24" };
  if (e.includes("sensed") || e.includes("allnocheats"))
    return { category: "flagship", color: "#6ee7b7" };
  if (e.includes("perception") || e.includes("beacon") || e.includes("landmark") ||
      e.includes("cuereflex") || e.includes("stage1") || e.includes("stage2") || e.includes("stage3"))
    return { category: "perception arc", color: "#93c5fd" };
  if (e.includes("pfc")) return { category: "PFC working memory", color: "#c4b5fd" };
  if (e.includes("hippo") || e.includes("realcurriculum") || e.includes("partialfreeze"))
    return { category: "Phase C / curriculum", color: "#f472b6" };
  if (e.includes("sleep") || e.includes("replay") || e.includes("nrem"))
    return { category: "sleep replay", color: "#a5b4fc" };
  if (e.includes("ada") || e.includes("perda") || e.includes("wta") || e.includes("lrboost") ||
      e.includes("rpe") || e.includes("surprise"))
    return { category: "Phase B refinement", color: "#fcd34d" };
  // 2026-05-09: post-Phase-1.4-BRANCH-A capability arc. Recent
  // experiment families that were falling through to "other".
  if (e.includes("phase_1_5") || e.includes("phase-1-5") ||
      e.includes("interference") || e.includes("long_tail") ||
      e.includes("sequential_expansion") || e.includes("retention_over_time"))
    return { category: "Phase 1.5 eval suite", color: "#fb923c" };  // orange
  if (e.includes("phase_1_4") || e.includes("phase-1-4") ||
      e.includes("continual_forgetting") || e.includes("forgetting") ||
      e.includes("branch_a") || e.includes("branch-a"))
    return { category: "Phase 1.4 BRANCH A", color: "#34d399" };  // green
  if (e.includes("phase_1_3") || e.includes("phase-1-3") ||
      e.includes("consolidation"))
    return { category: "Phase 1.3 consolidation", color: "#67e8f9" };  // cyan
  if (e.includes("phase_2") || e.includes("phase-2") ||
      e.includes("path_f") || e.includes("path-f") ||
      e.includes("surrogate") || e.includes("bptt") ||
      e.includes("shakespeare"))
    return { category: "Phase 2 (path-f-hybrid)", color: "#a78bfa" };  // purple
  if (e.includes("tier_2_3") || e.includes("tier2.3") ||
      e.includes("phrase"))
    return { category: "Tier 2.3 phrases", color: "#fde68a" };  // pale yellow
  if (e.includes("tier_2_1") || e.includes("tier2.1") ||
      e.includes("synonym") || e.includes("12word") ||
      e.includes("16word"))
    return { category: "Tier 2.1 / capacity", color: "#fbbf24" };  // amber
  if (e.includes("chat_demo") || e.includes("chat_synonym_demo") ||
      e.includes("chat_continual_demo") || e.includes("chat_learn_demo") ||
      e.includes("chat_speak_demo"))
    return { category: "Track 3 chat demos", color: "#f472b6" };  // pink
  // 2026-05-13: concept pool architecture (diversity beyond 4 motors).
  // noun_pool_APPLE, verb_pool_GO etc — addresses conversational ceiling.
  if (e.includes("concept_pool") || e.includes("concept_compose") ||
      e.includes("noun_pool") || e.includes("verb_pool"))
    return { category: "Concept pool architecture", color: "#60a5fa" };  // sky blue
  if (e.includes("smoke") || e.includes("test")) return { category: "smoke", color: "#5f6770" };
  return { category: "other", color: "#9aa3ad" };
}

/* ─── Stats helpers ──────────────────────────────────────────────────── */

export function mean(arr) {
  if (!arr || !arr.length) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function stdev(arr) {
  if (!arr || arr.length < 2) return null;
  const m = mean(arr);
  const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
}
