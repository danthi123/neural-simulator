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
  // Match the seed segment
  const m = stem.match(/^g11_seed\d+(?:_(.+))?$/);
  if (!m) return "(other)";
  return m[1] || "default";
}

/**
 * Categorize an experiment into a research milestone (best-effort heuristics).
 * Returns { category, color }.
 */
export function categorizeExperiment(experimentName) {
  const e = experimentName.toLowerCase();
  if (e.includes("baseline") || e === "default") return { category: "baseline", color: "#9aa3ad" };
  if (e.includes("cheat5") || e.includes("v3lateral") || e.includes("v3.1"))
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
