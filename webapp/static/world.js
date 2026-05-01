// Phase 2: 2D top-down playback of a recorded run.
// Phase 2.5: live mode — attach to an in-flight runner via WebSocket and
// animate the agent in real time.
//
// Renders the gridworld using Canvas 2D — an 8×8 (or grid_size) grid with:
// - Agent (animated along trajectory)
// - Goal beacon (with intensity-falloff halo)
// - Landmark (if recorded as fixed at grid center, default for runs with --landmarks)
// - Trajectory trail (fading)
// - Live distance chart (Phase 2.5 enhancement) — recent_dist over time
//
// Canvas 2D is deliberate over Three.js for the flat 2D gridworld — overkill
// for a top-down 8×8. Phase 3 swaps to Three.js when the world is 3D
// (PyBullet integration).

import { makeLineChart, PALETTE_EXPORT as P } from "/static/charts.js";

const $ = (sel) => document.querySelector(sel);

// Configuration
const CELL_PX = 56;   // pixels per grid cell
const PADDING = 24;   // canvas padding
const TRAIL_LEN = 30; // trajectory trail samples

// Canvas 2D requires string colors, so we mirror the CSS vars from :root.
// Single source of truth lives in style.css; if a var changes there, update
// the matching entry here. Canvas colors are kept identical to vars to keep
// the Canvas-rendered map visually consistent with the rest of the dashboard.
const C = {
  codeBg: "#0a0c10",      // matches --code-bg
  border: "#2a2f3d",      // matches --border
  fgMuted: "#5f6770",     // matches --fg-muted
  accent: "#6ee7b7",      // matches --accent
  accentWarn: "#fbbf24",  // matches --accent-warn
};

// State
let world = {
  data: null,
  step: 0,
  playing: false,
  speed: 5,
  rafId: null,
  lastFrameTime: 0,
  // Live-mode state (Phase 2.5)
  live: false,
  liveRunId: null,
  liveSocket: null,
  livePoints: [],   // [{step, total, pos, goal, recent_dist}]
  liveChart: null,  // makeLineChart instance for live recent_dist
  // Whether the currently-displayed run was launched with --landmarks. The
  // landmark icon (blue ring at grid center) and the "Landmark" legend row
  // are clutter for runs without this flag. Default true so the no-run-loaded
  // empty state shows the full legend (preview of all icon meanings).
  usedLandmarks: true,
  // Frozen-elapsed cache (client-side): {run_id: elapsed_sec_at_completion}.
  // Belt-and-suspenders for the elapsed-tick-after-done bug. The server
  // tries to set finished_at on completion, but if the drain_log task
  // crashed / never ran, the API will return a ticking elapsed_sec.
  // We cache the FIRST elapsed_sec we see for a run that's no-longer-running
  // and display that value forever after. Cleared when a new run with the
  // same run_id is observed (running=true again).
  _elapsedFrozen: new Map(),
};

export function setupWorldTab() {
  // Lazy-init when tab first activated
  document.querySelectorAll("nav button").forEach((btn) => {
    if (btn.dataset.tab === "world") {
      btn.addEventListener("click", () => {
        if (!world._initialized) initWorld();
        // If the user opened Live mode earlier in this session and then
        // navigated away, the auto-refresh interval self-stopped. Restart
        // it now so the step counts under each run_id resume ticking.
        if (world._liveModeOpened && !_pickerRefreshInterval) {
          openLiveModePicker();
        }
      });
    }
  });
}

/** Load a run by name into the World tab. Used by the Runs tab's "Play in
 *  World viz" button so users can jump straight from a run detail to its
 *  animated playback. */
export async function loadRunIntoWorld(name) {
  if (!world._initialized) initWorld();
  // Wait one frame for initWorld to finish populating the run list, then
  // call loadRun without a list-item (no highlight).
  await new Promise((r) => setTimeout(r, 50));
  await loadRun(name, null);
}

function initWorld() {
  world._initialized = true;
  const canvas = $("#world-canvas");
  const grid = 8; // default; updated when run is loaded
  resizeCanvas(canvas, grid);
  drawEmptyGrid(canvas, grid);

  $("#world-load-run").addEventListener("click", () => {
    document.querySelector('nav button[data-tab="runs"]').click();
  });
  $("#world-live-mode").addEventListener("click", toggleLiveMode);
  $("#world-play").addEventListener("click", play);
  $("#world-pause").addEventListener("click", pause);
  $("#world-speed").addEventListener("change", (e) => {
    world.speed = parseFloat(e.target.value);
  });
  $("#world-scrubber").addEventListener("input", (e) => {
    if (!world.data) return;
    world.step = parseInt(e.target.value, 10);
    // In live mode, manual scrubbing detaches the scrubber from following
    // the latest live step. Click Latest to re-attach.
    if (world.live) world.scrubberFollowsLatest = false;
    renderFrame();
    updateScrubberStepLabel();
  });

  setupScrubberStepLabel();
  setupHudCollapse("legend-toggle", "world-legend", "hud-legend-collapsed");
  // RUN HUD: collapsed by default (lots of fields, takes up canvas real
  // estate; user can expand once they want detailed run stats).
  setupHudCollapse("runhud-toggle", "world-runhud", "hud-runhud-collapsed", true);

  $("#scrubber-latest-btn")?.addEventListener("click", () => {
    if (!world.data) return;
    if (world.live) {
      // In live mode, "latest" = highest step we've seen so far. Re-enable
      // auto-following so future progress events keep us at the latest.
      world.scrubberFollowsLatest = true;
      world.step = (world.livePoints.length
        ? world.livePoints[world.livePoints.length - 1].step
        : 0);
    } else {
      world.step = (world.data.trajectory || []).length - 1;
    }
    $("#world-scrubber").value = String(world.step);
    renderFrame();
    updateScrubberStepLabel();
  });

  loadWorldRunList();
}

/** Wire the HUD collapse toggle button. Stores the collapsed state in
 *  localStorage under the given key so it persists across reloads.
 *  defaultCollapsed (default false) controls what's shown when the user
 *  has no prior preference in localStorage. */
function setupHudCollapse(buttonId, panelId, storageKey, defaultCollapsed = false) {
  const btn = document.getElementById(buttonId);
  const panel = document.getElementById(panelId);
  if (!btn || !panel) return;
  const apply = (collapsed) => {
    panel.classList.toggle("hud-collapsed", collapsed);
    btn.textContent = collapsed ? "+" : "−";
    btn.setAttribute("title", collapsed ? "Expand" : "Collapse");
    btn.setAttribute("aria-label", collapsed ? "Expand" : "Collapse");
  };
  const stored = localStorage.getItem(storageKey);
  const initial = stored == null ? defaultCollapsed : stored === "1";
  apply(initial);
  btn.addEventListener("click", () => {
    const next = !panel.classList.contains("hud-collapsed");
    apply(next);
    localStorage.setItem(storageKey, next ? "1" : "0");
  });
}

/** Set up the click-to-edit step indicator below the scrubber. */
function setupScrubberStepLabel() {
  const label = document.getElementById("scrubber-step-label");
  if (!label || label._bound) return;
  label._bound = true;
  label.addEventListener("click", () => {
    if (!world.data) return;
    const cur = world.step;
    const max = scrubberMax();
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = String(max);
    input.step = "1";
    input.value = String(cur);
    input.className = "scrubber-step-input";
    label.replaceWith(input);
    input.focus();
    input.select();
    const commit = () => {
      const v = parseInt(input.value, 10);
      if (!isNaN(v)) {
        const clamped = Math.max(0, Math.min(max, v));
        world.step = clamped;
        $("#world-scrubber").value = String(clamped);
        if (world.live) world.scrubberFollowsLatest = false;
        renderFrame();
      }
      input.replaceWith(label);
      updateScrubberStepLabel();
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { commit(); ev.preventDefault(); }
      else if (ev.key === "Escape") {
        input.replaceWith(label);
        updateScrubberStepLabel();
      }
    });
  });
}

/** Maximum scrubber index for the currently-loaded data. */
function scrubberMax() {
  if (!world.data) return 0;
  if (world.live) {
    return world.livePoints.length
      ? world.livePoints[world.livePoints.length - 1].step
      : 0;
  }
  return (world.data.trajectory || []).length - 1;
}

function updateScrubberStepLabel() {
  const cur = document.getElementById("scrubber-step-current");
  const tot = document.getElementById("scrubber-step-total");
  if (cur) cur.textContent = String(world.step);
  if (tot) tot.textContent = String(scrubberMax());
}

let _pickerRefreshInterval = null;

/** Open a picker modal showing in-flight runs the server is tracking, and
 *  when one is chosen, attach the World tab as a live viewer for that run.
 *
 *  Auto-refreshes every 1 second while the picker is visible AND the
 *  World tab is active AND we're not currently attached to a run. Lets
 *  the step counts under each run_id update in real time without forcing
 *  the user to re-click "Live mode". */
/** Toggle between Live mode (in-flight runs picker, auto-refreshing) and
 *  the default past-runs list. Clicking the Live mode button while
 *  already in Live mode returns to past runs (also detaches if currently
 *  attached to a live run). */
function toggleLiveMode() {
  const liveActive = !!_pickerRefreshInterval || world.live;
  if (liveActive) {
    // Exit Live mode: detach if attached, stop picker refresh, restore
    // past-runs list.
    if (world.live) closeLiveSocket();
    if (_pickerRefreshInterval) {
      clearInterval(_pickerRefreshInterval);
      _pickerRefreshInterval = null;
    }
    world._liveModeOpened = false;
    document.getElementById("world-live-mode")?.classList.remove("active");
    loadWorldRunList();
  } else {
    document.getElementById("world-live-mode")?.classList.add("active");
    openLiveModePicker();
  }
}

async function openLiveModePicker() {
  world._liveModeOpened = true;
  document.getElementById("world-live-mode")?.classList.add("active");
  await refreshLivePicker(/* showHeading= */ true);
  // Set up the auto-refresh ticker (idempotent — clears any existing).
  if (_pickerRefreshInterval) clearInterval(_pickerRefreshInterval);
  _pickerRefreshInterval = setInterval(() => {
    const tabActive = document.querySelector("#tab-world")?.classList.contains("active");
    if (!tabActive) {
      // Only stop refreshing when leaving the World tab. When attached
      // to a run, KEEP refreshing so the picker shows live step counts
      // for all running runs (lets the user track the others without
      // detaching).
      clearInterval(_pickerRefreshInterval);
      _pickerRefreshInterval = null;
      return;
    }
    refreshLivePicker(/* showHeading= */ false).catch(() => {});
  }, 1000);
}

async function refreshLivePicker(showHeading) {
  const list = $("#world-runs-list");
  // Track existing item nodes by run_id so we can update them in place
  // (avoids the "buttons flash" on every 1s refresh tick that came from
  // replaceChildren + rebuild). New items are appended; obsolete items
  // are removed.
  if (!list._itemsByRunId) list._itemsByRunId = new Map();
  if (showHeading) {
    // Fresh open: wipe DOM AND tracked items together so we rebuild from scratch.
    list.replaceChildren();
    list._itemsByRunId.clear();
  }

  try {
    const res = await fetch("/api/runs/launch");
    const data = await res.json();
    if (!data.runs.length) {
      // Replace whole content on empty state
      list.replaceChildren();
      list._itemsByRunId.clear();
      const heading = document.createElement("div");
      heading.className = "muted";
      heading.textContent = "No runs in flight. Launch one from the Launch tab.";
      list.appendChild(heading);
      return;
    }
    const seenIds = new Set();
    for (const r of data.runs) {
      seenIds.add(r.run_id);
      let item = list._itemsByRunId.get(r.run_id);
      const isNewItem = !item;

      // Compute the display strings (always, since they may have changed)
      const status = r.running ? "RUNNING" : `done (rc=${r.returncode})`;
      const progress = r.latest_progress
        ? ` step ${r.latest_progress.step}/${r.latest_progress.total}`
        : "";
      // Client-side elapsed-freeze: when a run transitions from running -> done,
      // capture the FIRST elapsed_sec we observe (or use the value we last saw
      // while running) and display that forever after. Workaround for server-side
      // bug where finished_at is occasionally not set, causing the API to return
      // a ticking elapsed_sec.
      let displayElapsed = r.elapsed_sec;
      if (r.running) {
        world._elapsedFrozen.set(r.run_id, r.elapsed_sec);
      } else {
        if (world._elapsedFrozen.has(r.run_id)) {
          displayElapsed = world._elapsedFrozen.get(r.run_id);
        } else {
          world._elapsedFrozen.set(r.run_id, r.elapsed_sec);
        }
      }
      const smallText = `${status}${progress} · elapsed ${Math.round(displayElapsed)}s`;

      if (isNewItem) {
        // Build all child elements ONCE for new items
        item = document.createElement("div");
        item.className = "world-run-item";
        item._runId = r.run_id;
        list._itemsByRunId.set(r.run_id, item);

        const name = document.createElement("div");
        name.className = "run-name";
        // Badge logic:
        //   ★ interactive → full per-trial control (goal/reward override + pause)
        //   ↻ batch       → raw-spawned runner that only emits a sidecar; no
        //                   live telemetry, no attach. Pause/kill still work.
        //   (none)        → standard webapp-launched batch run; attachable but
        //                   no control file.
        let badge = "";
        if (r.interactive) badge = " ★ interactive";
        else if (r.pause_only) badge = " ↻ batch (pause only)";
        name.textContent = r.run_id + badge;
        item.appendChild(name);

        const small = document.createElement("div");
        small.className = "small";
        small.textContent = smallText;
        item.appendChild(small);
        item._small = small;

        // Only wire click-to-attach for runs that actually stream telemetry.
        // Non-attachable runs (raw-spawned replicated batch) get a hover hint
        // explaining why; clicking does nothing.
        if (r.attachable !== false) {
          item.addEventListener("click", () => attachLive(r.run_id, item));
          item.style.cursor = "pointer";
        } else {
          item.style.cursor = "default";
          item.title = "Raw-spawned runner — no live telemetry. Pause and Kill still work.";
        }
      } else {
        // Existing item: only update text in place
        if (item._small.textContent !== smallText) {
          item._small.textContent = smallText;
        }
      }

      // Pause + Kill buttons: present iff running. Create on first running tick;
      // remove if the run transitions to done. Wrapped in a flex row so
      // Pause sits on the left and Kill on the right, in the same row.
      const wasRunning = !!item._pauseBtn;
      if (r.running && !wasRunning) {
        const btnRow = document.createElement("div");
        btnRow.className = "run-item-buttons";
        // Pause/resume toggle. Sets paused=true|false in the run's
        // interactive control file; runner polls and sleeps at env-step
        // boundaries while paused. Useful for freeing GPU for other work
        // without killing the run.
        const pauseBtn = document.createElement("button");
        pauseBtn.className = "pause-btn";
        const setLabel = (paused) => {
          pauseBtn.textContent = paused ? "▶ Resume" : "⏸ Pause";
          pauseBtn.title = paused
            ? "Click to resume this run"
            : "Click to pause this run (frees GPU; resume later without losing progress)";
        };
        const cached = world._pauseStateCache && world._pauseStateCache.get(r.run_id);
        // Synchronous initial label from cache (or default to Pause) so the
        // button never renders empty.
        setLabel(cached === true);
        // Refresh from server only if we don't have a recent cached state.
        if (cached == null) {
          fetch(`/api/runs/launch/${r.run_id}/control`)
            .then((res) => (res.ok ? res.json() : null))
            .then((state) => {
              const paused = !!(state && state.paused);
              if (!world._pauseStateCache) world._pauseStateCache = new Map();
              world._pauseStateCache.set(r.run_id, paused);
              setLabel(paused);
            })
            .catch(() => {});
        }
        pauseBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          const current = !!(world._pauseStateCache && world._pauseStateCache.get(r.run_id));
          const next = !current;
          // Optimistic update: flip label immediately
          if (!world._pauseStateCache) world._pauseStateCache = new Map();
          world._pauseStateCache.set(r.run_id, next);
          setLabel(next);
          try {
            await fetch(`/api/runs/launch/${r.run_id}/control`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ paused: next }),
            });
          } catch (e) {
            // Roll back on failure
            world._pauseStateCache.set(r.run_id, current);
            setLabel(current);
          }
        });
        btnRow.appendChild(pauseBtn);
        item._pauseBtn = pauseBtn;

        const killBtn = document.createElement("button");
        killBtn.className = "kill-btn";
        killBtn.textContent = "✕ Kill";
        killBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          if (window.killLaunchedRun) await window.killLaunchedRun(r.run_id);
          setTimeout(() => openLiveModePicker(), 800);
        });
        btnRow.appendChild(killBtn);
        item._killBtn = killBtn;
        item.appendChild(btnRow);
        item._btnRow = btnRow;
      } else if (!r.running && wasRunning) {
        // Transitioned to done: remove the whole button row in one shot
        if (item._btnRow) {
          item._btnRow.remove();
          item._btnRow = null;
        }
        item._pauseBtn = null;
        item._killBtn = null;
      }

      // Append new items at the end. Existing items already in the DOM stay
      // in place (no re-append needed).
      if (isNewItem) {
        list.appendChild(item);
      }
    }

    // Remove items for runs that have disappeared from the API response
    for (const [runId, item] of list._itemsByRunId.entries()) {
      if (!seenIds.has(runId)) {
        item.remove();
        list._itemsByRunId.delete(runId);
        world._elapsedFrozen.delete(runId);
        if (world._pauseStateCache) world._pauseStateCache.delete(runId);
      }
    }
  } catch (e) {
    list.textContent = `Error: ${e.message}`;
  }
}

async function attachLive(runId, listItem) {
  pause();
  closeLiveSocket();
  document.querySelectorAll("#world-runs-list .world-run-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  world.live = true;
  world.liveRunId = runId;
  world.livePoints = [];
  world.liveStartedAt = performance.now();
  // Live mode uses runId for the HUD's "name" field; clear any saved-run
  // name that lingered from a prior loadRun().
  world._loadedRunName = null;
  // Use a synthetic data shape compatible with renderFrame(). grid_size
  // is updated below from the run's cmd args (--grid-size N) once we
  // fetch them; default 8 matches g11_bg_runner.
  world.data = {
    grid_size: 8,
    trajectory: [],
    goal_log: [],
    action_log: [],
    reward_log: [],
    phase_stats: [],
  };
  world.step = 0;

  // Show live-mode toolbar controls (LIVE badge + ETA + Detach). The
  // scrubber row's Play/Pause/Speed/Latest are visible in BOTH modes
  // since they're useful for reviewing earlier moments of a live run.
  const liveControls = document.getElementById("live-controls");
  const progressBar = document.getElementById("world-progress-bar");
  if (liveControls) liveControls.style.display = "inline-flex";
  if (progressBar) progressBar.style.display = "block";
  world.scrubberFollowsLatest = true;
  const detachBtn = document.getElementById("world-detach");
  if (detachBtn && !detachBtn._bound) {
    detachBtn.addEventListener("click", () => {
      closeLiveSocket();
      $("#world-run-name").textContent = "Detached — no run loaded";
    });
    detachBtn._bound = true;
  }

  $("#world-run-name").textContent = `LIVE: ${runId}`;
  const canvas = $("#world-canvas");
  resizeCanvas(canvas, 8);
  renderFrame();
  // Show + initialize the live distance chart
  const row = $("#world-livechart-row");
  if (row) {
    row.style.display = "block";
    world.liveChart = makeLineChart($("#world-livechart"), {
      title: `Agent distance from goal — rolling 100-step mean (yellow dots = goal moved)`,
      yLabel: "distance",
      yMin: 0,
      yMax: 14,  // max Manhattan on 8x8 grid is 14
    });
    world.liveChart.updateData([{ values: [], color: P.accent, label: "recent_dist" }]);
  }
  // Detect whether this run is interactive (was launched with an
  // interactive_* preset). If so, show the control panel and wire the
  // canvas click handler.
  try {
    const ctrlRes = await fetch(`/api/runs/launch/${runId}/control`);
    if (ctrlRes.ok) {
      const ctrl = await ctrlRes.json();
      world.interactive = !!ctrl.interactive;
    }
  } catch {
    world.interactive = false;
  }
  // Detect whether this live run was launched with the landmark sensor
  // (canonical flag --enable-landmark-sensor; legacy --landmarks aliased)
  // so the landmark icon + legend row can be hidden when the cue isn't
  // actually in play. ALSO stash the cmd argv for the RUN HUD's flags +
  // seed fields. /api/runs/launch/{id} returns the full cmd list.
  //
  // Reset to false BEFORE the async fetch so attaching to a new run
  // that doesn't use landmarks immediately hides the icon, even if the
  // fetch is slow or fails. Only flip to true on confirmed landmark args.
  world._liveCmd = null;
  world.usedLandmarks = false;
  updateLegendVisibility();
  try {
    const statusRes = await fetch(`/api/runs/launch/${runId}`);
    if (statusRes.ok) {
      const status = await statusRes.json();
      const cmd = Array.isArray(status.cmd) ? status.cmd : [];
      world._liveCmd = cmd;
      // Match the canonical landmark flags exactly (token equality), not
      // substring containment. Substring containment was triggering on
      // unrelated args (e.g. anything containing the literal substring
      // "landmark") and was overly permissive for other contexts.
      world.usedLandmarks = cmd.some(
        (t) =>
          t === "--enable-landmark-sensor" ||
          t === "--landmarks" ||
          t === "--landmarks-replace-place",
      );
      updateLegendVisibility();

      // Parse --grid-size N from cmd so the canvas resizes correctly for
      // non-default grid sizes (e.g. 16x16 stress tests). Default 8 matches
      // g11_bg_runner. Apply BEFORE the WebSocket opens so the first frame
      // renders at the correct size.
      let gridFromCmd = 8;
      const gIdx = cmd.findIndex((t) => t === "--grid-size");
      if (gIdx >= 0 && gIdx + 1 < cmd.length) {
        const parsed = parseInt(cmd[gIdx + 1], 10);
        if (parsed > 0 && parsed <= 64) gridFromCmd = parsed;
      }
      if (world.data) world.data.grid_size = gridFromCmd;
      const canvas2 = $("#world-canvas");
      if (canvas2) resizeCanvas(canvas2, gridFromCmd);
    }
  } catch {
    // On fetch error, leave usedLandmarks=false (already set above). If a
    // run that needed landmarks gets the icon hidden, that's better than
    // showing it spuriously on every run.
  }
  setupControlPanel();
  openLiveSocket(runId);
}

/** Show or hide the interactive control panel based on world.interactive,
 *  wire button + canvas click handlers. */
function setupControlPanel() {
  const row = document.getElementById("world-control-row");
  const canvas = $("#world-canvas");
  if (!row || !canvas) return;
  if (!world.interactive) {
    row.style.display = "none";
    canvas.classList.remove("interactive");
    return;
  }
  row.style.display = "flex";
  canvas.classList.add("interactive");

  // Click in grid → teleport goal
  if (!canvas._interactiveBound) {
    canvas.addEventListener("click", onGridClick);
    canvas._interactiveBound = true;
  }
  // Buttons (use event delegation; handlers are idempotent across re-attach)
  const pauseBtn = document.getElementById("ctrl-pause-toggle");
  if (pauseBtn && !pauseBtn._bound) {
    pauseBtn.addEventListener("click", togglePause);
    pauseBtn._bound = true;
  }
  const clearBtn = document.getElementById("ctrl-clear-goal-override");
  if (clearBtn && !clearBtn._bound) {
    clearBtn.addEventListener("click", () => sendControl({ goal: null }));
    clearBtn._bound = true;
  }
  document.querySelectorAll("[data-reward]").forEach((btn) => {
    if (btn._bound) return;
    btn.addEventListener("click", () => {
      const v = parseFloat(btn.dataset.reward);
      sendControl({ inject_reward: v });
    });
    btn._bound = true;
  });
}

function onGridClick(ev) {
  if (!world.live || !world.interactive || !world.liveRunId) return;
  const canvas = $("#world-canvas");
  const rect = canvas.getBoundingClientRect();
  // Same coordinate math as cellToPx but inverted: client px → grid cell.
  const xPx = ev.clientX - rect.left - PADDING;
  const yPx = ev.clientY - rect.top - PADDING;
  const gridSize = worldGridSize();
  const gx = Math.max(0, Math.min(gridSize - 1, Math.floor(xPx / CELL_PX)));
  const gyFromTop = Math.floor(yPx / CELL_PX);
  const gy = Math.max(0, Math.min(gridSize - 1, gridSize - 1 - gyFromTop));
  sendControl({ goal: [gx, gy] });
  flashCell(canvas, gx, gy);
}

let pauseState = false;
async function togglePause() {
  pauseState = !pauseState;
  await sendControl({ paused: pauseState });
  const btn = document.getElementById("ctrl-pause-toggle");
  if (btn) {
    btn.textContent = pauseState ? "▶ Resume" : "⏸ Pause";
    btn.classList.toggle("active", pauseState);
  }
}

async function sendControl(update) {
  const status = document.getElementById("ctrl-status");
  if (status) status.textContent = `control: sending ${JSON.stringify(update)}…`;
  try {
    const res = await fetch(`/api/runs/launch/${world.liveRunId}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    if (status) status.textContent = `control: ${JSON.stringify(data.state)}`;
  } catch (e) {
    if (status) status.textContent = `control error: ${e.message}`;
  }
}

function flashCell(canvas, gx, gy) {
  // Brief visual confirmation that the click was registered. Drawn as a
  // ring expanding from the clicked cell.
  const ctx = canvas.getContext("2d");
  const [px, py] = cellToPx(gx, gy);
  const start = performance.now();
  function tick(now) {
    const t = (now - start) / 400;
    if (t > 1) return;
    renderFrame();
    ctx.save();
    ctx.strokeStyle = `rgba(110, 231, 183, ${1 - t})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(px, py, 14 + t * 30, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function openLiveSocket(runId) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/runs/${runId}`);
  world.liveSocket = ws;
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "progress") {
      handleLiveProgress(msg);
    } else if (msg.type === "done") {
      $("#world-run-name").textContent = `LIVE (done, rc=${msg.returncode}): ${runId}`;
    }
  };
  ws.onerror = () => {
    $("#world-run-name").textContent = `LIVE (socket error): ${runId}`;
  };
}

function closeLiveSocket() {
  if (world.liveSocket) {
    try { world.liveSocket.close(); } catch {}
    world.liveSocket = null;
  }
  world.live = false;
  world.liveRunId = null;
  world.livePoints = [];
  world.liveRecentEvents = [];
  world.liveChart = null;
  world.liveStartedAt = null;
  world.interactive = false;
  world._liveCmd = null;
  // Restore the legend to its full default state on detach so the user
  // sees all icon meanings again (matches the no-run-loaded state).
  world.usedLandmarks = true;
  updateLegendVisibility();
  pauseState = false;
  // Defensive: ensure the picker auto-refresh is running so the runs
  // list keeps ticking after detach. Idempotent — openLiveModePicker
  // clears any existing interval before starting a new one.
  if (world._liveModeOpened &&
      document.querySelector("#tab-world")?.classList.contains("active")) {
    openLiveModePicker().catch(() => {});
  }
  const chartRow = document.getElementById("world-livechart-row");
  if (chartRow) chartRow.style.display = "none";
  const ctrlRow = document.getElementById("world-control-row");
  if (ctrlRow) ctrlRow.style.display = "none";
  const canvas = document.getElementById("world-canvas");
  if (canvas) canvas.classList.remove("interactive");
  const pauseBtn = document.getElementById("ctrl-pause-toggle");
  if (pauseBtn) {
    pauseBtn.textContent = "⏸ Pause";
    pauseBtn.classList.remove("active");
  }
  // Hide live-only toolbar elements
  const liveControls = document.getElementById("live-controls");
  const progressBar = document.getElementById("world-progress-bar");
  if (liveControls) liveControls.style.display = "none";
  if (progressBar) {
    progressBar.style.display = "none";
    const fill = progressBar.querySelector(".progress-fill");
    if (fill) fill.style.width = "0%";
  }
}

/** A progress event represents 1 sim step (with --progress-print-interval=1
 *  baked into the launcher) or every Nth step otherwise. State is updated
 *  synchronously per event but RENDER calls are throttled via rAF so a
 *  burst of buffered events on attach collapses into a single jump-to-
 *  latest render — the agent appears at its current position immediately
 *  rather than fast-forwarding through history. */
let _liveRenderScheduled = false;

function scheduleLiveRender() {
  if (_liveRenderScheduled) return;
  _liveRenderScheduled = true;
  requestAnimationFrame(() => {
    _liveRenderScheduled = false;
    renderFrame();
    // Chart redraw also throttled here (chart canvas is a separate canvas
    // but same expense pattern).
    if (world.liveChart && world.livePoints.length > 0) {
      const latest = world.livePoints[world.livePoints.length - 1];
      const distAt = new Array(latest.step + 1).fill(null);
      for (const pt of world.livePoints) {
        if (pt.step <= latest.step) distAt[pt.step] = pt.recent_dist;
      }
      // Forward-fill nulls so the chart draws a continuous line across
      // gaps (e.g. when --progress-print-interval > 1 makes events sparse).
      // Without this, the chart's line-drawing code resets started=false
      // on every null, leaving isolated points with no connecting line.
      {
        let last = null;
        for (let i = 0; i < distAt.length; i++) {
          if (distAt[i] != null) last = distAt[i];
          else if (last != null) distAt[i] = last;
        }
      }
      // Goal-change markers — dots on the line at each step where the goal
      // differs from the previous event's goal. Lets the user visually
      // correlate recent_dist climbs/falls with phase boundaries.
      const goalChanges = [];
      let prevGoal = null;
      for (const pt of world.livePoints) {
        const g = pt.goal;
        if (prevGoal && (g[0] !== prevGoal[0] || g[1] !== prevGoal[1])) {
          goalChanges.push(pt.step);
        }
        prevGoal = g;
      }
      world.liveChart.updateData([
        {
          values: distAt,
          color: P.accent,
          label: "recent_dist",
          pointIndices: goalChanges,
          pointColor: P.warn,
        },
      ]);
    }
  });
}

function handleLiveProgress(p) {
  if (!world.data) return;
  world.livePoints.push(p);
  // Append the current pos to the synthetic trajectory; pad earlier steps
  // with the same pos so the array length matches step count.
  const traj = world.data.trajectory;
  const goalLog = world.data.goal_log;
  const actionLog = world.data.action_log;
  const rewardLog = world.data.reward_log;
  while (traj.length <= p.step) {
    traj.push(p.pos);
    goalLog.push(p.goal);
    // Per-step action + reward populated from progress event when present
    // (runner emits them at progress_print_interval). When --progress-print-
    // interval > 1 we get sparse events; intermediate steps are padded with
    // the most recent values rather than null so the HUD shows something
    // rather than "—" while the next event arrives.
    actionLog.push(typeof p.action === "number" && p.action >= 0 ? p.action : null);
    rewardLog.push(typeof p.reward === "number" ? p.reward : null);
  }
  // Update phase_stats so the overlay shows the right phase number
  if (!world.data.phase_stats.length || world.data.phase_stats.at(-1).goal[0] !== p.goal[0] || world.data.phase_stats.at(-1).goal[1] !== p.goal[1]) {
    world.data.phase_stats.push({
      step_start: p.step,
      goal: [p.goal[0], p.goal[1]],
    });
  }
  // Scrubber max grows with the run. Value follows latest UNLESS the user
  // has manually scrubbed back (scrubberFollowsLatest = false).
  $("#world-scrubber").max = String(p.step);
  if (world.scrubberFollowsLatest) {
    world.step = p.step;
    $("#world-scrubber").value = String(p.step);
  }
  $("#world-step-display").textContent = `step ${world.step} / ${p.total}`;
  updateScrubberStepLabel();

  // Live progress bar
  const fill = document.querySelector("#world-progress-bar .progress-fill");
  if (fill && p.total > 0) {
    fill.style.width = `${(p.step / p.total * 100).toFixed(1)}%`;
  }
  // ETA: extrapolate from a rolling 5-second window using SERVER-side
  // timestamps. Browser-receive timestamps are wrong here because the
  // server replays its full progress buffer on attach (so 900 events
  // arrive in the same 100ms WebSocket burst). Using `p.timestamp`
  // (Python time.time() at parse time, in seconds since epoch) means
  // even the burst events have correct spacing, and the cutoff trims
  // to "last 5 seconds of run wall-clock" rather than "last 5 seconds
  // of browser wall-clock".
  const eta = document.getElementById("live-eta");
  if (eta && p.step > 0 && p.timestamp) {
    world.liveRecentEvents = world.liveRecentEvents || [];
    world.liveRecentEvents.push({ t: p.timestamp, step: p.step });
    const cutoff = p.timestamp - 5;
    // Keep at least 2 events in the buffer — when --progress-print-interval
    // is large (e.g. 20), events can arrive >5s apart and the cutoff would
    // otherwise leave only 1 event, stalling the rate calc at 0.0.
    while (world.liveRecentEvents.length > 2 && world.liveRecentEvents[0].t < cutoff) {
      world.liveRecentEvents.shift();
    }
    let stepsPerSec = 0;
    if (world.liveRecentEvents.length >= 2) {
      const first = world.liveRecentEvents[0];
      const last = world.liveRecentEvents[world.liveRecentEvents.length - 1];
      const dt = last.t - first.t;
      if (dt > 0) stepsPerSec = (last.step - first.step) / dt;
    }
    const remaining = (p.total - p.step) / Math.max(0.01, stepsPerSec);
    const fmt = remaining > 60
      ? `${Math.round(remaining / 60)}m ${Math.round(remaining % 60)}s`
      : `${Math.round(remaining)}s`;
    eta.textContent = `${stepsPerSec.toFixed(1)} steps/s · ETA ${fmt}`;
  }

  // Schedule a render. Coalesces burst-replays into a single redraw so
  // attach is instant (jump-to-latest) instead of speed-replay.
  scheduleLiveRender();
}

async function loadWorldRunList() {
  const list = $("#world-runs-list");
  list.replaceChildren();
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    if (!data.runs.length) {
      list.textContent = "No runs available. Launch one from the Launch tab.";
      return;
    }
    // Sort by name (descending = newest first by date prefix in name).
    data.runs.sort((a, b) => b.name.localeCompare(a.name));
    for (const r of data.runs.slice(0, 100)) {
      const item = document.createElement("div");
      item.className = "world-run-item";
      const name = document.createElement("div");
      name.textContent = r.name;
      const small = document.createElement("div");
      small.className = "small";
      const sumStr = r.sum_finalQ != null ? r.sum_finalQ.toFixed(2) : "—";
      small.textContent = `seed=${r.seed ?? "?"} · sum=${sumStr} · phases=${r.n_phases}`;
      item.appendChild(name);
      item.appendChild(small);
      item.addEventListener("click", () => loadRun(r.name, item));
      list.appendChild(item);
    }
  } catch (e) {
    list.textContent = `Error: ${e.message}`;
  }
}

async function loadRun(name, listItem) {
  pause();
  closeLiveSocket();
  $("#world-run-name").textContent = `Loading ${name}…`;
  document.querySelectorAll("#world-runs-list .world-run-item").forEach((it) =>
    it.classList.toggle("active", it === listItem),
  );
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    world.data = data;
    world.step = 0;
    // Stash the filename so the RUN HUD can show it. The saved JSON itself
    // doesn't include its own name, so we capture it on load. Cleared on
    // attachLive() / closeLiveSocket() to avoid stale display.
    world._loadedRunName = name;
    // Fetch the sidecar to recover the launch cmd. The main /api/runs/{name}
    // payload does NOT include config_flags or cmd, so the only reliable way
    // to populate the RUN HUD's flags + (fallback) seed fields is via the
    // sidecar's `cmd` argv list. Older runs may not have a sidecar — fall
    // back gracefully.
    world._loadedSidecar = null;
    try {
      const scRes = await fetch(`/api/runs/${encodeURIComponent(name)}/sidecar`);
      if (scRes.ok) world._loadedSidecar = await scRes.json();
    } catch {
      // Older runs without a sidecar — leave null; HUD will show "—".
    }
    // Detect whether this saved run was launched with the landmark sensor.
    // Prefer the sidecar's cmd (canonical source); fall back to data.config_flags
    // for any runs whose sidecar load failed but main JSON still has flags
    // (older code path).
    const sidecarCmd = (world._loadedSidecar && Array.isArray(world._loadedSidecar.cmd))
      ? world._loadedSidecar.cmd : null;
    const flagSource = sidecarCmd
      || (Array.isArray(data.config_flags) ? data.config_flags : []);
    world.usedLandmarks = flagSource.some(
      (f) => typeof f === "string" && (f.includes("landmarks") || f.includes("landmark-sensor")),
    );
    updateLegendVisibility();
    const canvas = $("#world-canvas");
    resizeCanvas(canvas, data.grid_size || 8);
    const total = (data.trajectory || []).length;
    $("#world-scrubber").max = String(Math.max(0, total - 1));
    $("#world-scrubber").value = "0";
    $("#world-step-display").textContent = `step 0 / ${total - 1}`;
    $("#world-run-name").textContent = name;
    renderFrame();
    updateScrubberStepLabel();
  } catch (e) {
    $("#world-run-name").textContent = `Failed to load: ${e.message}`;
    world.data = null;
  }
}

function resizeCanvas(canvas, gridSize) {
  const size = gridSize * CELL_PX + PADDING * 2;
  canvas.width = size;
  canvas.height = size;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
}

function drawEmptyGrid(canvas, gridSize) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = C.codeBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid(ctx, gridSize);
}

function drawGrid(ctx, gridSize) {
  ctx.strokeStyle = C.border;
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridSize; i++) {
    const off = PADDING + i * CELL_PX + 0.5;
    ctx.beginPath();
    ctx.moveTo(PADDING, off);
    ctx.lineTo(PADDING + gridSize * CELL_PX, off);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(off, PADDING);
    ctx.lineTo(off, PADDING + gridSize * CELL_PX);
    ctx.stroke();
  }
  // Faint cell coordinates
  ctx.fillStyle = C.fgMuted;
  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "top";
  for (let x = 0; x < gridSize; x++) {
    ctx.fillText(String(x), PADDING + x * CELL_PX + 4, PADDING + 2);
  }
  for (let y = 0; y < gridSize; y++) {
    ctx.fillText(String(y), PADDING + 2, PADDING + y * CELL_PX + 12);
  }
}

function cellToPx(x, y) {
  // Grid (x, y) → canvas pixel center. Y is flipped so y=0 is at the bottom.
  return [
    PADDING + x * CELL_PX + CELL_PX / 2,
    PADDING + (worldGridSize() - 1 - y) * CELL_PX + CELL_PX / 2,
  ];
}

function worldGridSize() {
  return (world.data && world.data.grid_size) || 8;
}

function getCurrentGoal(stepIdx) {
  // goal_log: list of [gx, gy] per step (length = n_steps + 1)
  if (world.data && Array.isArray(world.data.goal_log) && world.data.goal_log[stepIdx]) {
    return world.data.goal_log[stepIdx];
  }
  // Fallback: phase_stats has the goal per phase
  const ps = (world.data && world.data.phase_stats) || [];
  let goal = null;
  for (const p of ps) {
    if (stepIdx >= (p.step_start || 0)) goal = p.goal;
  }
  return goal;
}

function getCurrentPhaseIndex(stepIdx) {
  const ps = (world.data && world.data.phase_stats) || [];
  for (let i = ps.length - 1; i >= 0; i--) {
    if (stepIdx >= (ps[i].step_start || 0)) return i + 1;
  }
  return 1;
}

/** Return [start, end) bounds for phase index p (0-indexed). end is the
 *  next phase's step_start, or n_steps for the final phase, or +Infinity
 *  if the run length is unknown (live, before total is known). */
function getPhaseBounds(p) {
  const ps = (world.data && world.data.phase_stats) || [];
  if (p < 0 || p >= ps.length) return [0, 0];
  const start = ps[p].step_start || 0;
  let end;
  if (ps[p].step_end != null) {
    end = ps[p].step_end;
  } else if (p + 1 < ps.length) {
    end = ps[p + 1].step_start || start;
  } else {
    // Last phase, no explicit end. Saved runs always have step_end; this
    // path is hit only in live mode, where the current phase's end is
    // either the run's total (if known) or unbounded.
    const data = world.data;
    end = (data && data.n_steps != null) ? data.n_steps : Infinity;
  }
  return [start, end];
}

/** Phase index (0-based) that step i belongs to, or -1 if before phase 0. */
function getPhaseIndexForStep(i) {
  const ps = (world.data && world.data.phase_stats) || [];
  for (let p = ps.length - 1; p >= 0; p--) {
    if (i >= (ps[p].step_start || 0)) return p;
  }
  return -1;
}

/** Compute aggregates as of step N from the trajectory walked from 0 to N
 *  inclusive. Replaces v2's habit of reading the whole-run pre-computed
 *  phase_stats / mean_distance_overall / n_steps_at_goal directly — those
 *  always show end-of-run values regardless of scrub position. The new
 *  walker matches the scrubber so saved-mode is "what the agent had seen
 *  by step N" and live-mode populates as the run progresses.
 *
 *  Conventions (small but documented):
 *  - Walks trajectory[0..N] inclusive (N+1 samples). That mildly differs
 *    from the runner's `dist_arr = distance_log[1:]` which skips the
 *    initial step. The end-of-run agreement is within rounding (see
 *    verification in commit message).
 *  - Phase membership uses phase_stats step_start <= i < step_end (or
 *    next phase's step_start, or n_steps for the open final phase in
 *    live mode).
 *  - Distance for step i uses the phase's goal (phase_stats[p].goal),
 *    not goal_log[i] directly. This sidesteps a one-step lag in goal_log
 *    at phase boundaries (goal_log[step_start] still has the OLD goal;
 *    goal_log[step_start+1] has the new goal). Using the phase's goal
 *    gives results that match the runner's per-phase aggregates.
 *  - Falls back to getCurrentGoal(i) when phase_stats is empty (defensive
 *    — saved runs always have phase_stats, but no harm).
 *
 *  Returns:
 *  - mean_dist: average Manhattan distance over [0, N], or null if no
 *    trajectory.
 *  - n_at_goal: count of steps in [0, N] where distance was 0.
 *  - finalQs: per-phase final-quarter-mean values for phases [0..cur]
 *    that have ≥ MIN_FINALQ_SAMPLES samples in their last-25% window
 *    AS OF step N. Phases without enough samples yet are OMITTED from
 *    the array — caller decides how to render.
 *  - finalQs_phase: 1:1 with finalQs, the 0-based phase index for each.
 *    Used by caller to know which phase corresponds to which value.
 *  - sum_finalQ: sum of finalQs values.
 *  - current_finalQ: the running finalQ for the CURRENT phase (the one
 *    containing step N), or null if not enough samples yet.
 *  - current_phase_idx: 0-based phase that step N is in, for caller
 *    correlation with the unfiltered phase_stats list.
 */
const MIN_FINALQ_SAMPLES = 10;

function computeAggregatesAtStep(N) {
  const data = world.data;
  const traj = (data && data.trajectory) || [];
  if (!traj.length) {
    return { mean_dist: null, n_at_goal: 0, finalQs: [], finalQs_phase: [],
             sum_finalQ: 0, current_finalQ: null, current_phase_idx: -1 };
  }
  // Clamp N to actual trajectory bounds. Defensive — scrubberMax should
  // already enforce this, but live mode's trajectory is built up
  // incrementally so a paranoid clamp is cheap insurance.
  const lastIdx = Math.max(0, Math.min(N, traj.length - 1));
  const ps = (data && data.phase_stats) || [];
  const curPhase = getPhaseIndexForStep(lastIdx);

  // Pre-compute distance + phase for every walked step.
  const dists = new Array(lastIdx + 1);
  const phases = new Array(lastIdx + 1);
  let sumDist = 0;
  let nAtGoal = 0;
  for (let i = 0; i <= lastIdx; i++) {
    const pos = traj[i];
    if (!pos) { dists[i] = null; phases[i] = -1; continue; }
    // Resolve goal: prefer the containing phase's goal so phase-boundary
    // off-by-ones don't leak into the per-phase aggregate. Fall back to
    // goal_log via getCurrentGoal if no phase info available.
    const phaseIdx = getPhaseIndexForStep(i);
    let goal = null;
    if (phaseIdx >= 0 && ps[phaseIdx] && ps[phaseIdx].goal) {
      goal = ps[phaseIdx].goal;
    } else {
      goal = getCurrentGoal(i);
    }
    if (!goal) { dists[i] = null; phases[i] = phaseIdx; continue; }
    const d = Math.abs(pos[0] - goal[0]) + Math.abs(pos[1] - goal[1]);
    dists[i] = d;
    phases[i] = phaseIdx;
    sumDist += d;
    if (d === 0) nAtGoal++;
  }

  const meanDist = (lastIdx + 1) > 0 ? sumDist / (lastIdx + 1) : null;

  // Per-phase finalQ. For each phase p in [0..curPhase], take its last 25%
  // window CLAMPED to step N. Skip phases with fewer than MIN_FINALQ_SAMPLES
  // in that window (== running value not yet meaningful).
  const finalQs = [];
  const finalQsPhase = [];
  let currentFinalQ = null;
  for (let p = 0; p <= curPhase; p++) {
    const [pStart, pEnd] = getPhaseBounds(p);
    // Effective end = phase_end OR step N+1, whichever is smaller.
    // (N+1 because we want steps i <= N to be eligible.)
    const effEnd = Math.min(pEnd, lastIdx + 1);
    const phaseLen = effEnd - pStart;
    if (phaseLen <= 0) continue;
    // Last-25% window. Use Math.floor(phaseLen * 3 / 4) to match the
    // runner's `len(p_dist)*3//4` exactly.
    const winStart = pStart + Math.floor(phaseLen * 3 / 4);
    let winSum = 0;
    let winN = 0;
    for (let i = winStart; i < effEnd; i++) {
      if (dists[i] != null) {
        winSum += dists[i];
        winN++;
      }
    }
    const isCurrent = (p === curPhase);
    if (winN < MIN_FINALQ_SAMPLES) {
      // Hide-when-too-few-samples: don't emit a value for this phase.
      // For the current phase that's not yet 75% elapsed, also leave
      // currentFinalQ as null so caller renders "—".
      if (isCurrent) currentFinalQ = null;
      continue;
    }
    const v = winSum / winN;
    finalQs.push(v);
    finalQsPhase.push(p);
    if (isCurrent) currentFinalQ = v;
  }
  const sumFinalQ = finalQs.reduce((a, b) => a + b, 0);

  return {
    mean_dist: meanDist,
    n_at_goal: nAtGoal,
    finalQs,
    finalQs_phase: finalQsPhase,
    sum_finalQ: sumFinalQ,
    current_finalQ: currentFinalQ,
    current_phase_idx: curPhase,
  };
}

function actionName(idx) {
  return ["N", "E", "S", "W"][idx] ?? "?";
}

function getStepAction(stepIdx) {
  if (!world.data) return null;
  if (Array.isArray(world.data.action_log) && stepIdx < world.data.action_log.length) {
    return world.data.action_log[stepIdx];
  }
  return null;
}

function getStepReward(stepIdx) {
  if (!world.data) return null;
  if (Array.isArray(world.data.reward_log) && stepIdx < world.data.reward_log.length) {
    return world.data.reward_log[stepIdx];
  }
  return null;
}

/** Show or hide the "Landmark" row in the legend HUD based on whether the
 *  current run was launched with --landmarks. The row has explicit id
 *  `legend-landmark` so we don't depend on CSS :has() support. */
function updateLegendVisibility() {
  const row = document.getElementById("legend-landmark");
  if (!row) return;
  row.style.display = world.usedLandmarks ? "" : "none";
}

function renderFrame() {
  const canvas = $("#world-canvas");
  const ctx = canvas.getContext("2d");
  const gridSize = worldGridSize();
  const data = world.data;

  ctx.fillStyle = C.codeBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!data) {
    drawGrid(ctx, gridSize);
    ctx.fillStyle = C.fgMuted;
    ctx.font = "13px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Pick a run from the right panel.", canvas.width / 2, canvas.height / 2);
    return;
  }

  const step = world.step;
  const goal = getCurrentGoal(step);

  // Beacon intensity field
  if (goal) {
    drawBeaconField(ctx, gridSize, goal);
  }

  drawGrid(ctx, gridSize);

  // Landmark (assume at grid center if --landmarks was used; we don't have
  // explicit landmark logging in JSON yet, so we just show a faint marker
  // at the center as a default visualization). Skipped entirely for runs
  // that weren't launched with --landmarks — the icon would be misleading
  // clutter (no landmark cue actually exists in the simulation).
  if (world.usedLandmarks) {
    drawLandmark(ctx, gridSize, [Math.floor(gridSize / 2), Math.floor(gridSize / 2)]);
  }

  // Trajectory trail
  drawTrajectoryTrail(ctx, data.trajectory || [], step);

  // Goal marker
  if (goal) drawGoal(ctx, goal);

  // Agent
  const traj = data.trajectory || [];
  const pos = traj[step];
  if (pos) drawAgent(ctx, pos, getStepAction(step));

  // RUN HUD shows run-level identity + per-step state + aggregate stats.
  // Updated here so it stays in sync with the scrubber for saved runs and
  // with live progress events for live runs. (Replaces the old AGENT
  // overlay — its agent/goal/distance/action/reward/phase fields are
  // covered by the RUN HUD's middle section.)
  updateRunHUD();
  // In live mode, the step display + scrubber are managed by
  // handleLiveProgress (which knows the runner's actual n_steps from the
  // progress event's `total` field). Don't overwrite with traj.length-1
  // here — the trajectory is being built up incrementally and would show
  // misleading "step N / N" values until the run completes.
  if (!world.live) {
    $("#world-step-display").textContent = `step ${step} / ${traj.length - 1}`;
    $("#world-scrubber").value = String(step);
  }
}

function drawBeaconField(ctx, gridSize, goal) {
  // Continuous radial gradient — more biological than per-cell shading.
  // Mirrors the runner's beacon model: intensity = 1 / (1 + falloff*d).
  const [gx, gy] = goal;
  const [cx, cy] = cellToPx(gx, gy);
  const maxRadius = Math.hypot(gridSize * CELL_PX, gridSize * CELL_PX);
  const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, maxRadius);
  grad.addColorStop(0, "rgba(110, 231, 183, 0.30)");
  grad.addColorStop(0.15, "rgba(110, 231, 183, 0.16)");
  grad.addColorStop(0.4, "rgba(110, 231, 183, 0.05)");
  grad.addColorStop(1, "rgba(110, 231, 183, 0.0)");
  ctx.fillStyle = grad;
  ctx.fillRect(PADDING, PADDING, gridSize * CELL_PX, gridSize * CELL_PX);
}

function drawLandmark(ctx, gridSize, lm) {
  const [px, py] = cellToPx(lm[0], lm[1]);
  ctx.save();
  ctx.strokeStyle = "rgba(147, 197, 253, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(px, py, 14, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(147, 197, 253, 0.4)";
  ctx.fill();
  ctx.restore();
}

function drawGoal(ctx, goal) {
  const [px, py] = cellToPx(goal[0], goal[1]);
  ctx.save();
  // Pulsing core
  ctx.fillStyle = C.accent;
  ctx.beginPath();
  ctx.arc(px, py, 8, 0, Math.PI * 2);
  ctx.fill();
  // Halo
  ctx.strokeStyle = "rgba(110, 231, 183, 0.7)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 18, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawTrajectoryTrail(ctx, trajectory, step) {
  if (!trajectory || trajectory.length === 0) return;
  const start = Math.max(0, step - TRAIL_LEN);
  ctx.save();
  for (let i = start; i < step; i++) {
    const a = (i - start) / TRAIL_LEN;
    const [px, py] = cellToPx(trajectory[i][0], trajectory[i][1]);
    ctx.fillStyle = `rgba(251, 191, 36, ${0.06 + a * 0.34})`;
    ctx.beginPath();
    ctx.arc(px, py, 4 + a * 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawAgent(ctx, pos, action) {
  const [px, py] = cellToPx(pos[0], pos[1]);
  ctx.save();
  // Body
  ctx.fillStyle = C.accentWarn;
  ctx.strokeStyle = C.codeBg;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 11, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  // Action arrow (heading the agent just went)
  if (action != null) {
    const dx = [0, 1, 0, -1][action] || 0; // N E S W → x delta
    const dy = [1, 0, -1, 0][action] || 0; // y up = N
    const tipX = px + dx * 18;
    const tipY = py - dy * 18; // canvas y is flipped
    ctx.strokeStyle = C.accentWarn;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    // Arrowhead
    const ang = Math.atan2(tipY - py, tipX - px);
    const ah = 5;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ah * Math.cos(ang - Math.PI / 6), tipY - ah * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(tipX - ah * Math.cos(ang + Math.PI / 6), tipY - ah * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fillStyle = C.accentWarn;
    ctx.fill();
  }
  ctx.restore();
}

/** Format an elapsed seconds value as a compact duration string. */
function formatElapsed(sec) {
  if (sec == null || isNaN(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return `${m}m ${r}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

/** Boring boilerplate flags injected by the launcher / runner harness that
 *  add no information about the experiment's research config — filtered out
 *  of the displayed flags string so the HUD focuses on the interesting bits
 *  (cluster flags, perception arc flags, etc). */
const _BORING_FLAGS = new Set([
  "seed",
  "n-steps",
  "out",
  "progress-print-interval",
]);

/** Compact rendering of a flags-bearing array. Accepts either a `cmd`-style
 *  list (full argv, e.g. `["python", "-m", "research.runners.g11_bg_runner",
 *  "--moving-goal", "--seed", "42", ...]`) or a pre-filtered config_flags
 *  list. Drops value tokens (anything not starting with `--`), strips the
 *  leading `--`, filters out boring launcher boilerplate, and returns a
 *  comma-joined short list. Returns null when nothing meaningful is left. */
function compactFlags(flags) {
  if (!Array.isArray(flags) || flags.length === 0) return null;
  // Use a Set to dedupe — when both a webapp preset and `extra_args` pass
  // the same --flag (e.g. --moving-goal), it shows up twice in cmd; dedupe
  // for HUD readability while preserving first-occurrence order.
  const seen = new Set();
  const out = [];
  for (let i = 0; i < flags.length; i++) {
    const t = flags[i];
    if (typeof t !== "string") continue;
    if (!t.startsWith("--")) continue;
    const name = t.replace(/^--/, "");
    if (_BORING_FLAGS.has(name)) continue;
    if (seen.has(name)) continue;
    seen.add(name);
    out.push(name);
  }
  if (out.length === 0) return null;
  return out.join(", ");
}

/** Parse `--seed N` from a cmd argv list. Returns null when not found. */
function parseSeedFromCmd(cmd) {
  if (!Array.isArray(cmd)) return null;
  for (let i = 0; i < cmd.length - 1; i++) {
    if (cmd[i] === "--seed") {
      const v = parseInt(cmd[i + 1], 10);
      if (!isNaN(v)) return v;
    }
  }
  return null;
}

/** Pull the current run's name + seed + n_steps + cmd from saved data /
 *  sidecar when available, or fall back to the live runId / progress / cmd.
 *  The `cmd` field holds the full launch argv list (e.g. `["python", "-m",
 *  "research.runners.g11_bg_runner", "--seed", "42", ...]`) and is the
 *  canonical source for both seed and flags — `data.config_flags` is NOT
 *  present in the /api/runs/{name} payload (it only appears in the runs-list
 *  summary), so we go through the sidecar instead. Returns an object with
 *  possibly-null fields — caller renders "—" when null. */
function getRunIdentity() {
  const data = world.data;
  const out = {
    name: null,
    seed: null,
    n_steps: null,
    cmd: null,
  };
  if (world.live) {
    out.name = world.liveRunId || null;
    // Live runs may have a `total` from the latest progress event.
    const last = world.livePoints && world.livePoints.length
      ? world.livePoints[world.livePoints.length - 1]
      : null;
    if (last && last.total) out.n_steps = last.total;
    // Cmd was fetched + stashed at attachLive time. Seed is parsed from it
    // (the live progress payload doesn't surface seed directly).
    out.cmd = Array.isArray(world._liveCmd) ? world._liveCmd : null;
    if (out.cmd) out.seed = parseSeedFromCmd(out.cmd);
    return out;
  }
  if (!data) return out;
  out.name = world._loadedRunName || null;
  out.n_steps = data.n_steps != null ? data.n_steps : null;
  // Sidecar `cmd` is the canonical source for flags. Saved JSON's `seed`
  // is also reliable when present, but fall back to parsing cmd otherwise.
  const sc = world._loadedSidecar;
  out.cmd = (sc && Array.isArray(sc.cmd)) ? sc.cmd : null;
  if (data.seed != null) {
    out.seed = data.seed;
  } else if (out.cmd) {
    out.seed = parseSeedFromCmd(out.cmd);
  }
  return out;
}

/** Update the bottom-left RUN HUD. Called from renderFrame() (saved-mode
 *  scrubbing) and from handleLiveProgress() / refreshLivePicker() (live).
 *  Each field gracefully shows "—" when the underlying data isn't
 *  available (e.g. live runs have no phase_stats yet). */
function updateRunHUD() {
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val == null || val === "" ? "—" : String(val);
  };

  const data = world.data;
  const ident = getRunIdentity();
  const step = world.step;

  // Identity row
  setText("runhud-name", ident.name);
  setText("runhud-seed", ident.seed);
  if (ident.n_steps != null) {
    setText("runhud-step", `${step} / ${ident.n_steps}`);
  } else {
    setText("runhud-step", `${step}`);
  }

  // Elapsed
  if (world.live) {
    if (world.liveStartedAt != null) {
      const sec = (performance.now() - world.liveStartedAt) / 1000;
      setText("runhud-elapsed", formatElapsed(sec));
    } else {
      setText("runhud-elapsed", "—");
    }
  } else if (data && data.elapsed_seconds != null) {
    setText("runhud-elapsed", formatElapsed(data.elapsed_seconds));
  } else {
    setText("runhud-elapsed", "—");
  }

  // Per-step state
  const traj = (data && data.trajectory) || [];
  const pos = traj[step] || null;
  const goal = getCurrentGoal(step);
  setText("runhud-pos", pos ? `(${pos[0]},${pos[1]})` : null);
  setText("runhud-goal", goal ? `(${goal[0]},${goal[1]})` : null);
  if (pos && goal) {
    setText(
      "runhud-dist",
      Math.abs(pos[0] - goal[0]) + Math.abs(pos[1] - goal[1]),
    );
  } else {
    setText("runhud-dist", null);
  }
  setText("runhud-phase", data ? String(getCurrentPhaseIndex(step)) : null);
  const a = getStepAction(step);
  setText("runhud-action", a != null ? actionName(a) : null);
  const r = getStepReward(step);
  setText("runhud-reward", r != null ? r.toFixed(2) : null);

  // Aggregates — computed AS OF step N (saved or live). v3 walks the
  // trajectory + per-phase goals to recompute these every render, instead
  // of v2's approach of reading whole-run pre-computed phase_stats /
  // mean_distance_overall (which made saved scrubbing show end-of-run
  // values regardless of position, and made live show all "—" because
  // the synthetic world.data has empty phase_stats for the upcoming
  // phases).
  if (data && Array.isArray(data.trajectory) && data.trajectory.length > 0) {
    const agg = computeAggregatesAtStep(step);
    setText(
      "runhud-finalq-cur",
      agg.current_finalQ != null ? agg.current_finalQ.toFixed(2) : null,
    );
    if (agg.finalQs.length > 0) {
      setText(
        "runhud-finalq-all",
        "[" + agg.finalQs.map((v) => v.toFixed(2)).join(", ") + "]",
      );
      setText("runhud-finalq-sum", agg.sum_finalQ.toFixed(2));
    } else {
      setText("runhud-finalq-all", null);
      setText("runhud-finalq-sum", null);
    }
    setText(
      "runhud-mean-dist",
      agg.mean_dist != null ? agg.mean_dist.toFixed(2) : null,
    );
    setText("runhud-at-goal", String(agg.n_at_goal));
  } else {
    setText("runhud-finalq-cur", null);
    setText("runhud-finalq-all", null);
    setText("runhud-finalq-sum", null);
    setText("runhud-mean-dist", null);
    setText("runhud-at-goal", null);
  }

  // Flags — derived from sidecar cmd (saved runs) or live launch cmd
  // (live runs). compactFlags handles both shapes (it skips value tokens
  // and the leading executable / -m / module-path). Filters out boring
  // launcher boilerplate (--seed, --n-steps, --out, --progress-print-interval).
  setText("runhud-flags", compactFlags(ident.cmd));
}

function play() {
  if (!world.data) return;
  world.playing = true;
  $("#world-play").disabled = true;
  $("#world-pause").disabled = false;
  world.lastFrameTime = performance.now();
  // Reset the fractional-step accumulator so a fresh play doesn't carry
  // residue from a previous play session.
  world._stepAccumulator = 0;
  tick();
}

function pause() {
  world.playing = false;
  $("#world-play").disabled = false;
  $("#world-pause").disabled = true;
  if (world.rafId) {
    cancelAnimationFrame(world.rafId);
    world.rafId = null;
  }
}

function tick() {
  if (!world.playing || !world.data) return;
  const now = performance.now();
  const dt = (now - world.lastFrameTime) / 1000;
  world.lastFrameTime = now;
  // Speed = steps per real second. Use a fractional accumulator so low
  // speeds (0.01, 0.25, 0.5) work correctly. The previous implementation
  // floored speed*dt then Math.max(1, ...)'d it, so any speed below ~60
  // still advanced 1 step per frame (= 60 steps/sec at 60fps).
  // Cap dt to prevent huge jumps after pause/tab-switch.
  world._stepAccumulator = (world._stepAccumulator || 0) +
    world.speed * Math.min(dt, 0.5);
  const stepsAdvance = Math.floor(world._stepAccumulator);
  if (stepsAdvance <= 0) {
    // Not enough accumulated yet; render the current frame and try again.
    renderFrame();
    world.rafId = requestAnimationFrame(tick);
    return;
  }
  world._stepAccumulator -= stepsAdvance;
  // In live mode the upper bound is the latest live step we've received,
  // not the saved-trajectory length. Manual playback in live mode walks
  // forward through the recorded history at the chosen speed; if it
  // catches up to the latest, we re-attach the scrubber to follow live.
  const maxStep = scrubberMax();
  world.step = Math.min(maxStep, world.step + stepsAdvance);
  if (world.live) world.scrubberFollowsLatest = false;
  $("#world-scrubber").value = String(world.step);
  renderFrame();
  updateScrubberStepLabel();
  if (world.step >= maxStep) {
    if (world.live) {
      // Caught up to live — re-attach so future progress events advance
      // the scrubber automatically, and stop the playback tick.
      world.scrubberFollowsLatest = true;
    }
    pause();
    return;
  }
  world.rafId = requestAnimationFrame(tick);
}
