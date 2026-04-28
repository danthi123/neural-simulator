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
};

export function setupWorldTab() {
  // Lazy-init when tab first activated
  document.querySelectorAll("nav button").forEach((btn) => {
    if (btn.dataset.tab === "world") {
      btn.addEventListener("click", () => {
        if (!world._initialized) initWorld();
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
  $("#world-live-mode").addEventListener("click", openLiveModePicker);
  $("#world-play").addEventListener("click", play);
  $("#world-pause").addEventListener("click", pause);
  $("#world-speed").addEventListener("change", (e) => {
    world.speed = parseFloat(e.target.value);
  });
  $("#world-scrubber").addEventListener("input", (e) => {
    if (!world.data) return;
    world.step = parseInt(e.target.value, 10);
    renderFrame();
  });

  loadWorldRunList();
}

/** Open a picker modal showing in-flight runs the server is tracking, and
 *  when one is chosen, attach the World tab as a live viewer for that run. */
async function openLiveModePicker() {
  const list = $("#world-runs-list");
  list.replaceChildren();
  const heading = document.createElement("div");
  heading.className = "muted";
  heading.style.marginBottom = "8px";
  heading.textContent = "Live runs (in-flight on this server):";
  list.appendChild(heading);

  try {
    const res = await fetch("/api/runs/launch");
    const data = await res.json();
    if (!data.runs.length) {
      const p = document.createElement("div");
      p.className = "muted";
      p.textContent = "No runs in flight. Launch one from the Launch tab.";
      list.appendChild(p);
      return;
    }
    for (const r of data.runs) {
      const item = document.createElement("div");
      item.className = "world-run-item";
      const status = r.running ? "RUNNING" : `done (rc=${r.returncode})`;
      const progress = r.latest_progress
        ? ` step ${r.latest_progress.step}/${r.latest_progress.total}`
        : "";
      const name = document.createElement("div");
      name.textContent = r.run_id + (r.interactive ? " ★ interactive" : "");
      const small = document.createElement("div");
      small.className = "small";
      small.textContent = `${status}${progress} · elapsed ${Math.round(r.elapsed_sec)}s`;
      item.appendChild(name);
      item.appendChild(small);
      // Kill button only for in-flight runs
      if (r.running) {
        const killBtn = document.createElement("button");
        killBtn.className = "kill-btn";
        killBtn.textContent = "✕ Kill";
        killBtn.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          if (window.killLaunchedRun) await window.killLaunchedRun(r.run_id);
          // Refresh the list after kill
          setTimeout(() => openLiveModePicker(), 800);
        });
        item.appendChild(killBtn);
      }
      item.addEventListener("click", () => attachLive(r.run_id, item));
      list.appendChild(item);
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
  // Use a synthetic data shape compatible with renderFrame()
  world.data = {
    grid_size: 8,
    trajectory: [],
    goal_log: [],
    action_log: [],
    reward_log: [],
    phase_stats: [],
  };
  world.step = 0;

  // Hide irrelevant playback controls; show live-mode controls instead.
  const pbControls = document.getElementById("playback-controls");
  const liveControls = document.getElementById("live-controls");
  const scrubberRow = document.querySelector(".scrubber-row");
  const progressBar = document.getElementById("world-progress-bar");
  if (pbControls) pbControls.style.display = "none";
  if (liveControls) liveControls.style.display = "inline-flex";
  if (scrubberRow) scrubberRow.style.display = "none";
  if (progressBar) progressBar.style.display = "block";
  const detachBtn = document.getElementById("world-detach");
  if (detachBtn && !detachBtn._bound) {
    detachBtn.addEventListener("click", () => {
      closeLiveSocket();
      $("#world-run-name").textContent = "Detached. No run loaded.";
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
      title: `recent_dist (rolling 100-step mean)`,
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
  world.liveChart = null;
  world.liveStartedAt = null;
  world.interactive = false;
  pauseState = false;
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
  // Restore playback controls visibility
  const pbControls = document.getElementById("playback-controls");
  const liveControls = document.getElementById("live-controls");
  const scrubberRow = document.querySelector(".scrubber-row");
  const progressBar = document.getElementById("world-progress-bar");
  if (pbControls) pbControls.style.display = "inline-flex";
  if (liveControls) liveControls.style.display = "none";
  if (scrubberRow) scrubberRow.style.display = "block";
  if (progressBar) {
    progressBar.style.display = "none";
    const fill = progressBar.querySelector(".progress-fill");
    if (fill) fill.style.width = "0%";
  }
}

/** A progress event covers ~100 sim steps. We extend the synthetic
 *  trajectory with the new point and re-render. We don't have intermediate
 *  positions, so the agent visually jumps each tick (every ~100 steps).
 *  Phase 3 with in-process bridge will get per-step granularity. */
function handleLiveProgress(p) {
  if (!world.data) return;
  world.livePoints.push(p);
  // Append the current pos to the synthetic trajectory; pad earlier steps
  // with the same pos so the array length matches step count.
  const traj = world.data.trajectory;
  const goalLog = world.data.goal_log;
  while (traj.length <= p.step) {
    traj.push(p.pos);
    goalLog.push(p.goal);
  }
  // Update phase_stats so the overlay shows the right phase number
  if (!world.data.phase_stats.length || world.data.phase_stats.at(-1).goal[0] !== p.goal[0] || world.data.phase_stats.at(-1).goal[1] !== p.goal[1]) {
    world.data.phase_stats.push({
      step_start: p.step,
      goal: [p.goal[0], p.goal[1]],
    });
  }
  world.step = p.step;
  $("#world-scrubber").max = String(p.total);
  $("#world-scrubber").value = String(p.step);
  $("#world-step-display").textContent = `step ${p.step} / ${p.total}`;

  // Live progress bar
  const fill = document.querySelector("#world-progress-bar .progress-fill");
  if (fill && p.total > 0) {
    fill.style.width = `${(p.step / p.total * 100).toFixed(1)}%`;
  }
  // ETA: extrapolate from elapsed wall-clock + steps so far
  const eta = document.getElementById("live-eta");
  if (eta && world.liveStartedAt && p.step > 0) {
    const elapsedMs = performance.now() - world.liveStartedAt;
    const stepsPerSec = p.step / (elapsedMs / 1000);
    const remaining = (p.total - p.step) / Math.max(0.01, stepsPerSec);
    const fmt = remaining > 60
      ? `${Math.round(remaining / 60)}m ${Math.round(remaining % 60)}s`
      : `${Math.round(remaining)}s`;
    eta.textContent = `${stepsPerSec.toFixed(1)} steps/s · ETA ${fmt}`;
  }

  // Live chart: index by step so the x-axis aligns with run progress.
  if (world.liveChart) {
    const distAt = new Array(p.step + 1).fill(null);
    for (const pt of world.livePoints) {
      if (pt.step <= p.step) distAt[pt.step] = pt.recent_dist;
    }
    world.liveChart.updateData([
      { values: distAt, color: P.accent, label: "recent_dist" },
    ]);
  }

  renderFrame();
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
    const canvas = $("#world-canvas");
    resizeCanvas(canvas, data.grid_size || 8);
    const total = (data.trajectory || []).length;
    $("#world-scrubber").max = String(Math.max(0, total - 1));
    $("#world-scrubber").value = "0";
    $("#world-step-display").textContent = `step 0 / ${total - 1}`;
    $("#world-run-name").textContent = name;
    renderFrame();
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
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid(ctx, gridSize);
}

function drawGrid(ctx, gridSize) {
  ctx.strokeStyle = "#2a2f3d";
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
  ctx.fillStyle = "#5f6770";
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

function renderFrame() {
  const canvas = $("#world-canvas");
  const ctx = canvas.getContext("2d");
  const gridSize = worldGridSize();
  const data = world.data;

  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!data) {
    drawGrid(ctx, gridSize);
    ctx.fillStyle = "#5f6770";
    ctx.font = "13px sans-serif";
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
  // at the center as a default visualization)
  drawLandmark(ctx, gridSize, [Math.floor(gridSize / 2), Math.floor(gridSize / 2)]);

  // Trajectory trail
  drawTrajectoryTrail(ctx, data.trajectory || [], step);

  // Goal marker
  if (goal) drawGoal(ctx, goal);

  // Agent
  const traj = data.trajectory || [];
  const pos = traj[step];
  if (pos) drawAgent(ctx, pos, getStepAction(step));

  // Update overlay
  updateOverlay(step, pos, goal);
  $("#world-step-display").textContent = `step ${step} / ${traj.length - 1}`;
  $("#world-scrubber").value = String(step);
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
  ctx.fillStyle = "#6ee7b7";
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
  ctx.fillStyle = "#fbbf24";
  ctx.strokeStyle = "#0a0c10";
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
    ctx.strokeStyle = "#fbbf24";
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
    ctx.fillStyle = "#fbbf24";
    ctx.fill();
  }
  ctx.restore();
}

function updateOverlay(step, pos, goal) {
  $("#overlay-pos").textContent = pos ? `(${pos[0]},${pos[1]})` : "—";
  $("#overlay-goal").textContent = goal ? `(${goal[0]},${goal[1]})` : "—";
  if (pos && goal) {
    const d = Math.abs(pos[0] - goal[0]) + Math.abs(pos[1] - goal[1]);
    $("#overlay-dist").textContent = String(d);
  } else {
    $("#overlay-dist").textContent = "—";
  }
  const a = getStepAction(step);
  $("#overlay-action").textContent = a != null ? actionName(a) : "—";
  const r = getStepReward(step);
  $("#overlay-reward").textContent = r != null ? r.toFixed(1) : "—";
  $("#overlay-phase").textContent = String(getCurrentPhaseIndex(step));
}

function play() {
  if (!world.data) return;
  world.playing = true;
  $("#world-play").disabled = true;
  $("#world-pause").disabled = false;
  world.lastFrameTime = performance.now();
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
  // Speed = steps per real second. Cap dt to prevent huge jumps after pause.
  const stepsAdvance = Math.max(1, Math.floor(world.speed * Math.min(dt, 0.1)));
  const total = (world.data.trajectory || []).length;
  world.step = Math.min(total - 1, world.step + stepsAdvance);
  renderFrame();
  if (world.step >= total - 1) {
    pause();
    return;
  }
  world.rafId = requestAnimationFrame(tick);
}
