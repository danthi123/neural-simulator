// brain3d.js — Three.js 3D visualization of the simulator
//
// Loaded via <script type="module"> at the bottom of index.html. Creates
// a scene with:
//   * One sphere per brain region, colored by family (sensory / V1 /
//     premotor / D1 / D2 / pallidum / thalamus / motor / PFC / etc).
//   * Curved lines for each declared region pathway, colored by
//     transmitter (excitatory / inhibitory / dopamine).
//   * Region-name labels via CSS2DRenderer.
//   * OrbitControls for free camera.
//   * Replay scrubber that animates region brightness from a run's
//     trajectory data — when the agent moves N at step T, cortex_N /
//     str_D1_N / motor_N light up.
//   * Live mode that polls /api/inflight every 1s.
//
// Three.js is loaded from unpkg via the <script type="importmap"> tag
// in index.html. No build step.
//
// Public API (used by app.js):
//   import { initBrain3D } from "/static/brain3d.js";
//   initBrain3D({
//     canvasContainer: document.getElementById("brain3d-host"),
//     scrubberEl: document.getElementById("brain3d-scrubber"),
//     stepLabelEl: document.getElementById("brain3d-step-label"),
//     ...
//   });

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// ─── Module state ──────────────────────────────────────────────────────
let layout = null;
let scene, camera, renderer, labelRenderer, controls;
let regionMeshes = {};   // name -> THREE.Mesh
let regionMaterials = {}; // name -> MeshStandardMaterial (so we can update emissiveIntensity)
let pathwayLines = [];   // {line: THREE.Line, fromKey, toKey, kind, baseOpacity}
let regionTargets = {};  // name -> [0..1] target activation; lerped each frame to current emissiveIntensity
let canvasContainer = null;
let labelLayer = null;
let animationActive = false;
let animationId = null;

// Replay state
let replayData = null;       // {trajectory, phase_stats, ...} from /api/runs/{name}
let replayStep = 0;
let replayTotal = 0;
let replayPlaying = false;
let replayPlayHandle = null;
let replayStepsPerSec = 5;

// Live state
let livePollHandle = null;
let liveMode = false;

// ─── Color helpers ─────────────────────────────────────────────────────
const KIND_COLOR = {
  exc: 0x60a5fa,  // blue (glutamate)
  inh: 0xf87171,  // red (GABA)
  da:  0xd946ef,  // magenta (dopamine)
  ach: 0xa3e635,  // green (acetylcholine)
};

function hexToColor(hex) {
  const c = new THREE.Color();
  c.set(hex);
  return c;
}

// ─── Scene setup ───────────────────────────────────────────────────────
async function loadLayout() {
  const res = await fetch("/static/brain3d_layout.json");
  if (!res.ok) throw new Error(`failed to load brain3d_layout.json: ${res.status}`);
  return res.json();
}

function createScene(width, height) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e1a);

  // Subtle "deep space" fog
  scene.fog = new THREE.Fog(0x0a0e1a, 25, 60);

  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.set(20, 12, 20);
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  // CSS2D label renderer (overlay)
  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(width, height);
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.left = "0";
  labelRenderer.domElement.style.pointerEvents = "none";

  // Lighting
  const amb = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(amb);
  const key = new THREE.DirectionalLight(0xffffff, 0.8);
  key.position.set(8, 12, 8);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x93c5fd, 0.35);
  fill.position.set(-8, -2, -4);
  scene.add(fill);

  // Floor grid (subtle reference)
  const grid = new THREE.GridHelper(40, 40, 0x1a2030, 0x1a2030);
  grid.position.y = -7;
  scene.add(grid);

  // Axes mini-helper
  const axes = new THREE.AxesHelper(2);
  axes.position.set(-13, -6.5, -12);
  scene.add(axes);
}

function createRegions() {
  const colors = layout._family_colors;
  const sphereGeo = new THREE.SphereGeometry(1.0, 32, 24);
  for (const [name, info] of Object.entries(layout.regions)) {
    const familyHex = colors[info.family] || "#888";
    const baseColor = hexToColor(familyHex);
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      emissive: baseColor.clone().multiplyScalar(0.4),
      emissiveIntensity: 0.5,
      metalness: 0.15,
      roughness: 0.45,
    });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.position.set(info.x, info.y, info.z);
    mesh.scale.setScalar(info.r);
    mesh.userData.name = name;
    mesh.userData.baseEmissiveIntensity = 0.5;
    mesh.userData.baseScale = info.r;
    scene.add(mesh);
    regionMeshes[name] = mesh;
    regionMaterials[name] = mat;
    regionTargets[name] = 0;

    // Label
    const labelDiv = document.createElement("div");
    labelDiv.className = "brain3d-label";
    labelDiv.textContent = name;
    const label = new CSS2DObject(labelDiv);
    label.position.set(0, info.r * 1.4, 0);
    mesh.add(label);
  }
}

function createPathways() {
  for (const p of layout.pathways) {
    const fromInfo = layout.regions[p.from];
    const toInfo = layout.regions[p.to];
    if (!fromInfo || !toInfo) continue;

    // Skip recurrent self-edges; render as small ring instead
    if (p.from === p.to) continue;

    // Curved Bezier between the two centers, slightly arched in Y
    const start = new THREE.Vector3(fromInfo.x, fromInfo.y, fromInfo.z);
    const end = new THREE.Vector3(toInfo.x, toInfo.y, toInfo.z);
    const mid = start.clone().add(end).multiplyScalar(0.5);
    // Lift the midpoint to give an arch; bigger arch for longer connections
    const dist = start.distanceTo(end);
    mid.y += Math.min(2.0, dist * 0.15);
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    const points = curve.getPoints(24);
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: KIND_COLOR[p.kind] || 0x888888,
      transparent: true,
      opacity: 0.18,
    });
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    pathwayLines.push({
      line, mat, curve,
      fromKey: p.from, toKey: p.to, kind: p.kind,
      baseOpacity: 0.18,
    });
  }
}

function setupOrbitControls() {
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.target.set(0, 0, 0);
  controls.minDistance = 5;
  controls.maxDistance = 60;
  controls.update();
}

// ─── Animation ─────────────────────────────────────────────────────────
function startAnimation() {
  if (animationActive) return;
  animationActive = true;
  const tick = () => {
    if (!animationActive) return;
    // Smoothly lerp emissiveIntensity toward target
    for (const [name, mesh] of Object.entries(regionMeshes)) {
      const target = regionTargets[name] || 0;
      const mat = regionMaterials[name];
      const baseE = mesh.userData.baseEmissiveIntensity;
      const cur = mat.emissiveIntensity;
      // Soft attack, slower decay
      const lerpRate = (target > cur) ? 0.35 : 0.07;
      mat.emissiveIntensity = cur + (baseE + target * 5.0 - cur) * lerpRate;
      // Pulse scale slightly with activation
      const targetScale = mesh.userData.baseScale * (1 + target * 0.2);
      mesh.scale.setScalar(mesh.scale.x + (targetScale - mesh.scale.x) * 0.2);
      // Decay target
      regionTargets[name] = target * 0.92;
    }
    // Pulse pathway opacity for active edges
    for (const p of pathwayLines) {
      const fromAct = regionTargets[p.fromKey] || 0;
      const toAct = regionTargets[p.toKey] || 0;
      const flow = Math.max(fromAct, toAct);
      p.mat.opacity = p.baseOpacity + flow * 0.7;
    }
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    animationId = requestAnimationFrame(tick);
  };
  tick();
}

function stopAnimation() {
  animationActive = false;
  if (animationId) cancelAnimationFrame(animationId);
}

// ─── Activity synthesis ────────────────────────────────────────────────
// Without per-region firing rate data in the run JSON, we synthesize
// region activation from action + reward at each replay step. When the
// real bridge instrumentation lands, this function gets replaced with a
// direct read from the per-step activity log.

const ACTION_SUFFIX = ["N", "E", "S", "W"];

function activateAction(actionIdx, gain = 1.0) {
  if (actionIdx < 0 || actionIdx > 3) return;
  const sfx = ACTION_SUFFIX[actionIdx];
  // Premotor cortex: strongest activation
  bumpActivity(`cortex_${sfx}`, 1.0 * gain);
  // Direct pathway (D1) wins
  bumpActivity(`str_D1_${sfx}`, 0.9 * gain);
  // Indirect pathway (D2) competes
  for (const o of ACTION_SUFFIX) {
    if (o !== sfx) bumpActivity(`str_D2_${o}`, 0.4 * gain);
  }
  // Pallidum cascade
  bumpActivity(`gpi_${sfx}`, 0.35 * gain);   // actually inhibited; show muted
  bumpActivity(`gpe_${sfx}`, 0.6 * gain);
  bumpActivity("stn", 0.5 * gain);
  // Thalamus released by D1 inhibition of GPi
  bumpActivity(`thal_${sfx}`, 0.95 * gain);
  // Motor cortex fires
  bumpActivity(`motor_${sfx}`, 1.0 * gain);
}

function activateReward(reward, hasDopamine = true) {
  if (!hasDopamine) return;
  if (reward > 0) {
    // SNc burst
    bumpActivity("snc", Math.min(1.0, 0.5 + reward * 0.5));
    // Reward-flash all D1 (LTP) — gentle, just shows DA broadcast
    for (const sfx of ACTION_SUFFIX) bumpActivity(`str_D1_${sfx}`, 0.3);
  } else if (reward < 0) {
    // SNc dip — show dim activation
    bumpActivity("snc", 0.15);
  }
}

function activateVisualPathway(usedVisualCortex) {
  // When visual cortex is enabled, retina/V1/V2/IT carry the load.
  // Animate as a steady flow during navigation.
  if (!usedVisualCortex) return;
  bumpActivity("retina", 0.7);
  bumpActivity("cortex_v1_simple", 0.5);
  bumpActivity("cortex_v1_complex", 0.4);
  bumpActivity("cortex_v2", 0.3);
  bumpActivity("cortex_it", 0.25);
}

function activateLanguagePathway(usedTextIO) {
  if (!usedTextIO) return;
  // During text I/O training, language_input is driven; PFC and cortex
  // pools receive activation; motor pools receive PFC-bypass drive.
  bumpActivity("language_input", 0.8);
  bumpActivity("dlpfc_wm", 0.5);
}

function activateGoalDrive(distToGoal) {
  // PFC working memory — held active when goal is far (still-pursuing).
  if (distToGoal == null) return;
  const drive = Math.min(1.0, distToGoal / 8.0);
  bumpActivity("dlpfc_wm", 0.3 + drive * 0.4);
}

function activateHippocampusEpisode(stepFraction) {
  // Periodic burst at episode boundaries — synthetic, matches the role
  // of CA3 pattern completion at trial onset.
  const phase = (stepFraction * 5) % 1;
  if (phase < 0.1) {
    bumpActivity("ec", 0.5);
    bumpActivity("dg", 0.4);
    bumpActivity("ca3", 0.35);
    bumpActivity("ca1", 0.3);
    bumpActivity("place_cells", 0.25);
  }
}

function bumpActivity(name, amount) {
  if (!regionTargets[name]) regionTargets[name] = 0;
  regionTargets[name] = Math.min(1.0, regionTargets[name] + amount);
}

function clearAllActivity() {
  for (const k of Object.keys(regionTargets)) regionTargets[k] = 0;
}

// ─── Replay ────────────────────────────────────────────────────────────
async function loadRun(name, listItem) {
  liveMode = false;
  if (livePollHandle) { clearInterval(livePollHandle); livePollHandle = null; }
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`${res.status}`);
    replayData = await res.json();
    const traj = replayData.trajectory || [];
    replayTotal = traj.length;
    replayStep = 0;
    // Detect features used
    replayData._usedVisualCortex = (replayData.config_flags || []).some(
      (f) => f === "--enable-visual-cortex");
    replayData._usedTextIO = (replayData.config_flags || []).some(
      (f) => typeof f === "string" && f.includes("language"));
    // Update UI scrubber
    const scrubber = document.getElementById("brain3d-scrubber");
    const stepLabel = document.getElementById("brain3d-step-label");
    const runNameEl = document.getElementById("brain3d-run-name");
    if (scrubber) {
      scrubber.max = String(Math.max(0, replayTotal - 1));
      scrubber.value = "0";
    }
    if (stepLabel) stepLabel.textContent = `step 0 / ${Math.max(0, replayTotal - 1)}`;
    if (runNameEl) runNameEl.textContent = name;
    // Render the first step
    renderReplayStep(0);
  } catch (e) {
    const runNameEl = document.getElementById("brain3d-run-name");
    if (runNameEl) runNameEl.textContent = `Failed: ${e.message}`;
  }
}

function renderReplayStep(step) {
  if (!replayData) return;
  const traj = replayData.trajectory || [];
  if (step < 0 || step >= traj.length) return;
  replayStep = step;
  const t = traj[step];
  clearAllActivity();
  // Visual pathway always shows some activity if vision is used
  activateVisualPathway(replayData._usedVisualCortex);
  // Animate the action (action index in trajectory: 0=N,1=E,2=S,3=W)
  if (t.action != null && t.action >= 0) activateAction(t.action);
  // Reward
  if (t.reward != null) activateReward(t.reward);
  // Goal pursuit
  if (t.pos && t.goal) {
    const d = Math.abs(t.pos[0] - t.goal[0]) + Math.abs(t.pos[1] - t.goal[1]);
    activateGoalDrive(d);
  }
  // Hippocampus on episode boundaries
  activateHippocampusEpisode(step / Math.max(1, replayTotal));
  // Update UI
  const scrubber = document.getElementById("brain3d-scrubber");
  const stepLabel = document.getElementById("brain3d-step-label");
  if (scrubber) scrubber.value = String(step);
  if (stepLabel) stepLabel.textContent = `step ${step} / ${replayTotal - 1}`;
}

function play() {
  if (replayPlaying || !replayData) return;
  replayPlaying = true;
  const updateBtn = (txt) => {
    const b = document.getElementById("brain3d-play");
    if (b) b.textContent = txt;
  };
  updateBtn("⏸ Pause");
  const interval = 1000 / replayStepsPerSec;
  replayPlayHandle = setInterval(() => {
    if (!replayPlaying) return;
    let next = replayStep + 1;
    if (next >= replayTotal) next = 0; // Loop
    renderReplayStep(next);
  }, interval);
}

function pause() {
  replayPlaying = false;
  if (replayPlayHandle) { clearInterval(replayPlayHandle); replayPlayHandle = null; }
  const b = document.getElementById("brain3d-play");
  if (b) b.textContent = "▶ Play";
}

function setSpeed(stepsPerSec) {
  replayStepsPerSec = stepsPerSec;
  if (replayPlaying) { pause(); play(); }
}

// ─── Live mode ─────────────────────────────────────────────────────────
function startLiveMode() {
  liveMode = true;
  if (livePollHandle) clearInterval(livePollHandle);
  pollLive();
  livePollHandle = setInterval(pollLive, 1000);
  const runNameEl = document.getElementById("brain3d-run-name");
  if (runNameEl) runNameEl.textContent = "Live mode (polling /api/inflight)";
}

function stopLiveMode() {
  liveMode = false;
  if (livePollHandle) { clearInterval(livePollHandle); livePollHandle = null; }
}

async function pollLive() {
  try {
    const res = await fetch("/api/inflight");
    const data = await res.json();
    const runs = data.inflight || [];
    if (!runs.length) {
      clearAllActivity();
      return;
    }
    const r = runs[0];  // Use the first in-flight run
    const p = r.progress || {};
    clearAllActivity();
    // Synthesize activity from progress kind
    if (p.kind === "swr_replay") {
      // Hippocampus replay during sleep — prominent CA3 + dopamine flashes
      bumpActivity("ec", 0.7);
      bumpActivity("dg", 0.7);
      bumpActivity("ca3", 0.95);
      bumpActivity("ca1", 0.7);
      bumpActivity("place_cells", 0.5);
      // Random-ish action + DA each tick to simulate replayed action-reward pairs
      const actIdx = Math.floor(Math.random() * 4);
      activateAction(actIdx, 0.5);
      bumpActivity("snc", 0.5);
    } else if (p.kind === "embodied_episode") {
      // Active language training — flow through visual + cortex + motor
      activateVisualPathway(true);
      activateLanguagePathway(true);
      const actIdx = Math.floor(Math.random() * 4);
      activateAction(actIdx);
      bumpActivity("snc", 0.4);
    } else if (p.kind === "step") {
      activateVisualPathway(true);
      const actIdx = Math.floor(Math.random() * 4);
      activateAction(actIdx);
    }
    // Update HUD with progress info
    const liveLabel = document.getElementById("brain3d-live-label");
    if (liveLabel) {
      let txt = `${r.name} · `;
      if (p.kind === "swr_replay") {
        txt += `Phase ${p.phase_num} SWR · event ${p.ev}/${p.ev_total}`;
      } else if (p.kind === "embodied_episode") {
        txt += `Phase ${p.phase_num} · ep ${p.episode}/${p.episodes_total} · ${p.correct_pct}%`;
      } else {
        txt += "(no progress markers yet)";
      }
      liveLabel.textContent = txt;
    }
  } catch (e) {
    /* ignore — just keep showing last frame */
  }
}

// ─── Public API ────────────────────────────────────────────────────────
export async function initBrain3D({ canvasContainer: container } = {}) {
  if (!container) throw new Error("brain3d: canvasContainer required");
  canvasContainer = container;
  // Don't double-init
  if (renderer) return;
  layout = await loadLayout();
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 500;
  createScene(w, h);
  container.style.position = "relative";
  container.appendChild(renderer.domElement);
  container.appendChild(labelRenderer.domElement);
  createRegions();
  createPathways();
  setupOrbitControls();
  startAnimation();

  // Resize observer — keep the canvas filling its container
  const ro = new ResizeObserver((entries) => {
    const e = entries[0];
    if (!e) return;
    const w2 = e.contentRect.width;
    const h2 = e.contentRect.height;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
    labelRenderer.setSize(w2, h2);
  });
  ro.observe(container);
}

export function brain3dLoadRun(name) { return loadRun(name); }
export function brain3dRenderStep(step) { return renderReplayStep(step); }
export function brain3dPlay() { return play(); }
export function brain3dPause() { return pause(); }
export function brain3dSetSpeed(s) { return setSpeed(s); }
export function brain3dStartLive() { return startLiveMode(); }
export function brain3dStopLive() { return stopLiveMode(); }
export function brain3dGetState() {
  return {
    initialized: !!renderer,
    replayLoaded: !!replayData,
    replayStep,
    replayTotal,
    replayPlaying,
    liveMode,
  };
}
