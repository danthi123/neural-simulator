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
// Bloom postprocessing — emissive regions and pulse particles glow.
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

// ─── Module state ──────────────────────────────────────────────────────
let layout = null;
let scene, camera, renderer, labelRenderer, controls;
let regionMeshes = {};   // name -> THREE.Mesh
let regionMaterials = {}; // name -> MeshStandardMaterial (so we can update emissiveIntensity)
let pathwayLines = [];   // {line: THREE.Line, fromKey, toKey, kind, baseOpacity, label}
let regionTargets = {};  // name -> [0..1] target activation; lerped each frame to current emissiveIntensity
let regionActivity = {}; // name -> current displayed activation (for tooltip read-out)
let canvasContainer = null;
let animationActive = false;
let animationId = null;

// Hover/click interaction
let raycaster = null;
let pointer = null;            // THREE.Vector2 mouse position in NDC
let pickableMeshes = [];       // sphere meshes (for raycasting)
let hoveredKey = null;         // currently hovered region name
let pinnedKey = null;          // clicked = pinned in info panel
let tooltipEl = null;          // floating tooltip div
let infoPanelEl = null;        // pinned info panel (right-side)
let usePlainLabels = true;     // toggle: friendly display_name vs technical key
let _lastLabelOpacity = 1;     // last applied label opacity (for change detection)

// Replay state
let replayData = null;       // {trajectory, phase_stats, ...} from /api/runs/{name}
let replayStep = 0;
let replayTotal = 0;
let replayPlaying = false;
let replayPlayHandle = null;
let replayStepsPerSec = 5;

// Live state (2026-05-03)
//
// Multi-live-run aware. Tracks all currently-alive runs from /api/inflight,
// lets the user pick which one to follow. Also accumulates the run's log
// progress markers into a synthetic trajectory so the scrubber works
// while the run is still going — user can scrub back through past steps
// even though the run hasn't finished.
let livePollHandle = null;
let liveMode = false;
let liveRunsList = [];        // current /api/inflight result (alive only)
let liveSelectedName = null;  // name of the run we're currently following
let liveTrajectory = [];      // synthetic per-step trajectory built from log progress
let livePartialBytes = 0;     // log bytes consumed so far (for incremental tail)

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

let composer = null;
let bloomEnabled = true;

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

  // Bloom postprocessing — glow on emissive materials and additive
  // particles. UnrealBloomPass picks up bright regions and blurs them
  // outward, giving the "active region is glowing" aesthetic.
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(width, height),
    0.6,    // strength
    0.8,    // radius
    0.15,   // threshold
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  composer.bloomPass = bloom;

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
    // Stash all the metadata on the mesh so the raycaster hover/click
    // handler can pull it without re-looking-up in layout.
    mesh.userData.name = name;
    mesh.userData.info = info;  // {display_name, function, neurons, description, ...}
    mesh.userData.baseEmissiveIntensity = 0.5;
    mesh.userData.baseScale = info.r;
    scene.add(mesh);
    regionMeshes[name] = mesh;
    regionMaterials[name] = mat;
    regionTargets[name] = 0;
    regionActivity[name] = 0;
    pickableMeshes.push(mesh);

    // Friendly label — uses display_name if defined, else falls back to
    // the technical region key. Toggle behavior via usePlainLabels.
    const labelDiv = document.createElement("div");
    labelDiv.className = "brain3d-label";
    labelDiv.textContent = (usePlainLabels && info.display_name) ? info.display_name : name;
    labelDiv.dataset.regionKey = name;  // for theme/visibility toggles
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
      label: p.label || null,
      baseOpacity: 0.18,
    });
  }
}

// ─── Hover / click interaction (2026-05-03) ────────────────────────────
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
  const arr = Array.isArray(children) ? children : [children];
  for (const c of arr) {
    if (c == null) continue;
    if (typeof c === "string" || typeof c === "number") {
      node.appendChild(document.createTextNode(String(c)));
    } else {
      node.appendChild(c);
    }
  }
  return node;
}

function setupInteraction() {
  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2();
  tooltipEl = document.createElement("div");
  tooltipEl.className = "brain3d-tooltip";
  tooltipEl.style.display = "none";
  canvasContainer.appendChild(tooltipEl);
  infoPanelEl = document.createElement("div");
  infoPanelEl.className = "brain3d-info-panel";
  infoPanelEl.style.display = "none";
  canvasContainer.appendChild(infoPanelEl);
  const dom = renderer.domElement;
  dom.addEventListener("mousemove", onPointerMove);
  dom.addEventListener("click", onClick);
  dom.addEventListener("dblclick", onDoubleClick);
  dom.addEventListener("mouseleave", () => {
    hoveredKey = null;
    tooltipEl.style.display = "none";
  });
}

// Double-click a region to focus the camera on it (zoom in close).
// Camera moves toward the region so it's the dominant subject.
function onDoubleClick(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickableMeshes, false);
  if (hits.length === 0) return;
  const mesh = hits[0].object;
  const target = mesh.position.clone();
  // Position camera on a 3/4 angle from current direction, ~6 units away
  const offset = camera.position.clone().sub(controls.target).normalize().multiplyScalar(6);
  const newPos = target.clone().add(offset);
  // Smooth fly using existing flyCamera infra — register a transient preset
  CAMERA_PRESETS.__focused__ = {
    label: "Focus",
    position: [newPos.x, newPos.y, newPos.z],
    target: [target.x, target.y, target.z],
  };
  flyCamera("__focused__", 600);
  // Also pin the info panel for the doubly-clicked region
  pinnedKey = mesh.userData.name;
  populateInfoPanel(infoPanelEl, mesh.userData.info, pinnedKey);
  infoPanelEl.style.display = "";
}

function onPointerMove(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickableMeshes, false);
  if (hits.length > 0) {
    const mesh = hits[0].object;
    const key = mesh.userData.name;
    if (key !== hoveredKey) hoveredKey = key;
    const tx = ev.clientX - rect.left + 18;
    const ty = ev.clientY - rect.top + 18;
    tooltipEl.style.left = `${tx}px`;
    tooltipEl.style.top = `${ty}px`;
    populateTooltip(tooltipEl, mesh.userData.info, key);
    tooltipEl.style.display = "";
    renderer.domElement.style.cursor = "pointer";
  } else {
    hoveredKey = null;
    tooltipEl.style.display = "none";
    renderer.domElement.style.cursor = "";
  }
}

function onClick(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickableMeshes, false);
  if (hits.length > 0) {
    const mesh = hits[0].object;
    pinnedKey = mesh.userData.name;
    populateInfoPanel(infoPanelEl, mesh.userData.info, pinnedKey);
    infoPanelEl.style.display = "";
  } else {
    pinnedKey = null;
    infoPanelEl.style.display = "none";
  }
}

function populateTooltip(parent, info, key) {
  parent.replaceChildren();
  if (!info) {
    parent.appendChild(el("strong", {}, key));
    return;
  }
  const display = info.display_name || key;
  const fn = info.function || "";
  const act = (regionActivity[key] || 0);
  const actPct = Math.round(act * 100);
  parent.appendChild(el("div", { class: "brain3d-tip-title" }, display));
  if (fn) parent.appendChild(el("div", { class: "brain3d-tip-fn" }, fn));
  parent.appendChild(el("div", { class: "brain3d-tip-stats" }, [
    el("span", {}, key),
    el("span", {}, `${info.neurons ?? "?"} neurons`),
    el("span", { class: "brain3d-tip-act" }, `act ${actPct}%`),
  ]));
  parent.appendChild(el("div", { class: "brain3d-tip-hint muted" },
    "click to pin · drag to orbit"));
}

function populateInfoPanel(parent, info, key) {
  parent.replaceChildren();
  if (!info) return;
  const display = info.display_name || key;
  const fn = info.function || "";
  const desc = info.description || "(no description)";
  const family = info.family || "";

  const closeBtn = el("button", { class: "brain3d-panel-close", title: "Close" }, "×");
  closeBtn.addEventListener("click", () => {
    pinnedKey = null;
    parent.style.display = "none";
  });
  parent.appendChild(el("div", { class: "brain3d-panel-header" }, [
    el("div", {}, [
      el("div", { class: "brain3d-panel-title" }, display),
      el("div", { class: "brain3d-panel-subtitle" },
        `${key} · family: ${family} · ${info.neurons ?? "?"} neurons`),
    ]),
    closeBtn,
  ]));

  if (fn) parent.appendChild(el("div", { class: "brain3d-panel-fn" }, fn));
  parent.appendChild(el("div", { class: "brain3d-panel-desc" }, desc));

  const fill = el("div", { class: "brain3d-panel-act-fill", id: "brain3d-panel-act-fill" });
  const valueEl = el("div", { class: "brain3d-panel-act-value", id: "brain3d-panel-act-value" }, "0%");
  parent.appendChild(el("div", { class: "brain3d-panel-act" }, [
    el("div", { class: "brain3d-panel-act-label" }, "Current activity"),
    el("div", { class: "brain3d-panel-act-bar" }, [fill]),
    valueEl,
  ]));

  const incoming = pathwayLines.filter((p) => p.toKey === key);
  const outgoing = pathwayLines.filter((p) => p.fromKey === key);
  const inList = el("ul", { class: "brain3d-panel-list" });
  if (!incoming.length) {
    inList.appendChild(el("li", { class: "muted" }, "(no incoming)"));
  } else {
    for (const p of incoming) {
      const li = el("li", {}, [el("span", { class: `brain3d-pw-${p.kind}` }, p.fromKey)]);
      if (p.label) li.appendChild(document.createTextNode(` · ${p.label}`));
      inList.appendChild(li);
    }
  }
  parent.appendChild(el("div", { class: "brain3d-panel-section" }, [
    el("div", { class: "brain3d-panel-section-title" }, `Inputs (${incoming.length})`),
    inList,
  ]));

  const outList = el("ul", { class: "brain3d-panel-list" });
  if (!outgoing.length) {
    outList.appendChild(el("li", { class: "muted" }, "(no outgoing)"));
  } else {
    for (const p of outgoing) {
      const li = el("li", {}, [el("span", { class: `brain3d-pw-${p.kind}` }, p.toKey)]);
      if (p.label) li.appendChild(document.createTextNode(` · ${p.label}`));
      outList.appendChild(li);
    }
  }
  parent.appendChild(el("div", { class: "brain3d-panel-section" }, [
    el("div", { class: "brain3d-panel-section-title" }, `Outputs (${outgoing.length})`),
    outList,
  ]));
}

function updatePinnedActivity() {
  if (!pinnedKey || infoPanelEl.style.display === "none") return;
  const fill = infoPanelEl.querySelector("#brain3d-panel-act-fill");
  const valueEl = infoPanelEl.querySelector("#brain3d-panel-act-value");
  const act = regionActivity[pinnedKey] || 0;
  const pct = Math.round(act * 100);
  if (fill) fill.style.width = `${pct}%`;
  if (valueEl) valueEl.textContent = `${pct}%`;
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

// ─── Traveling pulse particles (2026-05-03) ────────────────────────────
//
// Phase 3 of the original 3D viz design: render visible "spikes" flowing
// along synaptic pathway curves. When a region fires, particles spawn
// at its edge and travel along the pathway curve to the post-region.
//
// Implementation: one shared THREE.Points mesh with a pool of MAX_PULSES
// particles. Each frame:
//   1. Advance each live particle along its pathway curve (age += dt/life)
//   2. Update positions[]/colors[] BufferAttributes
//   3. Spawn new particles from pathways whose pre-region just activated
//
// Particle color = neurotransmitter (matches pathway color):
//   excitatory = blue, inhibitory = red, dopamine = magenta, ach = green
//
// Spawn rate is proportional to fromAct * 5 particles/sec/pathway when
// active. Lifetime = 1.0s (matches a typical 1-5ms propagation_delay
// scaled up for visibility — these are "metaphorical" spikes, not
// time-accurate).

const MAX_PULSES = 2000;
let pulsePoints = null;          // THREE.Points
let pulsePositions = null;       // Float32Array, MAX_PULSES * 3
let pulseColors = null;          // Float32Array, MAX_PULSES * 3
let pulseAlphas = null;          // Float32Array, MAX_PULSES
let pulseStates = null;          // Array of {curve, age, life, color}|null per slot

function setupPulseParticles() {
  pulsePositions = new Float32Array(MAX_PULSES * 3);
  pulseColors = new Float32Array(MAX_PULSES * 3);
  pulseAlphas = new Float32Array(MAX_PULSES);
  pulseStates = new Array(MAX_PULSES).fill(null);
  // Initialize positions far off-screen so dead slots aren't visible.
  for (let i = 0; i < MAX_PULSES; i++) {
    pulsePositions[i * 3 + 0] = 1e6;
    pulsePositions[i * 3 + 1] = 1e6;
    pulsePositions[i * 3 + 2] = 1e6;
    pulseAlphas[i] = 0;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pulsePositions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(pulseColors, 3));
  geom.setAttribute("aAlpha", new THREE.BufferAttribute(pulseAlphas, 1));
  // Vertex + fragment shader so we can fade per-particle via aAlpha.
  // Additive blending makes overlapping particles glow.
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    vertexShader: `
      attribute float aAlpha;
      varying float vAlpha;
      varying vec3 vColor;
      void main() {
        vAlpha = aAlpha;
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = 14.0 * (1.0 + aAlpha) * (50.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      varying vec3 vColor;
      void main() {
        // Soft circular sprite via distance-from-center
        vec2 d = gl_PointCoord - vec2(0.5);
        float r = length(d);
        if (r > 0.5) discard;
        float falloff = 1.0 - r * 2.0;
        falloff = falloff * falloff;
        gl_FragColor = vec4(vColor, vAlpha * falloff);
      }
    `,
    vertexColors: true,
  });
  pulsePoints = new THREE.Points(geom, mat);
  scene.add(pulsePoints);
}

const KIND_COLOR_VEC = {
  exc: new THREE.Color(0x60a5fa),
  inh: new THREE.Color(0xf87171),
  da:  new THREE.Color(0xd946ef),
  ach: new THREE.Color(0xa3e635),
};

function spawnPulse(pathwayLine, intensity) {
  // Find a free slot (state == null or age >= 1)
  let slot = -1;
  for (let i = 0; i < MAX_PULSES; i++) {
    if (pulseStates[i] == null) { slot = i; break; }
  }
  if (slot === -1) return;
  const color = KIND_COLOR_VEC[pathwayLine.kind] || KIND_COLOR_VEC.exc;
  pulseStates[slot] = {
    curve: pathwayLine.curve,
    age: 0,
    life: 0.6 + Math.random() * 0.4,  // 0.6-1.0s travel time
    intensity: Math.min(1.0, intensity),
  };
  pulseColors[slot * 3 + 0] = color.r;
  pulseColors[slot * 3 + 1] = color.g;
  pulseColors[slot * 3 + 2] = color.b;
  pulseAlphas[slot] = intensity;
}

function stepPulses(dt) {
  if (!pulsePoints) return;
  let dirtyPos = false;
  let dirtyAlpha = false;
  const _tmp = new THREE.Vector3();
  for (let i = 0; i < MAX_PULSES; i++) {
    const st = pulseStates[i];
    if (st == null) continue;
    st.age += dt / st.life;
    if (st.age >= 1.0) {
      pulseStates[i] = null;
      pulseAlphas[i] = 0;
      pulsePositions[i * 3 + 0] = 1e6;
      pulsePositions[i * 3 + 1] = 1e6;
      pulsePositions[i * 3 + 2] = 1e6;
      dirtyPos = true; dirtyAlpha = true;
      continue;
    }
    // Position along curve at param t=st.age
    st.curve.getPoint(st.age, _tmp);
    pulsePositions[i * 3 + 0] = _tmp.x;
    pulsePositions[i * 3 + 1] = _tmp.y;
    pulsePositions[i * 3 + 2] = _tmp.z;
    // Alpha rises quickly then fades — bell curve
    const t = st.age;
    const fade = Math.sin(t * Math.PI);
    pulseAlphas[i] = st.intensity * fade;
    dirtyPos = true; dirtyAlpha = true;
  }
  if (dirtyPos) pulsePoints.geometry.attributes.position.needsUpdate = true;
  if (dirtyAlpha) pulsePoints.geometry.attributes.aAlpha.needsUpdate = true;
}

function spawnPulsesForActiveFlows(dt) {
  // Each frame, randomly spawn particles for pathways whose
  // pre-region is active. Spawn rate proportional to fromAct.
  // Honors the user's pathway visibility toggles.
  for (const p of pathwayLines) {
    if (pathwayVisible[p.kind] === false) continue;
    const fromAct = regionActivity[p.fromKey] || 0;
    if (fromAct < 0.1) continue;
    // Expected spawns per second = fromAct * 6 (a "lit" region emits
    // ~6 visible spikes per second along each outgoing pathway)
    const spawnsPerSec = fromAct * 6;
    if (Math.random() < spawnsPerSec * dt) {
      spawnPulse(p, fromAct);
    }
  }
}

// ─── Camera presets (2026-05-03) ───────────────────────────────────────
//
// Named viewpoints that fly the camera to a specific subsystem. Each
// preset is { name, position, target } in scene coordinates. The
// transition is a smooth 800ms cubic-eased animation (not snap).

const CAMERA_PRESETS = {
  overview: {
    label: "Whole brain",
    position: [20, 12, 20],
    target: [0, 0, 0],
  },
  bg_cascade: {
    label: "BG cascade",
    // Look down +Y axis at the BG region; centered between cortex and motor
    position: [3.5, 14, 8],
    target: [3.5, -1.5, 0],
  },
  vision: {
    // Pan to the left (sensory in -> IT)
    label: "Vision pathway",
    position: [-12, 8, 12],
    target: [-7, 1.5, 0],
  },
  motor: {
    // Right side: thalamus + motor
    label: "Action output",
    position: [14, 8, 8],
    target: [7, 0.5, 0],
  },
  hippocampus: {
    // Look at the trisynaptic loop (off in +Z)
    label: "Hippocampus",
    position: [3, 8, 16],
    target: [2.5, -1, 7],
  },
  language: {
    // Top: PFC + language pathways
    label: "Language + PFC",
    position: [-1, 14, 12],
    target: [-1.5, 4.5, 2],
  },
  top_down: {
    label: "Top-down",
    position: [0, 30, 0.01],
    target: [0, 0, 0],
  },
};

function flyCamera(preset, durationMs = 800) {
  const p = CAMERA_PRESETS[preset];
  if (!p) return;
  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const endPos = new THREE.Vector3(...p.position);
  const endTarget = new THREE.Vector3(...p.target);
  const t0 = performance.now();
  function step() {
    const t = (performance.now() - t0) / durationMs;
    if (t >= 1) {
      camera.position.copy(endPos);
      controls.target.copy(endTarget);
      controls.update();
      return;
    }
    // Cubic ease-in-out
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    camera.position.lerpVectors(startPos, endPos, e);
    controls.target.lerpVectors(startTarget, endTarget, e);
    controls.update();
    requestAnimationFrame(step);
  }
  step();
}

// ─── Pathway type toggles (2026-05-03) ─────────────────────────────────
//
// Each pathway has a kind ("exc" / "inh" / "da" / "ach"). The user can
// toggle whole categories on/off, and additionally choose to show only
// pathways with flowing activity.

const pathwayVisible = { exc: true, inh: true, da: true, ach: true };
let onlyFlowing = false;

function setPathwayKindVisible(kind, visible) {
  pathwayVisible[kind] = !!visible;
  applyPathwayVisibility();
}
function setOnlyFlowing(value) {
  onlyFlowing = !!value;
  applyPathwayVisibility();
}
let pulsesEnabled = true;
function setPulsesEnabled(value) {
  pulsesEnabled = !!value;
  if (pulsePoints) pulsePoints.visible = pulsesEnabled;
}
function applyPathwayVisibility() {
  for (const p of pathwayLines) {
    const kindOn = pathwayVisible[p.kind] !== false;
    p.line.visible = kindOn;
  }
}

// Auto-fit camera to scene bounds. Computes the bounding box of all
// region meshes, places the camera so the whole thing is framed.
// padding=1.1 gives a tighter view so regions are larger on first
// load; user can zoom out manually to see the whole scene.
function fitCameraToScene(padding = 1.1) {
  const box = new THREE.Box3();
  for (const mesh of pickableMeshes) {
    const meshBox = new THREE.Box3().setFromObject(mesh);
    box.union(meshBox);
  }
  if (box.isEmpty()) return;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  const distance = (maxDim / 2 / Math.tan(fov / 2)) * padding;
  // Place camera at a 3/4 perspective: above + right + in-front
  const dir = new THREE.Vector3(0.7, 0.5, 0.7).normalize();
  camera.position.copy(center).addScaledVector(dir, distance);
  controls.target.copy(center);
  controls.update();
}

// ─── Animation ─────────────────────────────────────────────────────────
function startAnimation() {
  if (animationActive) return;
  animationActive = true;
  let _lastFrameTime = performance.now();
  const tick = () => {
    if (!animationActive) return;
    const now = performance.now();
    const dt = Math.min(0.1, (now - _lastFrameTime) / 1000);  // cap at 100ms
    _lastFrameTime = now;
    // Smoothly lerp emissiveIntensity toward target. Update regionActivity
    // to track the displayed (post-lerp) activation level — the tooltip
    // and info panel read from regionActivity, NOT regionTargets, so the
    // displayed bar matches the visual brightness.
    for (const [name, mesh] of Object.entries(regionMeshes)) {
      const target = regionTargets[name] || 0;
      const mat = regionMaterials[name];
      const baseE = mesh.userData.baseEmissiveIntensity;
      const cur = mat.emissiveIntensity;
      const lerpRate = (target > cur) ? 0.35 : 0.07;
      mat.emissiveIntensity = cur + (baseE + target * 5.0 - cur) * lerpRate;
      const targetScale = mesh.userData.baseScale * (1 + target * 0.2);
      mesh.scale.setScalar(mesh.scale.x + (targetScale - mesh.scale.x) * 0.2);
      regionTargets[name] = target * 0.92;
      // Activity = normalized emissiveIntensity above the baseline.
      regionActivity[name] = Math.max(0, Math.min(1.0, (mat.emissiveIntensity - baseE) / 5.0));
    }
    for (const p of pathwayLines) {
      const kindOn = pathwayVisible[p.kind] !== false;
      const fromAct = regionActivity[p.fromKey] || 0;
      const toAct = regionActivity[p.toKey] || 0;
      const flow = Math.max(fromAct, toAct);
      // "Only flowing" mode: hide pathway when its endpoints are quiet.
      const flowingFilter = onlyFlowing ? (flow > 0.05) : true;
      p.line.visible = kindOn && flowingFilter;
      p.mat.opacity = p.baseOpacity + flow * 0.7;
    }
    // 2026-05-03: traveling pulse particles — spawn from active
    // pathways, advance live ones along their curves. Skip the work
    // when the user has toggled them off (saves ~1ms/frame).
    if (pulsesEnabled) {
      spawnPulsesForActiveFlows(dt);
      stepPulses(dt);
    }
    // 2026-05-03: dynamic label visibility based on camera distance.
    // When zoomed far out, the labels overlap into a single illegible
    // blob; fade them out so the regions remain visible. When close,
    // show all. Threshold: 22 = "fully readable", 40 = "fully hidden".
    const camDist = camera.position.distanceTo(controls.target);
    const labelOpacity = Math.max(0, Math.min(1, 1 - (camDist - 22) / 18));
    if (Math.abs(labelOpacity - _lastLabelOpacity) > 0.05) {
      _lastLabelOpacity = labelOpacity;
      const labels = canvasContainer.querySelectorAll(".brain3d-label");
      labels.forEach((l) => { l.style.opacity = String(labelOpacity); });
    }
    updatePinnedActivity();
    controls.update();
    if (bloomEnabled && composer) {
      composer.render();
    } else {
      renderer.render(scene, camera);
    }
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
    // Hide mini grid + agent HUD if this run has no trajectory (text I/O)
    if (traj.length === 0 || !traj[0]?.pos) {
      if (miniGridCanvas) miniGridCanvas.style.display = "none";
      const hud = document.getElementById("brain3d-agent-state");
      if (hud) hud.style.display = "none";
    }
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

  // The runner serializes per-step data as parallel arrays:
  //   trajectory[i]  = [x, y]      agent position
  //   goal_log[i]    = [gx, gy]    goal at step i
  //   action_log[i]  = action idx  (0=N, 1=E, 2=S, 3=W)
  //   reward_log[i]  = reward
  // Build a unified `t` object with the keys the rest of the code uses.
  const pos = traj[step];
  const goalAt = (replayData.goal_log && replayData.goal_log[step])
                 || replayData.goal_pos;
  const action = replayData.action_log ? replayData.action_log[step] : -1;
  const reward = replayData.reward_log ? replayData.reward_log[step] : 0;
  const t = {
    pos: Array.isArray(pos) ? pos : null,
    goal: Array.isArray(goalAt) ? goalAt : null,
    action: typeof action === "number" ? action : -1,
    reward: typeof reward === "number" ? reward : 0,
  };

  clearAllActivity();
  activateVisualPathway(replayData._usedVisualCortex);
  if (t.action != null && t.action >= 0) activateAction(t.action);
  if (t.reward != null) activateReward(t.reward);
  if (t.pos && t.goal) {
    const d = Math.abs(t.pos[0] - t.goal[0]) + Math.abs(t.pos[1] - t.goal[1]);
    activateGoalDrive(d);
  }
  activateHippocampusEpisode(step / Math.max(1, replayTotal));
  const scrubber = document.getElementById("brain3d-scrubber");
  const stepLabel = document.getElementById("brain3d-step-label");
  if (scrubber) scrubber.value = String(step);
  if (stepLabel) stepLabel.textContent = `step ${step} / ${replayTotal - 1}`;
  // Mini gridworld + agent HUD — pass our reconstructed `t` plus access
  // to the previous trajectory slice for the trail.
  renderMiniGridworld(t, traj, step, replayData);
  updateAgentStateHud(t);
}

// ─── Mini gridworld inset (2026-05-03) ─────────────────────────────────
//
// Small 2D canvas overlay (top-left of the 3D scene) that shows the
// current agent position, goal, and recent trail when replaying a
// navigation run. Reuses the same conventions as the World tab —
// agent=yellow circle, goal=green dot, trail=fading yellow path.
//
// Hidden when no run is loaded or when the loaded run has no
// trajectory data (e.g. text I/O runs).

let miniGridCanvas = null;
let miniGridCtx = null;

function ensureMiniGridCanvas() {
  if (miniGridCanvas) return;
  miniGridCanvas = document.createElement("canvas");
  miniGridCanvas.className = "brain3d-minigrid";
  miniGridCanvas.width = 160;
  miniGridCanvas.height = 160;
  canvasContainer.appendChild(miniGridCanvas);
  miniGridCtx = miniGridCanvas.getContext("2d");
}

function renderMiniGridworld(t, traj, step, runData) {
  if (!t || !t.pos || !t.goal) {
    if (miniGridCanvas) miniGridCanvas.style.display = "none";
    return;
  }
  ensureMiniGridCanvas();
  miniGridCanvas.style.display = "";
  const ctx = miniGridCtx;
  const w = miniGridCanvas.width;
  const h = miniGridCanvas.height;
  // Determine grid size — prefer explicit grid_size from runData, else
  // derive from max coordinates seen in trajectory + goal.
  const gridSize = (runData && runData.grid_size)
    ? runData.grid_size
    : Math.max(8, Math.max(t.pos[0], t.goal[0], t.pos[1], t.goal[1]) + 1);
  const cell = Math.floor(Math.min(w, h) / gridSize);
  const padX = (w - cell * gridSize) / 2;
  const padY = (h - cell * gridSize) / 2;

  // Background + grid lines
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1a2030";
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridSize; i++) {
    ctx.beginPath();
    ctx.moveTo(padX + i * cell, padY);
    ctx.lineTo(padX + i * cell, padY + cell * gridSize);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padX, padY + i * cell);
    ctx.lineTo(padX + cell * gridSize, padY + i * cell);
    ctx.stroke();
  }

  // Trail — last 12 positions, fading. Each trajectory entry is [x,y].
  const trailLen = Math.min(12, step);
  for (let i = 1; i <= trailLen; i++) {
    const tp = traj[step - i];
    if (!Array.isArray(tp) || tp.length < 2) continue;
    const a = 0.6 * (1 - i / trailLen);
    ctx.fillStyle = `rgba(251, 191, 36, ${a.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(padX + tp[0] * cell + cell / 2,
            padY + tp[1] * cell + cell / 2,
            Math.max(2, cell * 0.18), 0, 2 * Math.PI);
    ctx.fill();
  }

  // Goal — green dot with halo
  const gx = padX + t.goal[0] * cell + cell / 2;
  const gy = padY + t.goal[1] * cell + cell / 2;
  ctx.fillStyle = "rgba(110, 231, 183, 0.25)";
  ctx.beginPath();
  ctx.arc(gx, gy, cell * 0.7, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#6ee7b7";
  ctx.beginPath();
  ctx.arc(gx, gy, cell * 0.3, 0, 2 * Math.PI);
  ctx.fill();

  // Agent — yellow circle
  const ax = padX + t.pos[0] * cell + cell / 2;
  const ay = padY + t.pos[1] * cell + cell / 2;
  ctx.fillStyle = "#fbbf24";
  ctx.beginPath();
  ctx.arc(ax, ay, cell * 0.35, 0, 2 * Math.PI);
  ctx.fill();
  // Action arrow if present
  if (t.action != null && t.action >= 0) {
    const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // N/E/S/W
    const [dx, dy] = dirs[t.action] || [0, 0];
    ctx.strokeStyle = "#0a0e1a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + dx * cell * 0.4, ay + dy * cell * 0.4);
    ctx.stroke();
  }
}

function updateAgentStateHud(t) {
  const hud = document.getElementById("brain3d-agent-state");
  if (!hud) return;
  if (!t || !t.pos) {
    hud.style.display = "none";
    return;
  }
  hud.style.display = "";
  const dist = t.goal
    ? Math.abs(t.pos[0] - t.goal[0]) + Math.abs(t.pos[1] - t.goal[1])
    : null;
  const actionLetter = t.action != null && t.action >= 0
    ? "NESW"[t.action] : "—";
  const rewardSign = t.reward > 0 ? "+" : "";
  const parts = [
    `pos=(${t.pos[0]}, ${t.pos[1]})`,
    t.goal ? `goal=(${t.goal[0]}, ${t.goal[1]})` : null,
    dist != null ? `dist=${dist}` : null,
    `action=${actionLetter}`,
    t.reward != null ? `reward=${rewardSign}${t.reward}` : null,
  ].filter(Boolean);
  hud.textContent = parts.join(" · ");
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
  // Hide mini gridworld + agent HUD when entering live mode — those
  // are only meaningful for replay of completed navigation runs.
  if (miniGridCanvas) miniGridCanvas.style.display = "none";
  const hud = document.getElementById("brain3d-agent-state");
  if (hud) hud.style.display = "none";
  pollLive();
  livePollHandle = setInterval(pollLive, 1000);
}

function stopLiveMode() {
  liveMode = false;
  if (livePollHandle) { clearInterval(livePollHandle); livePollHandle = null; }
  liveSelectedName = null;
  liveTrajectory = [];
  livePartialBytes = 0;
  // Reset run-name back to neutral
  const runNameEl = document.getElementById("brain3d-run-name");
  if (runNameEl) runNameEl.textContent = "No run loaded";
}

// Pick which live run to follow. Called from the Brain tab UI's
// live-run-picker dropdown. Resets the synthetic trajectory.
function selectLiveRun(name) {
  liveSelectedName = name;
  liveTrajectory = [];
  livePartialBytes = 0;
  const runNameEl = document.getElementById("brain3d-run-name");
  if (runNameEl) runNameEl.textContent = `LIVE: ${name}`;
  // Force an immediate poll to populate trajectory + UI
  pollLive();
}

async function pollLive() {
  try {
    const inflightRes = await fetch("/api/inflight").then((r) => r.json());
    const allRuns = (inflightRes.inflight || []).filter((r) => r.alive);
    liveRunsList = allRuns;
    refreshLiveRunPicker();

    if (!allRuns.length) {
      clearAllActivity();
      const liveLabel = document.getElementById("brain3d-live-label");
      if (liveLabel) liveLabel.textContent = "No active runs.";
      return;
    }

    if (!liveSelectedName || !allRuns.find((r) => r.name === liveSelectedName)) {
      liveSelectedName = allRuns[0].name;
      liveTrajectory = [];
      livePartialBytes = 0;
    }

    const r = allRuns.find((rr) => rr.name === liveSelectedName) || allRuns[0];

    // 2026-05-03 fix: rebuild the trajectory from the FULL log, not just
    // from the single-most-recent progress entry. This gives the
    // scrubber proper granularity — one trajectory sample per logged
    // progress line (every 10 eps for embodied, every 100 events for
    // SWR replay, every step for navigation).
    if (r.log_file) {
      try {
        const logText = await fetch(`/api/runs/launch/log/${encodeURIComponent(r.log_file)}`)
          .then((r2) => r2.ok ? r2.text() : "");
        rebuildTrajectoryFromLog(logText, r);
      } catch (logErr) {
        appendLiveStep(r, r.progress || {});
      }
    } else {
      appendLiveStep(r, r.progress || {});
    }

    if (livePlaying) {
      replayStep = Math.max(0, liveTrajectory.length - 1);
      renderLiveStep(replayStep);
    }
    syncScrubberToLive();

    const p = r.progress || {};
    const liveLabel = document.getElementById("brain3d-live-label");
    if (liveLabel) {
      let txt = `${r.name} · `;
      if (p.kind === "swr_replay") {
        txt += `Phase ${p.phase_num} SWR · event ${p.ev}/${p.ev_total}`;
      } else if (p.kind === "embodied_episode") {
        txt += `Phase ${p.phase_num} · ep ${p.episode}/${p.episodes_total} · ${p.correct_pct}%`;
      } else if (p.kind === "step") {
        txt += `step ${p.step}/${p.total}`;
      } else {
        txt += "(no progress markers yet)";
      }
      txt += ` · ${liveTrajectory.length} timeline samples`;
      if (!livePlaying) txt += ` · ⏸ paused at sample ${replayStep}`;
      liveLabel.textContent = txt;
    }
  } catch (pollErr) {
    /* ignore — keep showing last frame */
  }
}

// Re-parse the full log text to rebuild liveTrajectory. Cheap at 1Hz
// polling: regex over 32KB of log = sub-millisecond. Uses matchAll()
// to iterate matches.
const _LOG_EP_RE     = /\[(?:P\d\s+)?ep\s+(\d+)\/(\d+)\]\s+correct_moves=(\d+)\/(\d+)=([\d.]+)%/g;
const _LOG_SWR_RE    = /\[(?:P\d\s+)?[Ss][Ww][Rr](?:\s+ev)?\]?\s+(\d+)\/(\d+)/g;
const _LOG_STEP_RE   = /step\s+(\d+)\/(\d+)\s+pos=\((-?\d+),(-?\d+)\)\s+goal=\((-?\d+),(-?\d+)\)(?:\s+action=([NESW?]))?(?:\s+reward=([-+]?[\d.]+))?/g;
const _LOG_PHASE_RE  = /^={3,}\s*PHASE\s+(\d+):\s+(.+?)\s+(\d+)\s+(?:episodes|events)/gm;

function rebuildTrajectoryFromLog(logText, run) {
  const samples = [];

  // First pass: collect phase headers so each progress sample gets
  // tagged with the correct phase number.
  const phases = [];
  for (const m of logText.matchAll(_LOG_PHASE_RE)) {
    phases.push({ pos: m.index, num: +m[1], label: m[2].trim(), total: +m[3] });
  }
  function phaseAt(pos) {
    let cur = null;
    for (const ph of phases) { if (ph.pos <= pos) cur = ph; else break; }
    return cur;
  }

  // Embodied episode markers (Phase 2 of curriculum, or text_eval_embodied)
  for (const m of logText.matchAll(_LOG_EP_RE)) {
    const ph = phaseAt(m.index);
    samples.push({
      pos: m.index,
      progress: {
        kind: "embodied_episode",
        episode: +m[1], episodes_total: +m[2],
        correct_moves: +m[3], n_steps: +m[4], correct_pct: +m[5],
        fraction: (+m[1]) / Math.max(1, +m[2]),
        phase_num: ph?.num, phase_label: ph?.label,
      },
    });
  }

  // SWR replay markers
  for (const m of logText.matchAll(_LOG_SWR_RE)) {
    const ph = phaseAt(m.index);
    samples.push({
      pos: m.index,
      progress: {
        kind: "swr_replay",
        ev: +m[1], ev_total: +m[2],
        fraction: (+m[1]) / Math.max(1, +m[2]),
        phase_num: ph?.num, phase_label: ph?.label,
      },
    });
  }

  // Per-step navigation markers (g11_bg_runner)
  for (const m of logText.matchAll(_LOG_STEP_RE)) {
    samples.push({
      pos: m.index,
      progress: {
        kind: "step",
        step: +m[1], total: +m[2],
        pos: [+m[3], +m[4]], goal: [+m[5], +m[6]],
        action: m[7] ? "NESW".indexOf(m[7]) : -1,
        reward: m[8] ? +m[8] : 0,
        fraction: (+m[1]) / Math.max(1, +m[2]),
      },
    });
  }

  samples.sort((a, b) => a.pos - b.pos);
  const capped = samples.length > 5000 ? samples.slice(samples.length - 5000) : samples;
  liveTrajectory = capped.map((s) => ({
    ts: Date.now() / 1000,
    name: run.name,
    progress: s.progress,
  }));
}

// 2026-05-03 — public hook to skip to the latest live trajectory sample
// and re-engage auto-follow. Wired to the "Latest" button in the Brain
// scrubber row.
function liveJumpToLatest() {
  if (!liveMode) return;
  livePlaying = true;
  replayStep = Math.max(0, liveTrajectory.length - 1);
  renderLiveStep(replayStep);
  syncScrubberToLive();
}

// Track whether the user is letting playback follow live or paused on
// a past step. Live mode auto-advances when playing or when caught up;
// the scrubber lets the user move freely.
let livePlaying = true;

function appendLiveStep(run, progress) {
  // Build a synthetic step entry. The exact shape depends on the
  // progress kind so the scrubber animation can replay the same
  // activity pattern at each timeline position.
  const last = liveTrajectory[liveTrajectory.length - 1];
  const newSample = {
    ts: Date.now() / 1000,
    name: run.name,
    progress,
  };
  // Dedup: only add if progress fraction or kind/numbers changed.
  if (last && JSON.stringify(last.progress) === JSON.stringify(progress)) return;
  liveTrajectory.push(newSample);
  // Cap to last 1000 samples to keep memory bounded.
  if (liveTrajectory.length > 1000) liveTrajectory.shift();
}

function renderLiveStep(step) {
  const sample = liveTrajectory[step];
  if (!sample) return;
  const p = sample.progress || {};
  clearAllActivity();
  // Activity by progress kind — same logic as the original pollLive
  // but driven by a specific timeline sample.
  if (p.kind === "swr_replay") {
    bumpActivity("ec", 0.7);
    bumpActivity("dg", 0.7);
    bumpActivity("ca3", 0.95);
    bumpActivity("ca1", 0.7);
    bumpActivity("place_cells", 0.5);
    // Use deterministic action index from event count so scrubbing
    // shows the same animation each time
    const actIdx = (p.ev || 0) % 4;
    activateAction(actIdx, 0.5);
    bumpActivity("snc", 0.5);
  } else if (p.kind === "embodied_episode") {
    activateVisualPathway(true);
    activateLanguagePathway(true);
    const actIdx = (p.episode || 0) % 4;
    activateAction(actIdx);
    bumpActivity("snc", 0.4);
  } else if (p.kind === "step") {
    activateVisualPathway(true);
    const actIdx = (p.step || 0) % 4;
    activateAction(actIdx);
  }
}

function syncScrubberToLive() {
  // Update scrubber to match the live trajectory length.
  const scrubber = document.getElementById("brain3d-scrubber");
  const stepLabel = document.getElementById("brain3d-step-label");
  const total = Math.max(1, liveTrajectory.length);
  if (scrubber) {
    scrubber.max = String(total - 1);
    if (livePlaying) scrubber.value = String(total - 1);
  }
  if (stepLabel) {
    const cur = livePlaying ? total - 1 : replayStep;
    stepLabel.textContent = `live ${cur} / ${total - 1}`;
  }
}

function refreshLiveRunPicker() {
  const picker = document.getElementById("brain3d-live-picker");
  if (!picker) return;
  // Build options for each alive run; preserve the current selection.
  const previousVal = picker.value;
  picker.replaceChildren();
  if (!liveRunsList.length) {
    picker.appendChild(el("option", { value: "" }, "(no active runs)"));
    picker.disabled = true;
    return;
  }
  picker.disabled = false;
  for (const r of liveRunsList) {
    const opt = el("option", { value: r.name }, r.name);
    if (r.name === liveSelectedName) opt.selected = true;
    picker.appendChild(opt);
  }
  // If the picker had no selection but we have a selected run, sync.
  if (liveSelectedName && picker.value !== liveSelectedName) {
    picker.value = liveSelectedName;
  }
}

// Public hook for the scrubber UI: when in live mode, scrubbing past
// the end means "release follow"; scrubbing back means "show that step".
function liveSetStep(step) {
  if (!liveMode) return false;
  const total = liveTrajectory.length;
  if (step >= total - 1) {
    livePlaying = true;
    replayStep = total - 1;
  } else {
    livePlaying = false;
    replayStep = step;
  }
  renderLiveStep(replayStep);
  syncScrubberToLive();
  return true;
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
  setupInteraction();
  setupPulseParticles();
  // Auto-fit the camera to the scene bounds so the whole brain is framed
  // on first paint regardless of region layout changes.
  fitCameraToScene();
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
    if (composer) composer.setSize(w2, h2);
  });
  ro.observe(container);
}

function setBloomEnabled(value) {
  bloomEnabled = !!value;
}

export function brain3dLoadRun(name) { return loadRun(name); }
export function brain3dRenderStep(step) {
  // In live mode, scrubbing operates on the synthetic trajectory; in
  // replay mode, on the loaded run's trajectory.
  if (liveMode) return liveSetStep(step);
  return renderReplayStep(step);
}
export function brain3dPlay() { return play(); }
export function brain3dPause() { return pause(); }
export function brain3dSetSpeed(s) { return setSpeed(s); }
export function brain3dStartLive() { return startLiveMode(); }
export function brain3dStopLive() { return stopLiveMode(); }
export function brain3dSelectLiveRun(name) { return selectLiveRun(name); }
export function brain3dJumpToLatest() { return liveJumpToLatest(); }
export function brain3dFlyToPreset(name) { return flyCamera(name); }
export function brain3dFitCamera() { return fitCameraToScene(); }
export function brain3dSetPathwayKindVisible(kind, visible) {
  return setPathwayKindVisible(kind, visible);
}
export function brain3dSetOnlyFlowing(value) { return setOnlyFlowing(value); }
export function brain3dSetPulsesEnabled(value) { return setPulsesEnabled(value); }
export function brain3dSetBloomEnabled(value) { return setBloomEnabled(value); }
export function brain3dListPresets() { return CAMERA_PRESETS; }
export function brain3dGetState() {
  return {
    initialized: !!renderer,
    replayLoaded: !!replayData,
    replayStep,
    replayTotal,
    replayPlaying,
    liveMode,
    liveSelectedName,
    liveRunsList: liveRunsList.map((r) => ({ name: r.name, alive: r.alive })),
    liveTrajectoryLen: liveTrajectory.length,
  };
}
