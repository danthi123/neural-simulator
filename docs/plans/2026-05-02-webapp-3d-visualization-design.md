# Webapp 3D Neural Visualization — Design Doc

**Date:** 2026-05-02
**Status:** Design (no code yet)
**Successor to:** `viz/` (legacy DearPyGUI/OpenGL renderer in `neural-simulator.py`)

---

## Goal

Render the live (and replayed) state of the simulator in 3D in the browser:
~5,000 neurons across 50+ brain regions, ~175,000 synapses, with
real-time spike events, signal propagation, and weight changes — all
clearly readable AND aesthetically compelling.

Replace the legacy DearPyGUI/OpenGL viz (`neural-simulator.py` + `viz/`)
which has zero awareness of the modern brain-region framework, the
neuromodulator subsystem, the BG cascade, the visual cortex pipeline,
or any of the work since April 29.

Watcher should be able to:

1. **See where information flows.** Spot the retina → V1 → V2 → IT
   pathway lighting up when the agent looks at the goal.
2. **See decisions form.** Watch all 4 cortex_X pools compete, then
   one win and disinhibit GPi, releasing thalamus → motor.
3. **See learning happen.** Notice synaptic weights brighten on the
   pathway the agent is reinforcing.
4. **See replay.** Watch hippocampus burst-fire during sleep windows;
   see the tagged token-action pairs replay through the cortex.

---

## Design constraints

1. **Browser-native.** No native installs. Open URL → see brain.
2. **60 fps interactive** at 5K neurons + 175K static synapses + ~50
   simultaneous traveling pulses. Modern WebGL handles this trivially.
3. **Low data rate sim → browser.** ~10 Hz update budget, aggregated
   stats not per-spike. Per-spike details are computed client-side from
   a per-region rate.
4. **Replay AND live mode.** Same renderer for "load a recorded run"
   and "watch live training."
5. **Educational over comprehensive.** Better to render 50 regions
   beautifully than 5,000 individual neurons in a generic blob.

---

## Internal debate

### D1. Anatomical vs functional layout

**Anatomical** (regions placed in brain-like positions): pedagogically
beautiful, immediately legible to anyone with introductory neuroscience.
Real human brain has frontal/sensory/motor poles and subcortical
structures below — a recognizable layout helps newcomers map the
simulator to "real" brains.

**Functional cascade** (regions placed by information flow): great for
debugging — you see retina → V1 → V2 → IT in a horizontal river. Great
for explaining "why is the agent stuck?" sessions. But it abandons the
anatomical mental model.

**Decision:** *Hybrid*. Anatomically inspired, but cheat anatomy where
clarity wins. Specifically:

- Sensory in (retina) on the left; motor out (motor_X) on the right.
- Cortex on top, subcortical below — preserving the dorsal/ventral
  contrast.
- The 4-action BG cascade laid out as 4 stacked horizontal lanes
  (cortex_N → str_D1_N → ... → motor_N as a row). This is a deliberate
  anatomical lie — real BG is one structure with intermingled action
  channels — but it makes the parallel-cascade architecture *visible*.
- Hippocampus parked off to the side as its own subassembly (DG → CA3
  → CA1 → place_cells). Real hippocampus is medial-temporal, but
  spatially separating it makes the trisynaptic loop legible.

### D2. Color palette

Two competing principles: *each region distinct* vs *related regions
look related*. Going for ~50 individual hues is visual noise.

**Decision:** Color by **functional family**, with intra-family
distinguished by saturation/lightness gradients.

| Family | Hex | Members |
|---|---|---|
| Sensory input | `#4ade80` (green) | retina (ON+OFF channels) |
| Visual cortex | `#3b82f6` → `#1e3a8a` (blue gradient) | V1_simple, V1_complex, V2, IT |
| Premotor cortex | `#a855f7` (purple) | cortex_N/E/S/W |
| Striatum direct | `#ef4444` (red) | str_D1_N/E/S/W |
| Striatum indirect | `#fb923c` (orange) | str_D2_N/E/S/W |
| Pallidum/STN | `#92400e` (dark amber) | gpe_X, gpi_X, stn |
| Thalamus | `#67e8f9` (light cyan) | thal_X |
| Motor cortex | `#fbbf24` (yellow) | motor_X |
| PFC working memory | `#14b8a6` (teal) | dlpfc_wm |
| Hippocampus | `#ec4899` → `#831843` (pink gradient) | dg, ca3, ca1 |
| Place cells | `#f9a8d4` (light pink) | place_cells |
| Language | `#06b6d4` (cyan) | language_input/output |
| Dopamine (SNc) | `#d946ef` (magenta, special) | snc |
| Cerebellum | `#84cc16` (lime) | granule, purkinje, cf, mf |

These are colorblind-aware (no red/green collisions between pathways)
and chosen so the BG cascade reads as a coherent "warm-color river" vs
the cortex's "cool-color array."

### D3. Neuron representation: individual vs aggregate

**Individual** (one mesh per neuron): truthful but heavy on data feed
(5,000 spike states × 60 fps = 300K events/s). Most neurons in a region
are functionally interchangeable for visualization purposes — what
matters is *the region's firing rate*, not which exact neuron fired.

**Aggregate** (region-as-block): each region rendered as a labeled
slab/sphere whose size = neuron count, brightness = firing rate. Loses
the dazzle of "thousands of dots" but stays legible.

**Decision:** *Both, layered*. Default zoom-out view shows aggregate
glowing slabs. Click a region to drill in: orbit camera enters that
region, individual neurons appear as instanced glowing dots arranged in
a deterministic per-region grid (so a given dot at position [3,4]
always corresponds to the same neuron index — useful for connecting
specific spike events to specific dots). Per-neuron mode falls back to
streaming detailed spike events at 30 Hz over WebSocket only when the
camera is "inside" a region.

### D4. Synapses: 175K is too many

Three options:

1. **Render all 175K as faded lines.** Tested mentally — ugly.
   Region-internal density is high enough that regions become
   spaghetti.
2. **Render only declared region pathways (~80 of them).** Each
   pathway becomes a single bold curve from region A to region B with
   thickness ∝ pathway weight × density. This is the *graph* view.
3. **Render dynamic edges only when they fire** — particles travel
   along otherwise-invisible curves. Beautiful but viewers can't see
   the static structure when nothing is firing.

**Decision:** *Pathway-aggregate plus particle overlay*. Static layer:
faded thin curves for all declared pathways (alpha ~0.15 when
inactive). Dynamic layer: bright traveling particles spawned when a
pathway transmits a wave of spikes. Pathway curve brightens briefly
when its particle is in flight. This combines structural visibility
with motion beauty.

### D5. Signal propagation animation

Real action potentials propagate at ~1 m/s in unmyelinated axons,
30–120 m/s in myelinated. In our model: each pathway has a
`propagation_delay_ms` (typically 1–5 ms). When pre-region fires above
threshold at sim step T, post-region receives effect at sim step
T + delay/dt.

**Particle behavior:**

- Spawn at pre-region's edge (closest point on its bounding sphere
  toward post-region).
- Travel along the pathway curve.
- Arrive at post-region edge at exactly `propagation_delay_ms` of sim
  time later.
- Color = neurotransmitter:
  - Glutamate (excitatory cortical): `#60a5fa` light blue
  - GABA (inhibitory): `#f87171` red
  - Dopamine: `#d946ef` magenta
  - Acetylcholine: `#a3e635` green
  - Serotonin (if added): `#fb7185` pink
- Brightness ~ pre-region's recent firing rate (cap 1.0).
- Tail = motion blur trail of length proportional to delay.

A pathway with high firing rate looks like a continuous river of light;
a quiet pathway just shows the faded baseline curve.

### D6. Plasticity / weight changes

Weights drift slowly (10s of seconds for noticeable change). Two
overlay options:

1. **Persistent color tint.** Pathway base color = region color but its
   *saturation* shifts as weight magnitude changes. Strong pathways =
   vivid; pruned/floor pathways = washed out.
2. **Reward flash.** When the dopamine modulator fires (reward signal
   > threshold), all eligible pathways briefly glow gold for 200 ms,
   showing "this is what's getting reinforced."

**Decision:** *Both*. Color saturation gives a slow "where are weights
right now" picture; reward flashes give the dopamine moment its own
visual punch. Reward visualization is one of the most-requested-by-
users features for any neuromodulator simulation.

### D7. Tech stack

**Three.js (WebGL)** is overwhelmingly the right call. Alternatives:

- **D3 + SVG**: great for graph layouts but breaks at 4K+ animated
  particles.
- **Plain Canvas2D**: works for the existing 2D world view but can't
  do 3D depth, glow, or perspective camera.
- **WebGPU**: theoretically faster than WebGL2 but browser support
  still spotty (Safari 17+ only). Premature.
- **Babylon.js**: feature-equivalent to Three.js. Three.js is
  better-known and has more examples — easier to onboard contributors.

Three.js features we'll use:

- `InstancedMesh` for the 5K neurons as glowing dots (one geometry,
  5K transform matrices, single draw call).
- `LineSegments2` (from `examples/jsm/lines/`) for the pathway curves
  with variable thickness — `THREE.Line` width is GPU-driver-dependent
  and broken on most platforms.
- `Points` + custom shader for traveling pulses — particles with
  fragment-shader bloom.
- `OrbitControls` for free camera (orbit/pan/zoom).
- `EffectComposer` + `UnrealBloomPass` for the glow aesthetic.
- `CSS2DRenderer` for region labels that always face camera.

### D8. Data feed

The `sim/data_bus.py` channel system already publishes
`firing_rates`, `spike_events`, and `weights` from the bridge. For 3D
viz we need *per-region* aggregates, not per-neuron events.

**New publishes** to add to `bridge.py:_run_one_simulation_step`:

- `region_rates` — `{region_name: rate_hz}` per region, every 100 ms.
- `pathway_pulses` — `[{from, to, n_spikes, ts_ms}]` for pathways that
  transmitted ≥ 1 spike since last publish, every 100 ms.
- `nm_levels` — `{nm_name: concentration}` per neuromodulator.
- `reward_events` — `{ts_ms, magnitude}` when reward signal is emitted.

**Transport:**

- *Live mode*: WebSocket `/ws/sim_stream` from server to browser.
  Server consumes data-bus channels via in-process subscriber (already
  the pattern for `world.json`). One-way push at 10 Hz.
- *Replay mode*: pre-recorded into the run's HDF5 file as new datasets
  (`/region_rates`, `/pathway_pulses`, etc). Same renderer reads either
  the live stream or the replay file via a `DataSource` abstraction.

The runner's existing `--progress-print-interval` and trial JSON
already give enough for the static view; new datasets only needed for
the dynamic spike/pathway visualization.

### D9. Performance budget

Target: 60 fps on a mid-tier laptop GPU (Intel Iris Xe / M2 base / GTX
1650).

| Component | Cost | Budget |
|---|---|---|
| 5K instanced neurons | 1 draw call, ~0.2 ms | trivial |
| 175K static synapse lines | 1 draw call, ~3 ms | OK |
| 80 pathway curves | 1 draw call, ~0.1 ms | trivial |
| 50 active particles | 1 draw call, ~0.5 ms | trivial |
| Bloom post-processing | ~2 ms | OK |
| **Total render** | **~6 ms** | well under 16 ms |
| Region label CSS layer | ~1 ms | OK |

Bottleneck risk: WebSocket parse/dispatch in JS. At 10 Hz × ~50
regions × ~80 pathways = ~1300 numbers per packet, ~13 KB/s. Trivial.

### D10. Aesthetic direction

Reference: BlueBrain Project visualizations, Eyewire connectome
visualizations, Waterloo's *Spaun* model demos. Common attributes:

- **Dark navy/black background** (`#0a0e1a`) for depth and glow
  contrast.
- **Bloom/glow on active elements** — recently-fired regions appear
  to have inner light.
- **Subtle grid floor** (faint horizontal lines, `#1a2030`) for
  spatial reference without distracting from the brain.
- **Smooth camera transitions** (cubic-ease 800 ms) when clicking a
  region to zoom in.
- **No drop shadows** — they look dated. Stick to bloom and emissive
  materials.
- **Region labels** rendered as small monospace billboards above each
  region, alpha 0.7 when default-zoom, alpha 1.0 when camera is near.

Avoid:

- Cartoon stylization. This is a research tool, not a game.
- Over-saturated colors that flatten depth.
- Animation for animation's sake (random rotation, idle particle
  systems). Every motion should mean something is happening in the
  sim.

---

## Recommended architecture

### Layout

A single 3D scene anchored at origin, axes:

- **+X (right):** information flow forward (input → output).
- **+Y (up):** dorsal direction (cortex above subcortex).
- **+Z (out of screen):** action channel index for the BG cascade
  (action 0 = N is closest to camera, action 3 = W furthest).

```
   y
   |
   |  [V1] [V2] [IT]                    [PFC]
   |    \   \   /                        /|
   |  [retina]                         /  |
   |                          [cortex_N] [cortex_E] [cortex_S] [cortex_W]   z
   |                              |        |          |          |       /
   |                          [str_D1_N]                                /
   |                              |
   |                          [gpe/gpi/stn cluster]
   |                              |
   |                          [thal_N]
   |                              |
   |                          [motor_N]
   |   [snc]
   |   [hippocampus subassembly off-axis at (-X, +Y, +Z)]
   +--------------------------------------------- x
```

Position file: a JSON config `webapp/static/region_layout.json`
defining `{x, y, z, radius}` per region. Generated by a small
deterministic algorithm from the region name (sensory regions get
small +x; cortex gets +y; subcortical gets -y; per-action ones get +z
proportional to action index). Hand-overrides allowed.

Region radius ∝ √(n_neurons). A 64-neuron region is half the radius
of a 256-neuron region — visually tracks "this region matters."

### Files (proposed)

```
webapp/static/
├── viz3d.js                         # main entry; setupViz3D(canvas)
├── viz3d_layout.js                  # region positions + curves between
├── viz3d_neurons.js                 # InstancedMesh management
├── viz3d_pathways.js                # static curves + particle system
├── viz3d_overlays.js                # labels, region rates HUD
├── viz3d_camera.js                  # OrbitControls + smooth flyto
└── viz3d_data.js                    # WebSocket / HDF5 data source

webapp/static/region_layout.json     # data-only, hand-tunable

webapp/server.py                     # add /ws/sim_stream endpoint;
                                     # add /api/region_meta, /api/pathway_meta

sim/bridge.py                        # add per-region rate aggregation
                                     # publish region_rates, pathway_pulses
```

New HTML tab in `index.html`:

```html
<button data-tab="brain">3D Brain</button>
...
<section id="tab-brain" class="tab">
  <div class="toolbar">
    <button id="brain-load-run">Load run…</button>
    <button id="brain-live-mode">Live mode</button>
    <button id="brain-camera-overview">Overview</button>
    <button id="brain-camera-cascade">BG cascade</button>
    <button id="brain-camera-vision">Vision</button>
  </div>
  <canvas id="brain-canvas"></canvas>
  <div id="brain-region-detail">…</div>
</section>
```

### Data flow

```
SimulationBridge._run_one_simulation_step()
    │
    ├─► self.data_bus.publish("region_rates", {...})
    └─► self.data_bus.publish("pathway_pulses", [...])
         │
    server.py background subscriber
         │
         ├─► aggregator (debounce to 10 Hz)
         │
         └─► WebSocket /ws/sim_stream → browser
              │
              viz3d_data.js parses JSON
              │
              ├─► viz3d_neurons.js: update region brightness uniforms
              ├─► viz3d_pathways.js: spawn particles for each pulse
              └─► viz3d_overlays.js: update HUD
```

Replay path: same datapaths, but `viz3d_data.js` reads from the run's
HDF5 file (already mounted via `/api/runs/{name}/world.json`-equivalent
pattern). New datasets `/region_rates_history` and
`/pathway_pulses_history` written to the simrec h5 in real time.

---

## Implementation plan (phased)

The full thing is ~2 weeks of dev. Phased so each step delivers value
on its own.

### Phase 1: Static structure (3–5 days)

- `region_layout.json` with all 50+ regions positioned.
- Three.js scene loads, camera orbits, regions render as colored
  glowing spheres.
- Pathway curves drawn as thin static lines.
- Region labels.
- No data feed yet — purely structural.

**Deliverable:** "Here's the brain. Pretty." User can navigate the
architecture without any sim running.

### Phase 2: Static replay (3 days)

- Replay endpoint serves `region_rates_history` from a recorded run.
- Time slider: scrub through the run, see firing rates at each step.
- No particles yet — region brightness only.

**Deliverable:** "Watch the brain across a 1800-step navigation."
Seeking back/forward works.

### Phase 3: Particles (4 days)

- `pathway_pulses` published from bridge, written to HDF5.
- Particle system spawns pulses on transmission events.
- Color/brightness/trail logic.

**Deliverable:** "See information flow." Visually understand WHY the
agent picked the action it did.

### Phase 4: Live mode (2 days)

- WebSocket endpoint, debounced 10 Hz.
- Live mode toggle: `viz3d_data.js` switches data source from HDF5
  replay to WebSocket.
- ETA / pause / detach controls (mirroring 2D world tab).

**Deliverable:** "Watch live training." User launches a runner from
the Launch tab and sees the brain light up.

### Phase 5: Plasticity overlay (3 days)

- Pathway color saturation = current weight magnitude.
- Reward flash gold across eligible pathways.
- Per-neuromodulator concentration HUD.

**Deliverable:** "See learning happen." Visually distinguish
strengthening vs weakening pathways.

### Phase 6: Drill-in (3 days)

- Click a region: camera flies in.
- Per-neuron InstancedMesh appears.
- Per-spike streaming (high-rate WebSocket only when zoomed in).
- Esc key flies out.

**Deliverable:** "See individual neurons fire."

---

## What this replaces / deprecates

The legacy GUI (`neural-simulator.py` + `viz/renderer.py`) has:

- Zero awareness of `BrainRegion` (last meaningful update was April 29,
  before the framework was even being used).
- Zero awareness of declared `RegionPathway` (treats all synapses as
  flat sparse connectivity).
- Zero awareness of neuromodulators, clusters, visual cortex, text I/O,
  distributed motor pop, or SWR.
- DearPyGUI host: not maintained.
- OpenGL via PyOpenGL: works, but no contributor on the team is
  maintaining it.

After Phase 4 of this plan ships, the legacy GUI should be marked as
deprecated in `README.md`. The webapp + 3D viz becomes THE official
viewer. Phase 6 of this plan provides per-neuron drill-in that matches
the legacy GUI's main differentiating feature.

---

## Open questions

1. **Camera memory.** Should clicking "BG cascade" preset persist
   across reloads? Probably yes via `localStorage`.
2. **Region grouping for camera presets.** Current proposal: Overview,
   Vision, BG cascade, PFC + Hippocampus, Cerebellum. Add more as we
   add clusters.
3. **Layout for 4-action vs 8-direction (distributed motor pop).**
   Same +Z-by-action-index lane idea generalizes — just 8 lanes
   instead of 4, packed tighter. No special-case needed.
4. **Mobile.** Three.js works on phones but a 50-region brain is
   probably too detailed for a 6" screen. Defer; not a priority.
5. **Coloring synapses by post- or pre-region.** Convention chose
   post- (matches "where the signal is going"). Could try both A/B.
6. **Trial-by-trial replay marker.** Should there be a step counter
   shown? Yes — already in the world.json schema, just port over.

---

## Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | Hybrid anatomical/functional layout | Pedagogy + clarity |
| D2 | Color by functional family | Avoids 50-color noise |
| D3 | Aggregate by default, drill-in to per-neuron | Performance + clarity |
| D4 | Pathway aggregates + particles | Readable + animated |
| D5 | Pulses at neurotransmitter color | Educational |
| D6 | Saturation for weight, flash for reward | Two timescales |
| D7 | Three.js | Industry standard |
| D8 | New per-region channels via existing data_bus | Minimal bridge change |
| D9 | 60 fps on mid-tier laptop | Accessibility |
| D10 | Dark, bloomy, BlueBrain-like aesthetic | Established genre |

---

## Estimated effort

| Phase | Scope | Days |
|---|---|---|
| 1 | Static structure | 3–5 |
| 2 | Static replay | 3 |
| 3 | Particles | 4 |
| 4 | Live mode | 2 |
| 5 | Plasticity overlay | 3 |
| 6 | Drill-in to neurons | 3 |
| **Total** | | **18–20** |

---

## Next steps (immediate, while not blocking on this design)

- Add text I/O presets to the existing webapp Launch tab so
  text-experiment runners are launchable from the dashboard. (Tonight,
  ~30 min.)
- Add a results-loading endpoint that auto-reads `text_eval_*.json`
  files and shows them in the Findings tab. (Tonight if SWR run frees
  up time.)
- Next session: start Phase 1 of this design (`region_layout.json` +
  static structure).
