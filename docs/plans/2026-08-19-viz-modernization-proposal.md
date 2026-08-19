---
type: plan
status: parked
date: 2026-08-19
mechanism: viz-modernization
---
# Viz Modernization Proposal — Live 3D Console for the Human-Brain Spiking Substrate

**Status: PARKED (owner will pick this up later).** Produced by the `lit-and-viz-scout` workflow (2026-08-19), grounded
in surveys of FlyWire/Codex, the BRAIN-Initiative Connectome Data Explorer (=Codex), FruitFlyBrain/Neu3D + FlyConnectome
(natverse), VirtualFlyBrain (Geppetto), the human-brain heavyweights (Neuroglancer, Allen, HCP Connectome Workbench),
the modern web-viz field (NiiVue, BrainBrowser, EBRAINS siibra-explorer, three.js/WebGL2/WebGPU/deck.gl), and OUR
current setup. Vikunja carries a single parked pointer task; this doc is the reference.

## TL;DR
Keep the transport backbone we already have (it is genuinely good), but replace the renderer's representation and
modernize its rails. **Recommended stack: Three.js migrated to its WebGPU backend (WebGPURenderer + TSL) with an
automatic WebGL2 fallback, bundled via Vite, over a tiered live-activity streaming protocol.** Do NOT adopt Neuroglancer
as the live renderer (it is built for *static* connectome browsing, not time-varying spikes) — adopt its LOD/streaming
discipline and Codex/VFB's query + ontology UX. The single highest-leverage first move is small: wire the ChatBrain
through the existing `RegionActivityProbe` → `emit_activity` → WS pipe so talking to the brain lights up the 3D map,
vendor Three.js off the CDN, and auto-derive the layout from the live region set.

## 1. Recommended tech stack
- **Renderer: Three.js → WebGPU (WebGPURenderer / TSL), WebGL2 fallback.** Our console already IS Three.js (r0.160 via
  CDN) with OrbitControls, CSS2DRenderer labels, UnrealBloom. The WebGPU backend keeps the whole scene graph and adds:
  compute shaders (spike-decay / particle integration / per-region density on the GPU, off the CPU), storage
  buffers + instancing (millions of point sprites at 60 fps), and an automatic WebGL2 fallback via one renderer API.
  WebGPU is solid in Chromium/Edge, shipping in Safari 18+/Firefox — default to it, fall back to WebGL2.
- **Framework/build: plain ES-module SPA + Vite.** The current no-build step is what pins us to a stale CDN Three.js and
  blocks WebGPU/TSL + tree-shaking. Vite vendors/upgrades/code-splits Three.js and kills the internet-at-page-load
  dependency. Keep the hand-rolled tabbed SPA + hash routing; modularize only the 3D panel. Backend stays FastAPI.
- **Why not the surveyed tools as the live renderer:** Neuroglancer (Apache-2.0, WebGL2) is best-in-class for STATIC
  volumetric/mesh/segmentation browsing (precomputed chunked multi-res geometry, URL-encoded view state) and
  near-useless for animating live per-neuron voltage over a changing substrate. Adopt it LATER only as a complementary
  morphology/connectome inspector if per-neuron meshes ever exist. deck.gl (MIT) is a second engine we don't need.
- **The honest framing:** FlyWire is a static reconstruction of a small brain (~140K neurons) from precomputed EM
  meshes — impressive on fixed-geometry scale, never animates voltage. Our problem is the orthogonal axis:
  time-varying activity over a substrate heading toward human scale. We should NOT stream every neuron's voltage
  (infeasible + uninformative); the target is a perceptually-faithful, information-preserving rendering (region fields +
  representative sampled spikes + on-demand full rasters). Borrow FlyWire/Neuroglancer's LOD/stream-what's-in-view
  discipline and Codex/VFB's query/ontology UX; BUILD the live-spiking tiered temporal representation ourselves.

## 2. Features/UX to adopt (FlyWire · Codex · VirtualFlyBrain) and how to adapt
- **Codex query-driven selection** → a query bar over our region/pathway model (later population/cell-type):
  "highlight dopaminergic pathways", "regions active during the last chat turn"; color-by family/rate/neuromodulator;
  persistent selection sets co-visualized live.
- **Neuroglancer URL-encoded view state** → encode camera + selected regions + time-cursor + LOD tier in the hash
  router: a permalink to a brain state/moment ("watch the brain at t during this reply").
- **Codex connectivity/partner panel** → click a region → afferent/efferent pathways with LIVE flux + quantitative
  synapse counts; drill to population connectivity. Upgrades today's binary pathway toggles into a quantitative panel.
- **Neuroglancer multi-resolution LOD** → apply the principle to TEMPORAL/population tiers (below): stream the coarsest
  representation that conveys the view, refine on zoom/focus.
- **Time/layer scrubbing** (we have a replay scrubber) → extend to spike-event tiers + per-region rate timelines
  (small-multiples) synced to the scrubber; same scrubber in live and replay.
- **VFB ontology-driven navigation** → promote `brain3d_layout.json` to an ontology-lite (region → family → function,
  part-of hierarchy); navigate by structure/function; link regions ↔ our Findings/Experiments tabs.
- **VFB template alignment** → a canonical human-brain template layout as the coordinate frame onto which live
  activity / connectivity / (later) morphology register; AUTO-DERIVE the layout from the live region set (fixes the
  hand-regenerated-JSON drift).
- **Two console-native wins not in the surveys (matter most):** (1) wire the conversational brain — route ChatBrain's
  `cp_firing_states` through the same `RegionActivityProbe`/`emit_activity` path so talking drives the picture; (2)
  stream chat tokens (SSE/WS) + co-highlight the regions that activate as the brain "thinks/speaks." That is the
  difference between a dashboard and a window into a mind.

## 3. Reusable (open-source) vs build-new
- **Reuse/keep (all OSS):** Three.js (MIT, incl. WebGPURenderer/TSL/OrbitControls/CSS2DRenderer/UnrealBloom) upgraded +
  vendored; Vite (MIT); OUR whole transport pipeline (`sim/activity_probe.py RegionActivityProbe` → `sim/progress.py
  emit_activity` → `webapp/server.py` WS ring-buffer + latest-wins coalescing → client lerp — the crown jewel:
  bandwidth O(regions), sim never back-pressured); the data-driven layout model; FastAPI/uvicorn/pydantic; Neuroglancer
  (Apache-2.0) later as a complementary inspector + its precomputed-chunk/LOD format as a design pattern; uPlot (MIT)
  for fast spike-raster / rate small-multiples; MessagePack/protobuf/Cap'n Proto for binary spike-event framing.
- **Build-new (the actual product work):** the TIERED neural representation layer — Tier 0 region aggregates (exists;
  extend to ALL runners incl. ChatBrain); Tier 1 per-region rate/density fields as instanced GPU glyph clouds; Tier 2
  sampled representative neurons (deterministic bounded sample, delta-encoded binary spike-event stream, GPU point
  cloud w/ flash-decay); Tier 3 focused population (full raster + membrane traces + morphology). Plus: the binary
  spike-event protocol + host-side samplers/encoders; ChatBrain→activity wiring + chat token streaming + region
  co-viz; auto-derived layout; query/selection engine + URL-encoded view state + region↔findings cross-links; the
  sleek UI shell (dark theme, consistent motion, LOD/tier controls).

## 4. Phased path (rough effort)
- **Phase 0 — Foundation + highest-leverage wiring (~1–2 wk).** Add Vite; vendor + upgrade Three.js on the existing
  WebGL2 renderer; kill the CDN pin. Auto-derive `brain3d_layout.json` from the live region set. WIRE ChatBrain through
  `RegionActivityProbe`/`emit_activity` so chat lights up the region map. Add chat token streaming. *Ship this first.*
- **Phase 1 — Modern renderer + transport (~3–5 wk).** Migrate to WebGPURenderer (TSL) + WebGL2 fallback; binary WS
  channel (MessagePack/protobuf); Tier-1 per-region density/rate fields as instanced GPU glyph clouds; begin the UI
  shell refresh. *Stops looking like 90 spheres; WebGPU compute offloads particle/decay from the CPU.*
- **Phase 2 — Sampled per-neuron spikes / Tier 2 (~4–6 wk).** Host-side deterministic sampler + delta-encoded
  spike-event emitter; GPU point-cloud w/ flash-decay; sampling-budget controls; LOD auto-switch region-field ↔ sampled
  population on zoom. *Watch individual spikes fly at (toward-)human scale, bandwidth-bounded.*
- **Phase 3 — Query/selection/connectivity/ontology nav (~4–6 wk).** Codex/VFB features: query bar, persistent
  selection sets, color-by-attribute, quantitative connectivity panel w/ live flux, URL-encoded view state, region ↔
  Findings/Experiments cross-links, ontology-lite navigation. *Explorable + shareable, not just watchable.*
- **Phase 4 — Focused deep-dive / Tier 3 + optional Neuroglancer inspector (~4–8 wk, partly gated on the substrate
  exposing morphology).** On-demand full raster + voltage traces (uPlot small-multiples), optional dendrite/morphology,
  and — only if per-neuron meshes ever exist — an embedded Neuroglancer panel. *The neuroscientist's-microscope view
  for one region at a time, without ever paying its cost globally.*
- *Cross-cutting:* the sleek-shell design refresh threads through Phases 1–3; the "sim never in the render loop /
  latest-wins coalescing" invariant is preserved in every tier.

## The bar check (modern · sleek · performant)
- **Modern:** WebGPU + compute shaders + TSL, binary streaming, a build pipeline off the CDN pin, token-streamed chat.
- **Sleek:** one coherent engine (no second renderer), query + selection + shareable permalinks + ontology navigation
  borrowed from the best fly-brain tools, a design-system shell.
- **Performant:** bandwidth stays O(regions) at overview and bounded-by-sampling-budget for live spikes — independent
  of neuron count — with GPU-side particle/decay integration and the sim kept out of the render loop.

## Source anchors (for whoever picks this up)
`webapp/static/brain3d.js`, `webapp/static/brain3d_layout.json`, `webapp/server.py` (WS ring-buffer, `_try_parse_activity`,
`/api/brain-chat`), `sim/activity_probe.py` (`RegionActivityProbe.sample`), `sim/progress.py` (`emit_activity`),
`research/runners/brain_chat_tui.ChatBrain`. The legacy per-neuron desktop renderer (`viz/renderer.py`, GLUT/PyOpenGL)
is NOT on the path — its per-neuron ambition is realized instead by the Tier-2 WebGPU point cloud in the web console.
