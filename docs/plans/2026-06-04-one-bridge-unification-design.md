# One-bridge unification (spine item B) — design

> **Status:** DESIGN — awaiting owner approval before implementation (per the design-before-implement gate).
> **Goal:** collapse the conversational agent's three functional bridges (parser, composer, dlPFC) into ONE
> `SimulationBridge` with multiple interacting regions, turning the Python hand-offs into genuine synaptic
> `RegionPathway`s — "one interacting brain" rather than three brains orchestrated by Python.

## 1. Current structure (three bridges)

| bridge | role | size | dt | NMDA | plasticity |
|---|---|---|---|---|---|
| **parser** (`BridgeParser`) | comprehension: 6 conjunction units → 3×40 role ensembles; learned (position×voice)→role | 126 n | **1.0** | off | Hebbian on |
| **composer** (`build_bind_bridge`) | bind/unbind: role_ON/OFF + fill_ON/OFF sources → 4 AND coincidence banks | 8·D (6400 @ D=800) | **1.0** | off | none (fixed wiring) |
| **dlPFC** (`content_selection_spiking`) | dialogue planning: cortex_ctx ↔ dlpfc_wm loop, spreading activation | ~2×400 n | **0.5** | **on** | optional |

Cross-bridge signals are **Python hand-offs**: `parser.parse` returns `{role: word}` → the agent calls
`composer.store(...)`; `composer.kb` (a Python list of bound vectors) → `_assoc_graph` (a Python dict of fact-label
co-occurrences) → the dlPFC `SpikingSpreadingController`.

## 2. The core challenge — and a hard compatibility constraint

One bridge has **one neuron array, one step loop, one dt, one global config**. Two obstacles:

1. **Sequencing.** The three computations are orchestrated: each sets `cp_external_input_current` on a neuron slice,
   runs N steps, reads firing, resets. On a shared bridge they share the step loop, so e.g. the composer's
   reset-before-unbind would also reset the parser/dlPFC unless reset is made per-region (zero only the region's
   slice — a small change).
2. **dt + NMDA incompatibility (the load-bearing constraint).** Parser and composer run at **dt=1.0, NMDA off**;
   the dlPFC runs at **dt=0.5, NMDA on** (its bistable working memory is tuned to that regime). One bridge cannot
   have per-region dt. So **parser + composer are naturally compatible** (same dt, same NMDA-off, same Izhikevich)
   and **the dlPFC is the hard merge** — it would have to be re-tuned to dt=1.0 (risking its bistability) or kept on
   its own timing.

This constraint directly shapes the plan: merge the compatible pair first, treat the dlPFC as a separate harder step.

## 3. Approaches

**Approach 1 (RECOMMENDED) — one bridge, regions as index slices, time-multiplexed drives + synaptic cross-region
pathways.** Build one `SimulationBridge` whose `BrainRegion`s are the parser conjunctions/roles + the composer
source/AND banks (+ later the dlPFC). Each region keeps its internal wiring (parser Hebbian ensembles via
`inject_explicit_wiring`; composer coincidence banks likewise). The cross-region hand-off becomes a **synaptic,
gated pathway** (§4). The agent still orchestrates *timing* (drive parser → read roles → drive composer), but the
regions live on ONE bridge and interact through synapses — the owner's "one bridge with interacting regions." This
is achievable and incremental.

**Approach 2 (FUTURE north star) — continuous spiking flow.** A sentence drives the parser; role output flows
synaptically into the composer; bind/store/query flow without orchestrated reset/drive/read steps. The truest "one
brain", but it requires re-architecting the bind/unbind's *sequenced* reset→drive→read into continuous dynamics — a
much deeper change. Documented as the goal beyond Approach 1, not attempted first.

## 4. The key new wiring — the parser→composer hand-off as a gated route

The parser assigns a **role** (agent/action/patient) to each word *position*; the composer binds role⊗**code(word)**,
where the word's code is the substrate's own concept code (a graded ON/OFF current), not the parser's output. So the
hand-off is a **routing**: "this word's code binds to the role the parser selected." On one bridge this is realized
with the **already-shipped `transmission_gate`**: pre-wire each word-code drive → each role's fill bank, hold the
gates closed, and let the parser's role-ensemble firing **open** the gate for the selected role
(`couple_gate_to_pool`, the validated thalamocortical-gating primitive). Comprehension then *routes* composition in
spikes — exactly the seam we want to remove from Python.

## 5. Incremental build (each step validated; stop at any green step)

1. **Parser + composer on ONE bridge** (compatible: dt=1.0, NMDA off). Regions as slices, internal wiring preserved,
   Python orchestration unchanged. **Gate:** the 10 on-brain tests pass on the merged bridge (no capability
   regression). This alone is a real milestone: two of the three functional regions are one brain.
2. **The parser→composer hand-off as the gated synaptic route** (§4). **Gate:** `hear()` comprehends + stores via the
   spiking gate (the `{role: word}` Python hand-off removed), tests still green.
3. **The dlPFC region** — the hard one (dt/NMDA). Sub-options to evaluate then: (a) re-tune the dlPFC loop to dt=1.0
   and merge; (b) keep the association graph as a synaptic structure on the shared bridge with a readout; (c) honest
   boundary — if dt=0.5 bistability cannot survive dt=1.0, document that the dlPFC stays a separate-timing region
   (a real finding about why working-memory timescales differ from binding timescales). **Gate:** `elaborate()` works,
   or the boundary is documented.
4. **(Future) Approach 2** continuous flow.

## 6. Honest risks

- **Tuning perturbation.** The composer's coincidence bind is tuned (W_COINC=320, bias, drives) on its standalone
  bridge; a shared bridge means shared OU noise + shared step loop. Mitigation: per-region OU/reset; re-run the
  capability matrix as the gate. A regression is the measured cost (reported, not hidden).
- **Sequenced-reset interference** (§2.1) — fixed by per-region reset (zero the region's slice, not the whole array).
- **The dlPFC dt wall** (§5.3) — may be a genuine boundary; that itself is a biology-translatable finding (binding
  vs working-memory timescales).
- **Scale:** ~7K neurons + cross-region wiring — trivial for the GPU.
- **Effort:** multi-week; the increment means every step is independently valuable and the work can pause at any
  green gate.

## 7. Decisions for owner approval

1. **Approach 1** (time-multiplexed one-bridge, recommended) vs **Approach 2** (continuous flow) first? → I recommend 1.
2. **Incremental** order — parser+composer first (compatible), dlPFC last (the dt-wall)? → I recommend yes.
3. **dlPFC scope** — attempt the dt=1.0 re-tune (step 3a), or accept "dlPFC stays separate-timing" as an honest
   boundary if the bistability doesn't survive? → I recommend: try the re-tune cheap-first, accept the boundary if it
   fails (don't break the validated dialogue planning to force a merge).
