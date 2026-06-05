# Conversion Phase 4 (cheat D, association graph) — associative memory is SUBSTRATE-GENUINE; weight-SOURCE is the residual — 2026-06-05

Fourth/final phase of the cheat-conversion plan (`docs/plans/2026-06-05-conversational-cheat-conversion-plan.md`). The
audit flagged cheat D as "the dlPFC association graph is a Python dict from the kb." On inspection of the actual
dialogue-planning substrate (`research/runners/content_selection_spiking.py`), the picture is more favourable than the
one-line audit:

## The dlPFC associative memory is GENUINELY on the substrate (verified — not a cheat)
The dlPFC (`SpikingSpreadingController` / `SpikingLoopContextBuffer`) builds a real `SimulationBridge` with:
- **Concept ASSEMBLIES** — each concept = a 50-neuron firing assembly (`_cpat`/`_dpat`, `pattern_size=50`) in the
  cortex_ctx + dlpfc_wm regions (`build_loop_wm_bridge`). So concepts ARE firing pools on the bridge — the
  codes-vs-pools tension does not bite here.
- **Association SYNAPSES** — the c2d / d2c pathways + per-concept attractors are real bridge synapses
  (`set_pathway_weights("c2d", ...)`, line 205).
- **Spiking SPREAD** — `update()`/`read()` drive the assemblies and read the sustained firing
  (`_run_one_simulation_step`, `cp_firing_states`), decoding active concepts by cosine to the assembly patterns.

The spreading-activation + working-memory hold are GENUINE spiking computation on the bridge (multi-seed validated in
this module's prior findings: 220× concept specificity, ≥3-concept WM span). This is NOT a cheat.

## The residual cheat: the association weights are SET (outer-product), not LEARNED
The one non-biological residual is that the association/attractor weights are **set by an outer-product from the
co-occurrence graph** (line 204, `ww = full(..., attractor_weight)`), and that co-occurrence graph is the Python
recompute (`_assoc_graph`: concept→{concept: weight} from the kb facts). The module **already documents this exact
residual** (line 172, verbatim): *"Attractor weights are SET here (outer-product); learning them with the correct
rule is the documented next step."*

## The conversion (designed, buildable follow-on — NOT a fundamental boundary)
Hebbian-learn the association weights instead of setting them: build the dlPFC bridge eagerly; when a fact is stored,
drive its concept assemblies together for a window with Hebbian plasticity ON → the cross-assembly synapses (c2d)
strengthen for co-occurring concepts (Marr/Treves-Rolls CA3 autoassociation; Garagnani-Pulvermüller emergent spread).
The infrastructure exists (assemblies + the c2d pathway + `_run_one_simulation_step` + the bridge's Hebbian). The
de-risk: Hebbian-learned c2d reproduces the set-from-graph associates (the dlPFC picks the same top associate),
multi-seed. This is a real build but bounded — the assemblies and synapses are already there; the change is the
weight SOURCE (Python outer-product → Hebbian co-firing at store).

## The deeper BOUNDARY (measured, disclosed)
Cue-only associative recall (drive one concept, expect its associate) is **~27.5% multi-seed, barely above chance**
(`2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`) — the heteroassociative asymmetry: clean cue completion
needs sparse codes (capacity ∝ recurrent-synapses / sparseness). Co-STIMULATION (both concepts) reactivates both at
87.5%; the cue→associate DIRECTION is the boundary. The principled lever is SWR sleep-replay consolidation, not weight
hand-tuning. So even with Hebbian-learned weights, robust cue-direction recall is the named boundary.

## Honest status of cheat D
- The associative MEMORY (concept assemblies + association synapses) and the SPREAD are **substrate-genuine** — the
  bulk of D is already on the bridge, not a Python structure.
- The residual is the association-weight SOURCE (Python co-occurrence outer-product → Hebbian-learned at store) — a
  **designed, buildable follow-on** (the project's own documented next step), NOT a fundamental boundary.
- Robust cue-DIRECTION recall (27.5%) is the deeper measured boundary, addressable in principle by SWR consolidation.

## Artifact
No new code this phase — an honest substrate audit of `content_selection_spiking.py` (the dlPFC). The finding
reclassifies D from "Python associative memory" to "substrate-genuine associative memory with a documented
weight-source residual + a measured cue-direction boundary." NO sim/ edits.
