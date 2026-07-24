# DR-3 (new-direction Phase-0/1): a functional SELF-SCHEMA region — the brain reads + reports its own internal state, 6-seed GO on spikes (2026-07-23)

Second validated faculty of the owner's reframe (toward self-awareness). A Graziano Attention-Schema + higher-order
metacognition SELF region on ONE spiking `SimulationBridge`, reuse-by-import, NO `sim/` edit (verified). Three regions:
a `workspace` (real spiking GNW occupancy — K=4 content assemblies + shared inhibition, one ignites at a time) and a
`self_schema` region with three disjoint sub-blocks each reading ONE of the brain's OWN internal signals:
`attend[k]` (which workspace content is broadcast), `confid` (graded ∝ the real Bogacz-Brown familiarity confidence),
`author` (self-vs-heard). Its firing jointly represents + reports "what I'm attending to / how sure I am / whether I
authored it."

## 6-seed result — GO (all checks; seeds 42/43/44/100/101/102, on-bridge)
- **attention** acc **0.974** (chance 0.25) — reports which content is in the workspace.
- **confidence** Spearman **+0.980**, AUC **0.996** — monotone in the true familiarity-derived confidence.
- **authorship** acc **1.000** (chance 0.5).
- **self-lesion** (sever the schema's access; brain state unchanged) → all readouts collapse to chance (6/6) — it reads
  the REAL internal state, not a confabulation.
- **shuffled-internal-signal** → collapse (6/6). **state-not-content dissociation** |corr|<0.16 (6/6) — the schema is
  about the STATE, not the topic. **familiarity-gate lesion** flattens the confidence axis (6/6).
Runner `research/runners/_self_schema_region_derisk.py`; result `research/findings/raw/_self_schema_6seed.json`.

## Rigor
The first smoke exposed a real bug — a synchronous confidence pool (coarse ~4-level rate → tie-fragile hi/lo read
chance despite Spearman +0.76); fixed with per-neuron heterogeneity (desynchronizes into a graded rate code; verified
it doesn't break the workspace attractor) + a tie-robust AUC. Then GO.

## Honesty (load-bearing)
Framed throughout as a FUNCTIONAL self-model correlate — a region that reads + reports the brain's own internal-state
axes — explicitly NOT a claim of subjective experience / phenomenal consciousness (the emergent bet; arguably
untestable). Follow-ons: workspace-routed deliberation (P1.2); a spiking familiarity->confidence route (vs the graded
current). NO `sim/` edit.
