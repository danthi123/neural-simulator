# gap#5 (ii) emergent-DG — the E%-max FEEDFORWARD ca3 inhibition (robustness fix) is DOWNSTREAM of the amplification: on the DEFAULT runner config (ca3w=1.5, train=False, coincidence=False) NOTHING fires (sizes [0,0,0,0] feedforward OR feedback), so there is no firing to sparsify. The emergent-DG is a 3-LAYER problem; my ff-inhibition build is correct but premature without the amplification config. Honest scoping correction (I mis-tested layer 3 on the layer-1 raw config).

**2026-07-18.** Built the E%-max feedforward ca3 inhibition (`--ca3-ff`, a `ca3_ff_basket` driven by the DG afferent —
the divisive-normalization robustness fix for the emergent-DG's knife-edge amplification) and tested it on the
`_gap5_emergent_dg_selection_derisk` runner. Result corrects the scoping.

## Result (seed 42, n_ca3=400, the runner's default build)
| config | ca3 sizes | mean | sparsity | stability | sep_cos | moat | mossy-lesion |
|---|---|---|---|---|---|---|---|
| FEEDBACK-only (baseline) | [0,0,0,0] | 0.0 | 0.000 | 0.00 | 0.000 | 0 | 0 |
| FEEDFORWARD ff=10 | [0,0,0,0] | 0.0 | 0.000 | 0.00 | 0.000 | 0 | 0 |
| FEEDFORWARD ff=20 | [0,0,0,0] | 0.0 | 0.000 | 0.00 | 0.000 | 0 | 0 |

**Nothing fires in ANY config.** The ff-inhibition can't help because there is no CA3 firing to sparsify — it addresses
robustness of amplification, but the default runner has no amplification.

## Why: the emergent-DG is a 3-LAYER problem, and the runner's default is at LAYER-1 (the raw R0 boundary)
Re-reading `2026-07-18-gap5-emergent-DG-R0-...-BOUNDARY` precisely:
1. **Layer 1 — mossy→CA3 propagation (the raw R0 boundary):** the DG granule cells fire sparse single spikes (not
   bursts / detonator synapses), so the mossy volley alone drives ~0 CA3 cells. The runner's `_build_bridge` uses
   `ca3w=1.5, train=False, coincidence=False` → this raw regime → **0 firing** (what my test hit).
2. **Layer 2 — amplification (works, but was a SEPARATE probe):** the finding's assembly that fired 15-26 cells used
   `train=True, coincidence=True` (the dendritic-plateau read), a MODERATE recurrent `ca3w≈4`, a SYNCHRONIZED
   gamma-pulsed DG volley (2-3 on / 2-4 off), + the bistability keystone → the recurrent AMPLIFIES the mossy seed into
   a sparse separated assembly (input-11 → 15-26 cells, sep_cos 0.10-0.20). This is NOT in the runner's default `run()`.
3. **Layer 3 — robustness (my ff-inhibition build):** the amplification is a knife-edge (some inputs seed enough, some
   don't; too-strong `ca3w=5` saturates to all 2000). The E%-max feedforward inhibition makes the sparsity reliable
   across inputs — BUT only once layer-2 amplification produces firing.

## Status (honest)
- **My ff-inhibition build (`--ca3-ff`, `ca3_ff_basket`) is CORRECT + committed** (additive/default-None/byte-identical,
  4d3a2fee) — it is the right layer-3 mechanism. It is simply DOWNSTREAM of the amplification the default runner lacks.
- **I mis-scoped the TEST** (ran layer-3 on the layer-1 raw config → 0 firing → no signal). Correction recorded.
- **The genuine emergent-DG next build:** wire the LAYER-2 amplification config into `_gap5_emergent_dg_selection_derisk`
  (`train=True, coincidence=True, ca3w≈4`, a synchronized gamma-pulsed DG volley in `_select`, + the bistability
  keystone) so CA3 fires, THEN test the ff-inhibition for RELIABLE amplification across all inputs (vs the knife-edge
  feedback). A multi-layer build — the emergent-DG is genuinely the hardest gap#5 sub-item.
- **⇒ within gap#5, the SWR generative-replay (item i) is the MORE TRACTABLE priority** — it reads on the CLOSED
  completion (no amplification needed), whereas the emergent-DG (item ii) requires this layer-2 amplification build
  first. Do SWR replay FIRST when the GPU frees; the emergent-DG amplification+ff build follows.
