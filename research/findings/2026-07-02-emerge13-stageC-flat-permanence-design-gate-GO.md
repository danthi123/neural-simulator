# EMERGE-13 / rung-4 Stage C DESIGN GATE — GO (6/6 seeds): the Bouhadjar three-term learning rule self-organizes context-specific branch prediction on a FLAT `[post, pre]` PERMANENCE MATRIX (single conceptual segment per cell, dense cross-column potential pool, homeostatic allocation, NO segment lists, NO structural growth). ⇒ the Stage-C `sim/` realization is a clean flat-CSR permanence-update kernel over a pre-allocated coincidence potential pool + a WEIGHTED coincidence — NOT a per-segment port. Decisive, cheap-first, gates the protected-module kernel build.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge13_stageC_flat_permanence_derisk.py`. Reuse-by-import (EMERGE-9b task + floors); NO `sim/` edit; CPU/numpy; 6-seed 42/43/44/100/101/102.

## Why this gate exists
Stage B2 put the HTM Temporal Memory INFERENCE on the real bridge with the frozen EMERGE-9b connectivity loaded via `inject_explicit_wiring` (6-seed GO). Stage C is the last rung-4 piece: make the LEARNING run on the substrate too. But the bridge's `coincidence_detector` pathway is a flat CSR of per-`(pre,post)` synapses, while EMERGE-9b/9d use per-cell SEGMENT LISTS (a cell owns multiple distal segments for different contexts). Porting segment lists into a fixed-topology bridge pathway is hard; a flat CSR is the natural match. So the gating question: **does the three-term rule still self-organize context on a FLAT permanence matrix?** GO ⇒ the `sim/` kernel is simple (flat-CSR permanence dynamics); NO ⇒ multi-segment is genuinely required (a harder port). Either answer is decisive for the design.

## The flat realization tested
- `W[post, pre]` in [0,1] = permanence over cross-column potential synapses (no same-column, no self); CONNECTED = `W >= perm_conn`.
- **Prediction:** cell predictive iff its count of CONNECTED synapses from currently-active cells `>= act_th` (== the bridge coincidence with a connected-threshold; on the bridge the WEIGHTED coincidence — `c_drive = sum of active-synapse permanences` — approximates this).
- **Learning (Bouhadjar three-term, winner-based — the EMERGE-9b lesson):** potentiate `W[j, pre]` to prior WINNERS scaled by the per-cell dAP-rate homeostasis `hfac = 0.5 + 0.5*max(0, z* - z_j)`; depress the rest; grow from `p_init` toward prior winners; presynaptic-depress wrongly-predictive cells; per-cell `z` EMA.
- **Allocation without structural growth:** a burst with no matching connected segment -> the k FRESHEST cells (fewest wired synapses = the flat "committed" metric) become winners and potentiate toward the prior winners on their initially-sub-connected row -> distinct contexts land on distinct fresh cells. No segment lists, no runtime edge growth — exactly what a fixed-topology bridge pathway with a pre-allocated potential pool can do.

## Result — GO (6/6 seeds)
Overlapping sequences `[cue]+[shared middle L=4]+[branch]`, `n_seq=2`, `n_cells=16`, `act_th=3`, 80 epochs:
- **FLAT-3term branch 1.000** on all of 42/43/44/100/101/102 — the flat matrix self-organizes the same context-specific high-order prediction as EMERGE-9d's segment lists.
- **dAP-LESION 0.000** (prediction severed -> collapses; load-bearing).
- **untrained 0.000.**
- **>> Markov floor 0.500, >> chance 0.500.** No teacher.

## The one real bug (root-caused, not tuned)
First pass gave **0.000** (worse than chance = systematic context merge). Root cause: the homeostatic allocation keyed on `z`, but `z` only rises once a cell becomes PREDICTIVE (connected), so at cold-start `z=0` for every cell -> both contexts allocate onto the SAME first-k cells -> merge (exactly EMERGE-9's original failure). EMERGE-9d actually bootstraps allocation with a **committed-metric** (segment count), not z; the flat equivalent is the **wired-synapse count** (`W_wired_count`). With that, freshest = fewest-wired -> a second context lands on unwired cells -> disjoint -> 1.000. `z`-homeostasis MODULATES potentiation (`hfac`); it does not cold-start allocation. (Honest residual: the allocation is a host committed-metric bootstrap — as in 9d; a fully-neural allocation via homeostasis alone is the aspiration, not required for this rung.)

## Implication for the `sim/` Stage-C build (now de-risked)
The genuinely-new `sim/` piece is ONE additive/guarded fused `fused_htm_permanence_update` kernel (shaped like `fused_stdp_weight_update`) over a pre-allocated coincidence potential pool, plus a per-cell `z` EMA and a per-cell wired-count for allocation:
- per synapse `(pre, post)`: if `post` fired this step AND `pre` fired last step -> potentiate `W[post,pre] += lam_pot * hfac(post)`; if `pre` fired but not a winner-context -> depress; clamp [0,1].
- WEIGHTED coincidence (`cfg.coincidence_weighted_drive`) so `c_drive = sum of active-synapse permanences`, threshold calibrated to the connected-count `act_th`.
- default-off, byte-identical when off, byte-inertness test (the two-compartment-dAP precedent).
Then drive the SAME overlapping-sequence training loop on the bridge -> GO = the bridge self-organizes the branch prediction from scratch (== EMERGE-9d) -> **rung-4 COMPLETE**.

## Status of the rung-3 -> rung-4 arc
Rung-3 fully validated + fully spiking (EMERGE-9b/c/d). Rung-4: Stage A' (two-compartment dAP GO, `sim/`) -> Stage B2 (INFERENCE on-substrate GO, 6-seed) -> **Stage C DESIGN GATE (this, flat-permanence three-term == 9d, 6-seed GO)** -> Stage C build (the additive/guarded `fused_htm_permanence_update` kernel + on-bridge training loop) = rung-4 complete.

## Artifacts
`research/runners/_emerge13_stageC_flat_permanence_derisk.py`, `research/findings/raw/_emerge13_stageC_flat_permanence{,_6seed}.json`. Prior: `2026-07-02-emerge12-stageB2-bridge-tm-on-substrate-GO.md`, `2026-07-02-emerge10-stageAprime-two-compartment-dap-GO.md`.
