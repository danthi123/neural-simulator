# EMERGE-14 / rung-4 Stage C — GO (6/6 seeds UNANIMOUS): the whole unsupervised HTM Temporal-Memory sequence-LEARNING mechanism runs on the real `SimulationBridge`. The permanences LIVE in the bridge's `coincidence_detector` synapse weights (`cp_connections.data`) and are LEARNED FROM SCRATCH by the committed `sim/` `fused_htm_permanence_update` kernel (the Bouhadjar three-term rule) over a pre-allocated potential pool; PREDICTION is the bridge's OWN weighted coincidence recurrence (the dAP-lesion collapses it → load-bearing). ⇒ **rung-4 COMPLETE** — the single-spiking-substrate realization of the emergent, self-organizing sequence-memory cortex (INFERENCE + LEARNING), no teacher, multi-seed.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge14_stageC_onbridge_learning_derisk.py`. Reuse-by-import EMERGE-9b/12/13 + the committed `sim.kernels.fused_htm_permanence_update` + Stage-B2 `_prime_from_winners`. CPU numpy-backend; 6-seed 42/43/44/100/101/102. Supersedes the WIP finding `2026-07-02-emerge14-stageC-onbridge-learning-WIP-build-informative.md`.

## Result — GO (6/6 seeds)
Overlapping sequences `[cue]+[shared middle L=4]+[branch]`, `n_seq=2`, `n_cells=16`, `act_th=3`, 40 epochs, learned from scratch on the bridge:
- **ON-BRIDGE-LEARN branch 1.000** on all of 42/43/44/100/101/102 — the sim/ kernel self-organizes context-specific high-order prediction (cue0 → col6, cue1 → col7) in the bridge's own synapse weights.
- **dAP-LESION 0.000** — coincidence off → the bridge's prediction recurrence is severed → collapses. The on-substrate coincidence mechanism is LOAD-BEARING (a valid lesion control, unlike the earlier host-readout WIP).
- **untrained 0.000.**
- **>> Markov floor 0.500, >> chance 0.500.** No teacher.

## What is on-substrate (faithful) vs host (the acknowledged EMERGE-9d residual)
- **PERMANENCES** live in `cp_connections.data` (the bridge's synaptic weights) over a dense cross-column potential pool pre-injected as a `coincidence_detector` pathway at weight 0. ON-SUBSTRATE.
- **LEARNING** = the committed `sim/` `fused_htm_permanence_update` kernel applied to `cp_connections.data` each symbol (per-synapse `pre_last`/`post_now`/`hfac_post` gathered from the cached COO, exactly like `fused_stdp_weight_update`): causal potentiation to prior winners × dAP-rate homeostasis + presynaptic depression. Grows the right synapses from 0 to connected; depresses the rest. ON-SUBSTRATE.
- **PREDICTION** = the bridge's WEIGHTED coincidence recurrence (`c_drive = sum of active-synapse permanences`) via Stage-B2 `_prime_from_winners` (prior winners → `cp_prev_firing_states` → step → the primed apical-dAP cells). ON-SUBSTRATE (the lesion severs it).
- **WINNER SELECTION + committed-metric ALLOCATION** (fresh cells for a new context) = host-orchestrated, as in EMERGE-9d. The finding flags this residual; a fully-neural homeostatic allocation is the aspiration, not this rung.

## The debugging arc (systematic; root-caused not tuned)
1. **Dense-pool matching/allocation** — EMERGE-13's `perm>0` metrics are full for every cell on a pre-allocated pool → nothing allocates. Fix: matching/prediction use CONNECTED (`perm>=perm_conn`).
2. **Allocation RACE** — connected-count differentiates only after a cell connects (several epochs) → epoch-0 context merge (branch 0.000). Fix: the dense-pool "committed" metric = incoming perm ABOVE `p_init` (differentiates after ONE potentiation). → 0.500 (but host-readout, lesion didn't collapse).
3. **Host prediction → not faithful + lesion didn't collapse.** Reworked prediction through the bridge's weighted coincidence (`_prime_from_winners`). → 0.000 (the dense pool over-primed).
4. **Dense-pool background priming** — at `p_init=0.24`, EVERY cell gets `c_drive ≈ k_win*0.24 ≈ 0.96` from any active set, and 6-step sustained priming pushes all over threshold → everything primed (an isolated known-edge probe returned all columns). Fix: **`p_init=0.0`** — the pool is pre-allocated at weight 0, so the background `c_drive` is 0 and ONLY potentiated (learned) synapses contribute; the kernel grows synapses from 0. The isolated probe then returned EXACTLY the connected block; the full run → **1.000, lesion 0.000.**

## Status of the rung-3 → rung-4 arc — rung-4 COMPLETE
Rung-3 fully validated + fully spiking (EMERGE-9b discrete / 9c spiking-inference / 9d spiking-learning; capacity to 32 contexts). Rung-4 (the `sim/` two-compartment TM port): Stage A' (two-compartment dAP GO, `sim/`) → Stage B2 (INFERENCE on-substrate GO, 6-seed) → Stage C design gate (flat-permanence three-term == 9d, 6-seed GO) → Stage C kernel + config (committed, byte-inert) → **Stage C on-bridge LEARNING GO (this, 6-seed) = rung-4 COMPLETE.** The whole unsupervised, self-organizing, teacher-free HTM Temporal-Memory sequence cortex — inference + learning — runs on the real spiking `SimulationBridge`.

## Next frontier (rung-4 done → toward the language-sequence cortex)
- **Capacity/scale on-substrate:** the numpy TM scaled to 32 overlapping contexts (1.000); confirm the on-bridge learned TM holds at higher `n_seq`/longer middles (a scale run, not a new mechanism).
- **Autonomous step-loop learning:** wire `fused_htm_permanence_update` into `_run_one_simulation_step` behind the (already-added, byte-inert) `enable_htm_learning` flag + `cp_htm_z`, so the bridge learns AUTONOMOUSLY as it runs (the fully-on-substrate refinement; the per-symbol vs per-step STDP-window timing is the design point).
- **Close the host residuals:** fully-neural winner-selection (the per-column WTA, Stage-B1) + a fully-neural homeostatic allocation (replacing the committed-metric bootstrap).
- **Toward language:** this emergent sequence-memory primitive is the substrate for a simulated recurrent sequence/language cortex — scale toward real symbol sequences + connect to the communication goal.

## Artifacts
`research/runners/_emerge14_stageC_onbridge_learning_derisk.py`, `research/findings/raw/_emerge14_stageC_onbridge_learning{,_6seed}.json`, `sim/kernels.py` (`fused_htm_permanence_update`), `sim/config.py` + `sim/bridge.py` (`enable_htm_learning` + `cp_htm_z`, byte-inert). Prior: `2026-07-02-emerge13-stageC-flat-permanence-design-gate-GO.md`, `2026-07-02-emerge12-stageB2-bridge-tm-on-substrate-GO.md`, `2026-07-02-emerge10-stageAprime-two-compartment-dap-GO.md`.
