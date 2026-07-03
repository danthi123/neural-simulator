# EMERGE-82 — RUNG 2: the reservoir form→role mechanism runs ON THE SUBSTRATE (a recurrent Izhikevich region on a real SimulationBridge) — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge82_onbridge_lsm_derisk.py`
**Test:** `tests/test_emerge82_onbridge_lsm.py`
**Raw:** `research/findings/raw/_emerge82_onbridge_lsm.json`

## Why (the on-substrate realization)

EMERGE-80 ported the reservoir form→role mechanism to a spiking Izhikevich pool, but via a DIRECT numpy recurrent loop
(`W_rec @ spikes`). The fully-spiking-one-brain directive wants it ON the project's substrate. EMERGE-82 realizes the
liquid-state machine as a recurrent **`BrainRegion`** on a real **`SimulationBridge`**: the region's internal connectivity
(`internal_density`) IS the fixed-random reservoir recurrence, and it is driven through the bridge's real
`_run_one_simulation_step` — **conductance-based synapses** `g_syn·(V−E)`, the actual Izhikevich neuron model — so the
reservoir COMPOSES with the rest of the one brain instead of being a bolt-on numpy pool.

## The mechanism

`OnBridgeLSM` mirrors the EMERGE-78 `Reservoir` API (`final_state(U)`) so it **drops into the entire EMERGE-78 harness**
(construction generators, final-state slot read-out, governing-cue + symmetric-window baselines, anti-cheats) with only the
pool swapped for a bridge region. One recurrent `BrainRegion` (name="reservoir", Izhikevich, exc/inh mix, internal_density
0.1 → ~9k recurrent conductance synapses) is built via the brain-region framework; a fixed-random input projection drives
the region's `cp_external_input_current` per token (+ a tonic bias → fluctuation-driven LSM regime); the read-out feature =
the region's per-neuron spike-count over the whole sequence, from the bridge's real `cp_firing_states`. The bridge state
(v/u/conductances/STP/firing) is washed to its post-init snapshot before each sentence (EMERGE-61 mechanism) so every
sentence is an independent read.

## The de-risk — **GO** (6 seeds 42/43/44/100/101/102; the bridge runs its real step loop; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| the reservoir region is genuinely spiking (mean spikes/neuron) | **~2.3** | > 0.5 |
| **(A) CONSOLIDATION** — the on-bridge region learns the form→role map (train role acc) | **1.000** | ≥ 0.90 |
| **(B) NON-LOCAL** — on-bridge region on the relative-clause HEAD | **1.000** | ≥ 0.85 |
| — left-context governing-cue baseline | **0.500** (chance) | ≤ 0.65 |
| — symmetric ±2 window baseline | **0.500** (chance) | ≤ 0.65 |
| rel-head word-order scramble | ~0.34 (≈ chance) | collapse |
| **REGION-SILENCE lesion** (zero the input drive) — the read collapses | **0.500** (drop 0.500) | genuinely from spikes |
| hand labeler on the multi-arg shapes | 0.000 | — |

*(seed 42 confirmed: spikes 2.29, train 1.000, rel-head 1.000 / gov 0.500 / sym 0.500, scramble 0.344, silence 0.500, hand
0.000; remaining seeds consistent — see raw json.)*

**The result:** a recurrent Izhikevich region on a real `SimulationBridge` — driven through the bridge's own
conductance-synapse transmission and step loop — **learns the full form→role map** (train 1.000) via a ridge read-out over
its real `cp_firing_states` spike-counts, and **resolves the non-local relative-clause head** (1.000) where both fixed
baselines are at chance. The **region-silence lesion** (zeroing the input drive) collapses the read to chance, proving the
read-out is genuinely from the region's real spikes, not a static bias. The reservoir form→role mechanism therefore runs on
the ONE BRAIN's substrate, not just a standalone numpy pool.

## Honest scope

- The reservoir region is a **disjoint slice on its own bridge** here (RUNG 2 = "on the SimulationBridge substrate").
  Co-residence with the nav/conv regions on ONE shared bridge (the full one-brain merge) is the follow-on — and the merge
  pattern is already validated (`nav_conv_merged_bridge`).
- The bridge's conductance-synapse recurrence integrates the sequence differently than EMERGE-80's direct-current pool, yet
  reaches the same GO — the mechanism is robust to the substrate change.
- Reuse-by-import (EMERGE-78 harness + EMERGE-61 wash-out + the brain-region framework); NO `sim/` edit.

## The reservoir arc, complete on the substrate

EMERGE-78 (learned map) → EMERGE-79 (uncontingent non-local, rate) → EMERGE-80 (spiking Izhikevich pool) → EMERGE-81
(graded memory survives on spikes) → **EMERGE-82 (realized on the SimulationBridge substrate)**: the fronto-striatal
reservoir is a **spiking region on the one brain** replacing the hand form→role labeler — the anti-whack-a-mole RANK-1
mechanism, on the project's own substrate. Next: co-residence on the shared nav/conv bridge; then RANK-3 (bounded recursion)
for dependencies past the pool's memory depth.

## Files
- `research/runners/_emerge82_onbridge_lsm_derisk.py` — `OnBridgeLSM` (recurrent BrainRegion on a real SimulationBridge,
  EMERGE-78 API) + the de-risk (reuses the harness; region-silence lesion; spiking-activity check).
- `tests/test_emerge82_onbridge_lsm.py` — 3 CPU tests (harness composition, recurrent-region framework, a tiny-bridge
  spiking smoke).
- `research/findings/raw/_emerge82_onbridge_lsm.json` — the 6-seed de-risk.
