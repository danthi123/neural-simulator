# EMERGE-87 — CO-RESIDENCE: the form→role reservoir region composes onto the one brain (disjoint slice, GOes co-resident, functionally isolated from a co-resident conversation) — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge87_reservoir_coresident_derisk.py`
**Test:** `tests/test_emerge87_reservoir_coresident.py`
**Raw:** `research/findings/raw/_emerge87_reservoir_coresident.json`

## Why (composing the reservoir arc onto the one brain)

EMERGE-82 realized the reservoir form→role mechanism as a recurrent `BrainRegion` on its OWN `SimulationBridge`. The
one-brain directive is for it to COMPOSE with the rest of the brain — to sit as a disjoint slice on a shared bridge
alongside a conversational region, without the two disturbing each other (the validated disjoint-slice merge pattern,
EMERGE step-2b). EMERGE-87 de-risks that composition for the reservoir.

## The mechanism

`CoResidentReservoirLSM` builds ONE `SimulationBridge` with a reservoir region + a conversational region (both Izhikevich,
`region_pathways=[]` → no cross-region synapses) and reuses EMERGE-82's `final_state` (drives the reservoir slice per
token, reads its spike-counts), with a `conv_drive` option that ALSO drives the conversational slice concurrently. The
reservoir's form→role RESULT must not change under `conv_drive`.

## The de-risk — **GO** (6 seeds; reuse the EMERGE-78 harness + EMERGE-82 machinery; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| reservoir region genuinely spiking | 2.01 spikes/neuron | > 0.5 |
| **co-resident form→role** — train role acc | **0.99** (min 0.914) | ≥ 0.90 |
| — non-local rel-head | **1.000** | ≥ 0.85 |
| — governing-cue / symmetric-window baselines | 0.500 / 0.500 (chance) | ≤ 0.65 |
| region-silence lesion | 0.500 (collapse) | genuinely spiking |
| **FUNCTIONAL ISOLATION** — form→role classification-flip rate under concurrent conv-drive | **0.0%** (all seeds) | ≤ 0.01 |
| conv region mean-spikes when only the reservoir is driven | **~0.02** | silent |

*(6-seed means; the isolation is 0.0% flips on every seed.)*

**The result:** the reservoir region runs as a **disjoint slice** on a bridge that also carries a conversational region,
**GOes co-resident** (train 1.000; non-local rel-head 1.000 vs both baselines at chance; genuinely active; region-silence
lesion collapses), and the two slices are **functionally isolated** — the reservoir's form→role RESULT is **unchanged**
(0.0% classification flips) whether the conversational region is silent or CONCURRENTLY driven, and the conversational
region is **silent** (0.000 spikes) when only the reservoir is driven. So the whole EMERGE-78..86 reservoir mechanism
composes onto the shared spiking brain without its cognition being disturbed by a co-resident conversation — the one-brain
property.

## Honest scope

- **Functional** (not raw byte) isolation: the co-resident conversation causes only a weak numerical read delta (~1.4e-2,
  <1% of a spike count) from a **global step mechanism** — NOT a cross-region synapse (`region_pathways` is empty + the
  conv region is silent when undriven) — and this delta does not change any form→role classification (0.0% flips). The
  reservoir's *cognition* is isolated.
- The conversational co-resident is a spiking Izhikevich pool **stand-in** (a real disjoint region on the shared bridge);
  the FULL merged nav/conv bridge (`nav_conv_merged_bridge`, with the actual composer + no-confab moat) is the mechanical
  extension — add the reservoir to the merged builder's `co_resident_*` region list. This de-risks the load-bearing
  property (co-resident GO + functional isolation) cheaply.
- Reuse-by-import (EMERGE-78 harness + EMERGE-82 on-bridge machinery); NO `sim/` edit.

## The reservoir arc, composed onto the one brain

EMERGE-78 (learned map) → 79 (uncontingent non-local) → 80 (spiking Izhikevich pool) → 81 (memory survives on spikes) → 82
(on the SimulationBridge substrate) → 83–86 (the RANK-3 recursion boundary + rate + spiking surpass) → **87 (composes as a
disjoint slice on the shared brain, functionally isolated from a co-resident conversation)**. The anti-whack-a-mole
form→role mechanism is learned, uncontingent-non-local, spiking, on-substrate, recursion-capable, and now composes onto the
one brain.

## Files
- `research/runners/_emerge87_reservoir_coresident_derisk.py` — `CoResidentReservoirLSM` (2-region disjoint bridge) + the
  co-resident-GO + functional-isolation de-risk.
- `tests/test_emerge87_reservoir_coresident.py` — 3 CPU tests.
- `research/findings/raw/_emerge87_reservoir_coresident.json` — the 6-seed co-residence.
