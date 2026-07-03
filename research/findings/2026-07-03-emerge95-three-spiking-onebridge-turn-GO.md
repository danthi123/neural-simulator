# EMERGE-95 — RUNG A.3: ALL THREE spiking components on ONE bridge — the one-brain SUBSTRATE for the conversational turn — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge95_three_spiking_onebridge_turn_derisk.py`
**Test:** `tests/test_emerge95_three_spiking_onebridge_turn.py`
**Raw:** `research/findings/raw/_emerge95_three_spiking_onebridge_turn.json`

## Why (closing MAJOR-1 at the substrate level)

The EMERGE-90/91 capstone was honestly THREE separate spiking bridges with host-dict hand-offs — the adversarial-verify
MAJOR-1 ("NOT one brain"). The consolidation ladder: RUNG A.1 (EMERGE-92, producer on a shared bridge) → RUNG A.2
(EMERGE-93, composer + producer on ONE bridge) → EMERGE-94 (the spiking reservoir parses at dt=1.0) → **RUNG A.3
(EMERGE-95): all three spiking components on ONE bridge.** This meets the project's own substrate-consolidation bar
(`project_one_brain_substrate_vs_functional`): every spiking component of the conversational turn lives on ONE
`SimulationBridge`.

## The mechanism (three spiking slices, one bridge)

ONE `SimulationBridge` (dt=1.0) hosts `reservoir` (recurrent Izhikevich, comprehension) + `rf` (RF-phasor composer,
memory) + `slots` (Izhikevich producer, production) + the inert `_anchor`, as disjoint slices with `region_pathways=[]`.

- **Reservoir** — `SharedBridgeReservoirLSM` rebinds `OnBridgeLSM`'s `bridge`/`res_idx`/`W_in`/`_snap` to the shared
  bridge's `reservoir` slice; `final_state` drives/reads that slice unchanged (the EMERGE-87 pattern, dt=1.0 per
  EMERGE-94).
- **Composer** — `MergedRFComposer` (masked RF ops on the `rf` slice; the RF ops are a dt-agnostic separate loop).
- **Producer** — `RegistryProducer(shared_bridge=..., slot_region="slots")` (the EMERGE-92 slice).

Each component's wash-out restores the per-neuron state arrays (v/u/conductances/firing/STP) but NOT the composer's
complex RF synapses (`cp_rf_w_re/im`), so the reservoir's + producer's washes never disturb the composer's stored
memory. The three slices are index-disjoint with no cross-region synapses → functional isolation by construction.

## The de-risk — **GO** (6 seeds; reuse EMERGE-82/87/88/92/93 + MergedRFComposer; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **parse** — reservoir (slice) comprehends the heard transitive | **1.000** | ≥ 0.90 |
| **recall** — composer (slice) recalls the patient | **1.000** | ≥ 0.90 |
| **render_exact** — producer (slice) SPEAKS the answer — all three on ONE bridge | **1.000** | ≥ 0.90 |
| **no-confab MOAT** — unstored → false-accept / producer-invocations-on-abstain (gate-first) | **0.000 / 0** | ≤ 0.05 / == 0 |
| **comprehension-lesion** — reservoir collapsed → render collapses | **0.000** | ≤ 0.30 |
| **producer-no-learn** — learned spiking order removed → spoken order collapses | **0.000** | ≤ 0.60 |

*(seed 42 confirmed: parse/recall/render 1.000, moat 0/0, lesion 0.000, no-learn 0.000; the 6-seed aggregate is in the
raw json.)*

**The result:** every spiking component of the conversational turn — the reservoir that comprehends, the composer that
remembers, the producer that speaks — now co-resides on ONE `SimulationBridge`, and the whole turn (HEAR → comprehend →
store → ASK → recall → SPEAK) runs against the three shared slices, reproducing the EMERGE-90 separate-bridge result
(parse/recall/render 1.000) with the no-confab moat holding gate-first and both halves load-bearing. **The one-brain
SUBSTRATE for the conversational turn is achieved** — the deepest MAJOR-1 gap (three separate substrates) closed at the
substrate level.

## Honest scope (what "one brain" means here, precisely)

- This is the **one-brain SUBSTRATE** (co-location on ONE bridge, the project's `project_one_brain_substrate_vs_functional`
  bar for *substrate consolidation*): every spiking component is a disjoint slice on one `SimulationBridge`. It is NOT
  yet **functional** integration — the hand-offs are still **host-dict** (the reservoir's parsed roles → a python dict →
  the composer's `store`; the composer's recalled string → the producer's `decision`). Making those hand-offs
  **synaptic** (the comprehension role output DRIVES the composer via synapses; the recalled answer DRIVES the producer)
  is **RUNG B** — a genuine multi-week arc (a rate→firing adapter + the phasor→rate wall / A→W spell), correctly
  deferred. So: one substrate now; synaptic interaction next.
- The **word SURFACES are still host-token** (spiking ORDER; the A→W neural spell is the fully-spiking-words follow-on).
- Reuse-by-import (EMERGE-82/87/88/92/93 + MergedRFComposer); NO `sim/` edit; the shipped standalone paths byte-preserved.

## The one-brain-substrate ladder, complete

RUNG A.1 (producer on a shared bridge, EMERGE-92) → RUNG A.2 (composer + producer on one bridge, EMERGE-93) → the dt
probe (EMERGE-94) → **RUNG A.3 (all three spiking components on one bridge, EMERGE-95)**. The conversational turn's
substrate is consolidated onto one brain; RUNG B (synaptic hand-offs) is the deferred functional-integration arc.

## Files
- `research/runners/_emerge95_three_spiking_onebridge_turn_derisk.py` — `SharedBridgeReservoirLSM` + the 3-slice
  shared-bridge turn.
- `tests/test_emerge95_three_spiking_onebridge_turn.py` — 3 CPU tests.
- `research/findings/raw/_emerge95_three_spiking_onebridge_turn.json` — the 6-seed one-brain-substrate turn.
