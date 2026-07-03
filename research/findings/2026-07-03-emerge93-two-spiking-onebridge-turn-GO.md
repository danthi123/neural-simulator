# EMERGE-93 — RUNG A.2 toward the one-brain capstone: the two SPIKING components (composer + producer) fold onto ONE bridge — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge93_two_spiking_onebridge_turn_derisk.py`
**Test:** `tests/test_emerge93_two_spiking_onebridge_turn.py`
**Raw:** `research/findings/raw/_emerge93_two_spiking_onebridge_turn.json`

## Why (folding the capstone's two spiking bridges into one)

The EMERGE-90/91 capstone was honestly THREE separate substrates (the adversarial-verify MAJOR-1). RUNG A.1 (EMERGE-92)
proved the producer runs as a `slots` slice on a shared bridge. RUNG A.2 folds the capstone's TWO **spiking** bridges —
the RF-phasor composer (memory) and the Izhikevich slot producer (production) — onto ONE `SimulationBridge` as disjoint
slices, and runs the whole EMERGE-90 conversational turn against the shared slices. (The rate reservoir has no bridge,
so this is the honest "fold the two spiking bridges into one" — the composer + producer are no longer separate
substrates.)

## The mechanism (both co-residence realizations, on one bridge)

ONE bridge (dt=1.0) hosts an `rf` region (composer, sized `2*(K+3)*D` for the batched store-scan) + a `slots` region
(producer) + the inert `_anchor`, disjoint, `region_pathways=[]`. The composer is a `MergedRFComposer` (masked RF ops on
the `rf` slice — the step-2b co-residence); the producer is a `RegistryProducer(shared_bridge=..., slot_region="slots")`
(the EMERGE-92 slice). dt=1.0 works for both: the composer's RF ops are a **dt-agnostic separate loop**
(`rf_resonate_steps`), and the producer is dt=1.0. The composer's facts live in the complex RF synapses
(`cp_rf_w_re/im`, array-disjoint from `cp_connections` + NOT in the producer's wash-out snapshot), so the producer's
inter-utterance wash-out does **not** disturb the composer's memory.

## The de-risk — **GO** (6 seeds; reuse EMERGE-72/74 + MergedRFComposer + EMERGE-88; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **parse** — reservoir comprehends the heard transitive | **1.000** | ≥ 0.90 |
| **recall** — the composer (on the `rf` slice) recalls the patient | **1.000** | ≥ 0.90 |
| **render_exact** — the producer (on the `slots` slice) SPEAKS the answer — both on ONE bridge | **1.000** | ≥ 0.90 |
| **no-confab MOAT** — unstored → false-accept / producer-invocations-on-abstain (gate-first) | **0.000 / 0** | ≤ 0.05 / == 0 |
| **comprehension-lesion** — reservoir collapsed → render collapses | **0.000** | ≤ 0.30 |
| **producer-no-learn** — learned spiking order removed → spoken order collapses | **0.000** (≈ chance) | ≤ 0.60 |

*(seed 42 confirmed: parse/recall/render 1.000, moat 0/0, lesion 0.000, no-learn 0.000; the 6-seed aggregate is in the
raw json.)*

**The result:** the capstone's memory (RF composer) and production (Izhikevich producer) now co-reside as disjoint
slices on ONE `SimulationBridge`, and the whole turn — HEAR → comprehend → store (on the `rf` slice) → ASK → recall →
SPEAK (on the `slots` slice) — reproduces the EMERGE-90 separate-bridge result (parse/recall/render 1.000), with the
no-confab moat holding **gate-first** on the shared bridge (0 false-accepts, the producer never invoked on abstain) and
both halves load-bearing. Co-location changes nothing — the functional-isolation property (the two slices are
index-disjoint with no cross-region pathways) holds under the genuinely-active co-resident partner.

## Anti-cheats (all pass)

- **the shared-bridge turn == the EMERGE-90 separate-bridge turn** (parse/recall/render 1.000) — folding the two
  spiking bridges onto one does not change any result (functional isolation).
- **gate-first no-confab moat** — the composer on the `rf` slice abstains on unstored queries (0 false-accepts), and the
  producer is never invoked on an abstain (`production_count` unchanged) — the moat holds on the shared bridge.
- **comprehension-lesion** collapses the render (0.000); **producer-no-learn** collapses the spoken order — both halves
  are load-bearing even co-resident.

## Honest scope (the remaining ladder)

- This folds the **two SPIKING bridges** (composer + producer) into one. The **comprehension reservoir is the RATE
  reservoir** (no bridge) here — the EMERGE-91 spiking `OnBridgeLSM` is a THIRD bridge at dt=0.5, so folding it in too
  (all three on one bridge) is **RUNG A.3**, which must reconcile the reservoir's dt=0.5 with the producer's dt=1.0
  (cheap-first: probe whether `OnBridgeLSM` parses at dt=1.0). The words remain host-token (spiking ORDER); the A→W
  neural spell is the separate purity rung.
- **RUNG B (synaptic hand-offs)** — the hand-offs here are still host-dict (the composer's recalled string → the
  producer's decision); making them synaptic is the deferred multi-week follow-on.
- Reuse-by-import (MergedRFComposer + EMERGE-72/74 producer + EMERGE-88 comprehender); NO `sim/` edit.

## Files
- `research/runners/_emerge93_two_spiking_onebridge_turn_derisk.py` — the shared-bridge (rf + slots) turn.
- `tests/test_emerge93_two_spiking_onebridge_turn.py` — 2 CPU tests.
- `research/findings/raw/_emerge93_two_spiking_onebridge_turn.json` — the 6-seed fold.
