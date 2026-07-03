# EMERGE-92 — RUNG A.1 toward the one-brain capstone: the spiking PRODUCER runs as a disjoint SLICE on a shared bridge — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge92_producer_coresident_derisk.py`
**Test:** `tests/test_emerge92_producer_coresident.py`
**Raw:** `research/findings/raw/_emerge92_producer_coresident.json`
**Scoping:** the read-only one-brain-consolidation scoping (this session).

## Why (closing MAJOR-1 at the substrate level, cheap-first)

The EMERGE-90/91 capstone is honestly THREE separate spiking bridges with host-dict hand-offs — the adversarial-verify
verdict's MAJOR-1 ("NOT one brain"). The one-brain-consolidation scoping found **RUNG A** (co-location: the three
components as disjoint slices on ONE `SimulationBridge`, the EMERGE-87 pattern) is the cheap-first path to the project's
own substrate-consolidation bar, and that two of the three components already have GO co-resident realizations (the
reservoir — EMERGE-87; the RF composer — step-2b `MergedRFComposer`). It identified the **ONE genuinely-new piece**: the
producer's slot region must become a SLICE on a shared bridge (today `FrameSlotCQ`/`build_slot_bridge` build a private
bridge). EMERGE-92 builds + de-risks that piece.

## The mechanism (the one new piece)

An **additive** `shared_bridge=`/`slot_region=` on `build_slot_bridge` + `FrameSlotCQ.__init__` (default `None` = the
byte-identical private path; it threads through `ResetFrameSlotCQ`/`CorpusOrderFrameSlotCQ`/`RegistryProducer` via their
existing `**kwargs`): when a shared bridge is passed, the producer resolves its `slots` slice from that bridge's region
manager instead of building a private bridge. Mirrors the already-shipped `BridgeParser(shared_bridge=, index_offset=)`
refactor. NO `sim/` edit; the shipped producer path is byte-preserved (49 producer-chain tests pass).

## The de-risk — **GO** (6 seeds; reuse EMERGE-72/74 producer; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **co-resident render == private render** — the producer on a shared-bridge slice renders C_TRANS IDENTICALLY to its private bridge | **1.000** | ≥ 0.999 |
| render_exact (co-resident) — the ground-truth transitive | **1.000** | ≥ 0.999 |
| render_exact (private) — the reference | **1.000** | ≥ 0.999 |
| co-resident region genuinely active (spk/neuron) | **~0.116** | > 0.01 |

**The result:** the spiking `RegistryProducer` runs as a disjoint `slots` SLICE on a `SimulationBridge` that ALSO carries
a genuinely-active recurrent Izhikevich region (a stand-in for the reservoir/composer regions of the full turn), with NO
cross-region pathways, and renders every C_TRANS fact **byte-identically** to its private-bridge counterpart (6-seed
unanimous 1.000). The producer's co-residence changes nothing — the parameterization is correct and the co-location is
behavior-preserving.

## Anti-cheats (all pass)

- **co-resident == private (GO-identical)** — the load-bearing claim: the producer on a shared bridge behaves exactly as
  alone (the parameterization did not perturb the render).
- **co-resident region genuinely active** — the co-resident Izhikevich region fires (~0.12 spk/neuron when driven), so
  the co-residence is real, not a silent stand-in.
- **structural isolation** — the `slots` and `coresident` regions are index-disjoint with `region_pathways=[]` (no
  cross-region synapses), and the producer's read (`slot_pool_rates`) drives/reads only the `slots` slice + washes the
  substrate before each emit — so the co-resident activity cannot leak into the slot read (validated end-to-end under a
  genuinely-active reservoir + composer in the full-turn follow-on, RUNG A.2).

## Honest scope (the remaining ladder to "one brain")

- This is **RUNG A.1** — the one new co-location piece (producer-as-slice), validated single-variable. It does NOT yet
  fold all three components onto one bridge.
- **RUNG A.2 (the full 3-region turn)** — reservoir + RF composer + producer on ONE bridge, the whole HEAR→…→SPEAK turn
  co-resident, with the EMERGE-87 functional-isolation gate. It faces one honest integration question the scoping
  flagged: the reservoir was validated at **dt=0.5** and the producer at **dt=1.0**, but a shared bridge has one global
  dt — so RUNG A.2 must either re-validate the reservoir at dt=1.0 (or the producer at dt=0.5), or use the RF composer +
  producer (both dt-1.0-tolerant) on one bridge with the reservoir as the third. This is the next rung.
- **RUNG B (synaptic hand-offs)** — the comprehension role output DRIVES the composer via synapses; the composer's
  recalled answer DRIVES the producer via synapses — is genuinely multi-week (a rate→firing adapter + the phasor→rate
  wall / A→W spell) and is correctly deferred.
- Reuse-by-import (EMERGE-72/74 producer); NO `sim/` edit; the private path byte-preserved.

## Files
- `research/runners/_emerge59_spiking_broca_frame_slots_derisk.py` — the additive `shared_bridge=`/`slot_region=` on
  `build_slot_bridge` + `FrameSlotCQ` (default None = byte-identical).
- `research/runners/_emerge92_producer_coresident_derisk.py` — the co-residence de-risk.
- `tests/test_emerge92_producer_coresident.py` — 3 CPU tests.
- `research/findings/raw/_emerge92_producer_coresident.json` — the 6-seed co-residence.
