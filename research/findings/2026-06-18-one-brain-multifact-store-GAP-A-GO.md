# Roadmap phase 2, GAP A (step A1) — the MULTI-FACT store lives in synapses on ONE persistent bridge: GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** — uniform across K=8, 16, 32 (18/18 per K,
54/54 total). This is **GAP A** from the production scoping (`2026-06-18-production-one-brain-composer-scoping.md`),
the recommended FIRST cheap-first de-risk for the production `OneBrainComposer`: it reuses the validated
`_store_substrate` mechanism and needs no parser.

**The question.** The four prior de-risks store ONE fact in a resonate-and-fire (RF) register (`v`/`u` state) and query
it immediately — but a work-register reset erases that state. A real knowledge base holds MANY facts that persist
across turns AND across the per-operation work-register resets. The CYCLE-168 insight (from the fact-store-query
de-risk): **stored facts must live in SYNAPSES, not register state.** This de-risk builds that store and proves the
property.

**Runner:** `research/runners/_phaseB_onebrain_multifact_store_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_multifact_store_K{8,16,32}.json`

## Mechanism — K facts tiled into one persistent bridge's complex weights

One persistent bridge holds: work registers (the bind+bundle workspace), a STORE region of `K_max` tiled `(1+D)`
blocks (block `i` = one trigger neuron + a `D`-neuron readout), a query register, and `V` concept-score neurons.

- **STORE fact `i`:** reset the work registers; bind(agent) + bind(action) + bundle into the `acc` register (the
  already-GO step-2 chain, on-bridge); read `acc`'s composite phasor; **append** block `i`'s trigger→readout complex
  weights `(readout_i_k, trigger_i, composite_phasor[k])`. The composite now lives in the synaptic weights
  (`cp_rf_w_re`/`cp_rf_w_im`) — the Crawford–Eliasmith memory-in-weights — not in any register's `v`/`u`.
- **QUERY fact `i`, role `r` (on-substrate):** **zero ALL register state**, install the whole accumulated store, fire
  trigger `i` with a unit phasor → readout block `i` reconstructs composite `i` (every other block's trigger is at 0,
  so its readout stays 0 — per-block isolation); swap to the unbind synapse (readout_i → query register, conj role) →
  the query register recovers the role; swap to the cleanup synapse (query register → concept neurons, conj codebook)
  → the concept membranes are the matched-filter scores; argmax = the answer, max = the familiarity peak.

The store install **replaces** the bridge's complex CSR per op (RF weights are install-as-gate), so the host holds the
accumulated store-weight list and re-installs it for queries; the persistent thing is the **bridge** (neurons/arrays)
and the **fact values in those weights**, not a transient register phasor. Because a query zeroes ALL `v`/`u` first and
still recalls, the recall provably comes from the (re-installed) weights, not leftover register state — the
register-reset invariant, which is the GAP-A guarantee.

## Result — capacity sweep, 3 seeds × {D=64, D=128} per K

| facts K | recall == truth | recall == host oracle | store-block lesion (want collapse) | intact-after-lesion | moat clean-separation | stored peak / unused peak |
|---|---|---|---|---|---|---|
| 8  | **1.000** (6/6) | **1.000** | collapses (0.00) | 1.00 | 6/6 | ~1.3–3.0M / **0** |
| 16 | **1.000** (6/6) | **1.000** | collapses (0.00) | 1.00 | 6/6 | ~1.3–2.9M / **0** |
| 32 | **1.000** (6/6) | **1.000** | collapses (0.00) | 1.00 | 6/6 | ~1.3–2.9M / **0** |

Every one of the 18 configs per K (54 total across the sweep) is recall 1.00/1.00, lesion 0, intact 1, moat-sep 1, with
unused-block peaks exactly 0. The stored-fact peak is the SAME at K=8, 16, 32 (~1.3M at D=64, ~2.9M at D=128) — per-fact
recall is independent of how many facts are stored, the signature of per-block isolation (no cross-talk).

## Reading

- **The multi-fact store works on one persistent bridge:** K facts coexist in disjoint complex-weight blocks, each
  recalled exactly == the numpy `RFPhasorComposer` oracle (`_encode` + `_unbind_phases` + `_cleanup`).
- **Register-reset-safe (the GAP-A guarantee):** every query zeroes ALL register `v`/`u` first, yet recall is perfect —
  the facts are in synapses, not register state. A per-operation work-register reset (mandatory on a persistent bridge,
  CYCLE 168) cannot erase stored facts.
- **The store is load-bearing (lesion):** zeroing one fact's block weights collapses ONLY that fact's recall (its peak
  drops to the abstain floor), leaving the other facts intact — the store, not residual register state, carries recall.
- **The no-confab moat holds under the multi-fact store:** firing an UNUSED block (no weights) gives a peak of exactly
  0 (clean separation from stored facts' ~millions), so an absent fact abstains. The moat is preserved, not weakened.
- **Per-block tiling → no superposition cross-talk (confirmed to K=32):** the unused-block peak is exactly 0 (not a
  small leak) and the stored peak is unchanged across K=8/16/32, confirming the blocks are isolated. Unlike a single
  summed superposition (whose signal-to-noise ratio caps the fact count), the tiled store's capacity is bounded only by
  neuron count, not by matched-filter SNR — the sweep shows zero degradation from 8 to 32 facts, so the limit is the
  bridge size, not recall fidelity. (At very large K the per-bridge neuron budget is the cap; the validated 320-concept
  multi-bridge sharding route extends beyond it.)

## Honest scope + next

- The store-write reads `acc`'s composite phases to the host once to install the block weights (a store-time
  consolidation hop). The de-risk's load-bearing property is the multi-fact persistence + on-substrate query, NOT the
  elimination of the store-time host read; the all-synaptic store-write (drive the trigger's weights directly from
  `acc`'s firing) is a later refinement.
- This is the **per-fact-isolated** multi-fact store (GAP A). It is complementary to the step-3c **per-fact multi-role**
  coherence GO (a single composite carrying up to 4 bundled role-binds): together they cover both the across-fact and
  within-fact capacity of the persistent store.
- **Next (the production build):** GAP B — the **parser front-end** (drive the operand registers from the parser's role
  firing via gated routes, the `hear_synaptic`/`couple_gate_to_indices` precedent, adapted to drive an RF complex
  register), so comprehend→store→query→answer is one spiking flow with the host doing only text I/O. Then STEP A3 (wrap
  as a production `OneBrainComposer`, an `RFPhasorComposer` API-sibling), A4 (optional spiking winner-take-all), A5
  (make it the default + megakernel the persistent loop + retire legacy numpy from the runtime, keeping numpy as the
  test oracle).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_multifact_store_derisk \
    --seeds 42,43,44 --dims 64,128 --n-facts 8 --n-unused 3   # and --n-facts 16 / 32
```
