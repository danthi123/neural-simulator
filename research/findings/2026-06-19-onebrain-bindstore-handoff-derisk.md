# Tier-2 Phase A — the bind→store (H4) hand-off is SYNAPTIC on the persistent OneBrainComposer loop: GO

**Date:** 2026-06-19 (the Tier-2 persistent-integrated-loop arc; the owner's "real one brain" —
`project_one_brain_integrated_pipeline_and_cleanup`)
**Status:** **GO** (3 seeds × 2 D = 6/6 rows, every check unanimous). The bind→store DATA hand-off (H4) — today a host
read-then-write of the composite — is now a **register→register synaptic route** (`acc → store-block-readout` through a
unit complex synapse) on the production `OneBrainComposer` persistent bridge, with **NO `rf_read_phases()[acc]` and NO
host re-encode** of the bind output. A fact stored via the synaptic H4 recalls **IDENTICALLY** to the stock host store,
the severed-route lesion **collapses** recall, the permuted-store write **carries the content**, and the no-confab moat
**holds (0 breaches)**.
**Runner:** `research/runners/_phaseB_onebrain_bindstore_handoff_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_bindstore_handoff.json`
**Pre-registration:** `2026-06-19-tier2-persistent-integrated-loop-scoping.md` (commit `f1b551db`), Phase A — the
cheapest-first of the H0–H12 round-trip map. (Phase B — the on-substrate SEQUENCER — is OUT of scope here; the
controller takes it up separately.)

## The H4 host round-trip this removes

`OneBrainComposer` is already a persistent co-resident bridge with the FHRR bind/bundle/unbind/cleanup synaptic
on-bridge (H2/H3/H5/H6/H7), a persistent store in complex synapses, the megakernel + CSR cache resident. The residual
host round-trips are H1/H4/H8/H9. **H4 is the bind→store hand-off:**

- today `_compose_phases` settles the composite phasor in the `acc` register (block `2n`, on-bridge), then
  `rf_read_phases()[acc]` **reads it to numpy** and `_to_phasor` **re-encodes it on host**;
- `_write_block` then installs `complex(zc[k])` as the store block's trigger→readout weights.

The composite leaves the bridge (a phase read + a host re-encode) and is written back as weights. That read-then-write is
the round-trip.

## The synaptic hand-off (the register→register primitive, applied to H4)

The persistent store **is** a synapse (the memory must end as `cp_rf_w_re/im`), so this is not "remove the weight-write."
It is: remove the **host read + host re-encode of the bind output** that produces the store weight. The mechanism
(`SynapticH4Composer`, a thin subclass overriding only the two store methods):

1. `_compose_phases` binds + bundles into `acc` exactly as the stock composer (the same H2/H3 on-bridge path → `acc`'s
   v/u hold the byte-equal composite), then **leaves `acc` resident** and returns a sentinel (no `rf_read_phases()`).
2. `_write_block` installs the `acc → store-block-readout` complex synapse `(trig+1+k, P+acc·D+k, 1.0)` and resonates, so
   the composite flows from `acc` into the store block's readout neurons `trig+1+k` **through a unit complex synapse**
   (register→register; this is the GO primitive `2026-06-18-one-brain-register-handoff-GO.md`). The store weight is then
   captured at that **synaptic terminus** — the readout register `acc` drove — not from a host re-read of `acc`.

The store block, once written, is queried by the existing H5→H7 synaptic path unchanged. The capture window does **not
re-kick**, so the scoping doc's flagged sim/ edit #1 (mask the `rf_kick` spike-tracker re-init) was **not needed** — the
route window's resonate makes `trig+1+k` cross from the kick-reset phase-0 baseline `_compose_phases` left.

## Result — 3 seeds × {D=64, D=128}, K=8 facts (all 4 roles: agent/action/patient + AFFIRM polarity)

| check | result | reading |
|---|---|---|
| **recall == host** (patient / agent / yes-no) | **8/8 · 8/8 · 8/8, every row** | the synaptic store recalls IDENTICALLY to the stock host read-then-write store, on all three query types |
| syn-correct == host-correct (patient) | **1.000 == 1.000** | the synaptic store retrieves the actually-stored fact, == the host store's accuracy |
| **severed-route LESION** (zero `acc→store`) | **recall 0/8, every row** | the on-bridge hand-off is LOAD-BEARING — no parallel host write is silently doing the work |
| **permuted-store** (route `("cat","go","river")` into block 0, read block 0 directly) | **block 0 == the routed fact, every row** | the synaptic write CARRIES the content (not a leak); the distinct routed patient is unambiguous |
| **no-confab MOAT** (4 unstored `query_patient` + 3 unstored `ask_yes_no`) | **0 breaches total** (all `None` / `"unknown"`) | abstention intact — the moat is NOT weakened |

Unanimous GO: `all_recall_eq_host=true`, `syn_correct_eq_host=true`, `mean_lesion=0.0`, `permuted_follows_all=true`,
`moat_total_breaches=0`. ~22s/row on CPU (numpy); the FHRR algebra is exact, so CPU is the oracle path for this
exact/identity de-risk (parity, not a distribution — the merged-bridge precedent for exact effects applies).

## Why it matters

This confirms the register→register synaptic hand-off (GO at the abstract bind→unbind level on 2026-06-18) on the
**specific bind→store (H4) path of the production composer** — the highest-value single conversion in the H0–H12 map. A
between-op DATA hand-off now works fully synaptically on `OneBrainComposer`: the bind's output enters the persistent
store without round-tripping through `acc`'s host phases. The lesion + permuted-store + moat battery establish it is the
real on-bridge route doing the work, not residual host state.

## Verdict + next

**GO → the bind→store hand-off is synaptic on the persistent loop (H4 closed); recommend it as the composer default.**
The override is a thin, additive subclass (`SynapticH4Composer`) — folding it into `OneBrainComposer` is a small change
(`_compose_phases` leaves `acc` resident + returns a sentinel; `_write_block` routes + captures at the synaptic
terminus) that preserves the query path and the moat by construction, and needs **NO `sim/` edit**.

**Phase B (the on-substrate SEQUENCER) is the remaining deep piece** — and the scoping doc's load-bearing one: the
Python `for/if/return` that sequences the ops and gates answer-vs-abstain (H9). The data axis is mop-up of a proven
primitive (H1, H8, H11 remain, all extensions of this same GO mechanism + the existing spiking-WTA cleanup); the
sequencer is the research crux (result-conditioned op-selection in spikes). The controller takes it up separately.

## Honest scope

- Validated for the flat 4-role who/what store/query at K=8, 3 seeds × 2 D, on CPU (the exact-algebra oracle path). The
  recall==host parity is exact and mechanistically seed-/D-independent for this identity hand-off.
- This is a DATA hand-off (H4) only. It does NOT touch the orchestrator (H9) or the operand-in / argmax / render / clause
  round-trips (H1/H8/H10/H11) — those are the rest of the data axis (extensions of the same primitive) and the sequencer
  (the deep axis).
- The store remains a persistent complex-synapse weight (the FHRR way); "synaptic hand-off" means the bind output is
  routed register→register into the store readout and captured there, not host-read from `acc`.

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_bindstore_handoff_derisk --seeds 42,43,44 --dims 64,128
```
