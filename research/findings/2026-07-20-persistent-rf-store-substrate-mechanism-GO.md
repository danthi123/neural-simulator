---
type: finding
status: contributing
date: 2026-07-20
mechanism: fact-store
---

# Persistent RF store — the substrate mechanism for a device-synapse fact-store (GO, Phase 1)

**Date:** 2026-07-20 · **Status:** GO (Phase 1 = the `sim/` mechanism) — a PERSISTENT complex store CSR
(`cp_rf_store_re`/`im`), DISTINCT from the per-op `cp_rf_w_*`, summed ADDITIVELY into the RF matvec and installed via
a new `rf_set_store_weights`, so a fact-store can live IN the device synapses and PERSIST across per-op binds. This is
the `sim/` half of closing the composer fact-store's host-persistence (the last "not on the shared substrate" item;
the VSA idealization). BYTE-IDENTICAL when off; the megakernel is UNTOUCHED. The composer wire-in (`persistent_store`
flag) is Phase 2.

## The mechanism (design workflow's Approach A + megakernel BAIL; the 5 sim edits)

The last host residual after the single-substrate capstone was the composer fact-store: `store_conns`/numpy-kb are
HOST lists installed onto `cp_rf_w_*` transiently per read — because `rf_set_complex_weights` REPLACES the single
`cp_rf_w_*`, a per-op bind and a persistent store cannot both live there. Fix (chosen by an 8-agent design workflow
over 3 candidates, scored correctness+byte-identity then minimality):

- **A second, distinctly-named CSR** `cp_rf_store_re`/`cp_rf_store_im` (+ `cp_rf_store_dense`), default `None`,
  installed ONCE by `rf_set_store_weights` — which per-op `rf_set_complex_weights`/`rf_kick` NEVER touch, so the store
  survives every bind.
- **Summed ADDITIVELY** into `_rf_advance_one`'s matvec behind an independent `if cp_rf_store_re is not None:` guard,
  non-zero ONLY in store-readout rows DISJOINT from the op operator's rows (asserted in `rf_set_store_weights`), so
  `W@z + Store@z` writes non-overlapping neurons — no cross-corruption.
- **The megakernel BAILS to the per-step loop when a store is present** (`and cp_rf_store_re is None` on the dispatch
  guard) — so the `_RF_MEGASTEP_SRC` CUDA source is UNTOUCHED (no recompile, provably byte-identical off-path), and
  the loop applies BOTH the working and store matvecs.

## Byte-identity when off (provable + verified)

When `cp_rf_store_re` is `None`: the additive sub-block is SKIPPED (zero extra float ops → `_rf_re_new`/`_rf_im_new`
and everything downstream bit-identical); the megakernel guard gains `and True` (dispatch unchanged); the CUDA source
is unedited; `rf_set_complex_weights`/`rf_kick` never reference the store. Verified: **19 existing RF/composer tests
pass byte-identically** (`test_rf_megakernel` incl. the masked golden, `test_rf_neuron_mask_coexistence`,
`test_merged_rf_composer_coresident`, + the single-substrate suite), and the store-off RF path is deterministic
byte-identical across runs.

## On-path (verified)

- **The store SURVIVES a per-op bind + kick** (`rf_set_complex_weights` + `rf_kick` leave `cp_rf_store_re` byte-unchanged).
- **The loop applies Store@z** — the store-readout neurons are driven by the store matvec.
- **The megakernel BAILS** to the loop when a store is present (a spy confirms `_rf_resonate_steps_megakernel` is not
  called) — so the store term is never dropped.

CI: `tests/test_rf_persistent_store.py` (4 tests, GPU-path).

## Read-out

- **⇒ the substrate can now hold a PERSISTENT fact-store in device synapses** (disjoint from the per-op binds,
  byte-identical when off). This is the `sim/` mechanism the composer's fact-store host-persistence (the VSA
  idealization) needs to move on-substrate.
- **Phase 2 (composer wire-in) — the one open question is now RETIRED.** A `persistent_store=False` (default
  byte-identical) flag on `OneBrainComposer` installs `store_conns` via `rf_set_store_weights` once (instead of
  per-read `rf_set_complex_weights`) + bypasses the per-read store install. The ONLY on-path unknown was the
  staged→continuous READ FIDELITY (a persistent store keeps driving the readout THROUGH the unbind/cleanup windows vs
  the staged store swapped out). **De-risked (3-seed, `_gap_persistent_store_readfidelity_derisk.py`): on a minimal
  FHRR store→unbind→decode, the persistent store reads the filler IDENTICALLY to the staged store — `mean circular
  |Δphase| = 0.0000`, both decode the true filler at 0.0127 (chance ~0.25).** As reasoned: the RF read is PHASE-based +
  magnitude-invariant, so a readout refreshed at full magnitude and one decaying carry the SAME phase. ⇒ the composer
  wire-in produces IDENTICAL recall — a mechanical edit, no fidelity risk. CI `test_rf_persistent_store.py` (5 incl.
  the fidelity test). Perf note: an installed store forces the ~605ms loop over the ~96ms megakernel on the opt-in
  path (a follow-on lever: pre-merge the store into the op CSR, or a `use_store`-guarded kernel edit).

`sim/bridge.py` (EDIT 1 attribute defaults, EDIT 2 `rf_set_store_weights`, EDIT 3 additive matvec term, EDIT 4
megakernel BAIL guard, EDIT 5 invariant comment).
