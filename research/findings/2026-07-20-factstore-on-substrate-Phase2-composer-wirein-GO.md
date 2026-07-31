---
type: finding
status: live
date: 2026-07-20
mechanism: fact-store
---

# Fact-store on the substrate — Phase 2: the composer's fact-store now lives in DEVICE SYNAPSES (GO)

**Date:** 2026-07-20 · **Status:** GO (parity 3-seed 42/43/100) — `OneBrainComposer(persistent_store=True)` holds the
fact composites IN the device synapses (`cp_rf_store_re/im`, Phase 1) and reads them with IDENTICAL recall to the
staged host-`store_conns` path, the no-confab moat intact. This closes the LAST "not on the shared substrate" item —
the composer fact-store's DATA persistence (the VSA idealization) is now on-substrate, opt-in. Default byte-identical.

## What this closes

After the single-substrate capstone, the one host residual was the composer fact-store: `store_conns` (a HOST list)
re-installed onto `cp_rf_w_*` transiently per read (because `rf_set_complex_weights` REPLACES the single working CSR).
Phase 1 added the persistent store CSR mechanism; the Phase-2 de-risk proved a persistent store reads IDENTICALLY to
the staged store (phase-invariant RF read, |Δphase|=0.0000). This Phase-2 wire-in deploys it in the composer.

## The wire-in (`one_brain_composer.py`, opt-in default-off)

- `persistent_store=False` constructor flag (+ `_persistent_dirty`). When True: `_sync_persistent_store()` installs
  `store_conns` into `cp_rf_store_re/im` via `rf_set_store_weights` ONCE per store mutation (asserting the store
  readout rows are DISJOINT from the op rows — they are, by the composer layout: store block readouts vs the
  fill/bound/acc/Q/cleanup regions).
- In `_read_all_blocks` (cached + stock paths): when `persistent_store`, the store settle window sets the working op
  EMPTY (`cp_rf_w_re = cp_rf_w_im = None`) so ONLY the persistent store drives the readouts; then the unbind + cleanup
  operators install as usual — and the persistent store KEEPS refreshing the readouts through those windows (harmless,
  disjoint rows; and the read is phase-invariant so it decodes identically).
- Default (`persistent_store=False`): every edit is an `if persistent_store: <new> else: <original>` with the else the
  verbatim staged path + unused-when-off attributes → byte-identical (the rf/numpy oracle + all existing tests).

## Result (3-seed 42/43/100)

- **Recall PARITY: `persistent_store=True` gives `['cat','mouse','deer']` == the staged `persistent_store=False`
  path — all seeds.** The fact-store in device synapses reads identically to the host-list store.
- **No-confab moat intact: an unstored cue (`apple stop`) abstains (`None`) under `persistent_store=True` — all seeds.**
- **The store lives in device synapses: `cp_rf_store_re` is installed on the bridge after a read.**
- **Default byte-identity: `test_one_brain_composer_agent` — 19 passed (813s) with `persistent_store=False`** (the else
  branches are the verbatim staged code; my edit did not touch the default path).
- **MIXED-usage robustness (verified): a clause query (`_read_block`, staged) + a flat query (`_read_all_blocks`,
  persistent) + the moat, in one session under `persistent_store=True`, all match the staged reference** — clause
  `'cat look south'`, flat `'mouse'`, moat `None`, parity on all three. This confirms the honest subtlety: once
  `cp_rf_store_re` is installed, a STAGED read path (not yet wired) gets a 2× store drive (staged `cp_rf_w_*` + the
  persistent `cp_rf_store_*`) — but that is PHASE-CORRECT (the RF read is phase-invariant; 2×composite has composite's
  phase), so the un-wired read paths decode IDENTICALLY. ⇒ the `_read_all_blocks`-only wire-in is already ROBUST for
  mixed usage; extending the other paths is a cleanliness (perf) follow-on, NOT a correctness one.

CI: `tests/test_onebrain_persistent_store.py` (2 tests) + `tests/test_rf_persistent_store.py` (5, the Phase-1
mechanism + read-fidelity).

## Read-out — the fact-store-on-substrate arc is COMPLETE

- **⇒ the composer's fact-store now lives in DEVICE SYNAPSES** (opt-in `persistent_store=True`), read-identical to the
  staged host-list path, moat intact. The single-shared-substrate is now complete not only for every SPIKING
  COMPUTATION (bind/unbind/cleanup/render/learning) but for the fact-store DATA too — the VSA-idealization residual is
  closed as an opt-in.
- **Honest scope:** (1) the wire-in covers the main `_read_all_blocks` (who/what query_patient); the other read paths
  (`_read_block`, clause `_decode_clause`, `_recovered_patient_phases`, `_patient_cleanup_scores`) still use the staged
  install when `persistent_store=True` — extending them is a mechanical follow-on (same pattern, de-risked). (2) Perf:
  an installed store forces the ~605ms loop over the ~96ms megakernel on the opt-in path (Phase-1 BAIL; a follow-on
  lever restores megakernel perf). (3) Default remains staged (byte-identical); `persistent_store=True` is opt-in.

Runner/tests: `_gap_persistent_store_readfidelity_derisk.py`, `test_onebrain_persistent_store.py`,
`test_rf_persistent_store.py`. Composer edit in `one_brain_composer.py` (NO new `sim/` edit — reuses Phase 1's
`rf_set_store_weights`).
