---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) PREREGISTRATION AMENDMENT 4 -- the recall probe's first run showed A10's own recall is not free of history under the production-default composer, so, as AMENDMENT-3 requires, the recall is brought inside the isolation before any v4 arm runs. Also fixes two instrument defects of the probe's first run. Filed BEFORE the probe's second run and the v4 arms. Criteria (A), (B), (C) and (D) are NOT changed.
seeds: [7]
verdict: AMENDMENT only. No capability result is claimed here. The probe's first run read UNDEFINED on both composers (instrument preconditions); its measurements are reported below as measurements.
runner: research/runners/_reward_value_afferent_recall_probe.py
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/v4/recall_probe.json
  - research/findings/raw/_reward_value_afferent_derisk/v4/rf/recall_probe.json
---

# A10 pre-registration, amendment 4 (filed before the runs it governs)

Amends the pre-registration and AMENDMENT-1 to AMENDMENT-3. AMENDMENT-3 (19a00ce51) set a rule for the recall probe:
if it reads NO-GO, the v4 arms are not queued until the recall is brought inside the isolation or a further amendment
shows the dependence reaches no production read. This amendment takes the first branch.

## The probe's first run (19a00ce51, seed 7, numpy, the arms' env, LTM tier off)

Artifacts: `research/findings/raw/_reward_value_afferent_derisk/v4/recall_probe.json` (production-default composer)
and `research/findings/raw/_reward_value_afferent_derisk/v4/rf/recall_probe.json` (forced `rf`). Both read UNDEFINED, on instrument defects,
not on the measurement:
- The composer precondition compared the exact class name. The class the arms' env builds is a pool-bound subclass
  (`Pool1BoundOneBrainComposer`, `Pool1BoundComposer`), which is inside the family AMENDMENT-3 names.
- For `rf` no bridge was found, so the hash's sensitivity control did not run: `Pool1BoundComposer` runs its RF ops
  on the pool #1 substrate's bridge (`_pool1.bridge`), not on a `b` attribute.
- Deques and mappingproxies were hashed by type only (reported as `opaque`), so a change inside them was invisible.

What the first run measured, reported as measurements only:
- Production-default composer: the recalled value was "cat" on all four (dog, chase) recalls. The state after a
  recall did not converge (`state_after_converges` false), and a global generator moved on every recall
  (`rng_untouched` false). The first recall changed the composer attributes `_fact_shard`, `_fact_shard_built_K`,
  `_merged`, `_pool1`, `b`, `comp` and `parser`, and the bridge arrays `cp_membrane_potential_v`,
  `cp_recovery_variable_u` and the RF state and weight arrays.
- Forced `rf`: the value was stable, the state after a recall converged, the generators did not move; the first
  recall changed `_pool1` and `_scan_comps_cache`.

An exploration after the run (a traced recall on the same build, not an artifact; run 2 records the same trace as
`global_rng_callers`): the generator movement comes from `OneBrainComposer._spiking_select`. Its Izhikevich cleanup
bank draws OU noise from numpy's GLOBAL generator (`sim/bridge.py` `_draw_ou_noise_samples`) on every recall, and
at first use the bank is BUILT (`rf_phasor_composer._izh_bank`), and each build reseeds numpy and Python.

So with the flag on, A10's recall advances numpy's global generator before production's own recalls in the same
turn, moves the first-use bank build (and its reseed) earlier in the turn, and leaves composer state that production
code between A10 and production's first recall would read. That is a footprint on production.

## Design change: the recall runs inside the isolation

`webapp/reward_value_afferent_chat.isolated_recall` wraps `chat.inner.what_does`:
- a DEEP snapshot of everything reachable from `chat.inner` by attribute or container membership: every dense
  array copied; every list, dict, set, deque, bytearray and `__slots__` value recorded; every object's attribute
  bindings recorded; numpy and Python generator OBJECTS by state;
- the recall inside `_global_rngs_untouched` (numpy's and Python's global states restored; on cupy a private
  generator object is swapped in);
- a restore right after: arrays refilled in place, containers refilled, attribute bindings put back (attributes the
  recall added are deleted, so a lazily built bank or cache is discarded and production's own first recall builds
  it where it would with the flag off).
A10 does not drive when the snapshot cannot be taken (including when its array copies would exceed 4 GiB), when the
restore is not exact, or when a host generator changed. The record carries this under `reward_value.recall`.

Declared, not covered: module-level globals (not reachable by attribute); C objects with no Python state (locks,
kernels; counted as `opaque`); on cupy the recall's noise comes from a private generator, so A10's recalled value
can differ from production's in a noise-sensitive case; a concurrent writer to a snapshotted object between the
snapshot and the restore would be overwritten by the restore (the organ read has the same exposure).

## Instrument changes in the probe (run 2)

- The composer precondition checks the class FAMILY (the MRO names `OneBrainComposer`, resp. `RFPhasorComposer`) and
  that it is not a `TieredFactStore`.
- The bridge lookup includes `_pool1.bridge`; a precondition requires a bridge to be found.
- Deques and mappingproxies are hashed by content.
- New: `isolated_first` (the isolated recall from the fresh state, where A10 calls it) and `isolated_warm` (after the
  five production recalls), each comparing the hash of `chat.inner` and both generators with the state before. A
  second verdict, `verdict_isolated`, is GO iff both leave the hash and the generators unchanged with an exact
  restore and the isolated value equals the first production recall's value. Its preconditions include a
  sensitivity control that the UNISOLATED recall does leave a trace (hash or generator), so the check can fail.
- New: `global_rng_callers` on recalls #1 and #2.
- The first verdict (raw history-independence, as AMENDMENT-3 defined it) is unchanged.

Disclosed: smokes of the run-2 code wrote to a scratch path (not artifacts) before this amendment was committed. On
`rf` both verdicts read GO. On the default composer the raw verdict read NO-GO (state and generators, as in run 1)
and the isolated verdict read GO.

## Runs this amendment governs

1. Probe run 2, local under `tools/memcap.sh`, both composers, each writing a file named recall_probe_run2.json
   beside run 1 in `research/findings/raw/_reward_value_afferent_derisk/v4/` and `v4/rf/`.
2. The v4 arms exactly as AMENDMENT-3 describes them, at the commit that carries this amendment (or later), once
   run 2's isolated verdict reads GO on both composers.

Expected before run 2: the raw verdict NO-GO on the default composer (state and generators) and GO on `rf`; the
isolated verdict GO on both.
