---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) PREREGISTRATION AMENDMENT 2 -- fix round 2 after the review of 7d5c2743d. Declares the flag-ON side effect on the default-ON surprise organ measured in the v2 arms, makes the A10 read leave no footprint, and adds criterion (D). Filed BEFORE the v3 runs it governs. Criteria (A), (B) and (C) are NOT changed.
seeds: [7]
verdict: AMENDMENT only. No result is claimed here. Filed after the v2 seed-7 arms were seen; it adds a criterion that can only make GO harder, and it withdraws a proposed criterion that could not fail.
runner: research/runners/_reward_value_afferent_derisk.py
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_on_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_les.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_pre_off_main.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_arms_on_a.json
---

# A10 pre-registration, amendment 2 (filed before the v3 runs)

Amends `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (57f0ebfd0) and its
AMENDMENT-1 (d82ce8c8e). Filed on the A10 fix branch after the independent review of 7d5c2743d found that nothing in
the record mentioned a side effect its own v2 data show. Seed 7 stays a dev/calibration seed; the v2 data HAVE been
seen, so nothing here moves a threshold of (A), (B) or (C).

## What the v2 data show (the side effect, declared)

With `BRAIN_REWARD_VALUE_AFFERENT=1`, the production surprise read on the CONFIRM turn changed. The server's own
surprise block reads `surprise.surprise_hz` = 0.4050925925925926 Hz in off_a, off_b, les and the pre-patch reference
(`research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json`, `research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_les.json`, `research/findings/raw/_reward_value_afferent_derisk/v2/s7_pre_off_main.json`), but 0.3472222222222222 Hz in on_a, on_b
and rf/on_a (`research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_on_a.json`, `research/findings/raw/_reward_value_afferent_derisk/v2/rf/s7_arms_on_a.json`). The CONTRADICT read is equal in every arm. The A10
read itself reads 0.4050925925925926 Hz on CONFIRM: it is the organ's first read of the turn.

Cause. `da_mode_drives_chat.observe_turn` (and so the A10 read) runs before the server's surprise block in the same
turn, on the SAME process-shared organ. A read on the shared merged pool depends on read history: the pool bridge
has no `_rest_extra` snapshot, so `_hard_reset` leaves the surprise slice's adaptive thresholds, activity EMA and
refractory state where the previous read left them. A module-level probe on the production organ at seed 7 (the
arms' env, numpy) read CONFIRM 0.4050925925925926 Hz first and 0.3472222222222222 Hz second (exploration, not an
artifact; the v3 module-level check below records it). Reconsolidation is default-ON and gates on that read's
`surprised` flag, so it is exposed too; the arms ran it off, and no `surprised` decision flipped at seed 7.

It is also a second lesion asymmetry: the lesion arm's A10 read uses the standalone twin, so the production read in
the lesion arm stayed at 0.4050925925925926 Hz while the intact arm's moved.

## Statements this amendment withdraws

- The pre-registration: the lesion "does NOT touch ... the production `surprise_info`/`surprise_prefix` block
  computed later in the same turn". In the v2 code the INTACT A10 read did touch it (above). The lesion read did not.
- The finding's proposed next criterion (C'), `|lesion_hz - cuefree_hz(block)| < 1e-6` on the twin. With
  patient_expected->surprise zeroed, the twin has no route from the cue to the surprise pool, so once the read-time
  cut holds, the lesioned read equals the cue-free rate by construction. (C') tests the cut, not whether the live
  read carries the effect, and it would be scored on seed-7 data already seen. It is not a GO criterion for B2b.

## Design change (the mechanism)

The A10 read leaves no footprint (`webapp/reward_value_afferent_chat.py`). Right before the read it snapshots every
piece of state the read can mutate, and restores it right after:
- every array attribute of the bridge the read drives (the intact organ's pool bridge, or the twin under the
  lesion): dense per-neuron and per-synapse state, and the sparse weight data, indices and indptr;
- the bridge's scalar and container attributes, and attributes the read adds (deleted again);
- the runtime clock's scalars; the organ's host block bookkeeping (`_block`, `_cue_next`, `_novel_next`);
- the numpy and Python global RNG states.
Builds (`ensure_built`, the lesion twin) run before the snapshot and are kept. If the snapshot cannot be taken, the
read is not made and the turn falls back to the pre-existing afferent. Each record carries a `footprint` block: what
the read changed and whether the restore was exact.

Declared, not changed: cupy's global generator has no state accessor and is not restored (the read draws no random
numbers when the pool's noise is off; a first-use twin BUILD reseeds it, as production's `BRAIN_SURPRISE_LESION` path
already does). A first-use organ build happens where A10 first calls, earlier in the turn than the production
surprise block when no startup warm-up ran; the webapp startup warms the intact organ.

Not changed, and not fixed here: the production surprise read itself stays read-history dependent (two CONFIRM
assertions in one process read differently). That is a property of the default-ON organ on the merged pool, outside
this lane; it is logged in `research/FAILURE_LOG.md`.

## Criterion added: (D) no side effect on the default-ON surprise faculty

- (D): in the arms, the production `surprise` block AND the `reconsolidation` block of on_a and of les are equal
  (canonical JSON, timing keys dropped) to off_a's, whole block, on both turns.
- GO = A and B and C and D. (D) can only make GO harder. A missing `surprise` block on an OFF turn makes (D)
  unmeasurable, which is a precondition failure (UNDEFINED).
- (D) can fail: scored on the v2 arms it fails on CONFIRM (on_a 0.3472222222222222 vs off_a 0.4050925925925926), and
  a test pins that (`tests/test_reward_value_afferent.py::test_side_effect_check_fails_on_the_v2_arms`).
- Reported, not scored: each A10 read's `footprint` record; whether the intact A10 read equals the production read on
  each turn; the response paths that differ between on_a/les and off_a outside `da_drives`, `da_encoding`, `answer`.

## Module-level footprint check (instrument check of the fix, not a capability verdict)

`research/runners/_reward_value_afferent_footprint.py`, seed 7, the arms' env, on the production organ
(`get_organ(seed)` on the merged cortical pool). One process: REFERENCE (production CONFIRM then CONTRA, the flag-OFF
order); ISOLATED intact (A10 CONFIRM, production CONFIRM, A10 CONTRA, production CONTRA); ISOLATED lesion (the same
with the twin); RAW (the v2 A10 path: an unisolated read, then production CONFIRM); and, reported only, the v1
runner's in-process order (CONFIRM, CONTRA, CONFIRM), whose third read is the read v1's ON arm recorded. A sha256 over
the organ's read state is taken before and after every A10 read, and after every restore of the post-build snapshot.
- Preconditions: the organ is on the merged pool; the REFERENCE reads equal the v2 OFF arm's handler-level
  `surprise.surprise_hz`; every restore hashes back to the post-build hash; the RAW sequence shifts the production
  CONFIRM read (else the check could not fail).
- GO iff F1 every A10 read leaves the hash unchanged; F2 every production read in the isolated sequences equals the
  REFERENCE read for its turn; F3 each intact A10 read equals the production read that follows it.

## Runs this amendment governs

1. The module-level check above, local under `tools/memcap.sh`, at the commit that carries this amendment.
2. The v3 arms (off_a, off_b, on_a, on_b, les + block control) on the pool, production-default composer and forced
   `rf`, at the same commit, writing to `research/findings/raw/_reward_value_afferent_derisk/v3/` and `v3/rf/`.
3. A fresh pre-patch reference for each composer at the same main merge parent 306ef27d7, same env, on the same
   hardware class as the arms (the revision is provisioned only on the Xeon nodes pool1/pool2, where v2 ran), beside
   the v3 arms. The v2 pre-patch files stay as they are.
Expected, stated before the run: (B) and (C) read as in v2, because the A10 read is still the organ's first read of
the turn; the run exists to measure (D) and to re-measure (A) at the new code.

## Also recorded for B2b

- The merge adds one row to the default LBF registry: 38 -> 39 faculties, thin 2 -> 3 (the row is `thin` unless
  `BRAIN_REWARD_VALUE_AFFERENT` is set in the harness). The headline load-bearing fraction is unchanged.
- The meaningful next rung is a substrate-matched lesion with a within-block attribution (B'): zero the
  patient_expected->surprise edges on the intact organ's own pool slice for the read (restored after it) and compare
  the CONFIRM-block suppression (cue-free rate minus CONFIRM rate) intact vs lesioned on one substrate. It needs its
  own amendment and a mechanism change before any run, and B2b's gate seeds are the first data it may be scored on.
