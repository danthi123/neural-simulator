---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) PREREGISTRATION AMENDMENT 3 -- the follow-up round after the re-review of 4b6a9cf66 (verdict SOUND as a default-OFF merge, four problems named). Corrects a declaration, splits criterion (D) so its reconsolidation half can fail, adds one arm pair with reconsolidation on, and adds a module-level recall probe. Filed BEFORE every run it governs. Criteria (A), (B) and (C) are NOT changed.
seeds: [7]
verdict: AMENDMENT only. No result is claimed here. Filed after the v2 seed-7 arms and the v3 module-level footprint check were seen; the v3 arms queued at 621ace648 have not run. Nothing here moves a threshold of (A), (B) or (C), and the (D) change can only make GO harder.
runner: research/runners/_reward_value_afferent_derisk.py
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_off_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v2/s7_arms_on_a.json
  - research/findings/raw/_reward_value_afferent_derisk/v3/footprint_module.json
---

# A10 pre-registration, amendment 3 (filed before the runs it governs)

Amends `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (57f0ebfd0), AMENDMENT-1
(d82ce8c8e) and AMENDMENT-2 (621ace648). Filed on `research/a10-followups`, branched at 4b6a9cf66, the A10 head that
main merged as default-OFF. Seed 7 stays a dev/calibration seed.

## What the re-review of 4b6a9cf66 found

1. The lesion arm leaves the process-global generators changed, and AMENDMENT-2 names the wrong one. Every bridge
   build calls `sim/bridge.py` `_initialize_rng`, which reseeds cupy's, numpy's AND Python's global generators. The
   lesion twin's first-use build (`_ensure_les`) runs before the read's snapshot, so the snapshot's restore cannot
   undo it. AMENDMENT-2 said only cupy's generator is reseeded by that build.
2. The reconsolidation half of (D) cannot fail in the arms it governs. Every arm runs with
   `BRAIN_RECONSOLIDATION=0` (`_QUIET`), so both reconsolidation blocks are None and the comparison is None with None.
3. A10's own recall, `chat.inner.what_does` (under the default composer, `OneBrainComposer.query_patient`), runs
   outside the snapshot isolation, and nothing declares it.
4. A failed restore still drives the SNc: when the restore raised or was not exact, the read still returned
   `drives=True`.

## Statements this amendment corrects or withdraws

- AMENDMENT-2, "Declared, not changed": "cupy's global generator ... a first-use twin BUILD reseeds it". Corrected:
  the twin build reseeds all three global generators. Checked by hand on a standalone seed-7 organ (numpy; an
  exploration, not an artifact): the twin build changed both numpy's and Python's global state. The follow-up module
  now isolates that build (below), and a unit pin checks the real twin build leaves both unchanged.
- AMENDMENT-2's (D) as scored in its five arms: the reconsolidation comparison there is withdrawn as a scored
  criterion. It is reported only.
- The LBF row note (`research/runners/lbf_rows/reward_value_afferent.py`) said the A10 read "never perturbs the
  production surprise read". Scoped to what was measured: at the module level, seed 7, numpy, the organ's read-state
  hash was unchanged across every A10 read (`research/findings/raw/_reward_value_afferent_derisk/v3/footprint_module.json`). The handler level is (D), not yet measured.

## Design changes (the mechanism, `webapp/reward_value_afferent_chat.py`)

- The lesion twin's first-use build runs inside `_global_rngs_untouched`. Numpy's and Python's global states are saved
  and restored around it. On the cupy backend the build is handed a private cupy generator object, and the host's
  object is set aside and put back untouched. Cupy's RandomState has no get_state/set_state, and `cupy.random.seed`
  reseeds the current object in place, so only swapping the object keeps the host's stream intact. The twin build
  reseeds from its own seed, so the twin is the same either way (a unit pin compares it with an unguarded build).
  The record carries `footprint.twin_build`. If a host generator compares unequal after the build, the read does not
  drive.
- A restore that raises or is not exact returns `drives=False` (no snapshot, no read; no exact restore, no read).
- Declared, not isolated: the intact organ's first-use build (`ensure_built`). When no earlier caller built the organ,
  A10's call builds it, which reseeds all three generators at that point in the turn instead of at the production
  surprise block. The battery worker runs no startup warm-up; the webapp startup warms the organ. This build is common
  to the intact and lesion arms and absent from the OFF arm. Isolating it would not restore the flag-OFF sequence:
  production's own first build resets the generators at the surprise block, and A10 cannot reproduce that reset there.

## Criterion change: (D) split into two halves

- (D-s) surprise half, unchanged in substance: the production `surprise` block of on_a and of les equals off_a's,
  and that of on_rc equals off_rc's, whole block, on both turns.
- (D-r) reconsolidation half: on_rc's `reconsolidation` block equals off_rc's on both turns. It is MEASURED only when
  the rc pair is in the artifact and reconsolidation ran in off_rc on at least one turn (its block is a dict). In the
  five core arms it is reported (`reconsolidation_equal_unscored`), never scored.
- (D) = GO iff (D-s) holds AND (D-r) is measured AND (D-r) holds. GO = A and B and C and D, as before.
- An unmeasured (D-r) is a precondition of GO, not of NO-GO. If every other criterion reads GO and (D-r) is
  unmeasured, the verdict is UNDEFINED. A criterion measured and false is NO-GO whether or not (D-r) was measured.
- It can only make GO harder: before, (D-r) passed by construction; now it must be measured and hold.
- It can fail. Unit pins in `tests/test_reward_value_afferent.py`: a differing on_rc block reads NO-GO, a missing rc
  pair reads UNDEFINED with the rest GO, and the v2 arms still fail (D-s).

New arms (`research/runners/_reward_value_afferent_derisk.py`): `off_rc` and `on_rc`, identical to off_a and on_a
except `BRAIN_RECONSOLIDATION=1`, its production default. Expected before the run: on CONTRADICT the production read
is surprised and "fish" is not "cat", so reconsolidation runs in off_rc (a rewrite of the stored fact, after the
surprise read) and (D-r) is measured. On CONFIRM it does not run in either arm.

The v3 arms queued at 621ace648 have no rc pair and run the module before this round. Scored with this runner their
(D-r) reads unmeasured, so they cannot read GO on (D). Their (D-s) is scored as AMENDMENT-2 wrote it.

## The recall call (item 3): a module-level probe, then a declaration

`research/runners/_reward_value_afferent_recall_probe.py` measures whether `chat.inner.what_does` is history-
dependent on the chat the battery worker builds (`webapp.server._build_chat_brain("tiny-demo", "stub")`, the arms'
env, seed 7, numpy, LTM tier off). It hashes everything reachable from `chat.inner` before and after each recall
(arrays by bytes, sparse matrices by parts, objects by attributes; C objects such as locks by type only, counted).
It runs five recalls: (dog, chase) three times, (cat, eat), then (dog, chase) again. A sensitivity control checks
that the hash changes on a one-element change in the composer bridge and returns when it is undone.

- Reported: whether the value is the same on every (dog, chase) recall; whether the state a later recall leaves is
  the same after one or two recalls; whether the first recall changes state at all, and which composer and bridge
  attributes; whether the numpy or Python global generators move; the composer's `last_trace` after each call.
- Verdict: GO ("history-independent at the module level") iff the values are equal, the post-recall state converges
  and the generators do not move. Preconditions: the build, the composer class the arms use (OneBrainComposer by
  default, RFPhasorComposer with `--composer rf`), recall #1 returns "cat", the sensitivity control.
- The declaration in `webapp/reward_value_afferent_chat.py` cites the probe's result and its scope. If the probe
  reads NO-GO, the v4 arms are not queued until the recall is brought inside the isolation or a further amendment
  shows the dependence reaches no production read in the arms.

## Runs this amendment governs

1. The recall probe, local under `tools/memcap.sh`, both composers, each writing a file named recall_probe.json
   into `research/findings/raw/_reward_value_afferent_derisk/v4/` (production-default composer) and `v4/rf/`.
2. The v4 arms (the seven arms plus the block control) on the pool, production-default composer and forced `rf`, at
   the follow-up head, writing to `research/findings/raw/_reward_value_afferent_derisk/v4/` and `v4/rf/`.
3. The pre-patch reference for (A): the v3 references queued at 306ef27d7 (same revision, same env, same hardware
   class) serve v4. The OFF-arm env is unchanged by this amendment.

Expected, stated before the runs: (B) and (C) read as in v2, because the A10 read is still the organ's first read of
the turn and the twin build now leaves the host generators as they were. The v4 run exists to score (D) with both
halves measured and to re-measure (A) at the new code.

## Also recorded for B2b

The LBF registry row is unchanged (38 -> 39 faculties, `thin` unless `BRAIN_REWARD_VALUE_AFFERENT` is set in the
harness). The 6-seed gate stays in B2b (S28); default-ON would also need a SOUND independent review.
