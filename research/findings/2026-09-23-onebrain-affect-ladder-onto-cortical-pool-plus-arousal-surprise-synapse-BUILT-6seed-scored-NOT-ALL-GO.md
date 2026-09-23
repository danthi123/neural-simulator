---
type: finding
status: no-go
date: 2026-09-23
lane: one-brain/migration (charter D3)
mechanism: the Gate-B affect ladder (the production affect organ) moved onto the shared cortical pool as a 12th organ,
  plus a fixed cross-region synapse from its own latched arousal rungs onto the D2 surprise pool
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_onebrain_affect_pool_verify.py
artifacts:
  - research/findings/raw/_onebrain_affect_pool/smoke_seed42.json
  - research/findings/raw/_onebrain_affect_pool/calibrate_seed42.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed42.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed43.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed44.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed100.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed101.json
  - research/findings/raw/_onebrain_affect_pool/verify_M_seed102.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed42.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed43.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed44.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed100.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed101.json
  - research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed102.json
builds_on:
  - research/findings/2026-09-17-onebrain-wave3-organ-merge-ALL-11-organs-one-pool-GO.md
  - research/findings/2026-08-13-per-region-ou-wiring-affect-GO.md
  - research/findings/2026-09-02-crossedge-arousal-surprise-derisk-PARTIAL-smoke-go.md
---

# D3: the affect organ onto the shared cortical pool, with an arousal-to-surprise synapse. Built; ARM M scored 6/6 seeds, ARM X scored 6/6 seeds; FINAL SCORED VERDICT: NOT ALL-GO, 3/6 gate seeds GO

**Status: NO-GO — FINAL SCORED VERDICT (updated 2026-09-23, round-7 re-review; both the "1-seed smoke, gate
staged" wording AND the previous round's "2 of 6 arm-X seeds landed / mid-harvest, not a final verdict" wording
below are stale and superseded by this line -- the filename above previously still said "6seed-staged", which was
also stale; it is renamed with this fix).** Both arms are fully landed and scored on all 6 gate seeds. ARM M
(answer-preservation) — 5/6 pass M1-M7, seed 102 fails M3 (its graded tone level differs from the standalone
production ladder). ARM X (the v2 production-operating-point instrument, revision `c6fdf7be7`, all 6 seeds
committed to this branch) — X1 passes on 4/6 seeds (43: 5 flips, 44: 2, 101: 4, 102: 2 of 8) and fails on 2/6
(42 and 100: only 1 newly-flagged flip at S*, need >= 2); seed 102 is nonetheless not a GO seed because it fails M3.
The 3/6 figure below is the count of seeds that are fully GO, not the X1 count. Running the pre-registered `--aggregate` command over the complete, 6-seed-per-arm file set
(re-run 2026-09-23 in this fix round; per-seed table below) gives `ALL-GO: False`, **3/6 gate seeds fully GO (43,
44, 101)** — this is the final scored verdict over the pre-registered gate, not a mid-harvest read. The two GO
seeds beyond 43 sit at a thin margin: 44 and 102 (if 102 were not already disqualified by M3) both read exactly
2/8 flips at S*, the pre-registered minimum, with no slack. Everything here is DEFAULT-OFF. It is not wired into
production by default, and no default was flipped.

**Fix round (2026-09-23, after the adversarial review of `7b46d761e`).** Four corrections, detailed in the section of
that name below: the "gain-like, not a DC bias" claim is withdrawn; `XEDGE_W` = 0.05 is a hand-set constant, not a
calibrated value; the arm-X instrument now measures the surprise organ at its production operating point; and the
arm-X gate now scores a change of the organ's own surprise verdict instead of a 0.10 Hz rate floor. Arm M (the
answer-preservation gate, 6 seeds) is unchanged and still part of GO.

## Why affect is the next organ

The 11-organ Wave-3 pool (default-ON since `8ee5e6817`) holds the cortical organs. Among the charter-D3 core set,
the affect organ still ran on its own bridge in production. `affect_production_organ.read_differential` dispatches to
`_appraisal_interoceptive_ladder_derisk.get_ladder(seed)`, which is a standalone bridge of about 690 neurons. The
other organs that are still off the pool were less ready:

- `d6_multiref_wm` and `prospective_memory` run per session in production, so a process-shared pool would leak state
  between sessions.
- `source_provenance` needs an online re-open for its incremental `encode_fact`.

## What was built (commits `1fe3f56e`, `f9e03e44`, `f77556db`; no `sim/` edit)

- `research/runners/onebrain_affect_pool.py`
  - **`AFFECT_DESCRIPTOR`** is the production ladder's exact region and pathway spec, reused by import. A unit test
    checks that it matches the standalone ladder.
  - **`PoolAffectLadder`** runs the same settle, ramp, drive-off and read protocol on the pool slice. It uses a
    local per-neuron OU stream and local Hebbian-off inside `sequence_isolation`. The appraisal reaches the rungs
    only through the relay synapses, and the anti-cheat assertion is kept.
  - **`AROUSAL_XEDGE`** connects all 160 arousal-rung neurons to the `surprise` pool at a fixed, hand-set weight
    of 0.05. It sits behind the transmission gate `affect_arousal_to_surprise`.
- **Flags.** `BRAIN_ONEBRAIN_AFFECT_POOL` routes the production affect read and `get_merged_cortical_pool` onto the
  12-organ pool. `BRAIN_ONEBRAIN_AFFECT_XEDGE` adds the synapse. Both default to OFF and are read from an
  import-light module.
- **`CrossEdge.transmission_gate`.** An optional field on `CrossEdge`. When it is `None`, the dense dict is
  unchanged (tested).

## Measured (seed 42, numpy, N=7692 neurons on the 12-organ pool)

The smoke is `research/findings/raw/_onebrain_affect_pool/smoke_seed42.json`. All 7 of its checks pass:

- **Co-residence.** Affect read on the 12-organ pool matches affect alone on the superset config, with max |delta| = 0.0.
- **Tone levels.** The graded tone levels match the standalone production ladder at every production appraisal:
  (-3, -2, 0, 2, 3) on both.
- **Alive checks.** Correct sign at |a| >= 0.5. The neutral read is exactly 0.0. The affect_out lesion gives 0.0.
- **Determinism.** Two reads give identical results.
- **Edge isolation.** The arousal synapse leaves the ladder's own read unchanged (max |delta| = 0.0).

The continuous differentials differ from standalone, as expected: per-neuron het and OU are realized differently.
Pool vs standalone (rounded from the smoke artifact) <!--derived-->: -0.0772 vs -0.0833, -0.0394 vs -0.0389, 0.0347 vs 0.0378, 0.0800 vs 0.0864.

<!--derived-->
| effective arousal->surprise weight | contradict shift, a=+1 vs a=0 (Hz) | shift, a=-1 (Hz) | confirm at a=+1 (Hz; threshold 2.629) |
|---|---|---|---|
| 0.02 | +0.094 | +0.094 | 0.007 (same as base) |
| 0.05 (the hand-set `XEDGE_W`) | +0.224 | +0.224 | 0.007 (same as base) |
| 0.10 | +0.376 | +0.376 | 0.022 |
| 0.20 | +0.644 | +0.644 | 0.029 |
| 0.40 | +4.398 | +4.398 | 3.299 (confirm over threshold; runaway) |

Source: `research/findings/raw/_onebrain_affect_pool/calibrate_seed42.json`. The latched arousal rungs fire at 78.4 Hz during the surprise window, and the
relays are silent there (asserted on every step). At w=0.05 the contradict rate rises and confirm stays at its
0.007 Hz floor. That does NOT show a gain effect: confirm sits at a ~0 Hz floor, where a sub-threshold additive
(DC) depolarization would also leave it unchanged, and at w=0.4 confirm jumps to 3.30 Hz, which looks more like
DC. Gain vs DC was not tested and no claim is made. This sweep was also measured at a non-production operating
point (see the fix round), and at 600 pA every contradiction was already flagged at a=0 (all 8 per-block rates
2.66 to 5.21 Hz vs threshold 2.63), so no verdict could change.

## Honest residuals

1. **Mechanism level only.** Production organ reads are isolated per organ. The surprise read hard-resets the
   bridge, and the affect read runs inside `sequence_isolation`. So in a live turn the held arousal does not yet
   reach the surprise verdict. The next rung is a turn-scoped shared-state protocol. Until that exists, this synapse
   is not load-bearing on a reply.
2. **Symmetric by construction.** The +1 and -1 shifts are identical because the arousal relay is driven by
   |appraisal|. So "valence-independent" is a property of the design, not a discovery.
3. **Thin +0.5 margin.** At a=+0.5 the pool differential of 0.0347 is only 0.0047 above <!--derived--> the level-2 boundary of 0.03.
   The M3 check, where tone levels must equal standalone, could flip on another seed.
4. **Hebbian clip trap.** Found while building: with the pool's global Hebbian on, one read on an affect-only config
   clipped the `plastic=False` ladder edges to `hebbian_max_weight` (28 -> 1.0). This is the known plasticity BOUND
   TRAP. The read now turns Hebbian off locally, which is faithful because the standalone ladder has it off. M5
   checks the ladder weights across the whole 12-organ lifecycle.
5. **Seed 42 was the v1 sweep seed** and is also one of the 6 gate seeds. The sweep no longer sets anything: the
   weight is hand-set, and the v2 diagnostic sweep runs on non-gate seed 7.
6. **No same-count control.** No arm adds the same number of excitatory synapses from non-affect cells. So the arm-X
   result, if it passes, shows that the ladder's own held arousal, through this synapse, changes the surprise
   verdict. It does not show anything specific to arousal beyond "extra excitation onto surprise".
7. **Production strength is saturated.** Production drives every assertion at 600 pA. The v1 seed-42 read flagged
   every contradiction at that strength, so the v2 functional check reads at a marginal (weaker) strength. The
   production-strength verdict changes are reported, not gated. Making the edge matter at production needs graded
   assertion evidence, for example assertion drive set by the comprehension parser's confidence.

## Fix round (2026-09-23)

1. **Gain claim withdrawn** (review issue 1). "Gain-like, not a DC bias" is removed from this finding and from the
   module docstring. LC-NE adaptive gain (Aston-Jones & Cohen 2005) stays only as the biological motivation. The v2
   runner reports mean contradict Hz per assertion strength at a=0 and a=+-1 as raw data, with no gain claim.
2. **`XEDGE_W` provenance** (issue 2). 0.05 was hard-coded in the build commit `1fe3f56ef` (08:59:38). The seed-42
   sweep started at 08:55:41 and took about 1268 s, so it finished after the weight was set, and its result
   (`f77556db`) agreed with it. The weight is a hand-set constant. `--calibrate` is now a diagnostic on non-gate
   seed 7 and does not set it (`onebrain_affect_pool.py` comment corrected in `0f1f35ff1`).
3. **Operating point** (issue 3). v1 ran the surprise window inside the ladder's 8 pA OU on every pool neuron,
   while the production surprise read is noise-free. `local_ou(scope="affect")` now confines the OU current to the
   affect organ's neurons through the engine's `cp_ou_neuron_mask` seam. The surprise read uses the organ's own
   `_drive_read` sequence with `_step`. New check X0 requires that, at a=0 and at every tested strength, each trial's
   verdict equals the verdict of the organ's literal production read path.
4. **Functional gate** (issue 4). The 0.10 Hz floor (about 2% of a saturated 4.5 Hz baseline) is replaced by X1. S*
   is the largest grid strength at which the a=0 brain flags at most half the contradictions, chosen from the a=0
   read only. At S*, the held arousal must newly flag at least max(2, ceil(n/4)) contradictions at a=+1 and at
   a=-1. No S* means UNDEFINED, which is not a pass. X2 requires no confirm false alarm and no lost detection at the
   production 600 pA. Lesion, no-edge, intero-null, byte-off, held-arousal, determinism and OU-scope checks are
   relabelled I1-I7 integrity smokes: required, but they pass largely by construction. (The "and at a=-1" half and
   X2's evidential label are corrected in fix round 3 below.)
5. **Order robustness** (issue 8). `local_ou` now saves and restores the prior OU state instead of setting it to
   None (unit test `test_local_ou_restores_prior_ou_state_and_scope_mask`).
6. **Process** (issues 5-7). The shared `research/queue/.lane_waiver` written by the build agent was deleted. The
   ARM-M seed-42 job was moved up to the ARM-M block at the head of `pool.queue`. The v1 arm-X jobs on pool42
   (seeds 42, 44, 100) were stopped and the queued v1 seed-43 line removed; their instrument is superseded. The
   isolated-revision corpus gap is closed on main by the provisioner's corpus sync, and `tools/pool_sync.sh` now
   harvests revision-dir results.

The amended gate was committed in `cd5288018` before any v2 run, with an AMENDMENT LOG listing what had been seen.

**v2 instrument smoke (not a gate run).** A 2-organ pool (surprise + affect + the edge) on non-gate seed 7, run by
`research/probes/onebrain_affect_xv2_instrument_smoke.py`. Artifact:
`research/findings/raw/_onebrain_affect_pool/xv2_instrument_smoke_2organ_seed7.json`. Readings:

- **Operating point.** At a=0, every verdict at every grid strength equals the production read path, with max
  |delta Hz| = 0.0.
- **S*.** 350 pA. The a=0 brain flags 3 of 8 contradictions there, and all 8 at 600 pA.
- **Flips.** At S*, a=+1 newly flags 2 of 8 contradictions and loses none. That is exactly the pre-registered
  minimum of 2, so the margin is thin.
- **Specificity.** No confirm false alarm at 600 pA.

This shows the instrument runs and reads at the production operating point. It is not evidence for the 12-organ,
6-seed gate.

## Fix round 3 (2026-09-23, after the re-review of `4708ebdee`): arm-X gate v3

The gate was amended again in `254608eba`. That commit changed only the scoring and the attribution calls; the
measurement is unchanged. No 6-seed v2 arm-X result had been read when it was committed.

1. **Attribution restored.** v1 called `tools.lab` `lever` and `attributable_to` on the edge-lesion pair. The v2
   rewrite dropped both calls, and the BLOCK-class `attribution-required` gate went red on the runner. Both calls are
   back, now on X1's newly-flagged count:
   - `lever` compares the a=+1 verdicts at S* between the intact edge and the edge lesion.
   - `attributable_to` tests that count against the edge lesion, the intero-null and the no-edge pool.

   The results are reported per seed. The gate slipped through because pre-commit only passes ADDED files, so the
   gate itself gained a regression mode in `f693fcdb8`: a staged, modified runner that passed before and fails
   after is now blocked. `research/FAILURE_LOG.md` has a row for this.
2. **X1 counts one test, not two.** The arousal relay is driven by |appraisal|, and the arousal rungs get no input
   from the valence rungs. So the a=-1 read duplicates the a=+1 read by construction. X1 now requires the per-seed
   minimum at a=+1 only. a=-1 is reported, together with a flag for whether its per-trial Hz at S* equal a=+1's
   exactly, and is never counted.
   - The per-seed threshold max(2, ceil(n/4)) is **unchanged**. Raising it now would pick a threshold that the only
     v2 read (seed-7 smoke, 2/8) already fails. Nothing justifies lowering it.
   - What changes is how much evidence is claimed. X1 is ONE test of about 8 trials per seed. Its only replication
     is across the 6 seeds, and all 6 must pass.
3. **v2's X2 is now I8, a safety check.** It can barely fail at w=0.05:
   - At 600 pA the a=0 brain already flags every contradiction, and arousal only adds excitation.
   - Confirm sits at about 0 Hz, and w=0.05 does not lift it. In the v1 sweep, confirm crossed threshold only at
     w=0.4.

   It stays required, as a guard against regressions, but it is not evidence. The evidential set is now X0
   (operating point) plus X1 (function). Only X1 is evidence FOR the effect.
4. **One scorer.** Arm-X scoring is the pure function `score_x_arm`. `aggregate` re-scores every current-instrument
   record from its raw batteries with it and ignores the checks stored in the file. The v2 X jobs at revision
   `c6fdf7be7` are therefore scored by gate v3. A test pins their battery code as AST-identical to HEAD
   (`test_x_instrument_code_unchanged_since_the_staged_v2_revision`).
5. **Pool.** State as checked at 13:3x, from `ps` on the nodes:
   - No superseded v1 job is running. The v1 X101 and X102 on pool41 had already finished at 13:10 and 13:22; their
     files are superseded and were not opened. pool42 runs only the arm-M jobs, whose code is unchanged. pool40 is
     unreachable.
   - The seven v2 X/calibration lines were still queued. Two of them, X42 and X43, were dispatched to pool41 at
     13:38, before the amendment was committed. The other five were pulled out under the dispatcher's lock and
     re-queued unchanged, apart from a 4 GB virtual-memory cap (`ulimit -v`; the jobs measure about 0.8 GB VSZ).

## Fix round 4 (2026-09-23, `fdd8263b6`): two scorer loopholes closed in `aggregate()`

Re-review of `dcaaa2f0f` found two loopholes in `aggregate()` that would let it declare a false GO over the staged
xv2 harvest. Both are scoring/counting fixes only — no threshold, grid, S* rule, weight or measurement changed.

1. **SCORER LOOPHOLE.** `n_go` used to count every seed present in `by_seed`, including a non-gate seed (e.g. the
   diagnostic seed 7, whose verify-mode X file matches the harvest glob). A passing non-gate seed could stand in
   for a failing GATE seed and still read N/N GO. Fixed: the GO count and the ALL-GO denominator are now restricted
   to exactly the 6 registered `SEEDS = (42, 43, 44, 100, 101, 102)`; a non-gate seed is reported (and printed) but
   never counted, and ALL-GO additionally requires every gate seed present with no duplicate among them.
2. **Duplicate records.** More than one record contributing the same arm's checks (M, or a current-instrument X)
   for one seed used to resolve silently through `dict.update` last-wins — an order-dependent selection lever a
   rerun/retry file could exploit to overwrite a failing verdict with a later passing one. Fixed: a seed with
   duplicate records now reads `DUPLICATE-RECORDS -> UNDEFINED`, never GO, regardless of file order.

Pre-registered as a gate v3 amendment in the module docstring's AMENDMENT LOG, committed before any xv2/`verify_X_seed*.json`
result existed to be read. `tests/test_onebrain_affect_pool.py::test_aggregate_go_count_is_restricted_to_exactly_the_registered_gate_seeds`
and `::test_aggregate_duplicate_record_for_one_seed_reads_undefined_not_last_wins` both fail on the pre-fix `aggregate()` and pass after.

## The GO gate (pre-registered in the runner docstring) — SCORED 2026-09-23, NOT ALL-GO

The literal command is the `--aggregate` line in the runner's docstring:

```
python -m research.runners._onebrain_affect_pool_verify --aggregate \
    'research/findings/raw/_onebrain_affect_pool/verify_M_seed*.json' \
    'research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed*.json'
```

**Correction (this doc previously said these files "do not exist yet" — stale as of the mini-PC pool harvest
below).** ARM M has now landed on all 6 gate seeds, verified by provenance to all be at revision `bfc6978`
(`verify_M_seed42/43/44/100/101/102.json.prov.json` each carry `git_sha=bfc6978fada92e309013e4ec39fb1b9a44627f9d`):
`verify_M_seed42.json`, `verify_M_seed43.json`, `verify_M_seed44.json`, `verify_M_seed100.json`,
`verify_M_seed101.json`, `verify_M_seed102.json` (all under `research/findings/raw/_onebrain_affect_pool/`).

**Correction 2 (round-6 re-review, 2026-09-23): "its code path is unchanged since" was FALSE.** The claim implied no
later commit touched `onebrain_affect_pool.py` after `bfc6978`. `0f1f35ff1` (the same fix round that produced the v2
arm-X instrument, landed after `bfc6978`) rewrote `PoolAffectLadder.local_ou`: it now saves/restores the bridge's
prior OU-related attributes on exit instead of unconditionally setting them to `None`, and it added a `scope`
parameter (`"all"` vs `"affect"`). Checked whether this can affect the ARM-M reads above, by reading the diff rather
than assuming: every ARM-M call path (`_reads_all`/`_isolated_reads` for M1/M2/M4/M5/M6, and the two direct
`read_differential` calls used by M3/M7) omits `ou_scope`, so it takes the default `"all"` — unaffected by the
`scope="affect"` addition, which only `_onebrain_affect_pool_verify.py`'s arm-X code (`local_ou(scope="affect")` at
its X-arm block) requests. The save/restore-vs-`None` change is also a no-op for ARM M's sequential, noise-off,
single-scope reads: the affected bridge attributes cycle absent/`None` -> set -> `None` either way, since each
ARM-M read opens and closes its own `local_ou()` context with nothing else touching those attributes in between.
**Conclusion: the ARM-M verdicts recorded in the six files above are NOT affected by `0f1f35ff1`**, but the
parenthetical's literal wording was still false, and is corrected here rather than repeated.

**Correction 3 (round-7 re-review, 2026-09-23): the previous "2 of 6 landed / mid-harvest / not a final verdict"
framing directly below was FALSE at the time it was committed (`21225228c`, 18:04:46 EDT).** All six v2 ARM-X
files (revision `c6fdf7be7`) had already been harvested to the primary checkout's local disk well before that
commit -- by local filesystem birth time: seed 43 by 16:55:32, seed 42 by 17:11:06, seeds 44 and 101 by 17:42:03,
and seeds 100 and 102 by 17:57:59, all EDT, all more than 6 minutes before the 18:04:46 commit. (Using local
birth time rather than the files' preserved-from-remote mtime, per the lesson from a sibling lane's round-7 review
that mtime can be mistaken for local arrival time.) The doc's own two paragraphs above already admitted seeds
44/100/101/102 existed uncommitted in the primary checkout, which directly contradicted the "4 ... still
outstanding" and "mid-harvest, not a final verdict" language that followed -- both are withdrawn here. All six
files are now committed to this branch/finding, at `research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed{42,43,44,100,101,102}.json`
(+ `.prov.json` sidecars, each carrying `git_sha=c6fdf7be7673b264316888615be64883f23f48cf`, `git_dirty=false`,
`source_kind=git_archive`). `c6fdf7be7` remains an ancestor of this branch's HEAD with the scored battery
functions AST-identical since (pinned by
`tests/test_onebrain_affect_pool.py::test_x_instrument_code_unchanged_since_the_staged_v2_revision`; the three
commits between `c6fdf7be7` and HEAD touch only `aggregate`'s scoring/counting and doc/path text, not the
measurement). `aggregate` ignores arm-X checks from any record without the v2 `x_instrument` tag and re-scores
records that carry the tag with gate v3 (`score_x_arm`).

Running the pre-registered `--aggregate` command above (gate `v3-single-count-X1-X2-as-I8-attribution-2026-09-23`,
after the fix-round-4 scorer repair) over the complete, committed 6-seeds-per-arm file set gives, per seed:

| seed | ARM M | ARM X (X1 a=+1 flips @ S*, need >= 2) | GO | why not |
|---|---|---|---|---|
| 42 | pass (M1-M7) | fails (1 flip) | False | `X1_functional_verdict_flip_at_marginal_strength`: only 1 newly-flagged flip at S* |
| 43 | pass (M1-M7) | passes (5 flips) | **True** | — |
| 44 | pass (M1-M7) | passes (2 flips, exactly the minimum) | **True** | — |
| 100 | pass (M1-M7) | fails (1 flip) | False | `X1_functional_verdict_flip_at_marginal_strength`: only 1 newly-flagged flip at S* |
| 101 | pass (M1-M7) | passes (4 flips) | **True** | — |
| 102 | **fails** M3 | passes X1 (2 flips, exactly the minimum) but the seed is already disqualified by M3 | False | `M3_affect_tone_levels_equal_standalone` |

`ALL-GO (6/6 GATE seeds only, every M1-M7 + X0-X1 + I1-I8): False` — **3/6 gate seeds (43, 44, 101) fully GO. This
is the final scored verdict over the complete, pre-registered, 6-seed-per-arm gate — not a mid-harvest read; no
seed remains to land.** GO needs every one of M1-M7, X0-X1 and I1-I8 to hold; X1 fails outright on 2/6 seeds (42,
100). Of the two fully-GO seeds beyond 43, seed 44 sits at the pre-registered minimum with no slack (exactly 2/8
flips) while seed 101 has 4/8; seed 102 also reads exactly 2/8 on X1 but is disqualified by its M3 failure, which is the first actual
ARM-M counterexample to the "M is unchanged and still part of GO" framing above). Only X1 is evidence for the
effect, and it is one test per seed, replicated only across the 6 seeds -- of which 2/6 fail it outright and the
passing margin on the rest is thin. The gate's own verdict is NOT ALL-GO; the next methods, should this mechanism
be revisited, are:
- a weight derived from the seed-7 diagnostic calibration (`calv2/`), committed before any gate re-run;
- graded assertion evidence, so that the production strength is not saturated.
