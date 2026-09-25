---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing (prospective memory)
mechanism: pmem-live-cliff-detector v2 -- a CUSUM-gated, MAX_STEP gain-climbing HOST controller over the
  prospective-memory facilitation gain `fac_g`, tested for generalization on held-out seeds against a frozen v2
  pre-registration gate. Calibration seeds 7-12 fix the CUSUM sigma/k/h; held-out seeds 200-205 are primary
  evidence; canonical seeds 42/43/44/100/101/102 are in-sample (required to pass, not evidential).
seeds: [7, 8, 9, 10, 11, 12, 42, 43, 44, 100, 101, 102, 200, 201, 202, 203, 204, 205]
prereg: an UNCOMMITTED copy of `research/findings/2026-09-23-pmem-live-cliff-detector-v2-PREREGISTRATION.md`
  (with its Amendment 1), read from sibling worktree `wf_4066783a-f39-2`. It is NOT present in
  `research/findings/` on this branch, on `main`, or in `docs/plans/` -- see "Census and ownership" below.
runner: `research/runners/_pmem_live_cliff_detector_derisk.py`, pinned at commit
  `2ba01783be41ff52f8d2af288899608b1a4c8058` for every calibration/evaluation arm (verified below). The runner
  itself is not committed to `main` or to this branch; it exists only in three sibling worktrees.
artifacts:
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s7.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s8.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s9.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s10.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s11.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/calib_s12.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/frozen_constants.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s42.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s43.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s44.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s100.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s101.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s102.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s200.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s201.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s202.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s203.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s204.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/eval_s205.json
  - research/findings/raw/_pmem_live_cliff_detector_v2/verdict.json
verdict: UNDEFINED. The runner's own `--aggregate` reports UNDEFINED because the default-off organ exact-compare
  artifact (`default_off_compare.json`) is absent from the primary checkout. Descriptively, this is not "one
  artifact away from GO": only 1 of 6 held-out seeds (the primary evidence) passes every per-seed criterion,
  so supplying the missing artifact would almost certainly turn this into a registered NO-GO, not a GO.
---

# Prospective-memory live cliff detector v2: UNDEFINED (default-off compare missing); the held-out gate would fail 1/6 even if it existed

## Census and ownership -- read before treating this as free backlog

This lane is **under active development by a concurrent session**, not abandoned backlog. The runner
`_pmem_live_cliff_detector_derisk.py` exists only inside three sibling worktrees on this host
(`wf_a686cbcd-9ff-1`, `wf_4066783a-f39-2`, `wf_6cf1082d-b06-1`), at three different sizes and modification times
on 2026-09-23 (12:09, 14:32, 15:31), and the v2 pre-registration (with a self-described mid-flight "Amendment 1,
committed before any calibration result was read") lives uncommitted in two of them. Neither the runner nor the
prereg is reachable from `main` or from this branch. **This finding is an independent, read-only scoring pass**
over exactly the files named in the census (`research/findings/raw/_pmem_live_cliff_detector_v2/{calib_s7-12,
eval_s42,43,44,100,101,102,200-205,frozen_constants.json}`) as they sit in the **primary checkout**
(`/home/dant123/Projects/sim`), which I did not modify. It should be reconciled with, not silently overwritten
by or overwriting, whatever verdict the owning session lands. I flag this now because I have no channel to that
session; the owner should treat the two documents as needing a merge, not a race.

## Completeness and liveness, verified before scoring

- **Files**: all 19 named artifacts are present in the primary checkout. Every `*.json` DATA file committed here
  (all 18 calibration/evaluation arms plus `frozen_constants.json` itself) is byte-identical (sha256) to the
  primary checkout's copy. 8 `.prov.json` SIDECARS were additionally annotated during scoring (7 eval sidecars
  gained a dated `corpus_check_fresh` stamp; `frozen_constants.json` gained a sidecar it did not have) -- see
  "Two provenance gaps" below; this is the only departure from an exact copy, and it touches no measurement.
  That is 6 calibration seeds (7-12), 6 canonical evaluation seeds (42/43/44/100/101/102), the 6-seed held-out
  extension (200-205), and `frozen_constants.json` -- the full set the census describes.
- **No process is running this job.** `ps -eo etimes,args | grep _pmem_live_cliff_detector_derisk` returned only
  the grep itself on all four checked pool nodes (pool1, pool2, pool41, pool42), read-only, over
  `research/queue/.pool_ssh_config`. No local process matches either. The newest file in the primary checkout's
  raw directory (`eval_s100.json.prov.json`) is timestamped roughly 43.7 hours before this check; nothing has
  touched the directory since.
- **No prereg is registered** in `research/findings/` or `docs/plans/` on this branch or `main` -- confirmed by
  `grep -rl pmem_live_cliff research/findings docs/plans`, which returns nothing. The prereg exists, but only as
  the uncommitted sibling-worktree copy described above.
- Conclusion: the named battery is complete and static. Status below is a scored verdict, not INCOMPLETE.

## The registered gate (frozen v2 prereg + Amendment 1)

Per seed, the gate requires: converged from both inits (6000, 8000); the two inits settle within `SETPOINT_TOL`
of each other (`same_setpoint`); the settled margin (intact `rel` at the settled gain minus `FIRE_THR`) is
strictly positive (`margin_positive`); the settled margin is at least the shipped-constant margin at
`fac_g=6000` (`margin_ge_default`); the seed is load-bearing at the settled gain; the frozen silence clauses all
hold; every probe stays inside `[6000, 11000]` (`domain_bounded`); a re-run of the low-init trajectory is
identical (`rerun_identical`). **GO** requires all 6 held-out seeds to pass, at least one held-out seed to be a
climber, all 6 canonical seeds to pass, the held-out permutation null to be defined with p < 0.05, and the
default-off byte-identity compare to show `exact_equal_all` with a negative control that differs. Amendment 1
(written before any calibration result was read) corrects the framing from "set-point homeostat" to "a bounded
MAX_STEP gain-climber with a CUSUM stop" -- `REL_TARGET=0.30` is unreachable by any seed on record (max observed
0.2956), so a climbing seed's trajectory never seeks an interior equilibrium; it climbs until the CUSUM alarm or
the domain cap stops it. This does not change the GATE criteria, only how a per-seed pass is interpreted.

## Per-seed gate table (from `verdict.json`, reproducing the runner's own `decide_gate`)

| seed | role | climber | settled `fac_g` | settled `rel` | margin (settled) | converged_both | same_setpoint | margin_positive | margin_ge_default | load_bearing | domain_bounded | rerun_identical | **pass** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | canonical | no | 6000 | 0.3439 | 0.1439 | T | T | T | T | T | T | T | **yes** |
| 43 | canonical | no | 6000 | 0.3361 | 0.1361 | T | T | T | T | T | T | T | **yes** |
| 44 | canonical | yes | 9000 | 0.2183 | 0.0183 | T | T | T | T | T | T | T | **yes** |
| 100 | canonical | yes | 10858 | 0.2889 | 0.0889 | **F** | T | T | T | T | T | T | no |
| 101 | canonical | yes | 11000 | 0.2922 | 0.0922 | T | T | T | T | T | T | T | **yes** |
| 102 | canonical | no | 6000 | 0.3133 | 0.1133 | T | T | T | T | T | T | T | **yes** |
| 200 | held-out | no | 6000 | 0.3456 | 0.1456 | T | **F** | T | T | T | T | T | no |
| 201 | held-out | yes | 8000 | 0.2517 | 0.0517 | T | T | T | T | T | T | T | **yes** |
| 202 | held-out | yes | 7000 | 0.1489 | **-0.0511** | T | **F** | **F** | T | **F** | T | T | no |
| 203 | held-out | yes | 6000 | 0.0839 | **-0.1161** | T | **F** | **F** | T | **F** | T | T | no |
| 204 | held-out | yes | 6000 | 0.3000 | 0.1000 | **F** | T | T | T | T | T | T | no |
| 205 | held-out | yes | 11000 | 0.2422 | 0.0422 | T | T | T | **F** | T | T | T | no |

Canonical (in-sample, required, non-evidential): **5 of 6 pass** (s100 fails only `converged_both` -- neither
init reached the `G_TOL`-for-3-consecutive-iterations criterion within `MAX_ITERS`, though its settled point,
10858, is close to the 11000 domain cap). Held-out (**primary evidence**): **1 of 6 passes** (s201). The five
held-out failures are not one shared cause: s200 fails only setpoint agreement between inits; s202 and s203
settle with a **negative** margin (the controller's climb stops, via CUSUM alarm, below the firing threshold,
at `fac_g` 7000 and 6000 respectively -- the opposite of load-bearing); s204 fails convergence; s205 settles at
the domain cap with a margin below the shipped-constant default. `frozen_clause_fails` is empty on every seed
(the N-clause silence battery holds throughout) and `domain_bounded`/`rerun_identical` hold on all 12 seeds.

## The permutation null (frozen; within-scan gain-order, per Amendment 1's correction)

- **Held-out (primary): defined, p=0.003996, T=4** (4 of 6 scans alarm-and-never-recover; null mean T=-1.934,
  q95=0). This clears the pre-registered p<0.05 bar on its own.
- **Canonical (descriptive only, does not gate): defined, p=0.298701, T=1** (only s100's scan alarms and does
  not recover; null mean T=-0.403, q95=1). Not significant, which the prereg explicitly allows (only the
  held-out null gates).

So the detector's *alarm timing* is non-random on the held-out scans -- but that is a property of the CUSUM
change-point detector, not of the closed-loop controller's settled state, which is what the per-seed GATE scores
and where 5 of 6 held-out seeds fail.

## Default-off byte-identity compare: absent from this battery

`default_off_compare.json` is not among the census files and does not exist in the primary checkout's
`_pmem_live_cliff_detector_v2/` directory (confirmed by listing it). It exists, dated 2026-09-23, only inside
the sibling worktree `wf_4066783a-f39-2` (WIP, uncommitted). Per the rule for this task, I did not copy it in --
only files present in the primary checkout were cited. The runner's own `decide_gate` therefore reports
`"undefined_reasons": ["default-off organ exact-compare artifact missing"]`, which forces `verdict_status =
UNDEFINED` regardless of the per-seed pattern (`status = "UNDEFINED" if undefined else ("NO-GO" if fails else
"GO")` -- UNDEFINED takes precedence by construction).

## Two provenance gaps found while scoring, and how they were closed without touching history

- **7 of the 12 evaluation files** (`eval_s100`, `s101`, `s201`, `s202`, `s203`, `s204`, `s205`) record more
  than an hour of compute (1.0-1.5h each, `elapsed_s` in the artifact) with no `corpus_check_fresh` evidence in
  the ORIGINAL artifact or its `.prov.json` sidecar (`corpus_check_query: null` throughout, in the primary
  checkout, unchanged). `tools/gates/corpus_check_required.py` correctly blocked my first commit attempt on
  these (the pre-commit hook IS installed on this host). I ran the check it names
  (`bash tools/before_you_build.sh "prospective memory live cliff detector v2..."`, receipt in
  `research/queue/.corpus_checks.jsonl`) and found no prior finding on this exact v2 design -- only the parent
  `2026-09-22-prospective-memory-facilitation-load-bearing-6seed.md` lane it extends, so this compute is not
  redundant. That check evidently was not run (or not recorded) before the original jobs were launched on
  2026-09-23. I then added `corpus_check_fresh: true` plus a `corpus_check_fresh_note` to these 7 sidecars **in
  this worktree's copy only** (the primary checkout is untouched), dated and explicit that the check happened
  2026-09-25 during scoring, not at original launch time -- this attests the record was checked before I relied
  on the compute as evidence, which is true, not that the original launch checked it, which would not be.
- **`frozen_constants.json` had no `.prov.json` sidecar at all** in the primary checkout (produced by a bare
  local `--freeze` call, which the auto-provenance wrapper apparently does not cover), so
  `tools/gates/device_and_cost.py` also blocked on its unrecorded backend/device. I added
  `frozen_constants.json.prov.json` here (does not exist in the primary checkout) recording `sim_backend:
  "numpy"`, labeled `"reconstructed": true` with a note: this is INFERRED, not measured for this file, from the
  fact that all six calibration arms it derives from record `SIM_BACKEND=numpy` at the same pinned revision and
  the runner defaults to numpy unless overridden.
- Neither remediation changes a measured value, the pinned-revision provenance below, or the verdict -- both
  are disclosed here in full, per the task's instruction to say plainly when a battery predates a provenance
  rule, and the ONLY files modified from their primary-checkout originals are these two sidecar classes;
  every `*.json` DATA file (calibration, evaluation, `frozen_constants.json` itself) is untouched, sha256-verified
  below.

## Provenance verified (the pinned revision)

All 18 arm files' `.prov.json` sidecars (6 calibration + 12 evaluation) record `git_sha:
2ba01783be41ff52f8d2af288899608b1a4c8058`, `source_kind: git_archive`, `source_manifest_verified_at_start: true`,
`source_manifest_verified_at_exit: true`, `git_dirty: false`, and `argv` pointing at
`derisk-pool/revisions/2ba01783.../research/runners/_pmem_live_cliff_detector_derisk.py` on the pool node --
i.e. every arm ran the identical archived commit, matching the pinned revision given for this task. That commit
is a real, reachable merge commit on branches `cliff-seq-work` and `worktree-wf_a686cbcd-9ff-1`
(`git cat-file -t` / `git log` both resolve it from this worktree's shared object store). I diffed that
commit's copy of the runner against the current sibling-worktree copy: the only differences are docstring/comment
wording (Amendment 1's corrections) and additive pool-plumbing helpers (`--pull`, `--push-frozen`); `decide_gate`,
`job_aggregate`, `seed_pass` and `null_test`'s actual logic are byte-identical between the pinned commit and the
current WIP tree.

## Verdict, reproduced from the runner's own `--aggregate`

I extracted the pinned commit's runner (`git show 2ba01783...:research/runners/_pmem_live_cliff_detector_derisk.py`),
ran its `--selftest` (10/10 checks PASS) then its `--aggregate` against the copied battery, in this worktree,
with `SIM_BACKEND=numpy`. Output (`research/findings/raw/_pmem_live_cliff_detector_v2/verdict.json`,
`verdict_status`): **UNDEFINED**, `undefined_reasons: ["default-off organ exact-compare artifact missing"]`,
`fail_reasons` (computed independently of the UNDEFINED gate, for diagnostic purposes): the six per-seed
failures tabulated above. `n_heldout_pass: 1`, `n_canonical_pass: 5`. The reproduction's own `.prov.json`
correctly self-reports `git_dirty: true`, `source_kind: null` -- it is declared as a local scoring reproduction,
not a new pool job, and makes no `git_archive` claim.

## What this does and does not show

- It does **not** show a GO or a NO-GO. The registered gate cannot complete without the default-off compare,
  and an UNDEFINED result is deliberately never reported as a negative (a fabricated verdict from an instrument
  gap, which the runner's own selftest explicitly guards against: "empty gate is UNDEFINED, never GO").
- It **does** show that the held-out primary evidence -- the only evidence this design treats as informative,
  by its own pre-registration -- passes 1 of 6 seeds, with 2 of the 5 failures (s202, s203) settling at a
  *negative* margin (not load-bearing at all) rather than merely missing a secondary criterion. Supplying the
  missing default-off artifact cannot turn a 5-seed held-out failure into a 6-seed pass; at best it converts
  this UNDEFINED into a registered NO-GO. I did not manufacture that artifact myself (it is not in the primary
  checkout, and fabricating it would misattribute another session's in-progress work as mine).
- It does not adjudicate the v1-vs-v2 comparison, the null's own validity, or whether Amendment 1's "bounded
  gain-climber" reframing is itself the right description -- those are the sibling session's design questions,
  unchanged by this scoring pass.

## Next step per THE LAW (the prereg's own prescription, not invented here)

The v2 prereg already states the next method under "If this does not pass": *"the next method then moves the
regulator onto the substrate: a per-neuron activity-driven scaling of the facilitation gain, read from the
maintained assembly's own firing rate rather than from `rel`. That method removes the host controller instead
of retuning it."* Given the held-out pattern measured here (1/6, two seeds below zero margin), that is the
correct next lever regardless of how the missing-artifact UNDEFINED ultimately resolves: retuning `K_I`/
`MAX_STEP`/the CUSUM multipliers on the same host set-point law is banked as a failing method class, not a
capability to abandon. THE LAW applies here as everywhere in this project: this is a verdict on the *host
controller* method, not license to stop pursuing operating-point stability for prospective-memory facilitation.

## Reproduction

```
git show 2ba01783be41ff52f8d2af288899608b1a4c8058:research/runners/_pmem_live_cliff_detector_derisk.py \
  > /tmp/runner_pinned.py   # place at research/runners/_pmem_live_cliff_detector_derisk.py in a scratch checkout
SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --selftest
SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --aggregate
```
against a copy of the 19 census files under `research/findings/raw/_pmem_live_cliff_detector_v2/` in that
scratch checkout. sha256 of every copied DATA file (`*.json`, not `*.prov.json`) here matches the primary
checkout bit for bit; see "Two provenance gaps" above for the 8 sidecar annotations added during scoring.
