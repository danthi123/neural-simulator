# WORKLOG stage1-host-share (T22)

## 2026-10-02 00:41 EDT - sections 0 + 1 done
- Section 0: worktree `research/stage1-host-share` from main @
  PIN=40c43d520cbee67210b85ad767e88bdd176f88c1 (symlinks .venv + data); PINS.tsv + WORKLOG.md
  created. Tests `nice -n 10 pytest -q tests/test_stage1_host_share.py tests/test_host_share.py`
  -> **23 passed in 0.89s**; `--selftest` -> **SELFTEST OK**. Pin committed
  "ops(stage1-host-share): pin P=40c43d520". Provision request: state already **ok** (same pin as
  T18's request minutes earlier; all 4 nodes).
- Section 1 (step D): `make_jobs.sh dev 40c43d520...` -> dev_40c43d5.tsv **22 lines** (expected 22),
  committed. All 22 queued via `pool_queue.sh add --checked "... ; max_age_h=72"`: 0 failures,
  depth 403 -> 425. Shards land in research/findings/raw/_stage1_host_share/dev/s7/.
- NEXT: step 2 devcheck when all 22 shards are in (then step 3 gate lines, step 4 canary per
  RULING s1hs-canary + queue 132, step 5 wait, step 6 score). Note step 6 says push_both - per
  RULING push-credential (commit on the branch, never push). Queue is 425 deep (FIFO).

## 2026-10-02 05:58 EDT - section 2 first attempt: rc=1 (13/22 shards in), missing 9 all in flight
- devcheck rc=1 with ONLY "no shard missing" FAIL; the other 10 checks PASS. LIVE SEEDS line
  verbatim (reported only): {"C2": {"k-select": 1, "bg-select": 1, "organ-c": 1, "metacog": 1,
  "ignition": 1}, "C3": {"k-select": 0, "bg-select": 0, "organ-c": 0, "metacog": 0, "ignition": 0}}.
  Committed data + devcheck.txt as 45bab590e (PROVISIONAL, no verdict).
- Re-queue ONCE attempted for the 9 absent s1_lived outputs: every add REFUSED as already
  RUNNING (5: pool40 x3 since 04:57-05:09, wspool x2 since 05:36-05:46) or already queued (4) -
  all still in flight from the original 22-line queue; no FORCE_DUP used (a replication would be
  wrong).
- NEXT: when the 9 s1_lived shards land, re-run section 2 (attempt 2 of the ONCE rule; still rc=1
  then -> NEEDS-CLAUDE).
2026-10-02 stage1-host-share: section 2 attempt 2 DEVCHECK OK (22/22, all 10 checks) but commit blocked by corpus-check-required on 3 cells (s1_lived intact_a/intact_b/metacog_intervene, 1.0-1.2h compute, no fresh corpus check before their 04:57-06:09 runs). before_you_build ran 10:34 (record consulted; priors = the packet's own prereg); the 3 identical lines re-queued 10:40 so the consult precedes the re-run (dup-guard saw dead claims, no FORCE_DUP); attempt-1 copies at /opt/data/cache/scratch/t22_attempt1. Resume: devcheck + commit when they land.
2026-10-02 stage1-host-share: section 2 attempt 2 DEVCHECK OK (22/22, all 10 checks) but commit blocked by corpus-check-required on 3 cells (s1_lived intact_a/intact_b/metacog_intervene, 1.0-1.2h compute, no fresh corpus check before their 04:57-06:09 runs). before_you_build ran 10:34 (record consulted; priors = the packet's own prereg); the 3 identical lines re-queued 10:40 so the consult precedes the re-run (dup-guard saw dead claims, no FORCE_DUP); attempt-1 copies at /opt/data/cache/scratch/t22_attempt1. Resume: devcheck + commit when they land.

## 2026-10-02 13:33 EDT - section 2 complete (attempt 3)
- DEVCHECK OK rc=0, 22/22 shards, all 10 checks PASS (incl. `no shard missing`); data commit `358e52adf`.
- Corpus gate fix: 3 cells (s1_lived intact_a/intact_b/metacog_intervene, 1.0-1.2h compute) had no fresh corpus check before their 04:57-06:09 runs. before_you_build ran 10:34 (priors = the packet's own prereg), identical lines re-queued 10:40, re-ran 11:25/11:34/11:35 (landed 12:04/12:15/12:28); gate accepted via fallback (consult precedes re-run starts). attempt-1 copies: /opt/data/cache/scratch/t22_attempt1.
- LIVE SEEDS per criterion x seam (reported, not a pass condition): {"C2": {"k-select": 1, "bg-select": 1, "organ-c": 1, "metacog": 1, "ignition": 1}, "C3": {"k-select": 1, "bg-select": 1, "organ-c": 1, "metacog": 1, "ignition": 1}}
- Next: step 3 gate lines (132), step 4 canary + queue.
- Section 4: canary PASS recorded on pool42 (complete artifact 14:30, prov at_exit clean; pool41 mem_ok refused 7G, pool42 dispatch false-negative but setsid job completed); queued 132 gate lines (failures=132, max_age_h=168).
- Correction: the previous line's '(failures=132)' was the add-refusal count (canary record had the bare module name; re-recorded with research.runners.stage1_host_share); all 132 gate lines are now queued (final added=8, fail=0 on this retry).
- 2026-10-03 14:45 EDT - section 5 watch: gate = 128/132 (s42/s43/s44/s101/s102 complete at 22); the 4 remaining s100 cells (intact_a, intact_b, metacog_substitute, ignition_intervene) were DEAD CLAIMS: claimed on wspool 01:23/01:35/04:37/04:37 EDT, alive at the 13:35 T0 (running=4), gone by 14:40 with no exits.json record, no output, nothing queued/running on any node (pool40/41/42/wspool) - the 14:37 sandbox-state event killed them. ONCE requeue unused for these 4 (the 10:32 requeue covered only the 2 rc-failed s101/s102 cells, which landed 14:34). Section 3 block re-queued all 4 (`re-queued 4`, depth 203-206; queue's duplicate-claim detector confirmed each: "previous identical claim exists ... but is no longer alive"). Next: gate 132/132 -> step 5 wait done -> section 6 (score + finding + RESULT; note the ≤132 clause: write with what exists and flag if the missing re-runs do not land). Never push (sim-sync publishes main).
