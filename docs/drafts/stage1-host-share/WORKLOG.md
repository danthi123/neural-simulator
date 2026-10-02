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
