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
