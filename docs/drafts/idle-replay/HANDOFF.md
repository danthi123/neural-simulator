# HANDOFF: idle replay (Stage-2 C5 build), dev checks for Hermes

Written 2026-10-06 by Claude. Plan: `docs/drafts/idle-replay/PLAN.md`. Execute top to bottom. Do not change code, grid
rows, thresholds, turns or arms. Do not flip any default (Claude writes the flip after reading section 6). At most ONE
re-queue per missing result; anything not covered here is a NEEDS line `[s2-c5-build]` and you stop that line of work.

- Never merge to main, never `--no-verify`, never write waiver files. Dev seeds 1-8 only; no gate seed is ever run
  by this packet (the battery refuses them).
- `P=/home/dant123/Projects/neural-simulator`, `PY=$P/.venv/bin/python`, `W=$P/.claude/worktrees/stage2-exit-gate`.
- No brain-loading run on the server: every brain run below is a pool line.

## 0. Preconditions

1. `git -C $P merge-base --is-ancestor claude-review/s2-c5-build main && echo C5-ON-MAIN` prints `C5-ON-MAIN` (the
   orchestrating Claude merges the branch). If not: STOP, NEEDS.
2. The reply-level lines need a revision holding BOTH this organ (main) and the Stage-2 runner
   (`research/stage2-exit-gate`). Sync the gate branch the usual way: `cd $W && bash tools/agent_sync_branch.sh
   research/stage2-exit-gate`. A conflict: STOP, NEEDS (do not resolve). Do not touch `docs/drafts/stage2-exit/GATE_SHA`
   (the gate HANDOFF re-pins it in November).
3. `cd $W && SHA=$(git rev-parse HEAD) && git cat-file -e $SHA:research/runners/idle_replay_organ.py && git cat-file -e
   $SHA:research/runners/stage2_exit_gate.py && echo BOTH-OK` prints `BOTH-OK`.

## 1. Tests (server, CPU, seconds)

```bash
cd $W && SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_idle_replay_organ.py | tail -1        # 14 passed
cd $W && SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_stage2_exit_gate.py | tail -1         # all passed
git -C $W checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null
```

Any failure: STOP, NEEDS.

## 2. Queue (pool, by 2026-10-13)

```bash
cd $W && SHA=$(git rev-parse HEAD)
bash $P/tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
SIM_NO_PROVENANCE=1 $PY -m research.runners.idle_replay_battery --emit-organ-lines $SHA > /tmp/ir_organ.tsv   # 8 lines
SIM_NO_PROVENANCE=1 $PY -m research.runners.idle_replay_battery --emit-dev-lines $SHA > /tmp/ir_dev.tsv       # 34 lines
wc -l /tmp/ir_organ.tsv /tmp/ir_dev.tsv
```

Canary one reply line first (`bash $P/tools/pool_canary.sh run auto '<the s7 inner intact_a flag-1 line>'`); canary
rc != 0: STOP, NEEDS with its log tail. Then add every line with `bash $P/tools/pool_queue.sh add` (the 8 organ lines
are seconds each; the 34 reply lines are mem_gb 9, about 15-50 min each, about 20 pool job-hours). Also queue ONE
regression-battery line at the same revision (flag ON vs OFF across the default-on faculties):

```
cd ~/derisk-pool/revisions/$SHA && bash tools/mem_ok.sh 9 1 && bash tools/memcap.sh 10 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.onebrain_regression_battery --flag BRAIN_IDLE_REPLAY --out research/findings/raw/_regression_battery/battery_BRAIN_IDLE_REPLAY.json	idle-replay C5 build: integrated no-regression battery for the flip; mem_gb=9 max_age_h=168
```

## 3. Score (server, CPU, seconds; when the lines have landed in $P)

```bash
cd $W && SIM_NO_PROVENANCE=1 $PY -m research.runners.idle_replay_battery --score $P/research/findings/raw/_idle_replay/organ --op 0
cd $W && SIM_NO_PROVENANCE=1 $PY -m research.runners.idle_replay_battery --score-reply $P/research/findings/raw/_idle_replay/dev
```

Expected (the workstation dev record and smoke): organ `PASS` 8/8; reply `PASS` 8/8, `answer_rate.status` `OK`,
`s7_flag_on_vs_off.only_idle_turn_differs` true. Regression battery: `all_pass` true, 0 regress.

## 4. STOP conditions (each is a NEEDS line `[s2-c5-build]` with the printed JSON, then stop)

- organ verdict FAIL, or UNDEFINED after one re-queue of the missing seeds;
- reply verdict FAIL, or UNDEFINED after one re-queue; any row with `route` not `idle_replay`;
- `answer_rate.flags` contains `FLIP-HURTS-ANSWERS` (RULING flip-hurts-answers);
- `s7_flag_on_vs_off.differs` is anything but `["p_private"]` (the flag must change only the idle turn);
- the regression battery reports any regressed faculty;
- any shard with an errored turn, an `idle_replay` decode error, or peak RSS above 9.5 GB.

## 5. Report

Commit the result files by pathspec (`research/findings/raw/_idle_replay/`, the battery JSON) in a data commit on
main as usual, then one runbook NEEDS line: `[s2-c5-build] INFO: organ <verdict n/8>; reply <verdict n/8>; answer_rate
<status>; s7 on/off differs <list>; regression <all_pass, n regress>`. Do not write a finding and do not flip. Claude
reads it, writes the flip (section 6 of the PLAN) and the WORKLOG line.

## WORKLOG

- 2026-10-06 Claude: built on `claude-review/s2-c5-build`; organ dev battery PASS 8/8 at row 0 (workstation numpy,
  `docs/drafts/idle-replay/dev_organ_battery.json`); bar-sanity OK (`bar_sanity_record_c5_organ.json`).
