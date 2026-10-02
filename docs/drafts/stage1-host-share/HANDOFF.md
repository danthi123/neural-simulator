# HANDOFF: stage1-host-share (session-3 packet; execution packet for Hermes)

Written 2026-10-02 by the cloud Claude session 3. Execute top to bottom; use no judgment beyond the tables. Runbook
task: T22. Prereg (frozen groups, turns, arms, validity, credit lines):
`research/findings/2026-10-02-stage1-host-share-PREREGISTRATION.md`. Runner and scorer:
`research/runners/stage1_host_share.py` (imports `research/runners/host_share.py` + `host_share_schema.py`; tests
`tests/test_stage1_host_share.py`). Job lines: `docs/drafts/stage1-host-share/make_jobs.sh`.

What it measures: at the Stage-1 exit gate's C2 (`word`) and C3 (`lived`) probe turns, who owns the decision that
reaches the reply. Five seams (k-select, bg-select; call-graph checks organ-c, metacog, ignition), by intervention: the
neural state at the seam held fixed, only the host decision changed. The result is one credit line per criterion,
`PARTIAL (host decision: <seams>)` or one of the prereg section 6 alternatives, which the Stage-1 exit report prints
beside that criterion. It answers the Stage-1 exit HANDOFF's NEEDS-CLAUDE #10.

Rules for this packet: every step below is routine for you EXCEPT where a row says NEEDS-CLAUDE. You never edit the
runner, the instrument, the tests, the prereg, `make_jobs.sh` or a job line; you never flip a default. RULING
`s1hs-canary` lets you run the pool canary yourself (step 4). "WORKLOG <x>" means append <x> to `$K/WORKLOG.md` (step 0 creates it). The verdict you write is PROVISIONAL until Claude
confirms it; INVALID-OR-INCOMPLETE follows RULING `results-without-pass`.

Every block is ONE Bash call. Set first in each: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees; K=docs/drafts/stage1-host-share; RAW=research/findings/raw/_stage1_host_share`.

## 0. Start condition and worktree

Start when `git -C $P log --oneline -1 -- research/runners/stage1_host_share.py` prints a commit (the owner landed
`claude/session3-packets`). Then:

```bash
cd $P && git worktree add -b research/stage1-host-share $WT/stage1-host-share main && cd $WT/stage1-host-share && ln -s $P/.venv .venv && ln -s $P/data data
PIN=$(git rev-parse HEAD); mkdir -p $K && printf 'P\t%s\t%s\n' "$PIN" "$(date -u +%FT%TZ)" > $K/PINS.tsv
printf '# WORKLOG stage1-host-share (T22)\n' > $K/WORKLOG.md
SIM_NO_PROVENANCE=1 nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_stage1_host_share.py tests/test_host_share.py
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_host_share --selftest | tail -1
git add $K/PINS.tsv $K/WORKLOG.md && git commit -m "ops(stage1-host-share): pin P=${PIN:0:9}" && echo "$PIN"
```

| result | do |
|---|---|
| a test fails, or the selftest does not print `SELFTEST OK` | STOP. NEEDS-CLAUDE "[stage1-host-share] test <name> fails at <sha>" |
| all pass, commit made | provision P: `mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$PIN` (RULINGS `compute-placement`, `iw2-provision`), wait for `ok`/`already` in `/opt/host-status/sim-provision/$PIN.json`; a `refused` naming published main: `bash tools/agent_sync_branch.sh research/stage1-host-share`, re-pin (this block's PIN lines again), request the new head |

## 1. Step D: dev seed 7 (22 jobs, no verdict)

```bash
cd $WT/stage1-host-share && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && PY=$PY bash $K/make_jobs.sh dev $PIN
git add $K/jobs && git commit -m "ops(stage1-host-share): step-D dev lines at ${PIN:0:9}"
cd $P && bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why ; max_age_h=72" </dev/null || echo "ADD FAILED: $cmd"; done < '"$WT/stage1-host-share/$K/jobs/dev_${PIN:0:7}.tsv"
```

Expect 22 lines. Shards land in `$P/$RAW/dev/s7/` (22 files `<group>__<arm>.json`, `:` written as `_`).

## 2. Devcheck (when all 22 dev shards are in)

```bash
cd $WT/stage1-host-share && mkdir -p $RAW && cp -a $P/$RAW/dev $RAW/
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_host_share --devcheck $RAW/dev --seeds 7 | tee $K/devcheck.txt; echo rc=${PIPESTATUS[0]}
git add $RAW/dev $K/devcheck.txt && git commit -m "data(stage1-host-share): step-D dev seed 7 + devcheck (no verdict)"
```

| rc | do |
|---|---|
| 0 (`DEVCHECK OK`) | step 3. WORKLOG the `LIVE SEEDS` line verbatim (reported only; it changes nothing) |
| 1, only `no shard missing` FAILs | re-queue the missing lines ONCE (grep them in `jobs/dev_<P7>.tsv` by `--group` and `--arm`), re-run this block; still 1: NEEDS-CLAUDE |
| 1, any other FAIL | STOP. NEEDS-CLAUDE "[stage1-host-share] devcheck FAIL: <the FAIL lines> at <sha>" (a harness defect, Claude's) |

## 3. Step G: gate job lines (gate seeds)

```bash
cd $WT/stage1-host-share && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && PY=$PY bash $K/make_jobs.sh gate $PIN
git add $K/jobs && git commit -m "ops(stage1-host-share): step-G gate lines at ${PIN:0:9}"
```

Expect `gate_<P7>.tsv` with 132 lines; the script refuses any other count.

## 4. Canary, then queue the gate

The pool refuses gate-seed lines until a canary of the same runner module at the same revision has passed (runbook
section 3). RULING `s1hs-canary`: run it yourself, once (the lived group is the longer, heavier one), then queue:

```bash
cd $P && PIN=$(awk '$1=="P"{print $2}' $WT/stage1-host-share/$K/PINS.tsv) && J=$WT/stage1-host-share/$K/jobs/gate_${PIN:0:7}.tsv
bash tools/pool_canary.sh run pool41 "$(grep -m1 -- '--group s1_lived --arm intact_a ' $J | cut -f1)" --wait-s 7200; echo canary rc=$?
bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why ; max_age_h=168" </dev/null || echo "ADD FAILED: $cmd"; done < '"$J"
```

| result | do |
|---|---|
| canary rc 0, every add succeeds | continue; WORKLOG the time |
| canary rc 3 (inconclusive) | run it once more on pool42; still 3: NEEDS-CLAUDE |
| canary rc 1 (FAIL) | STOP. NEEDS-CLAUDE "[stage1-host-share] canary FAIL at <sha>: <its log path>" |
| an add fails | STOP that file, NEEDS-CLAUDE with the refusal text; never edit a line |

## 5. Wait

```bash
find $P/research/findings/raw/_stage1_host_share/gate -name '*.json' ! -name '*.prov.json' ! -name '*.part*' ! -name 'aggregate.json' | wc -l   # 132 when complete
```

Count stalled 12 h below 132: run step 6's aggregate (its `missing_shards` names them), re-queue exactly those lines
from `jobs/gate_<P7>.tsv` ONCE (by `--seed`, `--group`, `--arm`), wait again. Still missing: score anyway (INVALID-OR-
INCOMPLETE) and NEEDS-CLAUDE. **Window:** under 132 on 10-21: NEEDS-CLAUDE "[s1hs-window] <n>/132; queue depth <d>"
(RULING `aws-capacity`: Claude decides on AWS at the review; never start AWS yourself).

## 6. Score and write

```bash
cd $WT/stage1-host-share && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && mkdir -p $RAW && cp -a $P/$RAW/gate $RAW/
S1=$P/research/findings/raw/_stage1_exit/gate; [ -d "$S1" ] && S1ARG="--stage1-dir $S1" || S1ARG=""
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_host_share --aggregate $RAW/gate --seeds 42,43,44,100,101,102 --out $RAW/gate/aggregate.json $S1ARG > $K/report.md; echo rc=$?
head -3 $K/report.md
```

Then write the finding with this header and the report block unchanged, nothing else (PROVISIONAL until Claude confirms):

```bash
cd $WT/stage1-host-share && D=$(date +%F) && F=research/findings/${D}-stage1-host-share-RESULT.md && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && {
printf -- '---\ntype: finding\nstatus: live\ndate: %s\nlane: C-self-workspace (prove who owns the computation)\nmechanism: host share by intervention at the Stage-1 exit gate C2/C3 seams (T22), six seeds, scored by the registered aggregate\nseeds: [42, 43, 44, 100, 101, 102]\nrunner: research/runners/stage1_host_share.py\nartifacts:\n  - research/findings/raw/_stage1_host_share/gate/aggregate.json\n---\n\n# Stage-1 host share at the C2/C3 seams: result (%s, PROVISIONAL)\n\nPrereg: research/findings/2026-10-02-stage1-host-share-PREREGISTRATION.md (pin %s). Every number below is read from research/findings/raw/_stage1_host_share/gate/aggregate.json by `python -m research.runners.stage1_host_share --aggregate`.\n\n' "$D" "$D" "$PIN"
cat $K/report.md
} > $F && git add $RAW/gate $F $K/report.md $K/WORKLOG.md && git commit -m "data(stage1-host-share): six-seed host share at the Stage-1 C2/C3 seams (PROVISIONAL)" && bash tools/push_both.sh research/stage1-host-share
```

| rc | do |
|---|---|
| 0 (`MEASUREMENT-VALID`) | finding as above; NEEDS-CLAUDE "[stage1-host-share] RESULT MEASUREMENT-VALID at research/stage1-host-share @ <sha>; C2 <credit line>; C3 <credit line>; confirm + fold into the Stage-1 exit report" |
| 3 (`INVALID-OR-INCOMPLETE`), only missing shards | step 5's single re-queue first; still missing: finding as above, NEEDS-CLAUDE with the missing count |
| 3, any other validity failure | finding as above, merge per RULING `results-without-pass`, NEEDS-CLAUDE "[stage1-host-share] INVALID: <the failing V items>" |

Board line (WORKLOG + RESULT line; Claude folds the credit lines into the Stage-1 exit report, which is Claude's edit):
`<date> stage1-host-share: <verdict>; C2 <credit line>; C3 <credit line>. Finding: <path>.`

## Stop conditions

A gate blocks a commit for any reason other than the compute gates (`compute_idle_persistent`, `lane_starvation`: wait
30 min, retry up to 6 times); a test fails; a pool job fails twice outside the re-run rule; anything would need an edit
to the runner, the instrument, the prereg, `make_jobs.sh` or a job line: STOP, NEEDS-CLAUDE.
