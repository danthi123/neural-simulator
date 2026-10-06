# HANDOFF: B2d mouth arm on the workstation GPU, queued from the server (runbook T23)

Written 2026-10-02 by cloud Claude (session 4). For Hermes on S. It replaces section 5 of the B2d packet
(`$WT/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/HANDOFF.md`, written for the workstation) for the
18-job mouth arm only: queueing, re-runs, harvest and the mouth score, with S paths. Everything else in T9 (the stub
arm, scoring the stub arm, the finding of step 10) is unchanged. Rows here marked NEEDS-CLAUDE: write one line under
NEEDS CLAUDE in the runbook and stop this task.

**Why now.** The 3090 is idle and the pool is the bottleneck. This arm is registered (B2d prereg Amendment 1, A1.2,
`f1d296202`), ruled to run (RULING `b2d-mouth`), and GPU-only by its own prereg (`SIM_BACKEND=cupy`, the production
Qwen mouth). It is one of the Phase A exit items: the reply-level fraction at HEAD with the mouth.

**What does not change.** The 18 lines are the registered ones, byte for byte (`jobs/mouth_5818dc6.txt` on
`research/b2d-mouth-renderer`): each `cd`s into the workstation's export of P
(`~/derisk-pool/revisions/5818dc629c5c0cd3cbb370c0c6cc16b314805b03`, the `git archive` export A1.2 registers and the
scorer's rules E1-E3 check: `source_kind=git_archive`, manifest verified at start and exit), waits for RAM under
`tools/mem_ok.sh 10 4`, and runs under `tools/memcap.sh 16`. `tools/gpu_remote.sh add` puts `cd <per-job worktree>
&& ` in front; that worktree only carries the guard (`gpu_add.sh` at its revision: a clean checkout and, for gate
seeds, a committed `PREREG=`); the job then runs in the export, as registered. Claude decided this route (RULING
`b2d-mouth-gpu`); it is not a way around a refusal.

**Expected GPU time.** About 40 min per cell (4 arms of ~10 min on cupy, dev seed 7), 18 cells: about 12 GPU-h, plus
the 10-min preflight. Predicted outcome (A1.3, before any output): MOUTH-SILENT on most cells. That is a registered
reading, not a failure.

Every block is ONE Bash call. Set first in each:
`P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python; WT=$P/.claude/worktrees; W=$WT/b2d-mouth-renderer;
K=$W/research/packets/b2d-mouth-renderer; SH=research/findings/raw/_load_bearing/_shards;
PIN=5818dc629c5c0cd3cbb370c0c6cc16b314805b03; CORPUS_SHA=7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec`

## 0. Preconditions (once)

```bash
cd $W && git status --short --untracked-files=no | grep -v '^.. research/\(findings\|coordination\|queue\)/' | head -5
git -C $W cat-file -e HEAD:research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md && echo prereg-ok
wc -l < $K/jobs/mouth_5818dc6.txt; grep -c "revisions/$PIN " $K/jobs/mouth_5818dc6.txt
ls $P/$SH/b2d0929-mouth/s*/*/lb.json 2>/dev/null | wc -l
sha256sum $P/data/corpus/tinystories.txt | cut -c1-64
```

| result | do |
|---|---|
| the first command prints a path (uncommitted code in `$W`) | commit it per runbook section 3 if it is yours; else STOP, NEEDS-CLAUDE "[T23] uncommitted code in b2d worktree" |
| `prereg-ok` missing, or line counts not `18` and `18` | STOP, NEEDS-CLAUDE "[T23] packet branch lacks prereg or job file" |
| mouth `lb.json` count above 0 | STOP, NEEDS-CLAUDE "[T23] <n> mouth cells already present" (nothing was ever queued; Claude checks where they came from) |
| corpus hash differs from `CORPUS_SHA` | continue, but score in step 4 WITHOUT `--corpus` and say so in the WORKLOG (rule 4 then reads "not checked") |
| otherwise | step 1 |

WORKLOG: `research/packets/b2d-mouth-renderer/WORKLOG.md` on the packet branch (append; T9's file).

## 1. Preflight: the export exists and a cupy brain builds there (dev seed 7)

Two jobs: a 1-second check that the registered export is there at P, and the remap probe's production-mode lesion
arm (dev seed 7; the dev check the B2d packet's section 11 left for a GPU window; its reading feeds the open B2d
NEEDS-CLAUDE #3, it is not a gate cell).

```bash
cd $W && bash tools/gpu_remote.sh add "cd ~/derisk-pool/revisions/$PIN && grep -qx git_sha=$PIN .source_revision && .venv/bin/python tools/pool/provisioning/source_manifest.py verify --root . --manifest .source_manifest.sha256 --expected-sha256 \$(sed -n 's/^source_manifest_sha256=//p' .source_revision) && echo export-ok"
cd $W && bash tools/gpu_remote.sh add "$(sed -n 3p $K/jobs/dev7_moodrender_5818dc6.txt)"
cd $W && bash tools/gpu_remote.sh yield --max-hours 1
```

Next session: `bash tools/gpu_remote.sh log 60 | grep -E "export-ok|DONE\(rc=" | tail -4`.

| result | do |
|---|---|
| the check job ends `DONE(rc=0)` (its log shows `export-ok`) | step 2 (whatever the probe job did: it is dev-only) |
| the check job ends `DONE(rc!=0)` | STOP, NEEDS-CLAUDE "[T23] the export of P is missing or not at P on the workstation" (Claude re-exports with the packet's `export_local_revision.sh`) |
| `gpu_remote.sh add` itself refused | STOP, NEEDS-CLAUDE "[T23] gpu_remote refused: <message>" |
| neither has run yet | do another task; check next session (yield again if the window ended) |

The check is the export script's own (`export_local_revision.sh`: `git_sha=` in `.source_revision`, then the source
manifest verified against its recorded hash). If it fails, the NEEDS-CLAUDE line above applies: never edit the check
to make it pass, and never re-export yourself.

## 2. Queue the 18 registered lines, then free the card

```bash
cd $W && n=0; while IFS= read -r line; do bash tools/gpu_remote.sh add "$line" </dev/null || { echo "ADD-FAILED at line $((n+1))"; break; }; n=$((n+1)); done < $K/jobs/mouth_5818dc6.txt; echo "queued $n"
cd $W && bash tools/gpu_remote.sh yield --max-hours 12
```

WORKLOG "T23: queued <n>/18 mouth lines at <time>; yielded 12 h". `ADD-FAILED`: STOP, NEEDS-CLAUDE with the message
and how many were queued (never re-add the ones already queued). The owner may pause the queue to game at any time:
`pause --now` kills the running cell and puts its line back at the head of the queue; the line re-runs whole (no
resume flag is set), so nothing is needed from you. If the 12-h window ends with lines left, yield again the next
session (`--max-hours 12`).

## 3. While running (each session)

```bash
bash $P/tools/gpu_remote.sh status | head -8
bash $P/tools/gpu_remote.sh log 400 | grep -c "_shards/b2d0929-mouth/"
ls $P/$SH/b2d0929-mouth/s*/*/lb.json 2>/dev/null | wc -l
```

Results come back through `sim-sync.timer`; never poll the workstation. A line that ends `DONE(rc!=0)` is handled
by the score in step 4, not by hand.

## 4. Score the mouth arm (when 18 `lb.json` are in `$P`, or the queue holds no mouth line and none is running)

A tree at P for `--code-root` (its blobs are checked against `b2d_decision_fields.json`), with the corpus linked:

```bash
[ -d $WT/b2d-P ] || git -C $P worktree add --detach $WT/b2d-P $PIN; [ -e $WT/b2d-P/data ] || ln -s $P/data $WT/b2d-P/data
cd $W && $PY tools/b2d_mouth_score.py --tag b2d0929-mouth --pin $PIN --base $P/$SH --fields research/coordination/b2d_decision_fields.json --code-root $WT/b2d-P --corpus research/coordination/b2d_corpus_sha256.tsv --attempt1-dir $P/research/findings/raw/_load_bearing/_b2d0929_attempt1 --stub-tag b2d0929 --out $W/$SH/b2d0929-mouth/b2d_mouth_score.json | head -60; echo rc=$?
```

| result | do |
|---|---|
| `preconditions_failed` non-empty (rc=1) | STOP, NEEDS-CLAUDE "[T23] mouth score preconditions: <names>" |
| `rerun_cells` non-empty | step 5 for each listed cell, then score again |
| `rerun_cells` empty | step 6 |

## 5. One re-run of a listed cell (A1.4; one GPU, so the second attempt is on the workstation)

At most ONE re-run per cell; a cell that never ran (no file at all) is re-queued at most twice, nothing moved. For
cell `s<S>/<ROW>` (set `S=` and `ROW=`; `CAUSE=` its `fails` text from step 4):

5a. Never ran (`ls $P/$SH/b2d0929-mouth/s$S/$ROW` lists nothing): log a ledger row
`<cell> workstation never-ran: <CAUSE> workstation <line> <utc> requeued` in
`$W/research/coordination/b2d_mouth_reruns.tsv` (create it with the header
`cell first_host first_cause_and_verdict rerun_host rerun_command rerun_start_time rerun_result`, tab-separated, if
absent; refuse a third `never-ran` row for the same cell), then queue its unchanged line:
`cd $W && bash tools/gpu_remote.sh add "$(grep -F -- "--out $SH/b2d0929-mouth/s$S/$ROW/lb.json" $K/jobs/mouth_5818dc6.txt)"`.

5b. Ran (files exist): attempt 1 must be moved aside on the workstation BEFORE you move it here, or the 15-min sync
pulls it back. First queue the move on the workstation (seconds; no brain):

```bash
A=research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929-mouth/s$S
cd $W && bash tools/gpu_remote.sh add "cd ~/derisk-pool/revisions/$PIN && mkdir -p $A && test ! -e $A/$ROW && mv $SH/b2d0929-mouth/s$S/$ROW $A/ && echo moved-aside-s$S-$ROW"
```

When the log shows `moved-aside-s$S-$ROW` and `DONE(rc=0)` (yield 1 h if nothing runs), move it aside here too,
log the ledger row (`first_cause_and_verdict` = `<CAUSE>`; `rerun_result` = `queued`) and queue the unchanged line:

```bash
A=research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929-mouth/s$S
mkdir -p $P/$A && test ! -e $P/$A/$ROW && mv $P/$SH/b2d0929-mouth/s$S/$ROW $P/$A/ && echo moved-here
cd $W && bash tools/gpu_remote.sh add "$(grep -F -- "--out $SH/b2d0929-mouth/s$S/$ROW/lb.json" $K/jobs/mouth_5818dc6.txt)"
```

A `DONE(rc!=0)` on the move job, or `moved-here` not printed: STOP, NEEDS-CLAUDE "[T23] re-run move failed s$S/$ROW".
A cell already in the ledger with a non-`never-ran` row is never re-run: its state stands (the scorer finalizes it).
Then yield (`--max-hours 2` per re-run cell) and score again (step 4).

## 6. Commit the mouth data and score on the packet branch

```bash
mkdir -p $W/$SH/b2d0929-mouth && rsync -a $P/$SH/b2d0929-mouth/ $W/$SH/b2d0929-mouth/
[ -d $P/research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929-mouth ] && mkdir -p $W/research/findings/raw/_load_bearing/_b2d0929_attempt1 && rsync -a $P/research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929-mouth $W/research/findings/raw/_load_bearing/_b2d0929_attempt1/
cd $W && git add $SH/b2d0929-mouth research/coordination/b2d_mouth_reruns.tsv research/findings/raw/_load_bearing/_b2d0929_attempt1 research/packets/b2d-mouth-renderer/WORKLOG.md 2>/dev/null; git commit -m "data(B2d mouth arm): 18 cells at P=5818dc6 + b2d_mouth_score.json (GPU, queued from S; PROVISIONAL, unread by Claude)"
```

Then RESULT under NEEDS CLAUDE: "[T23] RESULT: mouth arm scored, verdict <verdict>, rows <per_row>; research/b2d-mouth-renderer @ <sha>;
T9 step 10 now needs only the stub arm". The finding is T9's step 10 (both arms); never write a mouth-only finding.
