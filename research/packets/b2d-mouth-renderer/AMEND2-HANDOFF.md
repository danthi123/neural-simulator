# HANDOFF: B2d Amendment 2 (a mood-engaged turn the Qwen mouth speaks), for Hermes on S

Written 2026-10-06 by Claude. Governing text: the B2d pre-registration's "Amendment 2"
(`research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md`); background:
`research/packets/b2d-mouth-renderer/QWEN-ROUTE-MAP.md`. It replaces RULING `b2d-mouth-remap`. The GPU runs on the
workstation through `tools/gpu_remote.sh add`, as for the A1.2 mouth arm (RULING `b2d-mouth-gpu`). Use no judgment
beyond the tables. A row marked NEEDS-CLAUDE: write one line under NEEDS CLAUDE in the runbook, starting `[A2]`, and
stop this task. Never edit the probe, the scorer, a job line, a seed or a rule. Never write the finding: Claude does.

Every block is ONE Bash call. Set first in each:
`P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python; WT=$P/.claude/worktrees; W2=$WT/b2d-amend2;
K=$W2/research/packets/b2d-mouth-renderer; R=research/findings/raw/_b2d_amend2; P2=$(cat $K/PIN_P2 2>/dev/null);
P7=${P2:0:7}`

WORKLOG: append to `$K/WORKLOG.md` on branch `research/b2d-amend2` (one dated line per step).

## 0. Branch and pin (once)

```bash
git -C $P worktree add $WT/b2d-amend2 -b research/b2d-amend2 main && cd $WT/b2d-amend2
git cat-file -e HEAD:tools/b2d_amend2.py && grep -q '^## Amendment 2' research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md && echo amend2-ok
SIM_NO_PROVENANCE=1 $P/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_b2d_amend2.py | tail -1
git rev-parse HEAD > research/packets/b2d-mouth-renderer/PIN_P2 && git add research/packets/b2d-mouth-renderer/PIN_P2 && git commit -m "ops(B2d A2): pin P2 = $(cut -c1-9 research/packets/b2d-mouth-renderer/PIN_P2) (main with Amendment 2)" && bash tools/push_both.sh research/b2d-amend2
```

| result | do |
|---|---|
| `amend2-ok`, tests pass, push verified | step 1 |
| no `amend2-ok` (main lacks the amendment) | STOP: the amendment is not merged yet; check again next session |
| a test fails, or a gate refuses other than `compute_idle_persistent`/`lane_starvation` (those: wait, retry) | NEEDS-CLAUDE "[A2] step 0: <message>" |

P2 is the commit `PIN_P2` names (the main commit), not the branch head: the job lines `cd` into the export of P2.

## 1. Export P2 on the workstation, with the valence map (once)

```bash
cd $W2 && bash tools/gpu_remote.sh add "bash research/packets/b2d-mouth-renderer/export_local_revision.sh $P2"
cd $W2 && bash tools/gpu_remote.sh add "cd ~/derisk-pool/revisions/$P2 && grep -qx git_sha=$P2 .source_revision && .venv/bin/python tools/pool/provisioning/source_manifest.py verify --root . --manifest .source_manifest.sha256 --expected-sha256 \$(sed -n 's/^source_manifest_sha256=//p' .source_revision) && test -s research/findings/raw/_affect_learned_valence_map.json && echo export-ok-with-map"
cd $W2 && bash tools/gpu_remote.sh yield --max-hours 1
```

Next session: `bash $P/tools/gpu_remote.sh log 80 | grep -E "exported|exists, manifest verified|REFUSED|export-ok-with-map|DONE\(rc=" | tail -6`.

| result | do |
|---|---|
| the check job prints `export-ok-with-map` and ends `DONE(rc=0)` | step 2 |
| the export printed `REFUSED ... does not contain origin/main` | main moved after P2: `git -C $W2 merge --no-edit main`, redo step 0's last line (new P2), then step 1. Second time: NEEDS-CLAUDE "[A2] main keeps moving under P2" |
| the check job ends `DONE(rc!=0)` (no map, or the manifest does not verify) | NEEDS-CLAUDE "[A2] export of P2 lacks the valence map or does not verify". Never re-export over an existing tree, never edit the check |
| `gpu_remote.sh add` refused | NEEDS-CLAUDE "[A2] gpu_remote refused: <message>" |
| neither job has run | other tasks; check next session (yield again if the window ended) |

## 2. Dev seeds: queue 30 lines

```bash
cd $W2 && $PY tools/b2d_amend2.py jobs --pin $P2 --phase dev --out $K/jobs/amend2_dev_$P7.txt && $PY tools/b2d_amend2.py jobs --pin $P2 --phase dev-lesion --out $K/jobs/amend2_devlesion_$P7.txt
cd $W2 && git add $K/jobs/amend2_dev_$P7.txt $K/jobs/amend2_devlesion_$P7.txt && git commit -m "ops(B2d A2): dev job files at P2 (24 + 6 lines; nothing queued yet)" && bash tools/push_both.sh research/b2d-amend2
cd $W2 && n=0; cat $K/jobs/amend2_dev_$P7.txt $K/jobs/amend2_devlesion_$P7.txt | while IFS= read -r line; do bash tools/gpu_remote.sh add "$line" </dev/null || { echo "ADD-FAILED at line $((n+1))"; break; }; n=$((n+1)); echo "queued $n"; done | tail -2
cd $W2 && bash tools/gpu_remote.sh yield --max-hours 6
```

Line counts must be 24 and 6. `ADD-FAILED`: NEEDS-CLAUDE with the message and how many were queued (never re-add
those). A job killed by a game pause (`rc=143`) is re-queued unchanged per RULING `game-mode-pauses`; that is not
its re-run.

## 3. While running (each session)

```bash
bash $P/tools/gpu_remote.sh status | head -6
ls $P/$R/$P7/dev/s*/*.json 2>/dev/null | grep -vc prov.json
```

Results come back through `sim-sync.timer`. When the queue holds no A2 line and none is running: step 4. If the
queue drained more than an hour ago and fewer than 30 files are in `$P`, but the job log shows them `DONE(rc=0)`:
NEEDS-CLAUDE "[A2] dev results did not sync".

## 4. Dev selection (outcome-blind)

```bash
cd $W2 && $PY tools/b2d_amend2.py dev --pin $P2 --base $P/$R/$P7/dev --out $W2/$R/$P7/dev/selection.json; echo rc=$?
```

| prints | do |
|---|---|
| `status: INCOMPLETE` with `missing_or_invalid` | once per listed arm: re-queue its unchanged line (`grep -F -- "--out $R/$P7/dev/s<S>/<ARM>.json" $K/jobs/amend2_dev*_$P7.txt`) with `gpu_remote.sh add`; WORKLOG it; yield; after they land, run this block with `--final` added. Dev arms are re-run at most once |
| `status: SELECTED` | commit (below), then step 5 |
| `status: NO-CANDIDATE` | commit (below), then NEEDS-CLAUDE "[A2] dev: no candidate; n_ok <the printed dict>". Queue no gate line |
| `status: LESION-SMOKE-FAILED` | commit (below), then NEEDS-CLAUDE "[A2] dev lesion smoke not route-clean". Queue no gate line |
| rc != 0 | NEEDS-CLAUDE "[A2] dev scorer: <message>" |

Commit (the selection must be committed BEFORE any gate line is queued; A2.8):
`mkdir -p $W2/$R/$P7/dev && rsync -a $P/$R/$P7/dev/ $W2/$R/$P7/dev/ && cd $W2 && git add $R/$P7/dev && git commit -m "data(B2d A2): dev seeds 1-8 at P2 + selection.json (<status>, <selected>)" && bash tools/push_both.sh research/b2d-amend2`

Do not read or report the dev lesion arms' replies: the selection is outcome-blind by registration.

## 5. Gate seeds: the six route arms first

```bash
cd $W2 && $PY tools/b2d_amend2.py jobs --pin $P2 --phase gate-route --out $K/jobs/amend2_gateroute_$P7.txt && git add $K/jobs/amend2_gateroute_$P7.txt && git commit -m "ops(B2d A2): gate route-arm job file at P2 (6 lines)" && bash tools/push_both.sh research/b2d-amend2
cd $W2 && while IFS= read -r line; do bash tools/gpu_remote.sh add "$line" </dev/null || { echo ADD-FAILED; break; }; done < $K/jobs/amend2_gateroute_$P7.txt
cd $W2 && bash tools/gpu_remote.sh yield --max-hours 2
```

When the six have landed (`ls $P/$R/$P7/gate/s*/route.json | wc -l` prints 6):

```bash
cd $W2 && $PY tools/b2d_amend2.py routecheck --pin $P2 --base $P/$R/$P7/gate --selection $W2/$R/$P7/dev/selection.json --out $W2/$R/$P7/gate/routecheck.json; echo rc=$?
```

| prints | do |
|---|---|
| `status: ROUTE-OK` | step 6 (a NOT-ENGAGED seed is fine here; it is scored later) |
| a seed failed with `E` or `M` | step 8's re-run for that seed's `route` arm, then this block again. Still failing after its re-run: NEEDS-CLAUDE "[A2] gate route arm s<S>: <why>" |
| a seed failed with `route-mismatch`, `mouth-silent` or `trace-absent` | NEEDS-CLAUDE "[A2] gate route arm s<S>: <why>". Queue nothing more |

## 6. Gate seeds: the other 48 arms

```bash
cd $W2 && $PY tools/b2d_amend2.py jobs --pin $P2 --phase gate-rest --out $K/jobs/amend2_gaterest_$P7.txt && git add $K/jobs/amend2_gaterest_$P7.txt $R/$P7/gate/routecheck.json && git commit -m "ops(B2d A2): route check OK; gate job file at P2 (48 lines)" && bash tools/push_both.sh research/b2d-amend2
cd $W2 && n=0; while IFS= read -r line; do bash tools/gpu_remote.sh add "$line" </dev/null || { echo "ADD-FAILED at line $((n+1))"; break; }; n=$((n+1)); done < $K/jobs/amend2_gaterest_$P7.txt; echo "queued $n"
cd $W2 && bash tools/gpu_remote.sh yield --max-hours 9
```

If the window ends with lines left, yield again next session. Step 3's watch applies (expect 54 gate files).

## 7. Score the gate

```bash
cd $W2 && $PY tools/b2d_amend2.py gate --pin $P2 --base $P/$R/$P7/gate --selection $W2/$R/$P7/dev/selection.json --attempt1 $P/research/findings/raw/_b2d_amend2_attempt1/$P7/gate --out $W2/$R/$P7/gate/a2_score.json; echo rc=$?
```

| prints | do |
|---|---|
| `rerun_arms` non-empty | step 8 for each listed arm, then score again |
| `needs_claude` non-empty | commit (below), then NEEDS-CLAUDE "[A2] <each entry>" |
| `verdict: COMPLETE`, `needs_claude` empty | commit (below), then RESULT under NEEDS CLAUDE: "[A2] RESULT: selected <cid>; rows <reading, defined, mouth_load_bearing, interval each>; research/b2d-amend2 @ <sha>; Claude writes the finding" |

Commit: `rsync -a $P/$R/$P7/gate/ $W2/$R/$P7/gate/ && [ -d $P/research/findings/raw/_b2d_amend2_attempt1 ] && rsync -a $P/research/findings/raw/_b2d_amend2_attempt1 $W2/research/findings/raw/; cd $W2 && git add $R/$P7/gate research/findings/raw/_b2d_amend2_attempt1 research/packets/b2d-mouth-renderer/WORKLOG.md 2>/dev/null; git commit -m "data(B2d A2): gate arms at P2 + a2_score.json (PROVISIONAL, unread by Claude)" && bash tools/push_both.sh research/b2d-amend2`.
Do not change the board's B2d text yourself beyond "A2 scored, awaiting Claude".

## 8. One re-run of a gate arm (A1.4; at most one per arm)

For arm `<ARM>` of seed `<S>` (set `S=` and `ARM=`). A file that never landed (rc != 0, or absent after the queue
drained): re-queue its unchanged line (`grep -F -- "--out $R/$P7/gate/s$S/$ARM.json" $K/jobs/amend2_gate*_$P7.txt`) at
most twice; nothing to move. A file that landed and reads E or M: move it aside on the workstation first, or the
15-minute sync brings it back:

```bash
A=research/findings/raw/_b2d_amend2_attempt1/$P7/gate/s$S; D=$R/$P7/gate/s$S
cd $W2 && bash tools/gpu_remote.sh add "cd ~/derisk-pool/revisions/$P2 && mkdir -p $A && test ! -e $A/$ARM.json && mv $D/$ARM.json $D/$ARM.json.prov.json $A/ 2>/dev/null; test -e $A/$ARM.json && echo moved-aside-s$S-$ARM"
```

When the log shows `moved-aside-s$S-$ARM` and `DONE(rc=0)`: `mkdir -p $P/$A && test ! -e $P/$A/$ARM.json && mv $P/$D/$ARM.json* $P/$A/ && echo moved-here`, then
re-queue the unchanged line. `moved-here` not printed, or the move job fails: NEEDS-CLAUDE "[A2] re-run move failed
s$S/$ARM". An arm that already has a file under `_b2d_amend2_attempt1` is never re-run again (the scorer reads it as
FAIL).

## What this does not do

No finding, no board status beyond "awaiting Claude", no change to A1.2's cells or score, no stub-arm work (T9 stays
as it is), no AWS, no pool lines (the arms need cupy and Qwen).
