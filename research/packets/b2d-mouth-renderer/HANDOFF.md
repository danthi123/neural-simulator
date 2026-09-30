# HANDOFF: Battery B2d (the #1 metric at reply level) + its mouth arm -- execution packet for the local model

Written 2026-09-30 by Claude. Branch `research/b2d-mouth-renderer`. Execute top to bottom. Use no judgment beyond
the tables: a row marked NEEDS-CLAUDE means STOP that thread, write one line under "NEEDS CLAUDE" in
`research/coordination/LOCAL_LLM_RUNBOOK.md` (on main) naming this packet and the row, then continue with any thread
that is not blocked. Every verdict you write is PROVISIONAL until Claude inspects it.

- Governing text: `research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md` (the main text
  plus its "Amendment 1" section, which registers the mouth arm, its predictions, the one-pin rule and this
  packet's tools). Do not change code, bars, turns, rows, seeds or arms. Never merge to main, never `--no-verify`,
  never write a waiver or a filler job, never set `POOL_PROVISION_ALLOW_STALE`, never start or stop AWS.
- **Each fenced block is ONE Bash call** and starts with `source .../env.sh` (the Bash tool keeps no state).
  `env.sh` sets `P` (primary checkout), `W` (this worktree), `K` (this packet), `PY`, `SH` (shard root), `PREREG`,
  and, once step 1 has run, `PIN_P`, `P7` and `TREE` (the workstation's export of P). Loops are wrapped in `bash -c`.
- Two arms, one pin P:
  - **stub arm**, tag `b2d0929`: 258 pool jobs (43 rows x 6 seeds, numpy, the stub renderer). Headline: S1.
  - **mouth arm**, tag `b2d0929-mouth`: 18 GPU jobs (3 affect rows x 6 seeds, cupy, `LB_ARM_RENDERER=production`,
    the production Qwen mouth). Reported separately, never averaged in.

## 0. State at handoff (done by Claude; do not redo)

- The harness change: `research/runners/onebrain_regression_battery.py` reads `LB_ARM_RENDERER` (`stub` default,
  byte-identical; `production` = the console's auto renderer). Tests: `tests/test_arm_renderer_production_path.py`.
- Scorers: `tools/b2d_score.py` (stub arm, already on main) and `tools/b2d_mouth_score.py` (mouth arm, new).
  Tests: `tests/test_b2d_score.py`, `tests/test_b2d_mouth_score.py`, `tests/test_b2d_mouth_packet.py`.
- Amendment 1 committed (`f1d296202`) before any mouth-arm output was read: predictions A1.3, SD re-check (prereg
  Step 1) closed, the wave script (Step 6) written, the re-pin audit recorded in
  `research/coordination/b2d_decision_fields.json` (`cited_blobs_at_pin`, `harness_blobs_at_pin`, `repin_audit`).
- Packet tools (this directory): `env.sh`, `export_local_revision.sh`, `make_jobs.py`, `repin.py`,
  `registry_fingerprint.py` + `registry_at_H.json`, `copy_results.sh`, `rerun_stub_cell.sh`, `rerun_mouth_cell.sh`,
  `draft_finding.py`. Wave script: `research/coordination/b2d_queue_next_wave.sh`.
- Dev seed 7: section 11. Nothing at a gate seed has been queued or run.

## 1. Pin P

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null; git status --short --untracked-files=no   # must print nothing
git fetch -q origin && { git merge-base --is-ancestor origin/main HEAD || git merge --no-edit origin/main; }
SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_arm_renderer_production_path.py tests/test_b2d_mouth_score.py tests/test_b2d_mouth_packet.py tests/test_b2d_score.py | tail -1
bash tools/push_both.sh research/b2d-mouth-renderer && git rev-parse HEAD > $K/PIN_P && cat $K/PIN_P
```

| result | do |
|---|---|
| `git status` printed tracked changes | STOP. NEEDS-CLAUDE "B2d: uncommitted tracked changes in the worktree" |
| the merge reports a conflict | `git merge --abort`; STOP. NEEDS-CLAUDE "B2d: merge conflict with origin/main: <files>" |
| the merge commit is blocked by `compute_idle_persistent` or `lane_starvation` | wait, retry. Any other gate: STOP, NEEDS-CLAUDE with its message |
| any test fails | STOP. NEEDS-CLAUDE with the failing test name |
| push verified on both remotes, PIN_P printed | continue |

## 2. Export P on the workstation; provision the pool at P

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
bash $K/export_local_revision.sh $PIN_P
cd $P && bash tools/pool_provision.sh --revision $PIN_P --isolated pool41 pool42
```

| result | do |
|---|---|
| export prints `exported ... -> ~/derisk-pool/revisions/<P>` (or `exists, manifest verified`) and the provisioner exits 0 | continue |
| `REFUSED ... does not contain origin/main` (main moved) | `rm -f $K/PIN_P`, go back to step 1. The second time: STOP, NEEDS-CLAUDE "B2d: main keeps moving under the pin" |
| any other export or provisioner failure | STOP. NEEDS-CLAUDE with the message |

Add AWS aliases (e.g. `pool1 pool2`) to the provisioner line only if they are already running (`bash tools/status.sh`
shows them); starting them is not yours (NEEDS-CLAUDE #1).

## 3. Re-pin to P, record the corpus hashes, commit

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && $PY $K/repin.py apply --pin $PIN_P --tree $TREE
```

| result | do |
|---|---|
| `REPIN APPLY OK: pin <P>` and a list of paths to commit | continue |
| `REPIN APPLY: N problem(s)` | STOP. NEEDS-CLAUDE "B2d repin: <the listed problems>" (a cited or harness file, or the registry, changed after the audit; Claude re-audits) |

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
bash -c 'set -euo pipefail; source '"$K"'/env.sh; T=$W/research/coordination/b2d_corpus_sha256.tsv
for n in pool41 pool42; do out=$(ssh -n -o BatchMode=yes -F $SSHCFG "$n" "cd ~/derisk-pool/revisions/$PIN_P && hostname && sha256sum data/corpus/tinystories.txt"); host=$(echo "$out" | sed -n 1p); sha=$(echo "$out" | sed -n 2p | cut -d" " -f1); printf "%s\t%s\tdata/corpus/tinystories.txt\t%s\t%s\tyes\t%s\n" "$n" "$host" "$sha" "$PIN_P" "$(date -u +%FT%TZ)" >> "$T"; done
sha=$(sha256sum $TREE/data/corpus/tinystories.txt | cut -d" " -f1); printf "workstation\t%s\tdata/corpus/tinystories.txt\t%s\t%s\tyes\t%s\n" "$(hostname)" "$sha" "$PIN_P" "$(date -u +%FT%TZ)" >> "$T"
tail -3 "$T"; tail -3 "$T" | cut -f4 | sort -u'
```

The last line must print exactly one hash, `7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec`. Any
other hash: delete that node's new row, re-provision that node once (step 2's provisioner line with that node only),
hash again; still different: that node takes no B2d line (NEEDS-CLAUDE "B2d corpus mismatch on <node>"). The
workstation row must match, or STOP (NEEDS-CLAUDE).

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && wc -l research/coordination/b2d_jobs.txt $K/jobs/mouth_$P7.txt     # 258 and 18
git add $K/PIN_P research/coordination/b2d_decision_fields.json research/coordination/b2d_jobs.txt $SH/b2d0929/PIN.txt $SH/b2d0929-mouth/PIN.txt $K/jobs/mouth_$P7.txt research/coordination/b2d_corpus_sha256.tsv
git commit -m "ops(B2d): pin P=$P7 (repin.py apply: registry and audited blobs unchanged); job files, PIN.txt, corpus hashes at P; nothing queued"
bash tools/push_both.sh research/b2d-mouth-renderer
```

Line counts other than 258 and 18: STOP (NEEDS-CLAUDE). Gate blocks: as in step 1's table. (This commit is a child of
P and changes no code; every job runs at P.)

## 4. Queue the stub arm (pool): wave 1 now, one wave per session after that

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && bash research/coordination/b2d_queue_next_wave.sh --dry-run | tail -2
bash research/coordination/b2d_queue_next_wave.sh; echo rc=$?
git add research/coordination/b2d_waves.tsv && git commit -m "ops(B2d): stub-arm wave queued (ledger)" && bash tools/push_both.sh research/b2d-mouth-renderer
```

| result | do |
|---|---|
| dry run: `43 line(s) would be queued as wave 1`; then `rc=0` and `queued ... well-formed` | continue; note the time in `$K/WORKLOG.md` |
| `REFUSED ... corpus-hash row` | step 3's corpus block did not run or was not committed: do it, then retry |
| `⛔ add failed` (a shape-check, prereg or GATE-SEED refusal) | STOP this arm. NEEDS-CLAUDE with the refusal text. Never edit a job line |
| `N fresh b2d0929 lines still queued (>= 10): nothing queued` | normal: the previous wave is still waiting |

**Every later session:** run the second line of the block again (and commit the ledger). It queues the next 43 lines
only when fewer than 10 fresh b2d0929 lines remain, and prints `no unqueued b2d0929 line left` once all 258 are in.
Wall time on pool41 + pool42 alone: the queue was about 446 lines (three to four days) deep when this was written,
then about 70 to 75 h of B2d work at one 8 GB job per node, so roughly a week; `max_age_h=168` on every line keeps
them fresh. With AWS the prereg sizes it at about 20 h and at most $40 (NEEDS-CLAUDE #1: the owner's call).

## 5. Queue the mouth arm (GPU) and yield the card

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && bash tools/gpu_queue.sh start >/dev/null 2>&1; bash -c 'while IFS= read -r line; do bash tools/gpu_queue.sh add "$line" </dev/null || { echo "ADD FAILED: $line"; exit 1; }; done < '"$K/jobs/mouth_$P7.txt"
bash tools/gpu_queue.sh status | head -5
```

Then free the card (the queue holds every job while the local-llm unit is active): `bash tools/local_llm/llm.sh
yield --max-hours 12`. Your session resumes by itself when the window drains. If it prints `refusing to yield: ...
not loaded`, the unit is already down and the queue is not held: nothing to do. `yield` refuses while a
`.claude/worktrees/draft-*` worktree has uncommitted tracked changes: commit those through the normal gates first,
or, if you cannot, NEEDS-CLAUDE "B2d mouth arm: yield refused (<its message>)". Never run a mouth line outside the
queue. Measured on dev seed 7: about 10 min per arm on cupy, 4 arms per cell, so about 40 min per cell and about 12 h
for all 18; if the window ends with lines left, yield again next session. `ADD FAILED`: STOP this arm, NEEDS-CLAUDE
with the refusal text.

## 6. While running (every session)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $P && bash tools/status.sh | sed -n '1,25p'
ls $P/$SH/b2d0929/s*/*/lb.json 2>/dev/null | wc -l          # stub cells landed (of 258)
ls $TREE/$SH/b2d0929-mouth/s*/*/lb.json 2>/dev/null | wc -l # mouth cells landed (of 18)
cd $W && bash research/coordination/b2d_queue_next_wave.sh --status
```

- A GPU job that ends `DONE(rc!=0)` in `research/queue/gpu_queue.log` (e.g. `mem_ok` refused, a pause killed it):
  wait until step 8 scores it, then use the mouth re-run row of step 9's table. Do not re-queue by hand.
- `gpu_queue.sh pause --now` (the owner gaming) re-queues the running job itself; nothing to do.
- A stub line the wave script reports stale: A1.4(c) (re-queue at most twice, after moving the stale line out of
  `research/queue/pool.queue` with `bash tools/pool_queue.sh` tooling); if unsure how: NEEDS-CLAUDE.

## 7. Harvest and commit the landed data (after all 258 + 18 have landed, or when step 8 lists only re-run cells)

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $P && bash tools/pool_sync.sh >/dev/null 2>&1; bash $K/copy_results.sh stub && bash $K/copy_results.sh mouth
cd $W && git add $SH/b2d0929 $SH/b2d0929-mouth research/findings/raw/_load_bearing/_b2d0929_attempt1 2>/dev/null; git commit -m "data(B2d): stub-arm and mouth-arm shards at P=$P7 (raw arm files + sidecars; not yet scored)" && bash tools/push_both.sh research/b2d-mouth-renderer
```

## 8. Aggregate and score

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $P && $PY tools/lb_shard.py aggregate --tag b2d0929 --seeds 42 43 44 100 101 102 --pin $PIN_P > /dev/null; $PY -c "import json;d=json.load(open('$SH/b2d0929/aggregate.json'));print('provenance', d.get('provenance',{}).get('status'))"
cd $W && $PY tools/b2d_score.py --tag b2d0929 --pin $PIN_P --seeds 42 43 44 100 101 102 --base $P/$SH --fields research/coordination/b2d_decision_fields.json --code-root $TREE --corpus research/coordination/b2d_corpus_sha256.tsv --reruns research/coordination/b2d_reruns.tsv --dispatch-log $P/research/queue/dispatch.log --attempt1-dir $P/research/findings/raw/_load_bearing/_b2d0929_attempt1 --compare-tag b2b0924-base --compare-pin $F --out $W/$SH/b2d0929/b2d_score.json | head -60
$PY tools/b2d_mouth_score.py --tag b2d0929-mouth --pin $PIN_P --base $P/$SH --fields research/coordination/b2d_decision_fields.json --code-root $TREE --corpus research/coordination/b2d_corpus_sha256.tsv --attempt1-dir $P/research/findings/raw/_load_bearing/_b2d0929_attempt1 --stub-tag b2d0929 --out $W/$SH/b2d0929-mouth/b2d_mouth_score.json | head -80
```

(If `lb_shard.py aggregate` writes its output elsewhere, read the path it prints; only `provenance.status` is used.)

## 9. Outcome -> action

**Stub arm (`b2d_score.py`).**

| prints | do |
|---|---|
| `preconditions_failed` non-empty | STOP. NEEDS-CLAUDE "B2d scorer precondition: <names>" (wrong `--code-root`, a pin mismatch, a host mismatch) |
| `aggregate provenance` is not `verified` | STOP. NEEDS-CLAUDE with the aggregate's invalid-cell list |
| `verdict: INCOMPLETE`, `not_defined` lists cells | for each cell ONCE: `bash $K/rerun_stub_cell.sh <seed> <faculty> <node> "<its fails text>"` with `<node>` = pool41 or pool42, whichever is NOT the first host the script names (run with `--dry-run` first). Commit `research/coordination/b2d_reruns.tsv`. After the re-runs land (step 6 counts), steps 7-8 again. A cell already re-run once stays as it is: INCOMPLETE is then final |
| a `not_defined` cell whose fails start `E: shard directory absent` and that has no dispatch record | a never-ran line: re-queue it (A1.4 c, at most twice) by re-running the wave script; it skips lines already handled |
| `verdict: FAIL` | final. Name each `fail_cells` row in the finding as a row the revision cannot measure as registered |
| `verdict: COMPLETE` (or INCOMPLETE final) | step 10 |
| `organ_error_cells` names a row other than comprehension-monitor | NEEDS-CLAUDE "B2d new organ error: <row> <error>" plus one row in `research/FAILURE_LOG.md` (`NOT-GATEABLE: found by B2d rule 7`); continue |
| `organ_error_cells` names comprehension-monitor | NEEDS-CLAUDE "B2d: the comprehension lesion-twin fix did not hold"; continue |

**Mouth arm (`b2d_mouth_score.py`).**

| prints | do |
|---|---|
| `preconditions_failed` non-empty | STOP. NEEDS-CLAUDE with the names |
| `rerun_cells` non-empty (MISSING, C-UNCONFIRMED, M-UNCONFIRMED) | for each cell ONCE: `bash $K/rerun_mouth_cell.sh <seed> <row> "<its fails text>"`, then yield (step 5), then steps 7-8 |
| a cell whose state is M twice | final FAIL for that cell; NEEDS-CLAUDE "B2d mouth: renderer did not resolve to Qwen twice at s<seed>/<row>" |
| `verdict: COMPLETE` | step 10. Read each row's `reading` against the predictions (Amendment 1, A1.3) |

## 10. The provisional finding

```bash
source /home/dant123/Projects/sim/.claude/worktrees/b2d-mouth-renderer/research/packets/b2d-mouth-renderer/env.sh
cd $W && $PY $K/draft_finding.py --pin $PIN_P --stub $SH/b2d0929/b2d_score.json --mouth $SH/b2d0929-mouth/b2d_mouth_score.json --out research/findings/$(date +%F)-B2d-battery-at-P$P7-PROVISIONAL.md
cp $P/$SH/b2d0929/aggregate.json $W/$SH/b2d0929/aggregate.json
git add research/findings/*-B2d-battery-at-P$P7-PROVISIONAL.md $SH/b2d0929/b2d_score.json $SH/b2d0929/aggregate.json $SH/b2d0929-mouth/b2d_mouth_score.json && git commit -m "results(B2d, PROVISIONAL): scored at P=$P7 -- see the finding; awaiting Claude review" && bash tools/push_both.sh research/b2d-mouth-renderer
```

Do not edit the numbers, labels or verdict lines the script writes. You may add one "Operational notes" section at
the end (what re-ran, what was slow). Then NEEDS-CLAUDE #2 (review + merge) and update the board card for B2d to
"scored, awaiting Claude review".

## 11. Dev-seed record (seed 7, done by Claude; re-run only if Claude asks)

See `$K/dev7/README.md` (the byte-identity check of the stub path before/after the change, and the mouth arm's
seed-7 cells scored by `tools/b2d_mouth_score.py`).

## NEEDS-CLAUDE (open; do not act on these)

1. **AWS lane for the stub arm (owner decision).** Pool-only is about a week behind the current queue; two
   r7i.4xlarge would take the prereg's ~20 h / <= $40 (cap $50/day). They would also serve the rest of the queue.
2. **Review + merge** `research/b2d-mouth-renderer` once scored (the renderer mode is default-off; merging changes no
   default). Until then the jobs run at the branch pin P, which contains origin/main.
3. **The mouth arm's next rung.** If A1.3's predictions hold (emo rows MOUTH-SILENT, affect-drives-response
   MOUTH-NOT-LOAD-BEARING), no registered turn is both mood-engaged and rendered. A remap (for example the `emo`
   session followed by a rendered recall turn in the same session, so the mood carries into a Qwen-rendered reply)
   needs a new probe turn, a dev-seed check and an amendment: Claude's.
4. Carried from the B2d HANDOFF on main, unaffected by this packet: the harness-level S1 identity replay (not
   blocking); the four genuinely new probe remaps; swap-drives-response's five uncommitted seed artifacts; the B2b
   seed-102 A1.4 ruling (the drift block reports those cells as they stand); `lb_shard.py aggregate`'s `ltm_mode`
   recorder is still blind (the finding labels LTM from the arm files' `source` field instead).
5. Any STOP row above.
