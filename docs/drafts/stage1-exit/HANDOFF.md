# HANDOFF — Stage-1 (INFANT) exit gate: execution packet for the local model

Written 2026-09-29 by Claude. Execute top to bottom. Do not change code, bars, turns or arms. At most ONE fix round
per issue; anything not covered here goes on the NEEDS-CLAUDE list at the end and you stop that line of work.

- Worktree: `/home/dant123/Projects/sim/.claude/worktrees/stage1-exit` (branch `research/stage1-exit-gate`). Below,
  `W=` that path, `P=/home/dant123/Projects/sim` (the primary checkout, on main; never switch its branch),
  `PY=/home/dant123/Projects/sim/.venv/bin/python` (the worktree has no `.venv`).
- Never merge to main. Never `--no-verify`. Never write waiver files. Push only with `bash tools/push_both.sh`.
- Set these first in every new shell:
  `W=/home/dant123/Projects/sim/.claude/worktrees/stage1-exit; P=/home/dant123/Projects/sim; PY=/home/dant123/Projects/sim/.venv/bin/python`
- The shell may be zsh: run every multi-line block exactly as given, inside `bash -c '...'` where shown.
- Prereg (frozen bars, arms, turns, predictions): `research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md`.
  Runner: `research/runners/stage1_exit_gate.py`. Everything is scored by `--aggregate`; never by hand.

## 0. State at handoff

- Three commits are on `research/stage1-exit-gate`, pushed to both remotes: commit 1 = the prereg alone; commit 2 =
  runner, tests, this packet, the FAILURE_LOG row, the dev smoke shards; commit 3 = the two pin files.
  `docs/drafts/stage1-exit/PREREG_SHA` holds commit 1 (the prereg's own commit); `docs/drafts/stage1-exit/GATE_SHA`
  holds commit 2 (the revision the gate runs at: it contains the prereg byte-unchanged plus the runner). If either
  file is missing, STOP (NEEDS-CLAUDE #1).
- Dev verification (dev seed 7) is DONE and PASSED (prereg section 11). Step 2 is only for a re-check.
- The gate has NOT been queued. Nothing in `research/findings/raw/_stage1_exit/gate/` exists yet.
- **Pre-data runner fix, 2026-09-30 (Claude; before any gate data; prereg byte-unchanged, no bar/arm/turn/seed
  changed).** The 204 lines queued at `4f6d9d7fb` could never have run: `run_shard` spawned `--worker` without
  `--allow-gate-seed`, and `main()`'s gate-seed guard applies to `--worker` too, so every gate shard died rc=2 before
  building a brain. A pool maintenance hold moved all 204 lines to `research/queue/pool.queue.held` before any
  dispatch; a pool canary of the same runner shape (stage1_c1_category at `bb6705de6`, pool42, inside the pool-job
  sandbox) printed `GATE seed 42 refused` -> `worker FAILED rc=2`. Fix: commit `38fd327f3` (the shard hands
  `--allow-gate-seed` and `--prereg` to its worker; failing-first test, now `9 passed`), main merged at `c238ff325`
  (no brain code changed: webapp/, sim/, research/runners/ untouched), GATE_SHA re-pinned to `c238ff325`, provisioned
  isolated on pool41+pool42 (LTM, corpus OK), and a pool canary of the held line seed 42 `val_novel`/`intact_a` at
  `c238ff325` ran to completion on pool41 inside the sandbox (rc 0, 17 turns, no errored turn, source `tiny-demo
  +LTM`, hash ok, corpus present, 740 s). The 204 replacement lines (same seed x group x arm set, `max_age_h=168`) are
  `research/queue/requeue_stage1_exit_gate_c238ff325.tsv` in the primary checkout; step 6's single re-queue greps THAT
  file, never `stage1_exit_gate_jobs.tsv` (it pins `4f6d9d7fb`).

## 1. When to start

Start step 3 when commits are possible: `cd $P && $PY tools/parallel_audit.py | grep -c "UNDER-PARALLELIZED"`
prints 0, or the pool queue depth (`bash tools/pool_queue.sh depth`) is below 40. Before that, only step 2 may run.

## 2. Dev verification (DONE by Claude, PASS; re-run only if a shard file is missing)

```bash
cd $W
ls research/findings/raw/_stage1_exit/dev/s7/*.json | grep -v prov       # expect word__intact_a, word__lesion_recruit, val_heldout__intact_a_2words, lived__intact_a, cat_disc__intact_a, val_novel__intact_a (prereg section 11)
$PY - <<'EOF'
import json, glob
for p in sorted(glob.glob("research/findings/raw/_stage1_exit/dev/s7/*.json")):
    if p.endswith(".prov.json"): continue
    d = json.load(open(p)); r = d["responses"]
    errs = [k for k, v in r.items() if "_error" in v]
    print("- `%s`: %d turns, sources %s, hash_ok %s, corpus %s, patch %s, errored %s, wall %ss" % (
        p.split("/")[-1], len(r), d.get("sources"), d.get("pythonhashseed_ok"), d.get("corpus_present"), d.get("patch"), errs, d.get("wall_s")))
EOF
```

- PASS if: every listed file exists; no errored turn; every source is `['tiny-demo +LTM']`; hash_ok True;
  corpus True; the `word__lesion_recruit` line shows `patch: {'patch': 'lesion_recruit', 'calls': N}` with N > 0.
- If a file is missing and no `stage1_exit_gate --worker` process is running (`ps aux | grep "[s]tage1_exit_gate --worker"`):
  re-run that one shard once with the command below (the worktree needs `data -> $P/data` linked: `ls $W/data/corpus/tinystories.txt`
  must exist; if not: `ln -s $P/data $W/data`), then re-check. If it fails again: STOP (NEEDS-CLAUDE #1).

```bash
cd $W && bash -c 'bash tools/mem_ok.sh 7 && bash tools/memcap.sh 9 -- '"$PY"' -u -m research.runners.stage1_exit_gate --shard --seed 7 --group GROUP --arm ARM --out research/findings/raw/_stage1_exit/dev/s7/GROUP__ARM.json'
# GROUP/ARM: word/intact_a ; word/lesion_recruit ; lived/intact_a ; cat_disc/intact_a ; val_novel/intact_a
```

- Any other failure of the PASS list: STOP (NEEDS-CLAUDE #1). Do not edit code.

## 3. Checks before provisioning (both must print OK)

```bash
cd $W
$PY -m research.runners.stage1_exit_gate --selftest | tail -1                    # must print: SELFTEST OK
SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_stage1_exit_gate.py | tail -1   # must print: 9 passed
git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null       # every runner appends to this shared log; never commit it
git status --short | grep -v "^??" | wc -l                                      # must print 0 (nothing uncommitted)
SHA=$(cat docs/drafts/stage1-exit/GATE_SHA); git merge-base --is-ancestor $SHA HEAD && echo GATE-SHA-OK
git diff $(cat docs/drafts/stage1-exit/PREREG_SHA) $SHA -- research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md | wc -l   # must print 0
git fetch origin main && git merge-base --is-ancestor origin/main $SHA && echo CONTAINS-MAIN || echo NEEDS-MERGE
```

If it prints `NEEDS-MERGE` (main moved after the packet was pushed): `git merge --no-edit origin/main`; if that
reports a conflict, `git merge --abort` and STOP (NEEDS-CLAUDE #2). After a clean merge, re-run the selftest and
the tests; any failure: STOP (NEEDS-CLAUDE #2). Then re-pin: `git rev-parse HEAD > docs/drafts/stage1-exit/GATE_SHA`,
commit it with the message `chore(stage1-exit): re-pin GATE_SHA after merging main (prereg unchanged)` and push
(`bash tools/push_both.sh research/stage1-exit-gate`). The prereg file must be byte-identical to commit 1:
`git diff $(cat docs/drafts/stage1-exit/PREREG_SHA) HEAD -- research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md`
prints nothing; if it prints anything, STOP (NEEDS-CLAUDE #2).

## 4. Provision the pool at the gate revision

```bash
cd $P && SHA=$(cat $W/docs/drafts/stage1-exit/GATE_SHA) && bash tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
for n in pool41 pool42; do
  ssh -o BatchMode=yes $n "ls ~/Projects/sim-data/knowledge_bundles/ 2>/dev/null | grep -q wikidata_core_15k && echo '$n LTM OK' || echo '$n LTM MISSING'; test -f ~/derisk-pool/revisions/$SHA/data/corpus/tinystories.txt && echo '$n CORPUS OK' || echo '$n CORPUS MISSING'"
done
```

- `REFUSED ... does not contain origin/main`: go back to step 3 (NEEDS-MERGE path). Twice: STOP (NEEDS-CLAUDE #2).
- `LTM MISSING`: `bash tools/pool_sync_assets.sh <node>` once, re-check; still missing: STOP (NEEDS-CLAUDE #3).
- `CORPUS MISSING`: re-run `pool_provision.sh` for that node once (it syncs `data/corpus/` from the primary
  checkout); still missing: STOP (NEEDS-CLAUDE #3). A node without the corpus builds a DEGRADED brain and every
  shard from it is UNDEFINED by the aggregate's precondition.

## 5. Queue the 204 gate jobs (one per seed x group x arm)

```bash
cd $W && SHA=$(cat docs/drafts/stage1-exit/GATE_SHA) && SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_exit_gate --emit-pool-jobs $SHA > /home/dant123/Projects/sim/research/queue/stage1_exit_gate_jobs.tsv && wc -l /home/dant123/Projects/sim/research/queue/stage1_exit_gate_jobs.tsv   # must print 204
cd $P && bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < research/queue/stage1_exit_gate_jobs.tsv'
bash tools/pool_queue.sh list | head -3
```

Any `ADD-FAILED`: re-run the loop once for those lines only (the duplicate guard refuses lines already queued).
Still failing: STOP (NEEDS-CLAUDE #4). Append to the WORKLOG: the SHA, the queue time, "204 jobs queued".

**ETA.** Measured on dev seed 7 (numpy, one thread, prereg section 11): build + first content turn 4.5-13 min, a
teach/recall turn 10-47 s, a feel turn ~2 s, and in group `lived` the forced BTSP write adds 8-18 min to the teach
turn and to each later verified recall (a `lived` arm took 47 min). Planning figure per job on a pool node:
`cat_disc` ~25 min, `cat_told` ~20 min, `word` ~21 min, `lived` ~65 min, `val_heldout` ~17 min, `val_novel` ~11
min; about 95 job-hours in total. Each job is ~6.3-7.7 GB RSS (`mem_gb=7`), so pool41/pool42 (15 GB each) run 1-2
jobs each: expect 25-50 h of wall time if nothing else uses
the nodes. **The queue is shared and was 330 deep at packet time (2026-09-29 22:10: 215 long-delay-credit sweep
lines at `mem_gb=2` and 104 host-share gate lines at `mem_gb=7`, ~40-60 min each), so on the pool alone these 204
lines start one to three days after queueing and finish one to two days later.** If the owner wants the gate
inside the October calendar window and the queue is still above ~150 when you reach this step, add NEEDS-CLAUDE
#13 (AWS lane: `bash tools/aws_cpu_launch.sh` under the $50/day cap; the prereg's section 10 sizes it at a few
hours and about $5-10) and continue queueing on the pool anyway. Record the start time; do not re-queue a running
job (check `ssh <node> 'ps aux | grep [s]tage1_exit_gate'` first; `POOL_JOB_MAX_AGE` is 12 h).

## 6. Wait for completion

Results sync back into `$P/research/findings/raw/_stage1_exit/gate/` (`pool-sync.service`, or
`bash tools/pool_sync.sh` by hand).

```bash
find $P/research/findings/raw/_stage1_exit/gate -name '*.json' ! -name '*.prov.json' ! -name '*.part*' ! -name 'aggregate.json' | wc -l   # 204 when complete
```

When the count stops rising for 6 h below 204: run step 7's aggregate (its `missing_shards` list names them),
re-queue exactly those lines from `research/queue/requeue_stage1_exit_gate_c238ff325.tsv` (section 0) ONCE (grep the line by its `--group`
and `--arm` and `--seed`), wait again. Still missing after that: score anyway (the verdict will be UNDEFINED
(INCOMPLETE)) and add NEEDS-CLAUDE #5.

## 7. Score with the registered scorer, write the result, commit

```bash
cd $W && mkdir -p research/findings/raw/_stage1_exit/gate && rsync -a $P/research/findings/raw/_stage1_exit/gate/ research/findings/raw/_stage1_exit/gate/
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_exit_gate --aggregate research/findings/raw/_stage1_exit/gate --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage1_exit/gate/aggregate.json > /home/dant123/Projects/sim/research/queue/stage1_exit_report.md
cat /home/dant123/Projects/sim/research/queue/stage1_exit_report.md
```

Write the finding with this exact header, then the report block unchanged, then one line per criterion from
section 8 (copy the sentence for its verdict; add nothing else):

```bash
cd $W && D=$(date +%F) && F=research/findings/${D}-stage1-exit-gate-RESULT.md && SHA=$(cat docs/drafts/stage1-exit/GATE_SHA) && {
printf -- '---\ntype: finding\nstatus: live\ndate: %s\nlane: integration-first (stage exits on the calendar)\nmechanism: Stage-1 (INFANT) exit gate on the integrated production brain, six seeds, scored by the registered scorer\nseeds: [42, 43, 44, 100, 101, 102]\nrunner: research/runners/stage1_exit_gate.py\nartifacts:\n  - research/findings/raw/_stage1_exit/gate/aggregate.json\n---\n\n# Stage-1 (INFANT) exit gate: result (%s)\n\nPrereg: research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md (gate revision %s). Scored only by `python -m research.runners.stage1_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_exit/gate/aggregate.json.\n\n' "$D" "$D" "$SHA"
grep -v '^-> ' /home/dant123/Projects/sim/research/queue/stage1_exit_report.md
printf '\n\n## Per-criterion reading (fixed sentences from HANDOFF section 8)\n\n'
} > $F && echo "now append one line per criterion from HANDOFF section 8 to $F"
```

Then commit and push (the shards, the aggregate, the finding, the packet notes):

```bash
cd $W && git add research/findings/raw/_stage1_exit/gate research/findings/*-stage1-exit-gate-RESULT.md docs/drafts/stage1-exit/WORKLOG.md
git commit -m "data(stage1-exit): six-seed Stage-1 exit gate on the integrated production brain -- verdict <paste the aggregate verdict>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && bash tools/push_both.sh research/stage1-exit-gate
```

If the commit is blocked ONLY by `compute_idle_persistent` or `lane_starvation`: wait 30 min and retry (max 6
tries), then STOP (NEEDS-CLAUDE #6). Any other gate failure: STOP (NEEDS-CLAUDE #6). Do not use `--no-verify`.

## 8. Outcome -> what to write (no other interpretation)

Stage line (first line of the report):

| report says | do |
|---|---|
| `Stage-1 exit verdict: PASS` | write the finding (step 7) and one board line (below). NEEDS-CLAUDE #7 (a first-ever stage exit must be adversarially verified before the board calls it passed). |
| `Stage-1 exit verdict: FAIL` | write the finding (step 7) and one board line. This is the expected outcome (prereg section 9); it is a measurement, not a stop. |
| `Stage-1 exit verdict: UNDEFINED (INCOMPLETE): ...` with only `missing shards` | re-queue the missing shards once (step 6), re-score. Still missing: finding marked UNDEFINED (INCOMPLETE) + NEEDS-CLAUDE #5. |
| `UNDEFINED (INCOMPLETE)` naming `null clean` | finding marked UNDEFINED; NEEDS-CLAUDE #8 (a non-deterministic build). Do not re-run. |
| `UNDEFINED (INCOMPLETE)` naming `expected brain` or `corpus` | finding marked UNDEFINED; NEEDS-CLAUDE #3 (a node built a different or degraded brain). |
| `UNDEFINED (INCOMPLETE)` naming `lesion applied` or `PYTHONHASHSEED` | finding marked UNDEFINED; NEEDS-CLAUDE #8. |

Per-criterion sentence (copy the one matching each PRIMARY group's verdict; `<crit>` = C1..C4, `<group>` = its
group; the numbers come from the report's table row):

| verdict | sentence |
|---|---|
| PASS | `<crit>` (`<group>`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6. |
| FAIL, "capability absent" | `<crit>` (`<group>`): FAIL, capability absent on <k> of 6 seeds. The integrated brain does not show this Stage-1 ability at production defaults; the gap named in prereg section 8 applies. |
| FAIL, "not load-bearing" | `<crit>` (`<group>`): FAIL, not load-bearing: the reply was correct but the <lesion> arm changed it on only <k> of 6 seeds. |
| FAIL, "permuted exposure" | `<crit>` (`<group>`): FAIL, not exposure-driven: the permuted exposure was tracked on only <k> of 6 seeds. |
| INCOMPLETE | `<crit>` (`<group>`): UNDEFINED, <n> of 6 seeds defined (reasons in aggregate.json). |

For the two diagnostic groups (`cat_told`, `val_novel`) add one line each: "`<group>` (diagnostic, not counted):
<verdict>, correct <n>/<n_defined>." For `val_heldout` also copy the per-seed r lines from the report verbatim.

Board line (never edit the primary checkout's files; append it to the WORKLOG, Claude moves it to the board):
`<date> Stage-1 exit gate (first integrated stage-exit attempt): <stage verdict>; C1 <v>, C2 <v>, C3 <v>, C4 <v> (r per seed: ...). Finding: <path>.`

## 9. Stop conditions (stop that line of work, write the NEEDS-CLAUDE item into the WORKLOG, do nothing else on it)

- GATE_SHA missing, or the prereg file differs from commit 1 at the pinned revision.
- A gate blocks a commit for any reason other than compute-idle / lane-starvation, or those persist past 6 retries.
- A merge conflict, or a selftest/test failure after a merge.
- Dev verification fails its PASS list twice.
- A pool node lacks the LTM bundle or the corpus after one re-sync.
- Queue adds fail twice.
- Any UNDEFINED reason in section 8 other than missing shards.
- Anything that would require editing `research/runners/stage1_exit_gate.py`, the prereg, a bar, a turn or an arm.

## 10. NEEDS-CLAUDE (judgment items; do not attempt)

1. Dev verification failure (an errored turn, a wrong source, a lesion patch never called, a missing shard twice).
2. Merge conflicts with main, tests failing after a merge, or the prereg not byte-identical at the pinned revision.
3. A pool node without the LTM tier or the corpus after one re-sync (a degraded or different brain).
4. Pool queue adds failing twice.
5. Gate shards still missing after one re-queue.
6. A gate failure on the result commit other than compute-idle / lane-starvation.
7. A PASS on any criterion, and above all a stage PASS: run `verify-go` adversarial skeptics (instrument first:
   the null, the lesion persistence, the permuted arm, the host share) before any board or ledger word changes.
8. A null failure (intact_a != intact_b), a hash-seed or lesion-applied precondition failure: diagnosis.
9. After the result lands: sync the summary docs + board (`sync-documentation`, `vikunja`); record the per-criterion
   table as the Stage-1 checklist under the roadmap's Phase C; open the integration slot each FAIL names
   (prereg section 8).
10. Host share at these seams: run the host-share instrument's intervention arms (branch
    `research/host-share-baselines`, `host_share_schema.py`) on the C2/C3 probe turns once that packet is on main;
    report F/S/D beside the criterion verdicts (prereg section 5).
11. A cupy replication of group `lived` WITHOUT the forced `BRAIN_EPISODIC_STORE=1` (the production GPU default),
    through `tools/gpu_queue.sh`, only after the pool gate has landed; informational.
12. The predicted C1 FAIL: choose the integration method (rung-3c learned member->context pathway vs the EMERGE-26
    shared superordinate code) for a default-off wiring into the strict route's recall miss.
13. Pool queue depth above ~150 at queue time: decide whether to run the gate on the AWS CPU lane (cap $50/day,
    auto-stop) so it lands inside the October window, or let it wait its turn on the pool.
