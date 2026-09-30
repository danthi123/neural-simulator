# HANDOFF — Stage-2 (CHILD) exit gate: execution packet for the local model

Written 2026-09-30 by Claude. Execute top to bottom. Do not change code, bars, turns or arms. At most ONE fix round
per issue; anything not covered here goes on the NEEDS-CLAUDE list at the end and you stop that line of work.

- Worktree: `/home/dant123/Projects/sim/.claude/worktrees/stage2-exit-gate` (branch `research/stage2-exit-gate`).
  Below, `W=` that path, `P=/home/dant123/Projects/sim` (the primary checkout, on main; never switch its branch),
  `PY=/home/dant123/Projects/sim/.venv/bin/python` (the worktree's `.venv` and `data` are symlinks to the primary's).
- Never merge to main. Never `--no-verify`. Never write waiver files. Push only with `bash tools/push_both.sh`.
- Set these first in every new shell:
  `W=/home/dant123/Projects/sim/.claude/worktrees/stage2-exit-gate; P=/home/dant123/Projects/sim; PY=/home/dant123/Projects/sim/.venv/bin/python`
- The shell may be zsh or fish: run every multi-line block exactly as given, inside `bash -c '...'` where shown.
- Prereg (frozen bars, arms, turns, predictions): `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md`.
  Runner: `research/runners/stage2_exit_gate.py`. Everything is scored by `--aggregate`; never by hand.
- The Stage-1 packet (`docs/drafts/stage1-exit/HANDOFF.md`, branch `research/stage1-exit-gate`) is the template this
  one follows; where the two differ, THIS file governs the Stage-2 gate.

## 0. State at handoff

- Commits on `research/stage2-exit-gate`, pushed to both remotes (newest last): `9e76e1299` the prereg alone;
  `e4bdf9f41` runner + tests + this packet + the WORKLOG; `8c934312a` the PREREG_SHA pin; then the 2026-09-30
  adversarial review: `ed956d08b` runner/test fixes (a gate-seed shard's worker was refused rc=2, so no gate job
  could ever have produced a shard; errored/organ-error arms now read UNDEFINED), `7e3b2972c` prereg amendment A1
  (section 12, instrument only, filed before any gate job), then this HANDOFF fix with PREREG_SHA re-pinned.
- `docs/drafts/stage2-exit/PREREG_SHA` holds the LAST commit that touched the prereg (the original plus its dated
  pre-dispatch amendments; today `7e3b2972c`). `docs/drafts/stage2-exit/GATE_SHA` holds the revision the gate runs
  at; it is written by Claude only after the dev-seed-7 smoke has been read. **GATE_SHA does not exist yet.** If
  either file is missing when you reach step 3, STOP (NEEDS-CLAUDE #1).
- Dev verification (dev seed 7): NOT yet read. The review queued the dev arms (section 2's list) on the pool with
  `FRONT=1` at the review's final revision; their shards land in `$P/research/findings/raw/_stage2_exit/dev/s7/`
  (`pool-sync.service`). Claude reads them, writes the WORKLOG lines, commits the shards and pins GATE_SHA. Until
  the WORKLOG says so, do not start step 3.
- The gate has NOT been queued. Nothing in `research/findings/raw/_stage2_exit/gate/` exists yet. It is queued in
  November (section 1), never earlier.

## 1. When to start (the calendar rule)

The Stage-2 exit attempt is due by 2026-11-21 (CLAUDE.md, integration-first + pivot dwell). Queue it so it lands
inside that window and after the Stage-1 result has been scored (so the two stage checklists are read together):

- Do NOT run steps 3-5 before **2026-11-01**, unless a Claude inspection note in this file's WORKLOG says otherwise.
- Run steps 3-5 (checks, provisioning, queueing) between **2026-11-03 and 2026-11-10**; target **2026-11-05**. The
  gate needs ~69 job-hours (prereg section 10): on pool41+pool42 alone that is 17-35 h of wall time AFTER the lines
  are reached, so a queue that is more than ~150 lines deep on the day you queue means the AWS decision
  (NEEDS-CLAUDE #13) must be raised the same day, with the pool lines queued anyway.
- If on 2026-11-10 the gate is still not queued for any reason, queue it that day and add NEEDS-CLAUDE #13.
- Scoring (step 7) happens as soon as the 186 shards have landed; the finding is written the same day, no later
  than **2026-11-21**. If the shards are incomplete on 11-21, score anyway (the verdict is UNDEFINED (INCOMPLETE))
  and write the finding as such; that is the calendar attempt.

Before queueing, also confirm commits are possible on the primary checkout: `cd $P && $PY tools/parallel_audit.py |
grep -c "UNDER-PARALLELIZED"` prints 0, or the pool queue depth (`bash tools/pool_queue.sh depth`) is below 40; if
neither holds, queue anyway (the calendar wins) and note it in the WORKLOG.

## 2. Dev verification (Claude reads it; you only re-check that the committed files exist)

```bash
cd $W
ls research/findings/raw/_stage2_exit/dev/s7/*.json | grep -v prov       # expect the files listed in prereg section 11
$PY - <<'EOF'
import json, glob
for p in sorted(glob.glob("research/findings/raw/_stage2_exit/dev/s7/*.json")):
    if p.endswith(".prov.json"): continue
    d = json.load(open(p)); r = d["responses"]
    errs = [k for k, v in r.items() if "_error" in v]
    orgs = [k for k, v in r.items() if any(isinstance(x, dict) and x.get("error") for x in v.values())]
    print("- `%s`: %d turns, sources %s, hash_ok %s, corpus %s, patch %s, errored %s, organ_errors %s, builtin_facts %s, process %s, wall %ss" % (
        p.split("/")[-1], len(r), d.get("sources"), d.get("pythonhashseed_ok"), d.get("corpus_present"), d.get("patch"), errs,
        orgs, len(d.get("builtin_facts") or []), d.get("process"), d.get("wall_s")))
EOF
```

- PASS if: every listed file exists; no errored turn; no organ error; every source is `['tiny-demo +LTM']`;
  hash_ok True; corpus True; builtin_facts > 0 on every arm; `process` shows `ltm_bundle` `wikidata_100k` and
  `peak_rss_mb` below 9216 (the memcap).
- If a file is missing and no `stage2_exit_gate --worker` process is running (`ps aux | grep "[s]tage2_exit_gate --worker"`):
  re-run that one shard once with the command below (the worktree needs `data -> $P/data` linked:
  `ls $W/data/corpus/tinystories.txt` must exist; if not: `ln -s $P/data $W/data`), then re-check. If it fails
  again: STOP (NEEDS-CLAUDE #1).

```bash
cd $W && bash -c 'bash tools/mem_ok.sh 7 && bash tools/memcap.sh 9 -- env SIM_NO_PROVENANCE=1 '"$PY"' -u -m research.runners.stage2_exit_gate --shard --seed 7 --group GROUP --arm ARM --out research/findings/raw/_stage2_exit/dev/s7/GROUP__ARM.json'
# GROUP/ARM (the review's dev list, 2026-09-30): the prereg's ten (the five intact_a ; ask/lesion_graded ;
#   inner/lesion_workspace ; conv/lesion_mouth ; emo/lesion_affect ; infer/transitive_chat) plus the null check
#   (ask/intact_b ; inner/intact_b) and the lesion arms no dev run had covered (ask/lesion_curiosity ;
#   inner/lesion_selfinit ; infer/lesion_chain_route)
```

- Any other failure of the PASS list: STOP (NEEDS-CLAUDE #1). Do not edit code.

## 3. Checks before provisioning (every line must print what its comment says)

```bash
cd $W
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage2_exit_gate --selftest | tail -1                    # must print: SELFTEST OK
SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_stage2_exit_gate.py | tail -1   # must print: 20 passed
git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null       # every runner appends to this shared log; never commit it
git status --short | grep -v "^??" | wc -l                                      # must print 0 (nothing uncommitted)
SHA=$(cat docs/drafts/stage2-exit/GATE_SHA); git merge-base --is-ancestor $SHA HEAD && echo GATE-SHA-OK
git diff $(cat docs/drafts/stage2-exit/PREREG_SHA) $SHA -- research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md | wc -l   # must print 0
test "$(git log -1 --format=%H $SHA -- research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md)" = "$(cat docs/drafts/stage2-exit/PREREG_SHA)" && echo PREREG-PIN-OK
git fetch origin main && git merge-base --is-ancestor origin/main $SHA && echo CONTAINS-MAIN || echo NEEDS-MERGE
```

If it prints `NEEDS-MERGE` (main moved after the packet was pushed; by November it will have): `git merge --no-edit
origin/main`; if that reports a conflict, `git merge --abort` and STOP (NEEDS-CLAUDE #2). After a clean merge, re-run
the selftest and the tests; any failure: STOP (NEEDS-CLAUDE #2). Then re-pin: `git rev-parse HEAD >
docs/drafts/stage2-exit/GATE_SHA`, commit it with the message `chore(stage2-exit): re-pin GATE_SHA after merging main
(prereg unchanged)` and push (`bash tools/push_both.sh research/stage2-exit-gate`). The prereg file must be
byte-identical to PREREG_SHA: `git diff $(cat docs/drafts/stage2-exit/PREREG_SHA) HEAD --
research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` prints nothing; if it prints anything, STOP
(NEEDS-CLAUDE #2). **A merge that changes a production default the gate reads is not yours to judge**: after the
merge, run `cd $W && SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_stage2_exit_gate.py -k
lesion_env_flags` (must pass) and note the merge SHA in the WORKLOG; Claude reads the diff at the next inspection.

## 4. Provision the pool at the gate revision

```bash
cd $P && SHA=$(cat $W/docs/drafts/stage2-exit/GATE_SHA) && bash tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
for n in pool41 pool42; do
  ssh -o BatchMode=yes $n "ls ~/Projects/sim-data/knowledge_bundles/ 2>/dev/null | grep -q wikidata_100k && echo '$n LTM OK' || echo '$n LTM MISSING'; test -f ~/derisk-pool/revisions/$SHA/data/corpus/tinystories.txt && echo '$n CORPUS OK' || echo '$n CORPUS MISSING'"
done
```

- `REFUSED ... does not contain origin/main`: go back to step 3 (NEEDS-MERGE path). Twice: STOP (NEEDS-CLAUDE #2).
- `LTM MISSING`: `bash tools/pool_sync_assets.sh <node>` once, re-check; still missing: STOP (NEEDS-CLAUDE #3).
- `CORPUS MISSING`: re-run `pool_provision.sh` for that node once (it syncs `data/corpus/` from the primary
  checkout); still missing: STOP (NEEDS-CLAUDE #3). A node without the corpus builds a DEGRADED brain and every
  shard from it is UNDEFINED by the aggregate's precondition.
- If pool40 is back (`ssh -o BatchMode=yes pool40 true` succeeds), add it to both commands.

## 5. Queue the 186 gate jobs (one per seed x group x arm)

```bash
cd $W && SHA=$(cat docs/drafts/stage2-exit/GATE_SHA) && SIM_NO_PROVENANCE=1 $PY -m research.runners.stage2_exit_gate --emit-pool-jobs $SHA --max-age-h 240 > /home/dant123/Projects/sim/research/queue/stage2_exit_gate_jobs.tsv && wc -l /home/dant123/Projects/sim/research/queue/stage2_exit_gate_jobs.tsv   # must print 186
cd $P && bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why" </dev/null || echo "ADD-FAILED: $cmd"; done < research/queue/stage2_exit_gate_jobs.tsv'
bash tools/pool_queue.sh list | head -3
```

Every line already carries `mem_gb=7 max_age_h=240` in its `--checked` reason (the dispatcher skips a line older
than 12 h without the token; 240 h is the cap). Any `ADD-FAILED`: re-run the loop once for those lines only (the
duplicate guard refuses lines already queued). Still failing: STOP (NEEDS-CLAUDE #4). Append to the WORKLOG: the
SHA, the queue time, the queue depth before and after, "186 jobs queued".

**ETA.** Prereg section 10: about 69 job-hours; each job ~6.3-7.7 GB RSS (`mem_gb=7`), so pool41/pool42 (15 GB each)
run 1-2 jobs each: 17-35 h of wall time once the lines are reached. Record the start time; do not re-queue a
running job (check `ssh <node> 'ps aux | grep [s]tage2_exit_gate'` first). If the queue depth before your add is
above ~150, add NEEDS-CLAUDE #13 (the AWS CPU lane, `bash tools/aws_cpu_launch.sh`, under the $50/day cap; about
4-6 h and $6-9) and continue queueing on the pool anyway.

## 6. Wait for completion

Results sync back into `$P/research/findings/raw/_stage2_exit/gate/` (`pool-sync.service`, or
`bash tools/pool_sync.sh` by hand).

```bash
find $P/research/findings/raw/_stage2_exit/gate -name '*.json' ! -name '*.prov.json' ! -name '*.part*' ! -name 'aggregate.json' | wc -l   # 186 when complete
```

When the count stops rising for 6 h below 186: run step 7's aggregate (its `missing_shards` list names them),
re-queue exactly those lines from `research/queue/stage2_exit_gate_jobs.tsv` ONCE (grep the line by its `--group`,
`--arm` and `--seed`), wait again. Still missing after that: score anyway (the verdict will be UNDEFINED
(INCOMPLETE)) and add NEEDS-CLAUDE #5.

## 7. Score with the registered scorer, write the result, commit

```bash
cd $W && mkdir -p research/findings/raw/_stage2_exit/gate && rsync -a $P/research/findings/raw/_stage2_exit/gate/ research/findings/raw/_stage2_exit/gate/
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage2_exit_gate --aggregate research/findings/raw/_stage2_exit/gate --seeds 42,43,44,100,101,102 --out research/findings/raw/_stage2_exit/gate/aggregate.json > /home/dant123/Projects/sim/research/queue/stage2_exit_report.md
cat /home/dant123/Projects/sim/research/queue/stage2_exit_report.md
```

Write the finding with this exact header, then the report block unchanged, then one line per criterion from
section 8 (copy the sentence for its verdict; add nothing else):

```bash
cd $W && D=$(date +%F) && F=research/findings/${D}-stage2-exit-gate-RESULT.md && SHA=$(cat docs/drafts/stage2-exit/GATE_SHA) && {
printf -- '---\ntype: finding\nstatus: live\ndate: %s\nlane: integration-first (stage exits on the calendar)\nmechanism: Stage-2 (CHILD) exit gate on the integrated production brain, six seeds, scored by the registered scorer\nseeds: [42, 43, 44, 100, 101, 102]\nrunner: research/runners/stage2_exit_gate.py\nartifacts:\n  - research/findings/raw/_stage2_exit/gate/aggregate.json\n---\n\n# Stage-2 (CHILD) exit gate: result (%s)\n\nPrereg: research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md (gate revision %s). Scored only by `python -m research.runners.stage2_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage2_exit/gate/aggregate.json.\n\n' "$D" "$D" "$SHA"
grep -v '^-> ' /home/dant123/Projects/sim/research/queue/stage2_exit_report.md
printf '\n\n## Per-criterion reading (fixed sentences from HANDOFF section 8)\n\n'
} > $F && echo "now append one line per criterion from HANDOFF section 8 to $F"
```

Then commit and push (the shards, the aggregate, the finding, the packet notes):

```bash
cd $W && git add research/findings/raw/_stage2_exit/gate research/findings/*-stage2-exit-gate-RESULT.md docs/drafts/stage2-exit/WORKLOG.md
git commit -m "data(stage2-exit): six-seed Stage-2 exit gate on the integrated production brain -- verdict <paste the aggregate verdict>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && bash tools/push_both.sh research/stage2-exit-gate
```

If the commit is blocked ONLY by `compute_idle_persistent` or `lane_starvation`: wait 30 min and retry (max 6
tries), then STOP (NEEDS-CLAUDE #6). Any other gate failure: STOP (NEEDS-CLAUDE #6). Do not use `--no-verify`.

## 8. Outcome -> what to write (no other interpretation)

Stage line (first line of the report):

| report says | do |
|---|---|
| `Stage-2 exit verdict: PASS` | write the finding (step 7) and one board line (below). NEEDS-CLAUDE #7 (a stage exit must be adversarially verified before the board calls it passed). |
| `Stage-2 exit verdict: FAIL` | write the finding (step 7) and one board line. This is the expected outcome (prereg section 9); it is a measurement, not a stop. |
| `Stage-2 exit verdict: UNDEFINED (INCOMPLETE): ...` with only `missing shards` | re-queue the missing shards once (step 6), re-score. Still missing: finding marked UNDEFINED (INCOMPLETE) + NEEDS-CLAUDE #5. |
| `UNDEFINED (INCOMPLETE)` naming `null clean` | finding marked UNDEFINED; NEEDS-CLAUDE #8 (a non-deterministic build). Do not re-run. |
| `UNDEFINED (INCOMPLETE)` naming `expected brain` or `corpus` | finding marked UNDEFINED; NEEDS-CLAUDE #3 (a node built a different or degraded brain). |
| `UNDEFINED (INCOMPLETE)` naming `lesion applied` or `PYTHONHASHSEED` | finding marked UNDEFINED; NEEDS-CLAUDE #8. |
| `UNDEFINED (INCOMPLETE)` naming `errored turn` or `organ error` | finding marked UNDEFINED; NEEDS-CLAUDE #8 (an instrument or organ crash; never read it as a lesion effect). Do not re-run. |
| `UNDEFINED (INCOMPLETE)` naming `grounding set` | finding marked UNDEFINED; NEEDS-CLAUDE #1 (the worker did not find the session's ChatBrain). |

Per-criterion sentence (copy the one matching each group's verdict; `<crit>` = C1..C5, `<group>` = its group; the
numbers come from the report's table row and the sub-clause line):

| verdict | sentence |
|---|---|
| PASS | `<crit>` (`<group>`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6. |
| FAIL, "capability absent" | `<crit>` (`<group>`): FAIL, capability absent on <k> of 6 seeds; sub-clauses: <copy the group's sub-clause line>. The integrated brain does not show this Stage-2 ability at production defaults; the gap named in prereg section 8 applies. |
| FAIL, "not load-bearing" | `<crit>` (`<group>`): FAIL, not load-bearing: the reply was correct but the <lesion> arm changed it on only <k> of 6 seeds. |
| FAIL, "permuted exposure" | `<crit>` (`<group>`): FAIL, not exposure-driven: the permuted exposure was tracked on only <k> of 6 seeds. |
| INCOMPLETE | `<crit>` (`<group>`): UNDEFINED, <n> of 6 seeds defined (reasons in aggregate.json). |

For C2 add the ownership line from the informational arms: "`infer` informational: qp_lesion conclusion intact on
<n>/6 (the route did not need the spiking read), transitive_chat derived-moth on <n>/6." For C3 copy the per-seed
ask-rate lines from the report verbatim; for C4 copy the per-seed discrete-emotion lines verbatim.

Board line (never edit the primary checkout's files; append it to the WORKLOG, Claude moves it to the board):
`<date> Stage-2 exit gate (second integrated stage-exit attempt): <stage verdict>; C1 <v>, C2 <v>, C3 <v>, C4 <v>, C5 <v>. Finding: <path>.`

## 9. Stop conditions (stop that line of work, write the NEEDS-CLAUDE item into the WORKLOG, do nothing else on it)

- GATE_SHA missing, or the prereg file differs from commit 1 at the pinned revision.
- A gate blocks a commit for any reason other than compute-idle / lane-starvation, or those persist past 6 retries.
- A merge conflict, or a selftest/test failure after a merge.
- Dev verification fails its PASS list twice.
- A pool node lacks the LTM bundle or the corpus after one re-sync.
- Queue adds fail twice.
- Any UNDEFINED reason in section 8 other than missing shards.
- Anything that would require editing `research/runners/stage2_exit_gate.py`, the prereg, a bar, a turn or an arm.

## 10. NEEDS-CLAUDE (judgment items; do not attempt)

1. Dev verification failure (an errored turn, a wrong source, a lesion patch never called, a missing shard twice).
2. Merge conflicts with main, tests failing after a merge, or the prereg not byte-identical at the pinned revision;
   also any November merge whose diff touches a production default one of the lesion flags reads.
3. A pool node without the LTM tier or the corpus after one re-sync (a degraded or different brain).
4. Pool queue adds failing twice.
5. Gate shards still missing after one re-queue.
6. A gate failure on the result commit other than compute-idle / lane-starvation.
7. A PASS on any criterion, and above all a stage PASS: run `verify-go` adversarial skeptics (instrument first:
   the null, the lesion persistence, the permuted arm, the host share) before any board or ledger word changes.
8. A null failure (intact_a != intact_b), a hash-seed or lesion-applied precondition failure: diagnosis.
9. After the result lands: sync the summary docs + board (`sync-documentation`, `vikunja`); record the per-criterion
   table as the Stage-2 checklist under the roadmap's Phase C, beside Stage 1's; open the integration slot each FAIL
   names (prereg section 8): the frame inventory + form-to-role comprehension (C1), the learning-progress trace on
   the ASK pool (C3), the appraisal -> category WTA in the ladder (C4).
10. Host share at these seams: run the host-share instrument's intervention arms (branch
    `research/host-share-baselines`) on the C2 conclusion probe, the C3 gap probes and the C5 chase probes once that
    packet is on main; report F/S/D beside the criterion verdicts (prereg section 5). C2 in particular: a PASS is
    host-owned at HEAD (the hop reads are a kb scan); decide whether the Phase-B content seam takes the chain route's
    hops first.
11. A cupy replication through `tools/gpu_queue.sh` (`--backend cupy`; the self-initiation organ then runs its full
    CA3 wander and the D5 write is on), informational, only after the pool gate has landed and only in a GPU window
    the local-model driver has yielded (`llm yield`).
12. The Qwen-mouth arm: the same five groups with the production renderer (`BRAIN_CHAT_RENDERER` unset on cupy), as
    B2d's mouth arm does; C1's grammar checker does not apply to open prose (the mouth would trivially pass it), so
    that arm reports grounding and the moats only. Informational.
13. Pool queue depth above ~150 at queue time: decide whether to run the gate on the AWS CPU lane (cap $50/day,
    auto-stop) so it lands inside the November window, or let it wait its turn on the pool.
14. The predicted C1 FAIL: choose between wiring the learned frame inventory behind the recall mouth and a
    form-to-role comprehension organ in front of `_maybe_acquire` (both default-off) as the first Stage-2 integration
    slot; the predicted C3 FAIL: the LP-trace design on the ASK pool; the predicted C4 FAIL: the WTA fold.
