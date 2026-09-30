# HANDOFF: Stage-1 C1 with the category-inheritance wiring (execution packet for the local model)

Written 2026-09-30 by Claude. Execute top to bottom. Use no judgment beyond the tables. Do not change code, bars,
turns or arms. At most ONE fix round per issue; anything not covered here is a NEEDS-CLAUDE item: write it in
`$K/WORKLOG.md` AND under "NEEDS CLAUDE" in `research/coordination/LOCAL_LLM_RUNBOOK.md` (that runbook's own rule),
then stop that line of work and continue with any other thread that is not blocked.

- Prereg (frozen bars, arms, predictions): `research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md`.
  Runner: `research/runners/stage1_c1_category.py` (the Stage-1 exit gate's own C1 group and scorer, imported).
  Wiring: `webapp/category_inherit_chat.py` + `research/runners/category_inherit_organ.py` (default OFF).
  Everything is scored by `--aggregate`; never by hand.
- Worktree `/home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category`, branch `research/stage1-c1-category`.
  Never merge to main. Never flip a default (the anchor `_CATEGORY_INHERIT_DEFAULT_ON` stays `False` on this branch).
  Never `--no-verify`. Never write a waiver file or a filler job. Push only with `bash tools/push_both.sh $BR`.
- **Each fenced block is ONE Bash call** and starts with `source .../env.sh`; loops are already wrapped in `bash -c`.
  `env.sh` sets `P` (primary checkout, on main; never switch its branch), `W` (this worktree), `K` (this packet dir),
  `PY` (the primary's python; the worktree has no `.venv`), `BR`, `PREREG`, `OUT`, `JOBS`, and the pins from
  `$K/PINS.tsv`: `PREREG_SHA` (the prereg's own commit) and `GATE_SHA` (the LAST GATE row: the revision the gate runs).
- `$W/data` must be a symlink to `$P/data` (the corpus is git-excluded): `ls $W/data/corpus/tinystories.txt` must
  succeed; if not: `ln -s $P/data $W/data`.

## 0. State at handoff

- Commits on `research/stage1-c1-category` (pushed to both remotes): the Stage-1 exit branch merged in (the C1 group
  and scorer are imported from it); the code commit(s); the prereg ALONE (`PREREG_SHA`); the dev shards + this packet
  (`GATE_SHA`); the pins. `$K/PINS.tsv` holds one PREREG row and one GATE row. If either is missing: STOP
  (NEEDS-CLAUDE #1).
- Dev verification is DONE (prereg section 11; `research/findings/raw/_stage1_c1_category/dev/`): the organ on dev
  seeds 1-8, and through the real handler on dev seed 7: `intact_a` -> "perhaps the canary eats the worm" (a flagged
  guess, `hypothesis_svo` [canary, eat, worm]); `lesion_inherit` -> abstain, only that turn changes; `permuted` ->
  "perhaps the canary eats the bug". The other dev-seed-7 arms (`off`, `intact_b`, the informational ablations) were
  NOT run: the workstation stayed memory-contended by sibling dev runs (`mem_ok.sh 9` refused for over an hour), so
  the launcher was stopped after the gate was queued. Declared risk: the real-handler null (`intact_a == intact_b`)
  was not checked at a dev seed (the organ's same-seed determinism was, 8/8); a null failure at the gate seeds reads
  UNDEFINED (NEEDS-CLAUDE #11), never a verdict. Do not re-run dev arms; the gate measures all of them.
- **The gate WAS queued by Claude on 2026-09-30 at 12:20 EDT**: main merged (366d131f7), GATE re-pinned to
  `bb6705de6259ef775ca265caa33e5799a587b97c` (the last GATE row), steps 1-2 run as written (CONTAINS-MAIN; six OK
  lines on pool41/pool42), 54 lines added with 0 ADD-FAILED, pool queue depth 526 after the add. **Start at step 4.**
  **Never run steps 1-3 again** (they are the record of how the gate was queued). Every shard of this gate comes
  from ONE revision, `bb6705de6`: never merge main, never re-pin, never regenerate `$JOBS`, never run step 3's loop.
  A missing-shard re-queue is step 4's own block, which re-adds only the absent lines from `$JOBS` as pinned. If a
  node lost the `bb6705de6` revision (the dispatcher logs "revision ... not provisioned"): STOP, NEEDS-CLAUDE #3.
- **Scorer hardening, 2026-09-30 (adversarial review, before any gate job was dispatched; no bar, arm, turn or prereg
  line changed):** a swallowed organ exception (an `error` in a turn's `category_inherit` trace) now makes that arm
  "not applied", so the seed reads UNDEFINED instead of FAIL "capability absent" (prereg section 4: no errored turn,
  the wiring applied). Commit 8ff2547ed; the scorer runs from the worktree HEAD in step 5, the shards from
  `bb6705de6`. Test count is now 25.

## 1. Checks before provisioning (every line must print what its comment says)

**Record only (done 2026-09-30 12:20 EDT). Do not run this step again: section 0.**

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && ls data/corpus/tinystories.txt                                                   # the path, no error
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_c1_category --selftest 2>/dev/null | tail -1        # SELFTEST OK
SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_category_inherit_organ.py tests/test_stage1_exit_gate.py 2>/dev/null | tail -1   # 25 passed
git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null; git status --short --untracked-files=no | wc -l   # 0
git merge-base --is-ancestor $GATE_SHA HEAD && echo GATE-SHA-OK                            # GATE-SHA-OK
git diff $PREREG_SHA $GATE_SHA -- $PREREG | wc -l                                          # 0
git diff $PREREG_SHA HEAD -- $PREREG | wc -l                                               # 0
grep -c "_CATEGORY_INHERIT_DEFAULT_ON = False" webapp/category_inherit_chat.py             # 1
git fetch -q origin main && (git merge-base --is-ancestor origin/main $GATE_SHA && echo CONTAINS-MAIN || echo NEEDS-MERGE)
```

| result | do |
|---|---|
| everything as commented, `CONTAINS-MAIN` | go to step 2 |
| `NEEDS-MERGE` (main moved) | run the merge block below once |
| the selftest or a test fails, a count is not as commented, or the default line reads anything but 1 | STOP, NEEDS-CLAUDE #2 with the failing line |

Merge block (only on `NEEDS-MERGE`):

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && git merge --no-edit origin/main || { git merge --abort; echo MERGE-CONFLICT; exit 1; }
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_c1_category --selftest 2>/dev/null | tail -1
SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_category_inherit_organ.py tests/test_stage1_exit_gate.py 2>/dev/null | tail -1
git diff $PREREG_SHA HEAD -- $PREREG | wc -l
```

If it prints `MERGE-CONFLICT`, anything but `SELFTEST OK` / `25 passed` / `0`: STOP, NEEDS-CLAUDE #2. Otherwise re-pin
and push:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null
printf 'GATE\t%s\t%s\tre-pin after merging main (prereg unchanged)\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" >> $K/PINS.tsv
git add $K/PINS.tsv && git commit -m "chore(stage1-c1): re-pin GATE after merging main (prereg unchanged)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && bash tools/push_both.sh $BR
```

Then return to the top of step 1 (the pins now name the merge commit). A second `NEEDS-MERGE` in the same session:
STOP, NEEDS-CLAUDE #2.

## 2. Provision the pool at GATE_SHA

**Record only (done 2026-09-30 12:20 EDT). Do not run this step again: section 0.**

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $P && bash tools/pool_provision.sh --revision $GATE_SHA --isolated pool41 pool42
for n in pool41 pool42; do
  ssh -o BatchMode=yes $n "ls ~/Projects/sim-data/knowledge_bundles/ 2>/dev/null | grep -q wikidata_core_15k && echo '$n LTM OK' || echo '$n LTM MISSING'; test -f ~/derisk-pool/revisions/$GATE_SHA/data/corpus/tinystories.txt && echo '$n CORPUS OK' || echo '$n CORPUS MISSING'; test -f ~/derisk-pool/revisions/$GATE_SHA/research/runners/stage1_c1_category.py && echo '$n RUNNER OK' || echo '$n RUNNER MISSING'"
done
```

| result | do |
|---|---|
| six OK lines | step 3 |
| `REFUSED ... does not contain origin/main` | back to step 1 (NEEDS-MERGE path); twice: STOP, NEEDS-CLAUDE #2 |
| `LTM MISSING` | `bash tools/pool_sync_assets.sh <node>` once, re-check; still missing: STOP, NEEDS-CLAUDE #3 |
| `CORPUS MISSING` or `RUNNER MISSING` | re-run `pool_provision.sh` for that node once; still missing: STOP, NEEDS-CLAUDE #3 |

## 3. Queue the 54 gate jobs (one per seed x arm)

**Record only (done 2026-09-30 12:20 EDT). Do not run this step again: section 0.**

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_c1_category --emit-pool-jobs $GATE_SHA > $JOBS && wc -l < $JOBS   # 54
grep -c "max_age_h=120" $JOBS                                                                                                # 54
cd $P && bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < '"$JOBS"
grep -c "stage1_c1_category" $P/research/queue/pool.queue                                                                    # 54 (`pool_queue.sh list` truncates commands)
```

Any `ADD-FAILED`: re-run the loop once (the duplicate guard refuses lines already queued). Still failing: STOP,
NEEDS-CLAUDE #4. Append to the WORKLOG: the date/time, `GATE_SHA`, "54 jobs queued", the queue depth
(`bash tools/pool_queue.sh depth`).

**ETA.** 54 jobs x ~25-30 min (numpy, one thread, ~7-7.7 GB RSS each; pool41/pool42 run 1-2 each) = ~25 job-hours,
12-25 h of pool wall time once the lines are reached. The queue was ~440 lines (three to four days) deep at packet
time, so expect the first shard two to four days after queueing. The lines carry `max_age_h=120`; do not re-queue a
line that is still waiting.

## 4. Wait for completion

Results sync into `$P/$OUT` (`pool-sync.service`, or `bash tools/pool_sync.sh` by hand).

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
find $P/$OUT -name '*.json' ! -name '*.prov.json' ! -name '*.part*' ! -name 'aggregate.json' | wc -l   # 54 when complete
```

When the count stops rising for 6 h below 54 and no `stage1_c1_category` line is still queued or running: re-queue
ONCE with this block. It re-adds exactly the `$JOBS` lines whose shard is absent (counted and informational arms),
pinned to `bb6705de6` as they are; it never touches a line whose shard landed.

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $P && grep -c "$GATE_SHA" $JOBS                                                    # 54 (every line at GATE_SHA)
grep -c stage1_c1_category research/queue/pool.queue research/queue/pool.running       # both ...:0
bash -c 'n=0; while IFS=$'"'"'\t'"'"' read -r cmd why; do out=${cmd##*--out }; [ -f "$out" ] && continue; n=$((n+1)); bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < '"$JOBS"'; echo "re-queued $n"'
```

If the first line prints anything but 54, or either count is not 0: do not run the loop; STOP, NEEDS-CLAUDE #5.
Then wait again. Still missing after that one re-queue: score anyway (the verdict reads UNDEFINED (INCOMPLETE)) and
NEEDS-CLAUDE #5. If no shard has landed by 2026-10-12: NEEDS-CLAUDE #10 (the Stage-1 calendar needs this before
2026-10-24). The lines carry `max_age_h=120`: after about 2026-10-05 12:20 EDT the dispatcher SKIPS them but leaves
them in `pool.queue`, so the block's guard (count not 0) stops you. If `stage1_c1_category` lines are still queued
after 2026-10-05 12:20 EDT with fewer than 54 shards landed: STOP, NEEDS-CLAUDE #10 (a fresh re-queue or another lane).

## 5. Score with the registered scorer, write the finding, commit

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && mkdir -p $OUT && rsync -a $P/$OUT/ $OUT/
SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_c1_category --aggregate $OUT --seeds 42,43,44,100,101,102 --out $OUT/aggregate.json > $P/research/queue/stage1_c1_category_report.md
cat $P/research/queue/stage1_c1_category_report.md
```

Write the finding with this exact header, then the report block unchanged, then the fixed sentences of section 6:

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && D=$(date +%F) && F=research/findings/${D}-stage1-c1-category-inheritance-RESULT.md && {
printf -- '---\ntype: finding\nstatus: provisional\ndate: %s\nlane: integration-first (Stage-1 exit criterion C1)\nmechanism: Stage-1 C1 category held-out inheritance on the integrated production brain with the default-off category-inheritance wiring switched on, six seeds, scored by the registered scorer\nseeds: [42, 43, 44, 100, 101, 102]\nrunner: research/runners/stage1_c1_category.py\nartifacts:\n  - research/findings/raw/_stage1_c1_category/gate/aggregate.json\n---\n\n# Stage-1 C1 with the category-inheritance wiring on: result (%s, provisional until Claude inspects)\n\nPrereg: %s (gate revision %s). Scored only by `python -m research.runners.stage1_c1_category --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_c1_category/gate/aggregate.json. The production default is unchanged (the wiring stays off by default).\n\n' "$D" "$D" "$PREREG" "$GATE_SHA"
grep -v '^-> ' $P/research/queue/stage1_c1_category_report.md
printf '\n\n## Reading (fixed sentences from the HANDOFF section 6)\n\n'
} > $F && echo "now append the section-6 sentences to $F"
```

Then commit and push (the shards, the aggregate, the finding, the WORKLOG):

```bash
source /home/dant123/Projects/sim/.claude/worktrees/stage1-c1-category/research/packets/stage1-c1-category/env.sh
cd $W && git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null
git add $OUT research/findings/*-stage1-c1-category-inheritance-RESULT.md $K/WORKLOG.md
git commit -m "data(stage1-c1): six-seed C1 with the category-inheritance wiring on -- verdict <paste the report's first line>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && bash tools/push_both.sh $BR
```

If the commit is blocked ONLY by `compute_idle_persistent` or `lane_starvation`: wait 30 min and retry (max 6 tries),
then STOP, NEEDS-CLAUDE #6. Any other gate: STOP, NEEDS-CLAUDE #6 with its message.

## 6. Outcome -> what to write (no other interpretation)

First line of the report:

| report says | write this sentence, then | NEEDS-CLAUDE |
|---|---|---|
| `... ON: PASS` and `Flip-readiness ...: READY` | "C1 with the wiring on: PASS on 6 of 6 seeds (correct, lesion effect reproduced, permuted exposure tracked); flip-readiness READY. Provisional until Claude's adversarial verification." | #7 (verify-go, then the flip and the Stage-1 amendment) |
| `... ON: PASS` and `NOT READY` | "C1 with the wiring on: PASS on 6 of 6 seeds; flip-readiness NOT READY (safety <n>/6, specificity <n>/6)." | #8 |
| `... ON: FAIL` with `capability absent on k of 6` | "C1 with the wiring on: FAIL, capability absent on <k> of 6 seeds; the per-seed organ reads are in aggregate.json rows.<seed>.organ_read_intact." | #9 |
| `... ON: FAIL` with `not load-bearing` or `not exposure-driven` | "C1 with the wiring on: FAIL, <copy the why clause verbatim>." | #9 |
| `UNDEFINED (INCOMPLETE)` with only missing shards | re-queue once (step 4); still missing: "UNDEFINED (INCOMPLETE): <n> shards missing." | #5 |
| `UNDEFINED (INCOMPLETE)` naming the null, a source, the corpus, PYTHONHASHSEED, or "not applied" | "UNDEFINED (INCOMPLETE): <copy the reason verbatim>." Do not re-run. | #3 (source/corpus) or #11 (null / not applied / hash) |

Then one line per informational arm, copied from the report's "Informational arms" line (no interpretation), and the
board line appended to the WORKLOG (Claude moves it to the board):
`<date> Stage-1 C1 with the category-inheritance wiring on (default off): <verdict>; flip-readiness <READY/NOT READY>. Finding: <path>.`

## 7. Stop conditions (stop that line of work, write the NEEDS-CLAUDE item, do nothing else on it)

- A PINS.tsv row is missing, or the prereg differs from `PREREG_SHA` at `GATE_SHA` or at HEAD.
- A merge conflict, a selftest/test failure after a merge, or a second NEEDS-MERGE in one session.
- A pool node lacks the LTM bundle, the corpus or the runner after one re-provision.
- Queue adds fail twice.
- Any UNDEFINED reason other than missing shards.
- A gate blocks a commit for any reason other than compute-idle / lane-starvation, or those persist past 6 retries.
- Anything that would require editing the runner, the organ, the wiring, the prereg, a bar, a turn or an arm; or
  flipping `_CATEGORY_INHERIT_DEFAULT_ON`.

## 8. NEEDS-CLAUDE (judgment items; do not attempt)

1. A PINS.tsv row missing.
2. Merge conflicts with main, a selftest/test failure, the prereg not byte-identical, or a second NEEDS-MERGE.
3. A pool node without the LTM tier, the corpus or the runner after one re-provision; or an UNDEFINED naming a source
   or the corpus (a degraded or different brain).
4. Pool queue adds failing twice.
5. Gate shards still missing after one re-queue.
6. A gate failure on the result commit other than compute-idle / lane-starvation.
7. C1 PASS + READY: run `verify-go` adversarial skeptics (the null, the lesion persistence `lesion_held`, the
   permuted arm, a frequency shortcut: 3 of the 5 `eat` facts are `worm`, which is exactly why the permuted arm must
   name `bug`); then (a) the flip commit on a flip branch with its no-regression checks (the load-bearing battery
   and the Stage-1 C2/C3/C4 groups unchanged by the flag), and (b) a dated amendment to the Stage-1 exit prereg that
   names `lesion_inherit` as C1's responsible-faculty lesion (the frozen `lesion_draw` cannot see this route), before
   any Stage-1 C1 re-attempt at the new production default.
8. PASS but NOT READY: diagnose the named safety/specificity turn (prereg section 9).
9. FAIL: the prereg section 9 row names where to look; the next method is Claude's.
10. Nothing landed by 2026-10-12: decide the lane (AWS CPU under the $50/day cap, or the local workstation under
    `mem_ok.sh`) so C1 lands inside the October Stage-1 window.
11. A null failure (intact_a != intact_b), the wiring or lesion "not applied" (including a swallowed organ error: the
    reason names the turn and the error), or a hash-seed precondition failure.
12. The bridging-member over-generalization (prereg section 12: a member that shares one frame from each category
    merges them; dev: the canary's `swim` then reached `lake`). Decide before any flip whether the moat needs a
    stricter admission (e.g. a frame reached only through the lateral association must clear a higher floor).
13. `cat_told` (the told-category diagnostic): this wiring does not read `isa` facts; whether EMERGE-26's told
    superordinate belongs on that route is a separate decision.
