# HANDOFF -- sleep transitive inference (plan 2026-09-25 step 5), execution packet for the local model

Written 2026-09-29 by the Claude build agent. Execute top to bottom. Do not tune anything, do not edit the prereg,
runner or grader, never use `--no-verify`, never write a waiver file. At any STOP, write one line to
`WORKLOG-sleep-transitive-inference.md` saying which step stopped and why, then hand back (NEEDS-CLAUDE).

All commands run from the worktree unless stated:

    W=/home/dant123/Projects/sim/.claude/worktrees/sleep-transitive-inference
    P=/home/dant123/Projects/sim            # primary checkout (queue tools, venv)
    PY=$P/.venv/bin/python
    O=research/findings/raw/_sleep_transitive_inference
    cd $W

## State at handoff

- Branch `research/sleep-transitive-inference` (from main `f52c6d246`), worktree `$W`. NOTHING is committed yet:
  commits were blocked by `gates/compute_idle_persistent` (non-exempt files).
- Files ready (uncommitted): `research/runners/sleep_transitive_inference.py` (runner),
  `research/runners/sleep_transitive_inference_grade.py` (registered scorer), `tests/test_sleep_transitive_inference.py`,
  `research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md` (the prereg),
  `research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md` (revision 5),
  `research/biology/sleep-relational-integration.md`, `WORKLOG-sleep-transitive-inference.md`, dev evidence under
  `$O/dev/`, and `research/findings/raw/_provenance/runs.jsonl` (appended by the dev runs).
- Queued on the GPU (`tools/gpu_queue.sh`): dev seeds 8, 9, 10, 11 (cupy, every arm) writing `$O/dev/cupy_seed{8,9,10,11}.json`.

## Step 1 -- wait for the four dev records

    ls $O/dev/cupy_seed8.json $O/dev/cupy_seed9.json $O/dev/cupy_seed10.json $O/dev/cupy_seed11.json
    grep -l "Traceback" $O/dev/cupy_seed*.log

- All four JSON files exist and no log has a Traceback: go to step 2.
- A job is still queued or running (`bash $P/tools/gpu_queue.sh status`): wait, check again in 30 min.
- A log has a Traceback, or a job ended without its JSON: STOP (NEEDS-CLAUDE 1).

## Step 2 -- the operating-point check (registered; it reads no inference result)

    $PY -m research.runners.sleep_transitive_inference_grade --devcheck $O/dev/cupy_seed8.json $O/dev/cupy_seed9.json $O/dev/cupy_seed10.json $O/dev/cupy_seed11.json; echo rc=$?

- rc=0 (`all_pass: true`, written to `$O/dev/opcheck.json`): go to step 3.
- rc=1: STOP (NEEDS-CLAUDE 2). Do not change any constant.

## Step 3 -- pre-commit checks

    $PY -m pytest -q tests/test_sleep_transitive_inference.py
    $PY -m research.runners.sleep_transitive_inference_grade --selftest | tail -1
    $PY tools/check_docs.py | tail -1
    $PY tools/claim_check.py research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md | tail -3
    $PY tools/biology_check.py | grep -i "sleep-relational\|✘" | head

All must pass (8 passed; SELFTEST PASS; "both document-structure rules pass"; "every measurement traces"; the biology
entry "✔"). Any failure: STOP (NEEDS-CLAUDE 3).

## Step 4 -- commit (two commits, in this order, pathspec only)

The compute-idle gate may block. Check first; if it reads UNDER-PARALLELIZED on compute, wait and retry every 30 min
(the coordinator expected commits to open by about 18:30 on 2026-09-29):

    (cd $P && $PY tools/parallel_audit.py | head -3)

Commit 1, code and dev data (no prereg in it):

    git add research/runners/sleep_transitive_inference.py research/runners/sleep_transitive_inference_grade.py tests/test_sleep_transitive_inference.py $O/dev research/findings/raw/_provenance/runs.jsonl
    git commit -m "feat(sleep-ti): runner, registered scorer, tests and dev evidence for offline-replay associative inference (plan step 5)" -m "Spiking CA3 replay store (Ecker AdEx preset, PV feedback, engine STDP); arms wake/sleep/off/dis/yok/wperm/scr/rev/cross/nohub/nostdp/nocomp/noadapt; dev seeds 7-11 only, no gate seed run. Operating-point check: $O/dev/opcheck.json." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

Commit 2, the preregistration and its documents (no raw artifact in it):

    git add research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md research/biology/sleep-relational-integration.md WORKLOG-sleep-transitive-inference.md
    git commit -m "prereg(sleep-ti): offline-replay associative inference with the prioritized-memory gates; design revision 5 closes the four open re-review items and registers WM11" -m "Registered before any gate seed. STI0-STI7 (STI7: premises survive the night), PM1-PM3, U1-U7, necessity table. Predicted: STI NO-GO (STI7) and PM NO-GO." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    SHA=$(git rev-parse HEAD); echo $SHA

- A gate other than compute-idle / lane-starvation blocks either commit: STOP (NEEDS-CLAUDE 4). Do not bypass.
- Blocked only by compute-idle / lane-starvation: leave the files staged, retry later (do not reorder the commits).

    bash $P/tools/push_both.sh research/sleep-transitive-inference

## Step 5 -- provision the pool at the prereg commit

    cd $P && bash tools/pool_provision.sh --revision $SHA --isolated pool41 pool42; echo rc=$?

- rc=0: go to step 6.
- It refuses because the revision does not contain origin/main: in `$W` run `git fetch origin && git merge --no-edit
  origin/main`. No conflict: re-run the pytest line of step 3, push, set `SHA=$(git rev-parse HEAD)` and provision
  again. A conflict: `git merge --abort`, STOP (NEEDS-CLAUDE 5).
- Any other failure (a node unreachable, an install error): STOP (NEEDS-CLAUDE 5).

## Step 6 -- queue the 30 gate jobs (6 seeds x 5 parts)

    cd $P
    for S in 42 43 44 100 101 102; do
      for PART in "a:wake sleep yok wperm scr rev" "b:off dis nostdp" "c:nocomp noadapt" "x:cross" "n:nohub"; do
        TAG=${PART%%:*}; ARMS=${PART#*:}
        LINE=": mem_gb=2 && cd ~/derisk-pool/revisions/$SHA && mkdir -p $O/parts && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.sleep_transitive_inference --seed $S --arms $ARMS --out $O/parts/seed${S}_${TAG}.json"
        bash tools/pool_queue.sh add "$LINE" || echo "REFUSED: $S $TAG"
      done
    done
    bash tools/pool_queue.sh list | tail -5

(`tools/pool_queue.sh add` runs the shared shape check, `tools/queue_job_shape_check.sh`, on every line itself.) Record
the queued lines in the WORKLOG. Any "REFUSED" line: STOP (NEEDS-CLAUDE 6).

ETA: each part 0.5-1.7 h on one pool core (part b is the largest); 30 parts on pool41+pool42 at the dispatcher's
concurrency: about 3-6 h wall.

## Step 7 -- collect and score (the registered scorer, nothing else)

`tools/pool_sync.sh` (pool-sync.service) pulls finished results into `$P/$O/parts/`. When all 30 part files exist:

    cd $P
    ls $O/parts/seed*_*.json | wc -l          # must be 30
    for S in 42 43 44 100 101 102; do
      $PY -m research.runners.sleep_transitive_inference_grade --merge $O/seed$S.json $O/parts/seed${S}_a.json $O/parts/seed${S}_b.json $O/parts/seed${S}_c.json $O/parts/seed${S}_x.json $O/parts/seed${S}_n.json
    done
    $PY -m research.runners.sleep_transitive_inference_grade --family $O      # writes $O/aggregate.json

Re-run budget: a seed whose merged record reads UNDEFINED with ONLY "U4 run error" may have its failed part re-queued
once (same line, same SHA). Any other UNDEFINED reason: no re-run.

## Step 8 -- outcome -> verdict (write exactly this, in a finding named
`research/findings/2026-09-3X-sleep-transitive-inference-6seed-<VERDICT>.md`, citing `$O/aggregate.json` and the six
`$O/seed*.json`; then STOP and hand back for Claude's review)

| `aggregate.json` family | what it means | what to write (the verdict word is the scorer's, never another) |
|---|---|---|
| verdict NO-GO, and in every seed the only false STI gate is STI7 | the predicted result: the night writes a direct A->D link that replay, its structure and its order are necessary for, but it erodes the premises | "STI NO-GO 0/6, carried by STI7 as registered: offline replay writes the inference and trades premises for it"; list PM1-PM3; list the necessity labels; next methods = prereg section 10, STI7 bullet |
| verdict NO-GO, any other gate false on some seed | a registered failure | name each false gate per seed from `per_seed`; copy the matching prereg section-10 bullet as the next method |
| verdict GO | every STI gate on all six seeds | "STI GO 6/6"; still list PM; do not call it "consolidation" |
| verdict UNDEFINED | a U-rule fired after the re-run budget | name the U-rule per seed; U6 -> prereg section 10 "U6 fires" bullet; U1/U2/U3/U7 -> instrument fix needed (NEEDS-CLAUDE 7) |
| verdict INCOMPLETE | a seed record is missing | go back to step 7 |
| pm_verdict | read the same way over PM1-PM3 | report it on its own line; it never changes the STI word |

Also paste the `necessity` table (label, k of 6) for every mechanism. Terms: `docs/TERMS.md` (no "consolidation", no
"works" for a weight read).

## Stop conditions (all hand back to Claude)

Steps 1-6 as marked; any change you would have to make to the prereg, runner, grader or a constant; any gate you would
have to bypass; any pool node or GPU error you cannot clear with the command shown; the family result in step 8.

## NEEDS-CLAUDE list

1. A dev job crashed (step 1): read the log; fix at most once; any fix before registration is disclosed in section 2.
2. The operating-point check failed (step 2): decide between a registered change of constant (disclosed) and
   registering the failure; never tune toward DI.
3. A pre-commit check failed (step 3).
4. A gate other than compute-idle / lane-starvation blocked a commit (step 4).
5. Provisioning failed or a merge with origin/main conflicted (step 5).
6. `pool_queue.sh add` refused a line (step 6).
7. A family UNDEFINED from U1/U2/U3/U7, or anything in step 8 that does not match a row of the table.
8. Always, after step 8: the finding's review (verify-go), the board and summary-doc sync (`sync-documentation`,
   `vikunja`), and the FAILURE_LOG line + gate for the phantom `ou_noise_sigma_pa` config field (spawned task
   task_8f9171ae, if not yet done).

## RULINGS (Claude, 2026-09-29)

1. Run the registered gate even though the premise gate is predicted NO-GO. The deliverable is the measured cost
   of the night to the premises; the next registered method (replay that favours what matters and protects premises)
   follows under no-defer.
2. The Ecker CA3 OU-noise plumbing bug (runs documented at 40 pA ran at 100 pA) is a separate local-model packet: fix
   the field so the engine reads it, and correct the stated conditions of every affected finding (list them).
   Reviewed at the next Claude inspection. Not a blocker for this gate.
