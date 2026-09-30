# HANDOFF: B2b seed-102 ruling + scoring, B2d prereg (for the local model)

Worktree `/home/dant123/Projects/sim/.claude/worktrees/b2-rulings`, branch `research/b2-rulings-2026-09-29` (from
main f52c6d246). Written 2026-09-29 by Claude. Execute top to bottom. Do not use judgment beyond the tables: any row
marked NEEDS-CLAUDE means stop and write one line under "NEEDS CLAUDE" in `research/coordination/LOCAL_LLM_RUNBOOK.md`.

Paths: `P=/home/dant123/Projects/sim` (primary checkout; shards land here), `W=$P/.claude/worktrees/b2-rulings`.
Run every command below with bash (the Bash tool is zsh: wrap loops in `bash -c`).

## 0. What is already done (do not redo)

- Ruling written: `research/coordination/b2b0924_s102_ruling.md` (text dates from 12:21 EDT 2026-09-29, before
  any re-run result). The 13:15/13:16 EDT 09-25 dispatches were attempt 1 (the 00:54 ones were B2a); attempt 1 is
  class E (results stranded on AWS pool1/pool2, stopped before sync). Reading B chosen (spend the one re-run).
- A1.4(a) local move done 12:14:49 EDT: attempt 1 is in `$P/research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s102/`.
- A1.4(b) re-runs, direct ssh, launched 12:15:07 (pool41, discourse-register) and 12:17:46 (pool42,
  episodic-memory). discourse-register finished rc 0 at 13:12, DEFINED, regressed (load-bearing). Its three arm
  files are equal to attempt 1's. episodic-memory: see 1-3 below.
- Checker: `tools/b2b_score.py` (+ `tests/test_b2b_score.py`, 11 pass). It re-derives A1.2 rules 1-5, A1.4 hosts,
  A1.5 states and verdict, the A1.9 B2a comparison, multiref.error counts, R3, and a per-cell sha256 manifest,
  without importing lb_shard, and cross-checks lb_shard's aggregate number by number.
- Staged, NOT committed (gates `compute_idle_persistent` and `lane_starvation` block every commit until compute is
  busy or the waiver budget frees ~18:30): the ruling, `b2b0924_reruns.tsv`, `b2b0924_reruns_commands.txt`,
  `tools/b2b_score.py`, `tests/test_b2b_score.py`, this file.
- B2c is CUT (coordinator ruling 2026-09-29). Branch `research/b2c-paired-flip-prereg-fixround1` left unchanged at
  09f6a1b94. Uncommitted fix-round-2 work kept in `git stash` ("B2c fix round 2 ...") and in
  `$P/.claude/worktrees/_b2c_fixround2_uncommitted_2026-09-29.patch`. Do nothing with it.

## 1-3. DONE 2026-09-29 14:10 EDT: both re-runs landed, B2b scored COMPLETE

episodic-memory finished rc 0 at 14:07:59 on pool42 (DEFINED, regressed). `tools/b2b_score.py` reads **COMPLETE**:
216/216 coverable cells DEFINED, 0 FAIL cells, robust core 28, union 29, mean 0.843, SD 0.014, no precondition
failed, and it agrees number by number with `lb_shard.py aggregate --pin F`. The TSV rows and the ruling's
"Outcome" section are filled in. Output: `research/findings/raw/_load_bearing/_b2b0924_score/b2b_score.json`;
registered aggregate copied to `research/findings/raw/_load_bearing/_shards/b2b0924-base/aggregate.json`.

To re-derive at any time (read-only; it must print `"verdict": "COMPLETE"` and `"preconditions_failed": []`):

```bash
cd $P && .venv/bin/python $W/tools/b2b_score.py --root $P --coord-root $W \
    --registered-aggregate research/findings/raw/_load_bearing/_shards/b2b0924-base/aggregate.json
```

| re-derivation prints | do |
|---|---|
| COMPLETE, no failed precondition | nothing |
| anything else | STOP. NEEDS-CLAUDE: "B2b re-derivation changed: <summary>" (a shard file changed on disk) |

## 4. Commit and push (only when gates allow; never --no-verify, never write a waiver)

```bash
cd $W && git add HANDOFF.md research/FAILURE_LOG.md tools/b2b_score.py tests/test_b2b_score.py research/coordination/b2b0924_s102_ruling.md \
  research/coordination/b2b0924_reruns.tsv research/coordination/b2b0924_reruns_commands.txt \
  research/findings/raw/_load_bearing/_b2b0924_score/b2b_score.json research/findings/raw/_load_bearing/_shards/b2b0924-base/aggregate.json
git commit -m "ops(B2b): seed-102 ruling (text written 12:21 EDT 2026-09-29, before any re-run result) carried out -- attempt 1 class E, one A1.4(b) re-run each on pool41/pool42; committed scorer tools/b2b_score.py; B2b COMPLETE (216/216 DEFINED; robust core 28, union 29, mean 0.843)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
bash tools/push_both.sh research/b2-rulings-2026-09-29
```

| commit result | do |
|---|---|
| succeeds, push verified | done with B2b; tell the owner the branch is ready to merge |
| blocked by `compute-idle-persistent` or `lane-starvation` | wait (budget frees about 18:30, or when B2d's jobs run); retry. Do not waive. |
| blocked by any other gate | STOP. NEEDS-CLAUDE with the gate's message |

## 5. Before AWS pool1 or pool2 is ever started again

Run the bottom block of `research/coordination/b2b0924_reruns_commands.txt` (and Steps 1a/1b above it) first. Do
not start pool1/pool2 just for this. NEEDS-CLAUDE (owner decision): start pool1/pool2 briefly to retrieve attempt
1's stranded rep0 + lb.json (a determinism cross-check, never scored), or terminate them.

## 6. B2d (prereg drafted; NOT queue-ready without Claude)

Worktree `$P/.claude/worktrees/b2d-prereg`, branch `research/b2d-prereg-2026-09-29` (from main 3c2907b85). Pin
H = `3c2907b85c10fe918ddc81b4c69de8888c9f321b`, tag `b2d0929`, 258 lines (same rows as B2b). Staged, not committed:
the prereg `research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md`,
`research/coordination/b2d_jobs.txt`, `research/coordination/b2d_decision_fields.json`,
`research/coordination/b2d_corpus_sha256.tsv`, `research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt`.
The exact commands with stop conditions are the prereg's last section, "Prepared commands", steps 0-9.

| prereg step | status | who |
|---|---|---|
| 0 job file + PIN | DONE (staged) | - |
| 1 re-check the self-describing-field calls (pragmatic-implicature first) | OPEN | NEEDS-CLAUDE |
| 2 commit + push the prereg | when gates allow | local model: `cd $P/.claude/worktrees/b2d-prereg && git add <the 5 staged files> && git commit` with the message below, then `bash tools/push_both.sh research/b2d-prereg-2026-09-29` |
| 3 provision H on pool41/pool42 (`--isolated`) | DONE 14:21 EDT 2026-09-29 | - |
| 4 corpus hash per node | DONE 14:22 (both 7a00272e..., matches primary; staged) | - |
| 5 build `tools/b2d_score.py` + identity replay on B2b | OPEN | NEEDS-CLAUDE |
| 6 queue waves (script, or the one-wave fallback in the prereg) | BLOCKED until steps 1, 2 and 5 are done | local model, exactly as the prereg's step 6 |
| 7-9 re-runs, aggregate, score | after the waves | local model, per the prereg |

B2d commit message (step 2):

```
prereg(B2d): full-registry load-bearing battery at H 3c2907b85 -- self-describing lesion fields (8 rows) excluded by a post-hoc arm-file scorer (S1, headline), reply-level secondary (S2), raw (S0) beside; comprehension-monitor lesion arm found crashing (B2b 6/6 is an organ error); job file + PIN + corpus hashes; nothing queued

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

Stop conditions for the local model on B2d: never queue before step 5's scorer is committed; never set
POOL_PROVISION_ALLOW_STALE; if a b2d0929 line fails `queue_job_shape_check`, STOP (NEEDS-CLAUDE).
Memory: one LB job per 15 GB node (an episodic-memory worker used 6.9 GB RSS), so pool-only wall time is about
50 h; AWS r7i.4xlarge x2 (about $1/h each) cuts it to about 15-17 h (owner cap about $50/day).

## NEEDS CLAUDE (open, for the owner or Claude; do not act on these)

1. (Informational) The Reading A / Reading B call in the ruling (chosen: B). Both re-runs are DEFINED and equal to attempt 1's landed arms, so it did not change the result.
2. pool1/pool2 AWS instances are `stopped`, not terminated, and still hold the stranded attempt-1 files and the
   pending node-side moves: retrieve-then-terminate, or leave.
3. No preregistered, ready-to-run 6-seed pool or GPU work exists besides B2d (survey 2026-09-29). B2d's own order
   puts the scorer (step 5) and the field re-check (step 1) before any queueing, so the pool stays idle until Claude
   does those, or the owner amends B2d to allow queueing first (scorer committed before any b2d0929 lb.json is read).
   The GPU half of the compute gate has no candidate at all.
5. B2b comprehension-monitor: 6/6 load-bearing is an organ crash in the lesion arm (see the ruling's Outcome and
   research/FAILURE_LOG.md 2026-09-29). Fix the organ (shape 7002 vs 1032) before B2d runs, or B2d scores that row
   UNDEFINED.
4. `lb_shard.py aggregate` reports `ltm_mode: off`, but every B2b intact_a arm file reads turn source
   `tiny-demo +LTM` (208/208, all four hosts). The recorder is blind (it keys on BRAIN_DATA_ROOT). Report B2b as
   run with the sim-data LTM bundle; fix the recorder before B2d is scored.
