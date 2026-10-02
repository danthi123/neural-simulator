# Cloud Claude sessions (credits until 2026-11-05)

The owner has $250 of Anthropic credits for cloud Claude sessions, expiring 2026-11-05. Spend them on the work only
Claude may do, so Hermes never waits on it: confirming a PASS, default flips, `sim/` code, preregs, bars, scorers,
seed lists, edits to existing findings, and writing well-specified work packages. Hermes does everything routine
(runbook section 2, RULINGs `routine-merges`, `results-without-pass`, `sync-branch`).

## What a cloud session can and cannot do

- It sees only GitHub (`danthi123/neural-simulator`): `main` (the server publishes it within minutes) and Hermes's
  `research/*` branches (published by the homelab publisher). It cannot reach the server, the pool, the GPU or
  Vikunja, and it has no corpora, knowledge bundles or `~/Projects/sim-catalog`.
- So it reads, reviews, writes code, specs and rulings, and runs small numpy smoke tests. Compute stays with Hermes:
  a cloud session queues nothing; it writes the job line into the packet or ruling and Hermes runs it.
- Gates: the cloud sandbox cannot run the gates that need server data. They run when the owner lands the branch
  (`tools/land_claude_branch.sh` merges through the full pre-commit suite). In the cloud, run the tests you touch and
  `python tools/check_docs.py`.

## Rules to paste at the start of every cloud session

```
You are the cloud Claude session for neural-simulator. Read CLAUDE.md and docs/CLOUD_CLAUDE_SESSIONS.md first.
The canonical repo lives on the owner's server; GitHub main mirrors it. Hermes (the server's driver agent)
does routine work; you do only what research/coordination/LOCAL_LLM_RUNBOOK.md section 2 reserves for Claude.
- Work on ONE branch, claude/<topic>, cut from origin/main. Never push main. Hermes's work is on origin/research/*.
- The live to-do list for you is the runbook's "## NEEDS CLAUDE" section plus docs/WEEKLY_CLAUDE_REVIEW.md.
- Rulings go into runbook section 6 as `- <KEY>: <ruling> (<date>)`. Do NOT delete NEEDS lines (Hermes appends
  there constantly, so deleting causes merge conflicts); add `- handled: [key1], [key2] (<date>)` and Hermes
  deletes them.
- Merging a research/* branch: `git merge origin/research/<x>` on your claude/<topic> branch, never on main.
- Faithfulness rules hold: six seeds for any generalization; the verify-go skill before any PASS; no host
  shortcuts; never --no-verify.
- Stop at about $60 of spend for this session. End with a REPORT: the branch, the FULL sha you pushed, what it
  lands, what Hermes should do next, and anything the owner must decide.
```

## The sessions, in order

1. **Backlog clearance (first, as soon as the publisher is ready).** Opus. Confirm or refuse the integration-wave2
   PASSes with verify-go (Hermes staged D/S/R verdicts, all PASS, flip list of three) and, if they hold, flip those
   three defaults: this is the unit of progress under integration-first. Then merge `research/draft-ecker-ou-field`
   (T15), rule on T3 sleep-TI (`research/sleep-transitive-inference`), T17 phantom-cfg, the test-suite triage, and
   write the LDC Step 4b prereg once the 24 pilot cells are in.
2. **Packet factory (before 10-10).** Opus. Write 4-6 complete work packages toward the Stage 1 exit (10-24): each a
   `docs/drafts/<packet>/HANDOFF.md`, a registered prereg with its bar, a scorer with tests, the seed list and the
   exact pool/GPU job lines, so Hermes can run each one start to finish with nothing reserved left open. Use
   `GAP_CLOSURE_MISSION.md` CURRENT STATE and the master roadmap for the order.
3. **Weekly reviews (10-08, 10-15, 10-22, 10-29, 11-05).** Sonnet unless a PASS or a design call needs Opus. Follow
   `docs/WEEKLY_CLAUDE_REVIEW.md`, then sync the summary docs (sync-documentation skill, Check I).
4. **Stage 1 exit attempt (10-24).** Opus. Review the attempt Hermes ran and rule on it.
5. **GPU work (10-02, session 4, done).** The pool is the bottleneck and the 3090 was idle: an audit found no
   queued pool task can move to cupy (every one pins numpy), so two GPU packets were written instead (runbook T23
   B2d mouth arm, T24 Stage-1 `lived` on cupy) plus `gpu_remote.sh add --rev`.

## Landing a cloud branch on the server (the owner, during a check-in)

The cloud session's REPORT gives the branch and the full sha. On the workstation run:

```bash
ssh homelab-server 'docker exec -u 99 -e HOME=/opt/data/home -e SIM_CLAUDE_REVIEW=1 -w /home/dant123/Projects/neural-simulator hermes bash tools/land_claude_branch.sh claude/<topic> <sha>'
```

It refuses unless the branch is exactly that sha, prints what it brings in, and merges through every gate. A
conflict or a failed gate leaves main unchanged; paste its message back to the cloud session, which merges the
latest `origin/main` into its branch, fixes the cause and pushes again.

## One-time setup (homelab publisher)

- Publish the server's `research/*` branches to GitHub (and Gitea) alongside `main`.
- Mirror GitHub's `claude/*` branches into a bare repo the hermes container can read, at
  `/opt/host-status/claude-import.git` (or pass `-e CLAUDE_IMPORT_REPO=<path>` in the command above).
