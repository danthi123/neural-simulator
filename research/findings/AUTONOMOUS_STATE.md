# AUTONOMOUS CONTINUATION STATE

> Durable cross-session pointer. Any re-trigger (scheduled watchdog, new
> session, post-compaction) reads THIS first and resumes the exact next
> action without re-deriving context. Update every cycle; commit+push
> both remotes. The conversation is NOT the memory — this file + git are.

**Updated:** 2026-05-19
**Mode:** continuous autonomous (24/7; no self-imposed stopping; only an
explicit user stop/pause or a true safety boundary halts work)

## Current objective

Build the catalog-grounded integrated-loop full spiking model to
instrument-soundness, then the decisive multi-seed lesion run, then the
staged compositional sequence — iterating following the project's
reference biology, no hand-back, no declare-unfit.

## Exact next concrete action

Background subagent `a4d4c9be7d13ca743` is running **pre-registered
biology-fidelity iteration 2**: compose the project's validated
homeostasis (per-stripe equalization) + validated temporal-credit /
dopamine-gated eligibility (cold-start break for the
`dlpfc_verb -> noun_pool` efferent) into `research/runners/integrated_loop_gate.py`,
re-tested on GPU against the SAME frozen gate (`v1` wm AND ep >= 0.90;
every drilled binding clears the byte-unchanged no-confab gate;
`full`+lesions novel probe byte-identical). Strengthen-only;
protected/frozen/moat byte-unchanged.

**On its completion notification:** controller trust-but-verify the diff
(commit scope = only the runner; protected set byte-empty `bd27292..HEAD`;
`integrated_loop_core.py` unchanged since `2048750`; no autograd; tests
25/25; moat 7/7), then:
- If acceptance met + committed -> dedicated adversarial re-review of the
  runner (Probe 8 BG-causal still intact; no query-time hard-feed
  introduced by the eligibility/reward wiring; lesions still faithful;
  homeostasis scoped not global-cheating) -> on CLEAR: no-harm phase ->
  Task 5 CONTROLLER-ONLY decisive multi-seed GPU run + mandatory
  anti-cheat smell-test + honest propagation both remotes.
- If faithful negative -> propagate honestly (findings + capability
  pillar stays PREDICTED + both remotes), then immediately begin the
  next cited biology-fidelity iteration (the subagent names it).

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

`718213a` (HEAD): durable state file. Runner committed state `6c2c055`
(logic == `d3a7ac3` + inert `--selfcheck-diag` harness). Iteration-2
background subagent `a4d4c9be7d13ca743` is IN FLIGHT (composing validated
homeostasis + temporal-credit into the runner on the local GPU); it
will commit only if `v1` wm AND ep >= 0.90 on GPU, else report a
faithful negative. On its completion: controller trust-but-verify ->
adversarial re-review -> no-harm -> Task 5 decisive run, or honest
propagation + next cited iteration.

## Pre-registered acceptance / frozen bars (NEVER tuned)

`integrated_loop_core.py` `_IL_*`: V1_MIN 0.90, SCI_MIN 0.80,
LESION_MAX 0.40, SCALE_TOL 0.10, ladder (2,4,8), MIN_SEEDS 3. No-confab
moat `research/runners/abstention_gate.py` + test 7/7 byte-identical.
GPU (CuPy) for every real/decisive run; numpy only for `--tiny-synth`.

## Continuation guarantee (TWO watchdogs — installed)

1. **LOCAL, GPU-capable** — Windows Scheduled Task `SimAutonomousWatchdog`
   runs `scripts/autonomous_watchdog.ps1` every 20 min. Conservative
   stall-gate: fires ONLY if no git commit for >40 min AND no active
   claude/python-sim process AND no fresh `.watchdog.lock`. On stall it
   re-invokes local `claude.exe -p` (bypassPermissions, `--add-dir` repo)
   with a prompt to read THIS file and continue the exact next action
   INCLUDING PENDING-LOCAL-GPU steps. Audit log:
   `research/findings/raw/autonomous_watchdog.log`. This is the primary
   guarantee for GPU-bound work. (Re-verify: `schtasks /query /tn
   SimAutonomousWatchdog`; re-register via `scripts/autonomous_watchdog.ps1`
   contract if missing.)
2. **REMOTE, cross-session non-GPU safety-net** — claude.ai routine
   `trig_01W7vwnpv4JYWUMjzwHaEKK6` (hourly, `0 * * * *`). Runs on a
   GitHub checkout with NO local/GPU access; advances only non-GPU
   pre-registered work (designs/plans/findings/propagation/state-file),
   pushes both remotes, marks GPU steps `PENDING-LOCAL-GPU`. Manage:
   https://claude.ai/code/routines/trig_01W7vwnpv4JYWUMjzwHaEKK6

If either watchdog is missing, RE-CREATE it before other work. Neither
watchdog nor the in-session discipline may stop on a promise: the
next-action tool call is always in the same turn.
