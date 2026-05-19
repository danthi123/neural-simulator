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

**Iteration 4 COMPLETE -> PRE-COMMITTED PROGRAM-LEVEL FINDING (honored,
propagated, NOT spun).** The project's OWN validated concept-binding
mechanism (embodied co-firing + topographic prior) was applied
faithfully in two forms; each recovered working-memory role-selectivity
but DESTROYED the loop's perfect episodic binding (ep 1.0->0.0); both
reverted, HEAD byte-restored. STRUCTURAL CAUSE (GPU-measured, not a
tuning gap): the validated concept-binding needs a SHUFFLED encode
order; the theta-gamma episodic store needs presentation-order ==
binding-index; in ONE online encode pass at N=2 these two validated
subsystems impose CONTRADICTORY encode-order requirements; no config
reconciles them. This REFUTES integrated-loop instrument-soundness AT
THIS MINIMAL SINGLE-PASS SLICE with the subsystems as currently
factored (does NOT refute the cheap-tier non-separability core nor the
hypothesis globally). Propagated: findings + capability (PREDICTED) +
both remotes. The pre-committed bound forbids a further config
iteration -- HONORED (no iteration 5; no tweak; no spin).

**NEXT PROGRAM STEP (genuinely-distinct architecture; NOT a config
iteration; its OWN proper design pass; autonomous, no hand-back, no
declare-unfit):** the encode-order contradiction is the signature of
the integration conflating two biologically-DISTINCT phases. Factor the
loop into (a) an ONLINE theta-ordered hippocampal episodic encode
(presentation-order == index, preserved) and (b) a SEPARATE OFFLINE
interleaved/shuffled-replay neocortical consolidation that builds the
concept selectivity -- reusing the project's VALIDATED multi-seed
strict-anti-cheat complementary-learning-systems consolidation
subsystem -- so each validated subsystem operates in the phase whose
order-requirement it satisfies. This is a distinct architectural
factorization, NOT a binding-rule tweak: it requires a PROPER design
pass (a fresh design -> writing-plans -> subagent-driven build with the
same adversarial + anti-cheat discipline + the SAME frozen acceptance:
`v1` wm AND ep >= 0.90 GPU; no-confab moat + protected + frozen
byte-unchanged; GPU/CuPy). It is deliberately NOT launched as a
reflexive same-loop patch subagent. Begin it at the design entry point
(brainstorm/writing-plans for the phase-factored consolidation
architecture), grounded in the catalog's complementary-learning-systems
factorization + the project's validated Phase-1.3 consolidation.

**PRE-COMMITTED bound for the phase-factored architecture (stated now):**
if a faithful phase-factored design (online theta-ordered encode +
offline shuffled-replay consolidation using the validated CLS
subsystem) ALSO cannot achieve `v1` wm AND ep >= 0.90 in the integrated
loop, that is a deeper program-level result to surface honestly with
its precise structural cause -- the next step then being the next
catalog-identified integration factorization, still autonomous, still
no hand-back. Stated in advance so no outcome is rationalized.

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Runner honest-wip `e02f692` (iter-3 foundation: temporal-credit +
homeostasis + documented non-zero init; episodic binding PERFECT
ep=1.0; verified-correct; iter-4 reverted so the runner is byte-exact
at this state). HEAD after the iter-4 program-level propagation commit
(findings + capability PREDICTED + this state file + plan, both
remotes). Iteration 4 produced the pre-committed PROGRAM-LEVEL finding
(validated concept-binding vs theta-gamma episodic store =
contradictory encode-order requirements at the N=2 single-pass slice);
honored (no iteration 5). Next program step = the phase-factored
consolidation architecture, via a PROPER design pass (NOT a reflexive
patch).

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
