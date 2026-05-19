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

**Program-level finding (iter-4) propagated + bound honored (849218f).
Phase-factored DESIGN (65f6c52) + TDD PLAN (412ccef) done + pushed.
FALSIFY-FIRST DE-RISK = GREEN (committed honest-wip `3300d3f`):** with
the offline shuffled-replay Phase-1.3 consolidation inserted on top of
the e02f692 foundation, GPU-verified (CuPy/RTX 3090) `v1 ep = 1.0` is
PRESERVED -- the relocated-contradiction risk did NOT materialize; the
phase factorization genuinely dissolves the iter-4 encode-order
contradiction at the v1 check; drilled wm queries role-correlated and
clear the byte-unchanged 650 gate (671-1591). NOT a soundness pass
(v1 wm not yet >=0.90 -- the genuine science question for the
controller-only decisive run).

**Exact next concrete action: dispatch the FULL Tasks 2-5 build
(subagent-driven) on top of `3300d3f`.** Build the full phase-factored
runner (online theta-ordered encode byte-unchanged + offline
shuffled-replay Phase-1.3 consolidation + consolidated WM/episodic-order
readouts; v1/full/8-lesions; SAME frozen gate `v1` wm AND ep >= 0.90;
full+lesions novel probe byte-identical) -> Task 3 DEDICATED
ADVERSARIAL REVIEW with MANDATORY primary scrutiny: (i) does
`no_cls_replay` GENUINELY collapse ep at full scale (its frozen
`_HELPER_EP` responsibility) or is it inverted -> then the frozen
verdict correctly VOIDs, surfaced as the honest program-level outcome
(NEVER edit the frozen verdict/partition/bars); (ii) is the
consolidated episodic-order readout a genuine spiking measurement, NOT
a strawman/hard-feed; (iii) lesions faithful (each = full minus exactly
one, identical per-trial RNG), validated subsystems reused
byte-unchanged, no autograd, WM selectivity learned-by-consolidation
not pre-wired -> Task 4 no-harm -> Task 5 CONTROLLER-ONLY decisive
multi-seed GPU run + anti-cheat smell-test + honest propagation both
remotes. GPU/CuPy real path (numpy only --tiny-synth); reuse-only
(net-new = phase controller + wiring).

**PRE-COMMITTED bound (in force):** if the faithful full phase-factored
build cannot achieve `v1` wm AND ep >= 0.90, OR a frozen lesion's
pre-registered responsibility is inverted (the frozen verdict correctly
returns VOID), that is surfaced honestly with its precise structural
cause as the next program-level result -> the next catalog-identified
integration factorization, autonomous, no hand-back, no config-crank,
no edit of the frozen verdict or no-confab moat. Stated in advance so
no outcome is rationalized.

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Runner honest-wip `3300d3f` = phase-factored falsify-first DE-RISK
GREEN (online theta-ordered encode byte-unchanged from e02f692 +
offline shuffled-replay Phase-1.3 consolidation + consolidated
WM/episodic-order readouts; GPU-verified v1 ep=1.0 preserved; NOT a
soundness pass). Built on iter-3 foundation e02f692 (temporal-credit +
homeostasis + documented non-zero init). Next = the full Tasks 2-5
build on top of `3300d3f`. HEAD after the iter-4 program-level
propagation commit (findings + capability PREDICTED + state file +
plan, both
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
