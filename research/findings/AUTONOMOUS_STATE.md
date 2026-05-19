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

**PENDING-LOCAL-GPU: iteration 3 (background subagent running locally on
the RTX 3090).** Apply the documented project fix: a small NON-ZERO
prior init (`weight_mean ~= 0.5`, `weight_jitter ~= 0.3`) on the net-new
`dlpfc_verb -> noun_pool_F*` RegionPathway the runner already adds, so
the (now-correctly-wired) temporal-credit reward can actually charge
STDP eligibility (a zero-init synapse carries no current -> the
documented CLAUDE.md zero-init gotcha; Barlow 1972 spontaneous baseline
weights). KEEP both already-faithful levers: LEVER 1 temporal-credit
(committed `5c27e99`, encode-only, validated idiom byte-unchanged) and
LEVER 2 homeostasis (committed `5c27e99`, verified working). SAME frozen
gate (`v1` wm AND ep >= 0.90 on GPU; every drilled binding clears the
byte-unchanged no-confab gate; `full`+lesions novel probe
byte-identical). Strengthen-only; protected/frozen/moat byte-unchanged;
GPU/CuPy (numpy only for `--tiny-synth`). Commit only if acceptance met;
else honest faithful-negative report.

**On the iteration-3 completion notification:** controller
trust-but-verify the diff (commit scope = only the runner; protected
byte-empty; `integrated_loop_core.py` unchanged since `2048750`; no
autograd; 25/25; moat 7/7), then:
- acceptance met + committed -> dedicated adversarial re-review (Probe-8
  BG-causal intact; no query-time hard-feed from the eligibility/reward
  or the non-zero init; lesions faithful; homeostasis scoped not
  global-cheating; the non-zero init does not itself leak the answer)
  -> on CLEAR: no-harm phase -> Task 5 CONTROLLER-ONLY decisive
  multi-seed GPU run + anti-cheat smell-test + honest propagation both
  remotes -> staged compositional sequence.
- faithful negative -> propagate honestly (findings + capability stays
  PREDICTED + both remotes); honest bound now in force: if soundness
  still fails AFTER the documented non-zero init on top of the two
  now-faithful mechanisms, this is a DEEPER ARCHITECTURE question to
  surface (a fundamentally different approach / explicit owner-facing
  architecture decision), NOT another config iteration.

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Runner honest-wip `5c27e99` = iteration 2: LEVER 1 (temporal-credit,
encode-only, validated idiom byte-unchanged) + LEVER 2 (homeostasis,
verified working) composed; acceptance NOT met (zero-init precondition
blocks LEVER 1 eligibility) so committed as honest wip, NOT a pass.
Iteration-2 findings + capability (PREDICTED) + this state file are the
next propagation commit (both remotes). Iteration 3 (documented
non-zero-init fix on top of `5c27e99`) is the in-flight next action.

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
