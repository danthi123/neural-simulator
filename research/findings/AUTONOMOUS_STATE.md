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

**PENDING-LOCAL-GPU: iteration 4 (the fundamentally-different,
project-validated binding approach the hard bound EXPLICITLY permits;
background subagent on the RTX 3090).** Iters 1-3 localized the
instrument-soundness gap precisely: episodic binding is now PERFECT
(ep=1.0) and the documented zero-init gotcha is resolved, but
role-selective WORKING-MEMORY binding does not form under a global
scalar three-factor (temporal-credit) signal -- triangulating with the
project's own 2026-05-05 verdict ("global scalar feedback fails at
biological scale; the credit-assignment RULE is the bottleneck, the
architecture is sufficient"). Its documented RESOLUTION is the
embodied-Hebbian co-firing + topographic-prior binding paradigm
(Tier-1 6x; v16 88.75% multi-seed -- the design Section-3 concept-layer
substrate). Iteration 4: carry the working-memory role-selectivity with
that VALIDATED co-firing+topographic mechanism (reused byte-unchanged
where it exists; net-new = only the in-loop wiring), with
temporal-credit relegated to credit/gating ONLY, KEEPING every faithful
part already committed in `e02f692`: Probe-8 `thal->dlpfc->noun_pool`
wiring, LEVER-2 homeostasis (working), the documented non-zero init,
perfect episodic binding. SAME frozen gate (`v1` wm AND ep >= 0.90 GPU;
every drilled binding clears the byte-unchanged no-confab gate;
`full`+lesions novel probe byte-identical). Strengthen-only;
protected/frozen/moat byte-unchanged; GPU/CuPy (numpy only for
`--tiny-synth`). Anti-hard-feed control: selectivity must be LEARNED by
co-firing, not pre-wired. Commit only if acceptance met; else honest
faithful-negative.

**PRE-COMMITTED iteration-4 bound (stated before the run):** if the
project's OWN validated co-firing+topographic binding mechanism ALSO
fails to produce role-selective working-memory binding in this
integrated loop at N=2, that is a genuine PROGRAM-LEVEL refutation of
integrated-loop instrument-soundness at this slice -- surfaced honestly
as a fundamental program decision, NOT a further iteration. Stated in
advance so the outcome cannot be rationalized.

**On the iteration-4 completion notification:** controller
trust-but-verify the diff (commit scope = only the runner; protected
byte-empty; `integrated_loop_core.py` unchanged since `2048750`; no
autograd; 25/25; moat 7/7), then:
- acceptance met + committed -> dedicated adversarial re-review (Probe-8
  BG-causal intact; co-firing selectivity LEARNED not pre-wired/hard-fed;
  lesions faithful; homeostasis scoped not global-cheating; non-zero
  init not leaking the answer) -> on CLEAR: no-harm phase -> Task 5
  CONTROLLER-ONLY decisive multi-seed GPU run + anti-cheat smell-test +
  honest propagation both remotes -> staged compositional sequence.
- faithful negative -> propagate honestly (findings + capability stays
  PREDICTED + both remotes) + surface the PRE-COMMITTED program-level
  refutation finding honestly (no further config/iteration spin).

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Runner honest-wip `e02f692` = iteration 3: documented non-zero
binding-agnostic init on the net-new `dlpfc_verb->noun_pool_F*`
efferent ON TOP OF iter-2's LEVER-1 (temporal-credit) + LEVER-2
(homeostasis). Result: zero-init gotcha RESOLVED -- episodic binding
PERFECT (ep=1.0), scores clear the 650 no-confab gate -- but
working-memory role-selectivity does NOT form under global-scalar
temporal-credit (wm=0.0); committed honest-wip, NOT a pass; hard bound
hit -> deeper-architecture finding. Iteration-3 findings + capability
(PREDICTED) + this state file are the propagation commit (both
remotes). Iteration 4 (the fundamentally-different validated
co-firing+topographic binding for the working-memory dimension, on top
of `e02f692`) is the in-flight next action.

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
