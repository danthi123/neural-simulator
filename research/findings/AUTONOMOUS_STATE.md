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

**Phase-factored architecture = VOID-BY-CONSTRUCTION (GPU-verified;
frozen verdict recomputed unchanged -> GATE=VOID; propagated; bound
honored).** Full build + adversarial review done faithfully. GPU
(CuPy/RTX3090, real N=2): `full` ep=0.0 (offline shuffled replay
DESTROYS episodic order) while `no_cls_replay` ep=1.0 (skip
consolidation -> online recall preserves order, does NOT collapse) ->
frozen `_HELPER_EP` duty of `no_cls_replay` INVERTED -> unchanged
frozen verdict = VOID (controller recomputed independently; integrity
clean; 28 tests pass). TWICE-CONVERGENT program-level reading: iter-4
(single-pass contradictory) + phase-factored (separating RELOCATES not
dissolves it) -> the evidence now points at the PRE-REGISTERED
NECESSITY PARTITION ITSELF as the likely falsified element (it encoded
a single-pass-conception necessity hypothesis biology-faithful
architectures do not realize; the frozen verdict correctly refuses to
certify rather than be edited). Propagated: findings + capability
(PREDICTED) + both remotes. Process lesson recorded: falsify-first
must probe the FULL mode jointly, not v1 alone.

**Exact next concrete action: proper DESIGN pass for the next
catalog-identified factorization (autonomous, no hand-back, no
config-crank, NOT a reflexive patch).** Deeper separation of relational
episodic binding from schema/concept abstraction along the
hippocampal-neocortical interaction: the episodic-order readout served
by the order-PRESERVING trisynaptic pattern-completion pathway
(catalog D.12/D.13) and the concept/WM readout by the order-INVARIANT
neocortical schema pathway -- genuinely DISTINCT readout pathways so
the necessity structure is examined against the catalog's documented
interaction rather than assumed. Proper design -> writing-plans ->
subagent build; its pre-registered falsify-first MUST probe the FULL
mode's wm AND ep jointly (the recorded process lesson), SAME frozen
gate, reuse byte-unchanged, GPU/CuPy, no autograd.

**PRE-COMMITTED TERMINAL bound (in force, stated in advance):** if the
next factorization ALSO reaches VOID by the same category mismatch
between the pre-registered necessity partition and biology-faithful
causal structure, the honest TERMINAL scientific conclusion is that the
pre-registered NECESSITY HYPOTHESIS -- not the local implementation --
is the refuted element; surface that as the program's honest result and
follow it with a catalog-grounded RE-DERIVATION of the necessity
hypothesis itself, still autonomous, still no hand-back, still no
frozen-verdict/moat edit. No outcome is rationalized.

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Runner `2582992` = full phase-factored build (faithful;
VOID-by-construction: full ep=0.0 / no_cls_replay ep=1.0 -> frozen
_HELPER_EP inverted -> unchanged frozen verdict = VOID). Plus `07eab87`
test pins. Built on the de-risk GREEN `3300d3f` / iter-3 foundation
e02f692. The phase-factored line is closed VOID-by-construction; next =
the next catalog-identified factorization (distinct order-preserving vs
order-invariant readout pathways) via a proper design pass. Built on
iter-3 foundation e02f692 (temporal-credit +
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
