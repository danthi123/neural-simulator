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

**THIRD CONVERGENT SIGNAL realized -> original necessity hypothesis
FALSIFIED (design 72e359a conclusion (b); propagated both remotes; the
pre-committed terminal path is now ACTIVE).** The distinct-pathways
design proved BEFORE any run that a biology-faithful architecture
cannot satisfy the original frozen `no_cls_replay`->`_HELPER_EP` duty
(episodic order = online trisynaptic store D.12/D.13; CLS consolidation
= order-INVARIANT neocortical schema, McClelland1995/Buzsaki2013/
validated Phase-1.3 -> removing consolidation collapses concept/WM not
episodic). Three faithful architectures, three directions, one
conclusion. Original frozen verdict NEVER edited (its VOID stands as
the honest record). Findings + capability (PREDICTED) propagated.

**Exact next concrete action: writing-plans for the corrected
approach.** FIRST work item = a NEW separately-pre-registered + frozen
catalog-grounded necessity module (SINGLE biologically-cited partition
correction: `no_cls_replay` -> working-memory/concept helper set, NOT
`_HELPER_EP`; numeric bars reused VERBATIM with original a-priori
justifications; instrument-validity-first; VOID strictly != FAIL;
created + frozen BEFORE any architecture). It is gated by a DEDICATED
ADVERSARIAL REVIEW whose PRIMARY mandate is goalpost-move detection
("legitimate catalog-derived correction or a rationalized repartition
to make the next candidate pass?" -- the latter BLOCKS the build). The
original frozen module is untouched; its VOID preserved permanently.
THEN the distinct-readout-pathways candidate is built + exercised
against the NEW module; its falsify-first MUST probe the FULL science
mode's wm AND ep JOINTLY at minimal load (the recorded process lesson);
reuse byte-unchanged; GPU/CuPy; no autograd.

**NEW PRE-COMMITTED bound (in force, stated in advance):** a faithful
build that reaches VOID/FAIL against the NEW catalog-grounded module is
an honest negative, propagated without spin; the next step is then the
next catalog-identified factorization -- autonomous, no hand-back, no
config-crank, NO further partition edits (one biologically-cited
correction only; a second would itself be goalpost-moving). No outcome
is rationalized.

DO NOT stop after any propagation/commit. The next concrete action
always begins in the same turn.

## Last durable commit

Design `72e359a` = distinct-readout-pathways design, conclusion (b):
the original necessity hypothesis is FALSIFIED (third convergent
signal). Findings + capability + this state file are the propagation
commit (both remotes). Runner remains at `2582992` (phase-factored,
closed VOID-by-construction) -- NOT the basis for the next build; the
next build is the distinct-pathways candidate exercised against a NEW
frozen catalog-grounded necessity module (writing-plans first work
item). Phase-factored line closed; superseded line =
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
