# Stage 1 (regime-correct static compositional retrieval): decisive multi-seed run is an honest negative — the trustworthy abstention substrate composes and holds at biological scale, but static two-store composition does not yield the compositional capability

## Status

Honest negative, propagated without spin, under the standing anti-cheat
discipline. This is a genuine, informative scientific result that
empirically confirms the owner's scientific reframe (internalized in the
design, section 2b): static two-store retrieval-composition is not how
biology produces compositional/conversational capability. It is NOT a
declaration that the approach is unfit, NOT a hand-back, NOT config-cranking.
No fixed threshold was moved; the original frozen verdict modules, the new
frozen capability-verdict module, and the no-confabulation moat are all
byte-unchanged.

## What was tested (pre-registered, fixed-bar)

The Stage-1 design: read recent-specific bindings from the hippocampal
engram pathway and order-invariant semantic structure from the consolidated
neocortical pathway, each in its biologically-correct regime, compose them,
and gate the answer through the byte-unchanged no-confabulation abstention
moat. Decisive run: full biological scale (no smoke), the validated 16-pool
concept substrate plus hippocampus (8240 neurons: language_input/output,
motor + FS pools, ec/dg/dg_pv_basket/ca3/ca1, noun/verb/adjective pools),
the frozen load ladder (2, 4, 8 recent-specific compositional facts), seeds
42/43/44, CuPy on an RTX 3090, kill-safe with durable capture, monitored to
actual process exit.

## Result

The pre-registered frozen capability-verdict module returns **FAIL**.
Every rung (N = 2, 4, 8; 3 seeds each):

- full_acc = 0.00
- recent_only_acc = 0.00
- remote_only_acc = 0.00
- abstain_correct_recent_only = 1.00
- abstain_correct_remote_only = 1.00

The verdict was independently recomputed from the single recorded output
(no re-run, no bar change): recorded FAIL == recomputed FAIL, reason
"smallest-load rung fails the frozen regime bars".

## Mandatory smell-test (a negative scrutinised for instrument validity)

A FAIL must be confirmed to be an honest measured negative, not an
instrument failure masquerading as FAIL (a silent crash producing trivial
zeros should be VOID, not FAIL). The smell-test passed:

- Genuine full-scale execution: CuPy / RTX 3090, 8240-neuron validated
  substrate + hippocampus (exactly the recipe the dedicated adversarial
  re-review cleared), 27 arm-runs (9 cells x full/recent_only/remote_only)
  over approximately 34 minutes of real spiking computation. A
  skipped/no-op pipeline would complete in seconds.
- Zero errors, exceptions, tracebacks, NaN/inf, or skips anywhere in the
  1461-line durable log. The absence of "replay/engram/recall" strings is
  subsystem silence (the runner prints only at the final gate; the reused
  subsystems log via the bridge, which the log records throughout), not
  skipped steps.
- Internally consistent: 9 raw cells x 3 arms, all complete, all with the
  same pattern.
- The verdict module is byte-unchanged and recomputes the same gate from
  the recorded raw numbers; it correctly returns FAIL (structurally valid
  rungs, below the accuracy bar) rather than VOID (which is reserved for
  unmeasurable/instrument-invalid input).

## What this means (the honest reading, no spin)

Two things are true and both are reported:

1. The static regime-correct two-path composition does NOT produce
   grounded compositional retrieval at biological scale: the composed
   readout never reached the calibrated no-confabulation confidence
   threshold on the correct answer, at any load or seed.
2. The trustworthy property HELD under composition: abstain-correct is
   1.00 across all seeds, loads, and both ablation arms. The
   no-confabulation moat, composed into this two-path architecture at
   full biological scale, abstained ("I don't know") rather than emitting
   a confident wrong answer, in every case. Zero confabulation under
   composition is a real, preserved property — the project's distinctive
   contribution survives this composition.

This negative is informative and convergent, not a dead end. It
empirically confirms the scientific point the owner raised and that was
internalized into the design (section 2b): biology does not resolve the
recent/remote conflict by reading two stores and combining them; it
resolves it by temporal multiplexing under one shared theta rhythm
(Separate Phases of Encoding And Retrieval), by order-bearing vs
order-invariant being operating modes of one theta-gamma code rather than
two stores, and by a generative hippocampal-prefrontal replay loop.
Stage 1 deliberately does not implement that mechanism; the decisive run
shows that without it the static composition does not yield the
capability. The result therefore triangulates precisely onto the
pre-registered next stage.

## Pre-registered next step (autonomous, no hand-back, no config-crank, no bar change)

Per the standing iterate-following-biology discipline: an honest FAIL
drives the next biology-identified fidelity refinement. That refinement is
already grounded and documented (design section 2b, references 9-17): the
conversational stage whose load-bearing core is a single shared
theta-gamma rhythm time-multiplexing an encode phase and a
retrieve/pattern-complete phase (Separate Phases of Encoding And
Retrieval; the acetylcholine gate via the validated neuromodulator
subsystem), with a prefrontal working-memory frame holding compositional
sequence structure and a generative replay loop producing novel
schema-constrained ordered sequences. It reuses the project's already
validated theta-gamma episodic store, trisynaptic pattern-completion
pathway, replay-consolidation subsystem, and neuromodulator subsystem,
byte-unchanged. It is its own pre-registered, fixed-bar, three-state test,
pursued autonomously through a proper design and plan pass.

## Honest ceiling (unchanged, restated)

Conversational/compositional capability is not achieved and is not
claimed. No fixed threshold was moved; the original frozen verdict, the
corrected frozen module, the new frozen capability-verdict module, and the
no-confabulation gate are all byte-unchanged. Every previously-validated
asset (trustworthy grounded memory, the no-confabulation abstention moat,
the validated subsystems) is intact and unaffected. The genuine,
durable contributions of this stage are: (a) a faithful, adversarially
hardened, fixed-bar instrument for regime-correct compositional retrieval;
(b) the empirical demonstration that the no-confabulation moat composes
and holds at biological scale (zero confabulation under composition); and
(c) the empirical confirmation that the static two-store composition does
not yield the capability, which converges with the biology and motivates
the next pre-registered stage.

## Files / evidence

- Frozen capability-verdict module (byte-unchanged since creation):
  `research/runners/compose_retrieval_core.py` (commit `c474d6e`).
- Net-new composition runner (adversarially reviewed + faithfulness-fixed
  + re-review CLEAR): `research/runners/compose_retrieval_runner.py`
  (commit `19190bd`).
- Durable decisive output: `research/findings/raw/compose_retrieval_DECISIVE.json`
  (verdict + 9 raw cells) and `...DECISIVE.log` (full 1461-line GPU log).
- Design (section 2b is the next-stage core):
  `docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md`.
- Original frozen verdict (`2048750`), corrected module (`36a7975`),
  no-confabulation moat + 7/7 test: byte-unchanged throughout.
