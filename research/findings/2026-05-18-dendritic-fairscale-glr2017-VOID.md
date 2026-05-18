# Owner-authorized fair-scale dendritic GLR-2017 MNIST run — honest VOID (the pre-registered THREE-STATE gate refused to fabricate a science verdict from a broken instrument)

## TL;DR

The owner explicitly chose Option 2 (deliberate, eyes-open, week-scale
local investment in the literature's discriminating regime). The
fair-scale run was built (subagent-driven TDD; Phase-A adversarially
APPROVED; dead-MNIST-URL defect fixed; GPU-accelerated via the
validated `sim.backend` as a verified pure-speed change with the
no-confab moat + committed rule + frozen `_DFAIR_*` byte-identical and
7/7 green throughout), and the decisive multi-seed run COMPLETED
cleanly on the 3090 (kill-safe, real MNIST 60000/10000 cache-verified;
not crashed by GPU contention -- it finished and exited).

**Result: GATE = VOID, 3/3 seeds.** Recomputed from the recorded JSON
(mandatory anti-cheat smell-test, no re-run, no bar-tuning): the
true-gradient `oracle` positive-control scored **0.1009 / 0.1135 /
0.1028** heldout (seeds 42/43/44) -- MNIST 10-class chance is 0.10.
Every condition (oracle, local_correct, local_wrongsign,
global_scalar, permuted) sat at chance. The pre-registered
instrument-validity gate **V1 (oracle >= 0.95)** was not met, so the
THREE-STATE verdict correctly returned **VOID -- explicitly NOT a
science PASS/FAIL**: no valid measurement of dendritic credit
assignment was made, because the measuring instrument (the
optimizer/harness) itself did not train even with the exact gradient.

This is the THREE-STATE design + the STRENGTHEN-only aggregate working
exactly as engineered -- they refused to emit a science verdict from a
non-functional instrument (the same class of catch as the Generator-S
false-PASS). Propagated WITHOUT spin.

## Honest scope (no overclaim, no underclaim)

- **VOID is NOT a science FAIL of dendritic credit assignment.** The
  science is UNTESTED: the positive control (true-gradient backprop)
  did not learn MNIST at the pre-registered config, so nothing about
  the biologically-local rule was validly measured. Reporting this as
  "dendritic learning fails" would be a lie; reporting it as a PASS
  would be a worse lie. It is exactly VOID: instrument not sound at
  the pre-registered config.
- **Diagnosed cause (honest engineering, not science):** plain SGD +
  Glorot-sigmoid init + lr=0.1 + 60 epochs on a 784-512-256-128-10
  sigmoid MLP under-trains even with the true gradient (classic
  deep-sigmoid + weak-optimizer non-convergence). The Phase-1
  `test_oracle_mode_is_positive_control_descends_loss` only validated
  a TINY 8-16-16-3 toy net on a 64-sample problem; it never validated
  that the SAME config trains the full MNIST net. The V1 gate exists
  precisely to catch this and it caught it (VOID, not a fake verdict).
- This is the SAME oracle-collapse instrument-non-constructibility
  that recurred across all four cheap probes (static-cosine ->
  W2-confound -> vanishing-sigmoid -> ReLU-overflow). It is now
  confirmed AT FAIR SCALE -- the FIFTH independent triangulation that
  a SOUND, discriminating instrument for biologically-local credit
  assignment is hard to construct at feasible local scale.

## Why this is decision-relevant + what the design pre-registered

The design pre-registered, verbatim: "An INSTRUMENT-VOID outcome (V1
or V2 unmet) is reported as 'instrument not soundly constructible even
at fair scale', NOT a science PASS/FAIL," and the prescribed response
to a VOID is to fix the INSTRUMENT (the optimizer/harness -- NEVER the
frozen science bars), then re-run. V1 was explicitly pre-registered as
a SOLVED ENGINEERING REQUIREMENT (a calibration precondition), not a
science result. So the honest, in-scope, NON-config-crank next action
(distinct from the falsify-cheaply cheap-probe "stop after 3" -- this
is the owner-authorized Option-2 investment whose design explicitly
prescribes the VOID->fix-instrument->re-run loop) is:

1. ONE rigorous instrument-engineering pass on `sim/dendritic_mlp.py`
   so the true-gradient positive control genuinely trains MNIST
   (standard, well-understood, non-fundamental fixes: input
   standardization, regime-appropriate init, momentum / hand-derived
   Adam -- all pure numpy/cupy, **NO autograd**), with the frozen
   `_DFAIR_*` science bars, the no-confab moat, the committed
   plasticity rule, the no-weight-transport + no-autograd invariants,
   and the THREE-STATE verdict ALL byte-UNCHANGED.
2. GATED by a CHEAP V1-only check (does properly-optimized true-
   gradient backprop reach >= 0.95 heldout on MNIST? a ~minutes
   check) BEFORE re-committing the full owner-authorized GPU run --
   the falsify-cheaply discipline applied to the instrument fix, so
   the 3090 is not spent again on an unsound instrument.
3. If even a properly-engineered backprop oracle cannot reach 0.95 on
   MNIST -> a stunning, genuinely decision-relevant signal (backprop
   MLP on MNIST is the most standard result in ML; failing it would
   indicate something far deeper). Far more likely: proper
   optimization makes V1 pass, the instrument becomes sound, and the
   full re-run finally yields a real science PASS/FAIL on the actual
   owner-authorized question.

This is calibrating the measuring instrument (the scale reads 0 for a
known 1 kg weight -> fix the scale; do NOT report "mass VOID, give up"
and do NOT "tune the scale to read what you want"). The frozen science
bars are untouched; only the optimizer is engineered, exactly as the
design pre-registered for a VOID.

## What is preserved / validated (unaffected)

Byte-UNMODIFIED + 7/7 green across the WHOLE fair-scale build incl. the
GPU port: the no-confabulation moat (`abstention_gate` +
`tests/test_abstention_gate.py`), every frozen `*_core` (incl.
`dendritic_fair_core` `_DFAIR_*`), `sim.dendritic_plasticity` (the
committed credit-assignment rule), `sim.train_checkpoint`,
`sim/bptt_snn*`, `sim/bridge.py`. The kill-safe checkpoints (all 15
legs) are intact; the GPU acceleration is a verified pure-speed change
(~4.4x at decisive scale; 39/39 green incl. moat + exact-1e-9
committed-rule faithfulness + every adversarial invariant on the cupy
path). The science/verdict/bars logic is byte-identical to the
adversarially-APPROVED Phase-A.

## Anti-cheat discipline (why this VOID is trustworthy)

The pre-registered THREE-STATE gate + the STRENGTHEN-only aggregate +
the V1 instrument-validity-FIRST design did exactly their job: a
non-functional instrument produced VOID, never a fabricated PASS/FAIL.
Mandatory smell-test recomputed every number from the recorded JSON
(no re-run, no bar-tuning); MNIST provenance verified real. The honest
VOID is propagated WITHOUT spin; the design-prescribed instrument
fix is gated by a cheap V1 check (no GPU re-spend on an unsound
instrument); the frozen science bars are NOT tuned; nothing is
config-cranked. The validated no-confab moat -- the project's
distinctive contribution -- remained byte-identical and 7/7 green the
entire time.

## Files / evidence

- Result: `research/findings/raw/g11_bg/dendritic_fair_gate.json`
  (GATE VOID; per-seed oracle 0.1009/0.1135/0.1028 ~chance;
  instrument_valid False all seeds; MNIST provenance 60000/10000
  cache-verified).
- Build: `sim/dendritic_mlp.py` (GPU via sim.backend),
  `research/runners/dendritic_fair_{core,gate}.py`, kill-safe
  checkpoints `research/findings/raw/g11_bg/dfair_ckpt/` (15 legs).
- Design/plan:
  `docs/plans/2026-05-17-dendritic-fairscale-glr2017-{design,implementation}.md`.
- Prior triangulations: the dendritic faithful-instrument terminus +
  the LLM-teach-then-wean steelman addendum
  (`2026-05-17-dendritic-faithful-instrument-TERMINUS.md`).
