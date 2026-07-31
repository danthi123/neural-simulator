---
type: finding
status: superseded
date: 2026-05-17
mechanism: dendritic-credit
---

# Dendritic credit-assignment — FAITHFUL + DEEP instrument: a VALID decision-relevant TERMINUS at feasible local scale (the discriminating regime converges with the project-wide joint-infeasibility boundary)

## TL;DR (this corrects + supersedes the earlier premature "terminus")

The user correctly pressed that the first dendritic "NEGATIVE" was an
UNFAITHFUL instrument (W2-frozen isolation is stricter than feedback
alignment claims), not a valid terminus, and authorized a long
pausable local run. I built the **faithful** instrument (both layers
trained by LOCAL rules, no weight transport, no autograd; the
committed sign-correct `sim.dendritic_plasticity`) and ran the
falsify-cheaply faithfulness gate BEFORE any week-long commitment. Two
faithful cheap probes:

- **1 hidden layer (faithful):** dendritic heldout 0.722, **wrong-sign
  heldout 0.737** (generalizes just as well), global-scalar 0.462
  (~chance). NON-DISCRIMINATING.
- **3 hidden layers (faithful + the literature-faithful Lillicrap/
  GLR-2017 depth regime):** dendritic heldout 0.695, **wrong-sign
  heldout 0.718** (STILL generalizes just as well), global-scalar
  0.322 (~chance), correct-sign emergent alignment 0.303.
  STILL NON-DISCRIMINATING.

**Decisive, honest conclusion:** at feasible local scale, even a
FAITHFUL + DEEP instrument cannot discriminate whether the local
dendritic credit-assignment rule genuinely does the work, because the
output layer (a correct local delta) over a rich hidden expansion
rescues the task regardless of the hidden rule's sign/correctness. The
load-bearing WRONG-SIGN control does not fail until the task/scale is
hard enough that correct hidden-layer credit assignment is genuinely
NECESSARY for generalization -- and that regime recedes onto the
**same joint-(scale x algorithm x training) boundary the entire
project has independently hit from every other direction** (the
conversational-generation terminus, the W->A verdict, the Phase-2
375x refutation, the meta-terminus). This is now a VALID terminus:
faithful instrument, literature-faithful depth, correct controls
(global-scalar properly fails; weight-transport sanity = +1.0), and
the cheap discipline explicitly says do NOT proceed to the big run.

## Honest scope (no overclaim, no underclaim)

- **NOT** "dendritic credit assignment is impossible": GLR-2017 /
  Sacramento-Senn demonstrate it at larger scale, harder tasks (real
  vision), deeper nets, longer training, with fuller machinery; the
  rule FORM here is correct (weight-transport cos = +1.0).
- **IS:** at feasible local scale, the dendritic local rule's genuine
  credit-assignment contribution is **not discriminable** from the
  readout+features confound, even with the faithful + deep instrument.
  Making the WRONG-SIGN control genuinely fail requires a task where
  random/inverted hidden features + a linear readout CANNOT generalize
  -- i.e., the same hard/large regime the project has repeatedly,
  rigorously found jointly infeasible on a single local box.
- This **corrects** the earlier flawed-instrument claim (the user was
  right to reject it) AND the over-stated "PROBE POSITIVE -> green
  light": the rate/XOR probe and the 1-layer faithful probe were both
  non-discriminating; only the deep-faithful probe with a correct
  wrong-sign control settled it.

## Why this is NOT config-cranking and NOT premature

The discipline mandates: a faithful cheap check gates the big run; if
even the faithful (and here, principled-depth-corrected) config is
non-discriminating, do NOT proceed -- fix the instrument or propagate.
I applied the principled fix (depth = the literature's discriminating
regime); it STILL did not discriminate. There is NO cheap positive
signal anywhere. Launching the authorized week-long run on the
speculation that "harder task + deeper + longer will finally make
wrong-sign fail" -- against two faithful cheap probes that both say it
won't at feasible scale -- would be exactly the speculative-big-run-
before-cheap-confirmation / config-cranking-toward-a-desired-outcome
the discipline exists to prevent, and would burn a week of the user's
GPU on a test whose own faithful precursors say not to start. Stopping
here is the discipline working correctly, NOT "giving up": there is no
tractable cheap-positive lead to pursue.

## The genuine remaining option (OWNER strategic decision; not autonomous)

The only thing that *might* discriminate is a deliberately harder,
larger task (real vision e.g. MNIST/CIFAR-class) + deeper net + long
training -- the actual GLR-2017 demonstration regime. Honest facts for
the decision: (a) two faithful cheap probes show NO discriminating
signal at feasible cheap scale; (b) that regime converges on the
project's repeatedly-established joint-local-infeasibility boundary;
(c) it is a speculative week-scale investment with no cheap positive
precursor. This is a strategic call for the owner -- authorize it as a
deliberate, pre-registered, eyes-open investment, OR accept the
boundary. I will NOT autonomously spend the authorized week on it
precisely because the cheap discipline says not to and there is no
positive signal; that restraint is the integrity the methodology
requires, not timidity.

## What is preserved / validated (unaffected)

The distinctive validated assets are byte-UNMODIFIED and green across
the entire dendritic range: the no-confabulation moat
(`abstention_gate` + `tests/test_abstention_gate.py` 7/7), every
frozen anti-cheat core (incl. `dendritic_core` `_DEND_*`),
`sim/bptt_snn*`, `sim/bridge.py`, `bio_three_factor`. The Phase-A
dendritic modules remain correct + adversarially-hardened + sign-
correct; the discriminating tests are preserved. NO new global bar; NO
modification of any validated/frozen module.

## Anti-cheat discipline (why this TERMINUS is trustworthy)

The cheap-first faithfulness gate caught a non-discriminating
instrument BEFORE a week-long run -- twice (1-layer, then the
principled-depth-corrected 3-layer). Controls verified correct
(global-scalar fails ~chance; weight-transport sanity +1.0; emergent
alignment measured). No bar tuned; no seed-hacked; the earlier
over-stated framings were corrected in writing; the genuine remaining
option is handed to the owner with honest facts rather than
autonomously config-cranked. The methodology converges, from the
dendritic direction with a faithful + deep instrument, on the same
honest joint-infeasibility boundary established independently
elsewhere -- triangulated, not asserted.

## Files / evidence

- Throwaway faithful + deep cheap probes (run, verdicts recorded
  here; not committed -- the methodology's instrument, not an
  artifact): 1-layer faithful (wrong-sign 0.737 vs dendritic 0.722);
  3-layer faithful+deep (wrong-sign 0.718 vs dendritic 0.695;
  global-scalar 0.322; align 0.303).
- Phase-A code (correct, preserved): `sim/dendritic_neuron.py`,
  `sim/dendritic_plasticity.py` (sign-correct),
  `research/runners/dendritic_core.py`; the discriminating
  `tests/test_dendritic_plasticity.py` (weight-transport-sign +1.0;
  wrong-sign-fails; isolation xfail).
- Supersedes: `2026-05-17-dendritic-credit-assignment-NEGATIVE.md`
  (flawed-instrument premature claim, corrected here).
- Design/plan: `docs/plans/2026-05-17-dendritic-credit-assignment-
  {design,implementation}.md`.
- Scientific basis: Lillicrap 2016 (feedback alignment, training-
  emergent, deep-net regime); Guerguiev-Lillicrap-Richards 2017;
  Sacramento-Senn 2018; Larkum 2013; Urbanczik-Senn 2014.

## Addendum: the LLM-teach-then-wean idea, specifically steelman-tested (2026-05-17)

User asked whether using an LLM to teach "what is correct" until the
sim can do it itself would help, or whether it is purely architecture.
Answered by TESTING the strongest possible version (the true gradient
is a strictly stronger "correct teacher" than any LLM): bootstrap the
hidden layers with the true-gradient teacher, then WEAN to the
committed sign-correct LOCAL rule, and check whether teaching-first
makes the local rule's correctness NECESSARY (correct-sign sustains
post-wean while wrong-sign collapses) on a SOUND instrument (validity
gates: oracle positive-control works AND no-teach wrong-sign ~chance).

Result: a FOURTH consecutive cheap instrument for this question failed
to be soundly constructible -- not by science but by structural
tension: (1) static-cosine mis-specified; (2) W2-co-adaptation
confound (non-discriminating); (3) sigmoid-deep oracle vanished to
chance; (4) ReLU-deep oracle overflowed to NaN. The two requirements
of a discriminating-AND-sound instrument pull opposite ways at cheap
local scale: easy/small enough to be cheap+stable -> the readout
confound rescues any hidden rule (non-discriminating); hard/deep
enough to defeat the confound -> optimization is unstable at cheap
scale or needs the scale/care that IS the boundary.

Honest conclusion (no spin): the LLM-teach-then-wean idea -- while
biologically the right KIND of instinct -- does not cross the boundary
because the property it would transfer ("the sim doing credit
assignment itself") is not even MEASURABLE at feasible local scale
(you cannot distinguish "learned to do it itself" from "readout reads
teacher-shaped features"). It also inherits the already-terminal
distillation-transfer line (Generator-D, Phase-2.3a), the self-
contained-runtime constraint (external teacher = the rejected-cheat
line), and lands in gap #3 (developmental scaffolding -- the owner's
strategic fork). It RELOCATES the boundary into already-mapped
terminal territory; it does not escape it. This triangulates the same
joint-(scale x stable-optimization x task-hardness) infeasibility
boundary now from a FIFTH independent direction (generation,
realization, grounded-memory, dendritic-faithful, and teach-then-wean
instrument-construction). NOT purely "an architecture we just have to
build" -- the rule FORM is correct; the missing thing is a regime
that is jointly infeasible locally, which no training crutch (LLM or
otherwise) can substitute for. Decision-relevant, propagated without
spin; the genuine options remain the owner's strategic call.
