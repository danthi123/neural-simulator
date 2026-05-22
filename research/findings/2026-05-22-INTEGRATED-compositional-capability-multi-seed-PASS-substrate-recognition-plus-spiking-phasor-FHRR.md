# Integrated compositional capability -- multi-seed PASS: the project's validated concept-recognition substrate + the spiking-phasor FHRR composition subsystem, end-to-end, clears the frozen 0.80 compositional bar at multi-seed mean 0.96-0.99 across loads {2,3,5}, with the no-confabulation abstention moat preserved -- the first multi-seed-validated compositional capability in the project after eight architectures could not produce one

## Status

The integration milestone. The cheap-first trilogy established FHRR
composition is reachable, noise-tolerant, and abstention-carrying; the
spiking-phasor FHRR subsystem was built and validated on abstract
symbols; this runner is the genuine end-to-end integration with the
project's substrate. Controller-only; seeds 42/43/44; reuse-by-import;
no protected/frozen/moat module modified; no autograd.

## The architecture

A two-system pipeline:
- **Recognition front-end**: the project's validated v14/v16 +
  hippocampus substrate (the 800-event multi-seed Phase-1 caches).
  For each concept word it drives the word and reads which concept
  pool fires -- the validated direct-binding readout. The recognized
  pool is the substrate's concept identity for that word.
- **Composition back-end**: the spiking-phasor FHRR subsystem
  (`research/runners/spiking_phasor_fhrr.py`). Each concept pool gets
  a fixed deterministic spiking-phasor symbol; a word's symbol is the
  symbol of the pool the substrate RECOGNIZED it as.
- They join at the concept-identity level. Recognition error
  propagates honestly: a misrecognized word gets the wrong symbol.

The compositional task: encode L (cue, adjective) facts via the FHRR
subsystem (bind + bundle), query each via unbind, clean up over the
adjective vocabulary with the abstention moat, against the project's
frozen 0.80 bar.

## Result (pre-registered; no bar tuned; seeds 42/43/44, 300 trials/load)

```
                   recognition   L=2      L=3      L=5
seed 42            15/16 = 93.8%  1.000    1.000    1.000
seed 43            13/16 = 81.2%  0.967    0.936    0.887
seed 44            13/16 = 81.2%  0.998    0.993    0.993
-----------------------------------------------------------------
integrated mean                   0.988    0.976    0.960   (all >= 0.80)
composition-only mean              0.989    0.981    0.969

VERDICT -> INTEGRATED MULTI-SEED PASS
```

The integrated two-system pipeline clears the frozen 0.80
compositional bar at multi-seed mean at every load -- and not
marginally: 0.96-0.99. The lowest single cell is seed 43 at L=5
(0.887), still comfortably above the bar.

The **composition-only accuracy** (restricted to facts whose words
were all correctly recognized by the substrate) is 0.97-1.00 -- it
confirms the FHRR composition itself is essentially perfect on the
substrate-recognized symbols. The integrated accuracy's only shortfall
from 1.0 is recognition error propagating: where the substrate
misrecognizes a task word, the wrong symbol enters composition.

## What this is

This is the first multi-seed-validated end-to-end compositional
capability in the project. Eight architectures + four diagnostic
probes + two substrate variants of the biology-grounded rate-coded
substrate could not produce compositional retrieval that cleared the
bar -- they plateaued at ~0.46, the readout at the noise floor. The
two-system pipeline -- validated recognition substrate + spiking-phasor
FHRR composition -- clears it at 0.96-0.99 multi-seed, and the
no-confabulation abstention moat is carried natively in the FHRR
clean-up (the abstention probe established clean groundable/
ungroundable separation).

The renewed-focus arc, end to end: the dynamics-gating fix class was
exhausted (8 architectures + the staged-recurrence variant, all
negative, all converging on "the readout needs a structured decodable
object"); owner-directed external research surfaced Orchard & Jarvis's
spiking-phasor FHRR; three cheap-first probes de-risked it; the
subsystem was built; this integration validates it end-to-end on the
project's substrate.

## Honest scope and caveats (stated plainly, not buried)

1. **Two-system architecture.** This is a rate-coded recognition
   substrate plus a phase-coded composition subsystem, joined at the
   concept-identity level -- NOT a single unified substrate. Whether a
   two-system recognition+composition architecture is the
   brain-analogue direction the owner wants (vs a unified substrate)
   is a standing paradigm question for the owner.
2. **The interface is identity-level, not activity-level.** The
   substrate recognizes a word to a discrete pool label; the FHRR
   symbol is a fixed lookup keyed to that label. The substrate's
   actual neural activity does not itself flow into the FHRR layer. A
   deeper activity-level integration is not done.
3. **The phasor neuron models are biology-inspired engineering.**
   Theta-gamma phase coding is real, well-characterized biology; the
   phase-sum / phase-subtraction integrator neurons are function-first
   engineered devices (Orchard's design), not derived from a
   biological neuron model.
4. **3 seeds; the project's compositional task at loads {2,3,5}.**
   This is small-load compositional retrieval -- reliable
   cue-to-attribute recall -- explicitly NOT fluent open-ended
   language.
5. **Integrated accuracy is recognition-bounded.** Composition is
   essentially perfect (composition-only 0.97-1.00); the integrated
   number reflects which seed's recognition errors land on task
   words. The pipeline's remaining gap is the substrate's
   direct-binding recognition accuracy -- a validated capability with
   known improvement paths (more Phase-1 training raised it to 85.4%
   multi-seed; the v14 baseline reached 88.75%).

## Discipline check

Pre-registered frozen 0.80 bar; not tuned. Reuse-by-import only
(`test_one_checkpoint` recognition + the `spiking_phasor_fhrr`
subsystem); no protected/frozen/moat module imported or modified; no
autograd. Honest propagation both remotes.

This is a nominal PASS, so it is scrutinized: the task is above chance
(clean-up over 4 fillers, chance 0.25, result 0.96-0.99); recognition
is genuinely in the loop (seed 43's recognition misses propagate, the
integrated accuracy drops to 0.887 where they hit task words);
composition-only isolates and confirms the FHRR layer; the symbols
are random per pool and assigned independently of the facts (no
answer leak). A dedicated adversarial review of the integration glue
is the pre-registered next discipline step before any capability-
status claim.

## Files / evidence

- Integration runner: `research/findings/raw/spiking_phasor_integration.py`
- Result: `research/findings/raw/spiking_phasor_integration.json`
- Log: `research/findings/raw/spiking_phasor_integration.log`
- Subsystem: `research/runners/spiking_phasor_fhrr.py`
- Cheap-first trilogy: `fhrr_numpy_probe`, `spiking_phasor_fhrr_probe`,
  `fhrr_abstention_probe`.

## Next step

A dedicated adversarial review of the integration (is the PASS an
artifact; is recognition genuinely load-bearing; is the abstention
moat genuinely carried; is anything leaking) -- the standard
discipline for a load-bearing PASS, scrutinised harder than a FAIL.
If the review clears it, the integrated compositional capability is a
genuine validated milestone and warrants a capability_status.json
pillar with the honest two-system framing. Then the deeper arcs:
activity-level integration (the substrate's neural activity feeding
the phasor layer directly), and scaling beyond the small-load task.
