---
type: finding
status: partial
date: 2026-08-08
mechanism: sleep-replay-consolidation-self-generated-hippocampal-engram
lane: breadth / teacher-loop / memory
runner: research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py
builds-on: research/runners/_teacher_loop_scaling_derisk.py
attacks-baseline: teacher-loop SCALING de-risk (frac_recalled ~ 1/N; artifact teacher_loop_scaling.json, commit cf4b4ff4e)
biological-pattern: hippocampal->cortical systems consolidation (McClelland/McNaughton/O'Reilly 1995; Wilson/McNaughton 1994); replay-cortical-consolidation gate line v1..v3 (distinct spiking substrate, cited not imported)
artifacts:
  - research/findings/raw/teacher_loop_sleep_replay_s42.json
  - research/findings/raw/teacher_loop_sleep_replay_s42.json.prov.json
---

# Self-replay of the brain's OWN hippocampal engrams beats catastrophic forgetting in the sequential teacher-loop (single-seed SMOKE GO, teacher/world absent during replay)

## The crux this attacks

<!--derived-->
(this paragraph quotes the PRIOR teacher-loop SCALING finding's measured baseline numbers, not this run's artifact)

The teacher-loop SCALING de-risk measured the breadth wall: teaching N distinct facts SEQUENTIALLY into ONE brain
via corrective e-prop on a shared leaky-readout retains ~1 fact (`frac_recalled ~ 1/N`; every fact learned
perfectly at immediate ~0.995, then OVERWRITTEN). The INTERLEAVED control (the teacher re-presents old facts
alongside the new one) retains 8/10 at N=10 on the SAME net -> capacity is adequate; the failure is sequential
INTERFERENCE on the shared readout. The mitigation had to raise SEQUENTIAL retention toward that 8/10 ceiling
WITHOUT the teacher re-presenting old facts (that would just BE the interleaved crutch).

## The mechanism (brain-based, self-generated)

Hippocampal->cortical systems consolidation. Teaching remains WAKE (teacher present, percepts drawn from the
world). At the same time a HIPPOCAMPUS captures a compressed engram of the episode = the MEAN of the percepts the
brain experienced, tagged with the taught class (a lossy one-shot trace, the brain's own memory). Then an OFFLINE
SLEEP phase self-replays: the hippocampus reactivates each stored engram and GENERATES a replay pattern from it
(engram + brain-owned internally-generated variability -- a generative replay from the hippocampal prototype,
using a BRAIN-owned RNG, never env's true prototype or env's noise process), interleaves the self-generated old
facts with the new one, and re-consolidates them into the shared readout via the SAME e-prop rule.

The replayed patterns are SELF-GENERATED from the brain's own store, NOT the teacher re-presenting:
`_self_replay_consolidate(net, hippocampus, ...)` and `Hippocampus.generate_replay(...)` take NO `env` parameter
and contain NO `env` token in their code bodies (grep-verified) -- the teacher and the world are absent during
consolidation.

## Result (single-seed SMOKE, seed 42, N=10, numpy backend on a 46-neuron net)

<!--derived-->

| arm | N=1 | N=5 | N=10 frac_recalled | what it is |
|---|---|---|---|---|
| NOREPLAY (baseline) | 1/1 | 2/5 | **1/10 (0.10)** | the 1/N catastrophic-forgetting wall (== consolidation lesioned) |
| REPLAY (self-replay) | 1/1 | 5/5 | **9/10 (0.90)** | self-generated hippocampal replay, teacher/world absent |
| SCRAMBLE (content-lesioned) | 1/1 | 0/5 | **1/10 (0.10)** | IDENTICAL extra compute, engram labels shuffled |

- REPLAY 9/10 MEETS AND EXCEEDS the interleaved 8/10 ceiling, WITHOUT the teacher re-presenting.
- REPLAY immediate acquisition = 1.000 (learning the new fact is not broken).
- The N=8 smoke (numpy) gave the same shape: NOREPLAY 0.12, REPLAY 0.75, SCRAMBLE 0.00, GO.

## Teeth (all pass -> Verdict GO; preconditions block in the artifact)

- **(a) retention RISES** vs no-replay sequential: 0.10 -> 0.90 on the SAME net/epochs (`reaches`, moved=True).
- **(b) load-bearing**: NOREPLAY (== lesion the consolidation phase) forgets to 1/10 -> the replay phase carries it.
- **(c) immediate acquisition stays perfect**: REPLAY mean immediate acq = 1.000 >= 0.9 floor.
- **(d) the STORE is the source** (self-generated + load-bearing): SCRAMBLE replays the store's content lesioned
  (labels shuffled) with the IDENTICAL extra compute and collapses to 1/10. So the retention rise is carried by
  the STORED ENGRAM CONTENT, not by the extra gradient steps and not by the teacher. `attributable_to` reads 100%
  of the replay-vs-scramble effect on the manipulation; the self-generation margin (replay - scramble) = +0.80.
- **grep-verify teacher/world absent**: the consolidation + replay code paths hold no `env` token.

The SCRAMBLE arm doubles as the compute-fairness control: it is the mechanism's own answer to "isn't the rise
just more training?" -- same compute, corrupted content, forgetting returns. This matches the 2026-04-27
sleep-replay-infrastructure finding's lesson that replay needs the right CONTENT, not just the right machinery.

## Scope / honest boundary

- Single-seed SMOKE (seed 42). The 6-seed claim needs 6/6 -- command below.
- N=10 world of noisy perceptual prototypes -> single-class patient mapping on the OnBridgeEpropNet transport-free
  e-prop substrate (the exact scaling-baseline substrate; reuse-by-import, NO sim/ edit). Scaling to hundreds of
  facts (the real breadth target) is the declared NEXT step; consolidation compute grows ~quadratically in N
  (fact i replays i+1 engrams), which is biologically faithful (repeated nightly replay) but a throughput item to
  characterize at larger N.
- The engram is a MEAN-prototype trace (a compressed generative replay). A richer hippocampal engram (multiple
  exemplars / a pattern-completing attractor store) is the faithful next refinement; the mean-prototype already
  clears the 8/10 ceiling here.
- backend: numpy (a 46-neuron net; cupy launch overhead makes GPU slower for this size). The 6-seed command below
  gives cupy per the standard recipe; the mechanism is backend-independent.

## Reproduce

Single-seed SMOKE (as run):
```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_sleep_replay_consolidation_derisk --seed 42 \
    --n-max 10 --milestones 1 5 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
    --settle-steps 25 --test-n 40 --out research/findings/raw/teacher_loop_sleep_replay_s42.json
```

6-SEED (GO needs the retention rise 6/6 at 42..47; one seed per process, in parallel):
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_sleep_replay_consolidation_derisk --seed $s \
  --n-max 10 --milestones 1 5 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
  --out research/findings/raw/teacher_loop_sleep_replay_s$s.json & done; wait
```
