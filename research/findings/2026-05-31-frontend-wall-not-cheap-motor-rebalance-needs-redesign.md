# The 28-word front-end wall is NOT a cheap motor-rebalance -- it needs redesign (owner-strategic) -- 2026-05-31

## Question (cheap-first de-risking)
The cheating audit quantified the front-end wall: 28-word pool-label recognition 0.571 (vs 0.812 at 16
words). The v17 finding diagnosed "motor pools dominate the argmax" (22/28 words have a motor pool as the
top off-target). If the wall were just that readout dominance, a cheap inference-time fix (down-weight the
motor pools before the argmax) would lift pool-label -- distinguishing a CHEAP fix from a needs-retrain
(owner-strategic) one. No retrain; reuse the existing 28-word bridge.

## Method
Probe research/findings/raw/_frontend_motor_dominance_probe.py: load the existing 28-word bridge
(_v17_28word_seed42), capture per-pool firing for each word, then (1) measure whether each CONCEPT word wins
among CONCEPT pools only (motors excluded as candidates), and (2) sweep a motor down-weight factor f and
recompute the full pool-label.

## Result -- the cheap fix FAILS; the wall is deeper than motor dominance
- Baseline pool-label (f=1.0): 0.571 (reproduces the audit).
- Among CONCEPT pools only, the correct concept word wins just **13/24 = 0.542**. So even with motor
  competition removed, the concept pools do NOT cleanly separate concept words (54%, far above the 24-pool
  chance 4% -- they discriminate, but weakly).
- Motor down-weight sweep: f=1.0 -> 0.571, but f<=0.5 -> 0.464 (WORSE). Down-weighting the motor pools HURTS,
  because the 4 motor words (north/east/south/west) then lose their own pools. There is no readout
  reweighting that fixes the concept words without breaking the motor words.

## Verdict: NOT a cheap readout fix -> the front-end wall is genuinely architectural (owner-strategic)
The 28-word recognition wall is NOT simply motor-pool readout dominance (the v17 framing was incomplete).
Two facts establish this: (a) concept pools only discriminate concept words at 54% among themselves -- a
real separability limit, not just motor crowding; (b) the obvious cheap fix (suppress motors at readout)
makes it WORSE, because motor and concept words have opposite needs. So the front-end needs a genuine
architectural redesign -- e.g. larger/balanced concept representations, more lang_input dimensions, smaller
motor pools, or a concept-only architecture -- NOT an inference-time rebalance. This CONFIRMS the
owner-strategic framing: pushing learned recognition past ~28 words is a real retrain/redesign, not a quick
win.

## Refines the v17 finding (honest correction)
v17 said "motor pools dominate; 22/28 words have a motor pool as top off-target." True, but incomplete: the
deeper limit is that the CONCEPT pools themselves only separate concept words at 54%. Motor dominance is a
symptom; the cause is the concept-representation separability at 28 words (the same separability frontier the
DG-biologization arc characterized as fundamental on this substrate). The biological fix is richer concept
representations (intrinsically less-overlapping reps via richer training), which is the documented
owner-strategic, months-scale direction -- not autonomously launched.

## Net (this cheap-first did its job)
It distinguished cheap-fix vs needs-retrain: the answer is NEEDS-REDESIGN. So no cheap autonomous lever
remains on the 28-word front-end wall; the next step is an owner decision on the architectural/richer-training
direction. The validated composition + recognition-at-small-scale + real-substrate (variance-limited,
temporal-integration-fixable) results all stand; the front-end at scale is the honestly-bounded frontier.
