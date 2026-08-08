---
type: finding
status: contributing
date: 2026-08-08
mechanism: pragmatic-success-readback-wta-speaker
lane: D-pragmatics
runner: research/runners/_pragmatic_success_readback_leg2_derisk.py
builds_on: research/findings/2026-08-08-pragmatics-communicative-success-neural-coincidence-detector-leg1-6seed-GO.md
artifacts:
  - research/findings/raw/_pragmatic_success/leg2_smoke.json
superseded_by: research/findings/2026-08-08-pragmatics-readback-leg2-v2-oracle-RESOLVED-convergence-NEGATIVE-critic-value-fallback.md
---

# Leg 2 (read success back to TRAIN a WTA speaker) is a NEGATIVE — the WTA choice is not weight-controllable; the declared fallback is a spiking value-critic

<!--derived-->

**One line.** Reading the Leg-1 neural coincidence-success back as three-factor DA does change the intent→utterance
weights correctly, but the speaker's winner-take-all CHOICE cannot be moved by those weights — a hand-set **30×
oracle** weight still fails to select the target utterance (mean readout acc **0.1667** across 6 seeds, below the
1/3 chance). So the mapping is UNLEARNABLE in this WTA architecture regardless of the credit signal. This is an
honest NEGATIVE on the METHOD; the CAPABILITY is carried forward to the declared fallback (a spiking value-critic).

## What was built (and what worked)

One spiking bridge with the Leg-1 coincidence evaluator co-resident: `intent[K]` → (PLASTIC, three-factor,
reward-modulated) `utterance[K]`-WTA → the winning utterance selects the RSA listener's belief response →
`belief[K]` + `intent[K]` → the fixed coincidence `success[K]` → RPE = success − running baseline delivered as
`current_reward_signal`. Coincidence-contingent DA (a mismatch → no coincidence → RPE below baseline → the choice
is weakened) — the negative arm the 2026-08-03 vocal-credit v1 lacked.

**Two things worked.** (1) A necessary fix: eligibility must build from pre×post COACTIVITY
(`reward_eligibility_from_coactivity=True`), not from STDP/Hebbian weight-change (both off here) — without it the
eligibility trace stays 0 and the three-factor rule has nothing to convert (diagnosed 2026-08-08). With it,
eligibility builds (max ~0.36, ~14k synapses) and the intent→utterance weights change substantially under reward
(max |dw| ~26 over 40 trials). So the credit path is live.

## Why it is a NEGATIVE (the decisive tooth)

<!--derived-->

The learning does not move behaviour: the (underpowered, single-seed → verdict UNDEFINED) smoke's trained /
untrained / yoked choice accuracies are all **0.3333** (chance for K=3) — training changes weights but not the WTA
winner. The teeth-backed negative rests on a 6-seed oracle-weight probe (this was a PROSE probe in this v1 session; it is
re-issued as a COMMITTED, provenance-stamped code path in the v2 finding's `--oracle-probe`, which is where the
auditable oracle numbers now live). The decisive
isolation is an
ORACLE-WEIGHT readout probe: set intent[t]→utter[t] to weight 30 and every
other intent→utter block to 1 (a 30× differential the credit rule could never need to exceed), tonic drive 0, and
read the greedy WTA winner. If the choice tracked the afferent weights this would score 1.0; instead it is
**0.1667 mean across all six seeds (42 43 44 100 101 102), below the 0.333 chance** <!--derived-->. The winner-take-all winner is
dominated by per-neuron heterogeneity and the shared-FS latch dynamics — the first assembly to ignite suppresses
the others via the shared inhibition — so the plastic afferent weights cannot select it. The failure is in the
READOUT architecture, not the credit signal: even a perfect teacher of the weights would not produce the right
utterance.

This reproduces, in the speaker setting, the documented Gate-B / integrated-loop wall (a global scalar credit
cannot carry a selective policy;
`2026-05-19-integrated-loop-iter3-...global-scalar-credit-cannot-carry-WM-selectivity`), and the WTA-under-
reward-only irreversibility noted in the roadmap's Gate-B history. The oracle-weight probe adds the sharper point:
here even a NON-scalar, per-block oracle weight cannot steer the choice — the bottleneck is the WTA read, upstream
of any credit question.

## Honest scope

The credit half is live and correct (eligibility builds; weights move under contingent DA). The negative is
specific and teeth-backed: the WTA speaker's choice is not a function of the learned intent→utterance weights
(oracle-weight probe, 6 seeds). No metric is lifted from a passing arm; the smoke's chance-level accuracies and
the oracle-weight probe are the actual measurements. numpy-CPU; NO `sim/` edit. Leg 1 (the coincidence success
signal itself) remains a 6/6 GO and is unaffected.

## Declared fallback (spec) — the next method for the SAME capability

A **spiking value-critic baseline**: learn a graded value V(intent, utterance) from the coincidence success (the
Leg-1 signal, which is sound), and let the utterance choice follow the graded value directly — a competition the
learned value CONTROLS — rather than a heterogeneity-latched winner-take-all whose winner the weights cannot move.
Concretely: (a) a value population per (intent, utterance) trained by the coincidence RPE; (b) a choice read as a
graded/soft competition biased by V (not an all-or-none FS latch), or a lowered-inhibition WTA whose winner
demonstrably tracks the afferent drive (re-run the oracle-weight probe as the acceptance gate BEFORE training).
Still no `sim/` edit. The capability (read success back to shape speaking) is not abandoned — the WTA-latch method
is banked as failed and the value-critic method is next.

## Reproduce

```bash
# the negative smoke (nothing learned; trained=untrained=yoked=chance):
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_derisk --smoke --seed 42 \
    --json research/findings/raw/_pragmatic_success/leg2_smoke.json
# the decisive oracle-weight readout probe (v1 was a prose probe: build_speaker_bridge + _readout_policy with
# intent[t]->utter[t]=30, others=1, UTT_DRIVE_PA=0, 6 seeds). It is superseded by the v2 finding's COMMITTED
# --oracle-probe code path, which is where the auditable oracle numbers now live (graded regime -> 1.0/6).
# A full 6-seed training run would also be NEGATIVE for the same reason -- the oracle-weight probe already proves
# the choice cannot track the weights on all six seeds.
```
