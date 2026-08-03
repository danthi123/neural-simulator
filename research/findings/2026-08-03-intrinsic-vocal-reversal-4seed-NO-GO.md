---
type: finding
status: negative
date: 2026-08-03
mechanism: intrinsic-vocal-exploration-and-reversal
runner: research/runners/_developmental_vocal_intrinsic_reversal_derisk.py
artifacts:
  - research/findings/raw/developmental_vocal_intrinsic_reversal/seed42.json
  - research/findings/raw/developmental_vocal_intrinsic_reversal/seed43.json
  - research/findings/raw/developmental_vocal_intrinsic_reversal/seed44.json
  - research/findings/raw/developmental_vocal_intrinsic_reversal/seed100.json
---

# Intrinsic vocal exploration and same-brain reversal are not reliable

<!--derived-->
**Verdict: NO-GO at the four-seed development gate.** Replacing balanced
injected babbling with one shared neural arousal signal produced reliable
acquisition and reversal in only one of four brains. The result does not retire
the injected-babbling scaffold and was not promoted to the full control battery.

## Question

Can one continuously operating spiking brain generate its own vocal
exploration, learn a two-intent by two-referent convention from a listener's
consequences, and then unlearn and relearn the convention when the same listener
changes it?

The tested path used one target-independent arousal population, six symmetric
exploration routes, neural competition, reward-US to SNc dopamine bursts, and
negative-feedback to RMTg to SNc dips. No desired output channel or listener
mapping was injected into the brain.

## Result

Raw per-trial evidence starts at
`research/findings/raw/developmental_vocal_intrinsic_reversal/seed42.json` and
the sibling seed artifacts listed above.

| seed | initial convention | initial held-out | reversed convention | reversed held-out | old convention after reversal | explored actions | reward / error events | GO |
|---:|---:|---:|---:|---:|---:|---|---:|:---:|
| 42 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3/4 | 1 / 44 | no |
| 43 | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 | 4/4 | 154 / 317 | no |
| 44 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 4/4 | 66 / 350 | yes |
| 100 | 0.00 | 0.00 | 0.25 | 0.50 | 0.25 | 4/4 | 65 / 244 | no |

All measured synaptic changes remained inside the declared vocal-learning
routes. The artifacts embed the exact source hashes, commit
`8517338cb90028239c6eae0dfe0c37f32e578634`, CuPy backend, and RTX 3090 device.
Seeds 101 and 102 were deliberately not run after the development gate failed,
so they remain untouched for a later held-out test of a new architecture.

## What Failed

The shared arousal signal generated activity, but it did not cleanly identify
which neural action earned the later dopamine signal. In the decision events
retained by all four artifacts, both competing exploration populations fired in
essentially every speak, intent, and referent bank. Ties were also common. That
means local coactivity traces can mark the losing route as well as the route
that actually controlled the emitted action.

Exploration coverage was not the whole problem: seeds 43 and 100 tried all four
composite actions and still failed acquisition and reversal. Dopamine also
varied sharply by seed. The seed-44 success therefore shows that the ingredients
can align in one initialization, not that the mechanism is reliable.

Post-failure diagnostic variants strengthened cortical inhibition, changed
fatigue and recovery, scoped homeostasis, and altered sensory gain. Stronger
competition tended to lock in one structural winner; stronger fatigue could
cause long silence; and restored exploration still left ambiguous local credit.
Those variants were removed rather than tuned into a new unregistered gate.

## Controls Not Run

The full no-consequence, yoked-reward, dopamine-lesion, arousal-lesion,
RMTg-lesion, and pathway-attribution battery was not run. Once the main effect
failed three of four development seeds, those controls could not rescue the
claim and would have spent GPU time without changing the verdict.

## Decision

Stop tuning this cortical winner-take-all exploration path. The next mechanism
must separate neural action selection from performance evaluation and ensure
that only the executed action carries strong local eligibility when a global
dopamine burst or dip arrives. The preregistered design is
[`docs/plans/2026-08-03-neural-vocal-action-credit-design.md`](../../docs/plans/2026-08-03-neural-vocal-action-credit-design.md).
