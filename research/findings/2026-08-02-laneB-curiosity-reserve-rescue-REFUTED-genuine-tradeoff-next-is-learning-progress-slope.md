---
type: finding
status: contributing
date: 2026-08-02
mechanism: curiosity-reward-omission-veto
artifacts:
  - research/findings/raw/lanes/curiosity/rev_ctrl_s42.json
  - research/findings/raw/lanes/curiosity/rev_rect_s42.json
---

# lane B (curiosity reward-omission veto): the "protective reserve rescues the veto" lever is a GENUINE honest-negative, not a tuning gap — the depression-on-absence is LOAD-BEARING, so no scalar reserve setting separates a slow-learner from an unlearnable concept (they are per-ask identical); the base omission circuit stays 6/6 GO, and the record-grounded next mechanism is a spiking LEARNING-PROGRESS-SLOPE differentiator

<!--derived-->
**One-line verdict.** In the lane-B 6-seed result (`research/findings/raw/lanes/laneB_reserve_6seed.json`) EVERY base
gate passes 6/6 and `reserve_rescues` is the ONLY failing gate (0/6): an active decaying inhibitory reserve makes the
learnable-concept veto go UP, not down. Testing the finding's own named fix (rectify the reserve to
potentiation-dominated — no active erasure on reward-absence, a persistent "was-EVER-rewarding" trace) FLIPS the reserve
condition the intended way BUT breaks two previously-passing base gates (noisy-veto collapses below the stop-floor;
mastery falls). Root cause, confirmed at the mechanism level: the depression-on-absence is LOAD-BEARING — it clears the
reserve for genuinely-unlearnable concepts so their excitatory veto can grow; remove it and noisy concepts keep a
baseline inhibitory reserve that pins their veto below the floor. A zero-progress re-ask of a SLOW learner is per-ask
IDENTICAL (omit HIGH, no reward US) to an unlearnable concept, so no scalar reserve can separate them. This is a genuine
trade-off frontier, not a tuning miss. No `sim/` edit (the tracked runner is unchanged; the iteration was an env-gated
scratchpad copy).

## The iteration — rectifying the reserve flips the target gate but breaks the base circuit

<!--derived-->
Research-first: `before_you_build.sh` surfaced the base finding `2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed.md`,
which NAMED this exact lever ("a deeper/decaying protective reserve so a concept that was EVER rewarding is not vetoed on
later zero-progress re-asks") and pinned the root cause ("the only per-ask signal cannot separate mastered-but-still-novel
from unlearnable; that separation lives in the HISTORY"). That reframed the residual as a per-ask IDENTIFIABILITY limit,
not a tuning gap — so the test was whether a monotone reserve can beat the identifiability limit. It cannot:

<!--derived-->
| arm (seed 42) | real noisy_veto | real learn_veto | noisy_vetoed | mastered | reserve_rescues |
|---|---|---|---|---|---|
| control (= lane B default) | 20.5 | 3.2 | True | 8 | False |
| rectified (rpe lower-clip -1.0 → 0.0) | **8.0 (< floor 12)** | 1.0 | **False** | **2** | False |

<!--derived-->
Rectification flips condition 1 (real learn_veto 1.0 < reserve-lesion 1.5) but noisy-veto collapses below the stop-floor
(`noisy_stops=False`) and mastery falls 8→2 (`yoked=False`). The reserve-lesion arm is byte-identical across both runs
(clean isolation). A diagnostic (`scratchpad/diag_reserve.py`) confirms the mechanism engages correctly in isolation
(pure-learnable: reserve potentiates 91→295 Hz, veto 0.0; pure-noisy: reserve depressed -1.0/ask, veto grows to 39 Hz →
vetoed) — the flaw appears only for a SLOW learnable concept re-asked with zero progress, which hits `omit HIGH` and thus
depresses the reserve AND potentiates the excitatory veto SIMULTANEOUSLY, erasing the early-reward reserve exactly when
protection is needed. Artifacts `research/findings/raw/lanes/curiosity/rev_ctrl_s42.json` and
`research/findings/raw/lanes/curiosity/rev_rect_s42.json`; seed-42 scout is decisive (the broken base gates are
structural), so 3-seed was correctly not promoted.

## The record-grounded next mechanism (verdict on the METHOD, not the capability)

<!--derived-->
Put the HISTORY back where the base finding says the separation lives: replace the per-ask omission→reserve with a
spiking estimate of LEARNING-PROGRESS SLOPE — a slow leaky integrator of the reward-omission signal (tau ~ several asks,
matched to the mastering horizon) whose DERIVATIVE gates the veto, computed by a phasic-minus-tonic / delay-line
differentiator (the SNc/LHb temporal-difference motif this project already uses for `snc_B − snc_neutral`). That is the
spiking form of the host expected-learning-progress (ELP) low-pass tracker the omission circuit was meant to replace but
which discarded the temporal integration. Biology: eligibility-trace / synaptic-tag decay on a behavioural timescale;
Oudeyer-Kaplan learning-progress intrinsic motivation. A slope signal distinguishes a slow-but-improving learner
(positive slope → protect) from an unlearnable concept (flat slope → veto) — the separation a scalar reserve cannot make.

## Honest scope

<!--derived-->
The base reward-omission veto circuit is UNAFFECTED (6/6 core / 5/6 composite GO) — this is a verdict on the
reserve-RESCUE refinement METHOD (a scalar reserve cannot beat the per-ask identifiability limit), not on the curiosity
capability. The next mechanism (learning-progress-slope differentiator) is named + biology-cited, not built. A clean
demonstration of the research-first discipline: the corpus check reframed the residual from "tune the reserve" to "the
signal is per-ask non-identifiable — you need history/slope", which the iteration then confirmed by breaking the base
gates.
