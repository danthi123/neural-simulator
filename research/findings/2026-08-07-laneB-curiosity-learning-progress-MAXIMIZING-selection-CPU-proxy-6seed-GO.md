---
type: finding
status: contributing
date: 2026-08-07
mechanism: curiosity-learning-progress-maximizing-selection
runner: research/runners/_laneB_curiosity_lp_max_selection_derisk.py
artifacts:
  - research/findings/raw/lanes/curiosity/lp_max_selection_smoke.json
  - research/findings/raw/lanes/curiosity/lp_max_selection_6seed.json
---

# lane B curiosity: learning-progress-MAXIMIZING ask selection is immune to the noisy-TV trap by construction — 6-seed CPU-proxy GO

<!--derived-->
**One-line verdict.** Every existing lane-B runner selects the ask by `argmax want` (a novelty-driven
VTA/striosome salience read) and uses learning progress (LP) ONLY as a protective veto gate. The core
Oudeyer-Kaplan intrinsic-motivation thesis — that the drive should MAXIMISE learning progress itself — had never
been tested here as the ask-SELECTION drive. Tested as a CPU numpy proxy, it is a clean 6-seed GO: LP-max
selection allocates its ask budget to the learnable frontier and masters all learnable concepts with ~3-7 wasted
noisy asks, while the project's current novelty-max selector is captured by unlearnable high-novelty noise
(masters 3/6, ~80 noisy asks, never masters all). The advantage is LP-specific: permuting the trace redirects
the budget onto noise (masters 1-2/6) and lesioning LP loses the noise-avoidance edge. This is a CPU-proxy GO,
not the on-bridge spiking realisation, and it is a PROACTIVE complement to the already-GO reactive omission veto.

## Why this is distinct and un-tried (re-anchor result, 2026-08-07)

<!--derived-->
The lane-B record is mixed across epochs and the concluded pieces were NOT re-derived. Concluded: the
reward-omission veto (6-seed GO, `2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed.md`); the
reserve-rescue lever (genuine honest-negative, `2026-08-02-laneB-curiosity-reserve-rescue-REFUTED-...md`); the
LP-slope DIFFERENTIATOR as a protective gate (CPU proxy GO 6/6 + a first on-bridge `LP_SLOPE_GO` smoke,
`2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-...md`); and the on-bridge substrate-memory
promotion of that gate (NEGATIVE/PARTIAL 1/6, blocked pending a design decision on a seed-robust non-saturating
no-read control — deliberately NOT continued here). Reading both runners confirmed the gap: in
`_laneB_curiosity_learning_progress_slope_derisk.py` and `_curiosity_reward_omission_veto_derisk.py` the ask is
chosen by `argmax want`/novelty, and LP only ever GATES the veto. LP-as-the-SELECTION-drive — the actual
Oudeyer-Kaplan claim, whose whole point is proactive immunity to the "noisy-TV" problem — was never tested.

## Mechanism and controls

<!--derived-->
CPU-cheap, no `sim/` import; reuses the real Bogacz-Brown anti-Hebbian familiarity gate. The world has three
fast-learnable concepts (low observation jitter), three slow-learnable (high jitter, many asks to master), and
three unlearnable noisy concepts (a fresh random code every render, so novelty stays ~maximal forever). Each turn
one concept is asked, imprinted, and its progress read (`1 - post-ask novelty`) updates a phasic (fast EMA) minus
tonic (slow EMA) LP slope. Selection score = arm-specific exploitation + an identical count-based exploration
bonus (`beta / sqrt(count+1)`, novelty-agnostic), so the ONLY difference between arms is the exploitation term:

- `real` (LP-max): exploit = `max(0, LP slope)` — pick what is improving fastest.
- `novelty_max` (like-for-like control = the current selector's drive): exploit = novelty.
- `lp_lesion`: exploit = 0 (pure uniform exploration) — LP removed.
- `permuted_lp` (anti-cheat): exploit = LP slope read from a MIS-ASSIGNED concept (learnable reads noisy, noisy
  reads learnable) — same LP magnitudes, wrong owner.
- `novelty_min`: exploit = `-novelty` — shows LP-max is not merely "avoid novelty".

## Frozen 6-seed result

<!--derived-->
Command (deterministic; re-runs are byte-stable):

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_selection_derisk \
  --out research/findings/raw/lanes/curiosity/lp_max_selection_6seed.json
```

| arm | learnable mastered (of 6) | noisy asks | masters ALL within budget |
|---|---:|---:|---|
| real (LP-max) | 6/6 every seed | 3-7 | yes, in 70-78 asks |
| novelty-max (current selector) | 3/6 every seed | 78-83 | no (90 = budget) |
| lp-lesion (uniform) | 6/6 | 30 every seed | yes, but 4-10x more noise |
| permuted-lp (anti-cheat) | 1-2/6 | 85-88 | no |
| novelty-min | 1-3/6 | 0-3 | no |

Aggregate verdict: **6/6 seeds GO**. Gate flags pass 6/6: g1 mastery, g2 noise-avoidance (real noisy asks
<= 0.5x novelty-max), g3 efficiency (real masters all, novelty-max never does), g4 LP-load-bearing (lp-lesion
noisy asks >= 1.5x real), g5 LP-specificity (permuted-lp redirects to noise / breaks mastery). The final
LP slopes separate cleanly: fast/slow learnable ~ +0.13 each, noisy ~ +0.01 to +0.06.

## Interpretation and honest scope

<!--derived-->
The result is exactly the Oudeyer-Kaplan prediction: novelty-max curiosity is captured by an unlearnable
high-novelty stimulus (the noisy concept's novelty never falls), so it burns ~90% of its budget on noise and
only masters the trivially-easy fast concepts. LP-max is immune BY CONSTRUCTION — a noisy concept yields ~zero
progress slope, so it is never a high-value target and no reactive veto is needed to escape it. Three honest
caveats. (1) This is a CPU numpy PROXY: the LP traces are numpy EMAs, not spiking pools — like the LP-slope
differentiator that preceded it, the on-bridge spiking realisation is the next step and is NOT claimed here. (2)
The lp-lesion (uniform) arm still MASTERS all six given the budget, so LP-max's demonstrated value is proactive
NOISE AVOIDANCE and budget EFFICIENCY, not mastery-possibility per se. (3) This is a PROACTIVE COMPLEMENT to the
already-GO reactive omission veto, not a replacement — the two attack the noisy-TV problem from opposite ends
(select-away vs veto-after).

## Next step (named, not built)

<!--derived-->
On-bridge spiking realisation: drive ask selection from the spiking `lp_fast`/`lp_tonic` progress pools already
built for the LP-slope veto gate (read fast-minus-tonic as the SELECTION value, not just the veto-protection
signal). Caveat carried forward: the substrate-memory promotion of those pools is known-fragile (NEGATIVE/PARTIAL
1/6), so the on-bridge selection realisation inherits that open substrate-expressivity question — it should be
attempted only alongside, or after, a seed-robust homeostatic/normalised LP readout.
