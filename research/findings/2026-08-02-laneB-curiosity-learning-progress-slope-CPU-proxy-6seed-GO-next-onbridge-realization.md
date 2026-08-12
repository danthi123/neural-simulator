---
type: finding
status: contributing
date: 2026-08-02
mechanism: curiosity-learning-progress-slope
runner: research/runners/_laneB_curiosity_learning_progress_slope_derisk.py
artifacts:
  - research/findings/raw/lanes/curiosity/lp_slope_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_6seed.json
  - research/findings/raw/lanes/curiosity/lp_slope_onbridge_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_memory_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_memory_6seed_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_s44_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_s100_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_confidence_s100_smoke.json
  - research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_confidence_s100_harder_nlearn8.json
---

# lane B curiosity: learning-progress slope separates slow-but-improving from unlearnable in a 6-seed CPU proxy

<!--derived-->
**One-line verdict.** The reserve-rescue negative was a real identifiability limit of a per-ask veto signal. A
phasic-minus-tonic learning-progress trace fixes that limit in the cheap lane-B proxy: six seeds all pass. Slow
learnable concepts remain askable while they improve, noisy/unlearnable concepts stay vetoed, and the slope lesion,
permuted-history, curiosity-lesion, and omission-only controls all collapse. This is a **CPU mechanism GO**, not yet
the full on-bridge BrainRegions realization.

## Mechanism

<!--derived-->
The runner keeps the existing Bogacz-Brown anti-Hebbian familiarity source and adds one per-concept history trace:

```text
progress read = 1 - post-ask novelty
fast pool     = fast EMA(progress)
tonic pool    = slow EMA(progress)
slope         = fast - tonic
```

The omission veto still accumulates on no-progress asks, but a concept whose slope remains positive is protected from
the veto. That is the Oudeyer-Kaplan learning-progress signal the earlier finding said was missing. The key anti-cheat
is history specificity: in the `permuted_history` arm, slow concepts read noisy traces and noisy concepts read slow
traces, so the mechanism should waste asks on noise and lose slow mastery.

## Frozen 6-seed result

<!--derived-->
Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_learning_progress_slope_derisk \
  --out research/findings/raw/lanes/curiosity/lp_slope_6seed.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | GO |
| seed GO count | 6/6 |
| slow concepts mastered | 5/5 on every seed |
| mean slow confidence | 0.534 |
| mean noisy confidence floor | 0.106 |
| mean protected slow asks | 73.2 |
| protected noisy asks | 0 total |
| omission-only slow mastery | 0/5 on every seed |
| slope-lesion slow mastery | 0/5 on every seed |
| curiosity-lesion asks | 0 on every seed |
| permuted-history noisy asks | 108.7 mean vs 8.2 real |

Per seed:

| seed | GO | slow conf | noisy floor | slow protected asks | noisy protected asks | omission mastered | slope-lesion mastered | permuted mastered | permuted noisy asks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | yes | 0.542 | 0.128 | 82 | 0 | 0 | 0 | 0 | 111 |
| 43 | yes | 0.528 | 0.094 | 64 | 0 | 0 | 0 | 0 | 109 |
| 44 | yes | 0.540 | 0.104 | 71 | 0 | 0 | 0 | 0 | 110 |
| 100 | yes | 0.517 | 0.102 | 68 | 0 | 0 | 0 | 1 | 106 |
| 101 | yes | 0.539 | 0.101 | 76 | 0 | 0 | 0 | 0 | 109 |
| 102 | yes | 0.539 | 0.109 | 78 | 0 | 0 | 0 | 1 | 107 |

## Interpretation

<!--derived-->
This cleanly answers the residual from the reward-omission veto finding. A scalar reserve cannot separate "slow but
learning" from "unlearnable", because their current ask can look identical. The missing variable is temporal history:
positive progress slope protects slow learners, flat slope leaves noisy concepts vetoed. The controls are load-bearing:
omission-only and slope-lesion both master 0/5 slow concepts, permuting the history burns the ask budget on noisy
concepts, and removing curiosity stops all asks.

## Honest scope and next step

<!--derived-->
This runner is deliberately CPU-cheap and does not import `sim`; it imports the real familiarity gate but implements
the phasic/tonic traces as numpy state. So it is not a full lane-B closure. The next build is the on-bridge realization:
add runner-local BrainRegions for the fast and tonic progress pools, gate the existing spiking omission-veto read from
their difference, and re-run the same six-seed controls before wiring the drive into the develop-loop teacher hook. The
first on-bridge readout smoke and the substrate-memory follow-up below update that ladder.

## On-bridge smoke follow-up

<!--derived-->
The first on-bridge promotion was added to `_curiosity_reward_omission_veto_derisk.py` behind `--lp-slope`: runner-local
`lp_fast`, `lp_tonic`, and `lp_gate` BrainRegions read the fast-minus-tonic history as a spiking protection signal for
the existing omission-veto candidate filter. The EMA history is still runner-side in this first bridge smoke, so the
result is not a full substrate-memory close.

Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --seeds 42 \
  --out research/findings/raw/lanes/curiosity/lp_slope_onbridge_smoke.json
```

Seed 42 LP-specific verdict was **LP_SLOPE_GO**, while the older full omission-runner verdict stayed **NO** because its
unrelated yoked and omit-lesion gates still fail under this slow-learner smoke. The LP-slope-specific evidence:

| metric | result |
|---|---:|
| real slow concepts mastered | 5/5 |
| omission-only slow concepts mastered | 2/5 |
| slope-lesion slow concepts mastered | 3/5 |
| protected slow asks | 10 |
| protected noisy asks | 0 |
| LP gate at slow protection | 9.95 Hz |
| final slow trace slope | 0.328 |
| final noisy trace slope | 0.003 |
| permuted-history protected noisy asks | 21 |
| base gap/want, ask-ratio, confidence, noisy-stop, curiosity-lesion, permutation, moat gates | pass |

Interpretation: the on-bridge readout/gate is doing the intended causal work in a one-seed smoke. The remaining lane-B
work is to move the EMA history itself out of runner state, then run the full six-seed control set.

## Substrate-memory follow-up

<!--derived-->
The next promotion added `--lp-substrate-memory` to `_curiosity_reward_omission_veto_derisk.py` as an opt-in on top of
`--lp-slope`. In this mode the runner no longer recalls LP history from a Python EMA table for candidate protection.
Instead, it writes concept-specific history into plastic cue->`lp_fast` and cue->`lp_tonic` pathways, then drives the
cue and reads the spiking `lp_gate` pool as the protection signal. The teaching scalar is still runner-supplied from
positive learning progress, so this is a substrate-memory promotion smoke, not the final fully endogenous LP reward.

Seed 42 smoke:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --seeds 42 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_memory_smoke.json
```

That single-seed smoke was **LP_SLOPE_GO**: real mastered 5/5 slow concepts vs omission-only 2/5 and slope-lesion 3/5,
with 36 protected slow asks at 56.65 Hz and 0 protected noisy asks. The learned memory rates separated slow from noisy
concepts (slow fast/tonic 265.74/282.41 Hz; noisy fast/tonic 37.81/77.93 Hz), and permuted history redirected
protection to noisy concepts.

The smoke-scale six-seed promotion did **not** hold:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_memory_6seed_smoke.json
```

Aggregate: **1/6 GO** and **1/6 LP_SLOPE_GO**. Per seed:

| seed | GO | LP_SLOPE_GO | real mastered | omission-only | slope-lesion | slow protected | noisy protected | final slow gate Hz | final noisy gate Hz | primary failure |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | yes | yes | 5 | 2 | 3 | 36 | 0 | 58.33 | 0.77 | passes |
| 43 | no | no | 3 | 5 | 5 | 19 | 0 | 25.00 | 0.77 | substrate LP hurts vs controls |
| 44 | no | no | 5 | 4 | 5 | 12 | 25 | 54.63 | 71.76 | noisy concepts protected |
| 100 | no | no | 5 | 4 | 5 | 0 | 0 | 0.00 | 0.00 | LP gate silent |
| 101 | no | no | 5 | 5 | 5 | 11 | 19 | 24.07 | 47.84 | noisy concepts protected / controls tie |
| 102 | no | no | 2 | 3 | 1 | 8 | 0 | 14.81 | 6.94 | protection too weak vs omission-only |

Interpretation: the substrate-memory path can express the intended separation, but the current operating point is
seed-fragile. The failures are not one knob: one seed is silent, two protect noisy concepts, and multiple seeds do not
beat omission-only or slope-lesion controls. Do not promote this to a full non-smoke six-seed run as-is. The next Lane B
mechanism should add a seed-robust neural/homeostatic equalizer or a less intrusive per-concept threshold read from
tonic LP before attempting another promotion.

## Homeostatic-read scout

<!--derived-->
An opt-in `--lp-homeostatic-read` scout was added after the 1/6 result. It requires `--lp-substrate-memory` and keeps
the plastic cue->`lp_fast`/`lp_tonic` substrate memory unchanged, but replaces the global `lp_gate` firing floor with a
per-concept fast/tonic memory-ratio read (`fast_over_tonic_x100`, with an 80 Hz fast-rate floor and an 87 ratio floor).
This is a diagnostic readout scout, not a final fully-neural comparator claim.

The targeted seed-44 scout directly tested the prior noisy-protection failure and passed:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --lp-homeostatic-read --seeds 44 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_s44_smoke.json
```

Seed 44 became **LP_SLOPE_GO**: real mastered 5/5 vs omission-only 4/5 and slope-lesion 4/5, protected slow asks 36,
protected noisy asks 0, final slow ratio score 87.5 vs noisy 0.0, and permuted history protected noisy 37.

The silent-readout stress seed stayed negative:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --lp-homeostatic-read --seeds 100 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_s100_smoke.json
```

Seed 100 was **NO**: real mastered 5/5 and protected 7 slow asks with 0 noisy-protected asks, but the LP lesion still
mastered 5/5 and the slow/noisy substrate memory slopes were not separated enough (-0.583 vs -0.571), so LP was not
load-bearing. This banks the scout as **PARTIAL**, not a promotion path. The ratio read can repair the noisy
false-positive mode, but the next mechanism must make the LP memory causally stronger against matched no-read controls
without losing the noisy guard.

## Confidence-teaching scout

<!--derived-->
A second opt-in scout added `--lp-memory-teach confidence`, still defaulting to the prior positive-LP teaching for all
existing artifacts. This writes cue->LP memory from post-ask confidence, matching the original CPU proxy's history
quantity, while the homeostatic ratio read guards noisy concepts.

Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --lp-homeostatic-read --lp-memory-teach confidence --seeds 100 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_confidence_s100_smoke.json
```

This improved the seed-100 substrate memory and noisy guard but still did **not** pass: real mastered 5/5, protected 35
slow asks, protected 0 noisy asks, and separated final slow/noisy ratio scores 102.9 vs 45.0 with slopes +0.060 vs
-0.540. The failing control is load-bearing: slope-lesion also mastered 5/5, so the LP read is not causally required
on that seed. Verdict: confidence teaching is a useful scout for memory strength, but the next pass needs a harder or
better-matched no-read control / operating point where LP can demonstrate causal benefit, not another threshold-only
tweak.

## Harder n-learn scout

<!--derived-->
The next cheap scout made the seed-100 task larger to look for a non-saturated matched no-read operating point:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._curiosity_reward_omission_veto_derisk \
  --smoke --lp-slope --lp-substrate-memory --lp-homeostatic-read --lp-memory-teach confidence \
  --n-learn 8 --n-noisy 3 --n-turns 200 --ask-budget 50 --seeds 100 \
  --out research/findings/raw/lanes/curiosity/lp_slope_substrate_homeostatic_confidence_s100_harder_nlearn8.json
```

This also stayed **NO**. The larger configuration (`n_learn=8`, `n_noisy=3`, `n_turns=200`, `ask_budget=50`, `d=512`)
did not create a clean causal window: real, omission-only, and yoked controls all mastered 8/8 slow concepts, while
the slope-lesion mastered 4/8. The confidence-teaching substrate memory still protected 10 slow asks but reopened the
noisy guard with 21 protected noisy asks; the permuted-history arm mastered 7/8 and protected 34 noisy asks. Verdict:
simply increasing the slow concept count is not a clean matched no-read operating point. Do not continue Lane B by
larger-task threshold tuning; the next Lane B attempt needs a genuinely better matched non-saturating control that does
not increase noisy cross-talk, or the CPU fallback should switch to Lane C.
