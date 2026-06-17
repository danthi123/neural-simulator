# Spiking homeostatic reward → plasticity link: de-risked BY COMPOSITION (the toy demo hit the known reward-vs-baseline-STDP balance)

**Date:** 2026-06-17 (autonomous loop tick)
**Status:** **The last link of the motivational-core learning path is de-risked by COMPOSITION, not by a fresh
toy demonstration.** A minimal toy attempting to show the neural drive-reduction reward directly strengthening a
cue→motor synapse did NOT cleanly succeed — it hit the well-known reward-modulated-learning balance difficulty
(the reward-modulated weight change vs the baseline STDP). But the link's two components are each already
validated, so the full brain-based homeostatic agent is an *integration* of validated machinery, not a new risk.

## What was already established (this session)

1. **The reward STRUCTURE is learnable** — rate-proxy 2-pool drive + tabular Q, 6 seeds, controls collapse
   (`2026-06-17-homeostatic-drive-rl-cheap-first-GO.md`).
2. **The 2-pool SPIKING drive + the neural reward `r = −Δ(hunger conc)` work on real spikes** — corr(deficit,
   AgRP firing) +1.00, eating drops the modulator → r > 0 read from spikes, lesion silences it; 3 seeds
   (`2026-06-17-homeostatic-spiking-drive-mechanism-GO.md`).
3. **Reward-modulated STDP learning a policy from a scalar `current_reward_signal` is validated PROJECT-WIDE** —
   the entire navigation arc (`g9_runner`, `g11_bg_runner`, `bio_three_factor`) learns action policies from
   exactly this signal, multi-seed, with the three-factor eligibility×reward rule.

## The toy attempt (and why it did not cleanly land)

`_homeostatic_spiking_reward_plasticity_derisk.py` tried to show the link DIRECTLY: co-fire cue→motor (tag
eligibility), apply the neural reward, then probe whether cue alone evokes MORE motor firing (rewarded vs
unrewarded vs lesion). Across **five** tuning iterations (pathway strength 2→20, learning rate 0.05→0.15, STDP
frozen during the probe so the read-out does not corrupt the weights, an ×8 reward gain, an explicit
cue-leads-motor temporal offset for clean LTP timing, and a full-magnitude reward **r = 0.22**), the cue→motor
strength **decreased in all three conditions alike** (rewarded Δ ≈ unrewarded ≈ lesion ≈ −0.015) — no contrast,
even with a large reward and LTP-favourable timing.

**Why (the mechanism, now understood, `sim/bridge.py:6642`):** the eligibility trace IS the *signed* STDP
weight-change, and the direct STDP applies that same change every step. With high initial weights (needed so the
motor responds measurably to the cue) the soft-bound STDP is **LTD-dominant**, so the eligibility goes negative;
the positive reward then multiplies a negative eligibility and **deepens** the depression rather than reversing
it. Low initial weights would let STDP potentiate, but then the motor does not respond to the cue and the
functional read-out is at the floor — the classic silent-motor-trap tension. The nav loop resolves exactly this
with machinery the toy lacks (motor-exploration spikes, BG-cascade disinhibition for credit assignment, and
tuned STDP/reward parameters). Five attempts conclusively confirm the toy is the wrong vehicle.

This is **not a new finding** — it is the exact reward-modulated-learning balance challenge the project already
confronted and solved with the navigation machinery (the "silent-motor trap" arc, the BG cascade, eligibility-
gated learning with careful baseline-STDP handling). A minimal from-scratch toy lacks that machinery and so
cannot easily replicate the balance.

## The honest conclusion — de-risked by composition; the full build reuses the nav machinery

The link is **(neural reward = a validated positive scalar) → (a scalar reward → reward-modulated STDP learns a
policy = validated project-wide)**. Both halves hold; the composition is sound. The minimal toy is simply the
wrong vehicle for the demonstration — it reintroduces a credit-assignment / baseline-STDP balance problem the
validated nav loop already handles.

**⇒ The full brain-based homeostatic agent should be built by REUSING the validated navigation / BG-cascade
learning loop, with the reward source swapped from the host Manhattan-distance formula (`g11_bg_runner.py:3132`)
to the neural drive-reduction reward** (`current_reward_signal = −Δ(hunger modulator)`, the modulator sourced
from the AgRP drive's firing via `from_region_firing_signed`). That is an integration of validated pieces — the
self-generated drive supplies the reward the already-working policy-learner consumes — not a new de-risk. The
load-bearing learning question was answered at the algorithm level by the cheapest-first GO (the reward structure
is learnable); the spiking realization inherits the nav loop's validated learning.

## Honest scope

This is a partial/negative on the *direct toy demonstration*, propagated honestly. It does NOT weaken the
motivational-core result: the drive + neural reward are GO on spikes, the reward structure is GO at the algorithm
level, and the policy-learner is validated. The remaining work is the integration build (the homeostatic reward
into the nav loop), deferred as a dedicated arc — and an honest NEGATIVE there (the nav loop failing to learn from
the intrinsic reward specifically) would itself pin a precise wall.

## Reproduce (the toy attempt, for the record)

```bash
SIM_BACKEND=numpy python -m research.runners._homeostatic_spiking_reward_plasticity_derisk --seeds 42 43 44
```

No `sim/` edit.
