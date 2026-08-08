---
type: finding
status: contributing
date: 2026-08-08
mechanism: pragmatic-success-readback-graded-wta-value-critic-speaker
lane: D-pragmatics
runner: research/runners/_pragmatic_success_readback_leg2_v2_derisk.py
builds_on: research/findings/2026-08-08-pragmatics-readback-leg2-WTA-speaker-NEGATIVE-value-critic-fallback.md
artifacts:
  - research/findings/raw/_pragmatic_success/leg2_v2_oracle_probe.json
  - research/findings/raw/_pragmatic_success/leg2_v2_summary_6seed.json
  - research/findings/raw/_pragmatic_success/leg2_v2b_critic_probe_6seed.json
  - research/findings/raw/_pragmatic_success/leg2_v2b_critic_probe_s42.json
---

# Leg 2 v2 — the oracle precondition is RESOLVED (1.0/6), the read-back trains above its controls, but convergence is a NEGATIVE: BOTH the actor-WTA and the learned critic value under-separate across seeds — the DA-trained value signal is too small/noisy, so value AMPLIFICATION is the next need

<!--derived-->

**One line.** The Leg-2 v1 negative had two named blockers: the WTA choice was not weight-controllable (a 30×
oracle weight scored 0.167) and a naive DA→three-factor loop over-reinforces the early-active utterance. v2
RESOLVES the first — a GRADED soft-WTA whose winner tracks the afferent scores **oracle-weight acc 1.000 on 6/6
seeds** (committed code path, closing the v1 provenance gap) — and the coincidence-contingent DA + spiking
value-critic DOES train a policy that beats both controls (trained 0.444 > untrained 0.222 > yoked 0.167; the
YOKED decisive tooth collapses to 0.000 on half the seeds). But convergence is a NEGATIVE: trained never reaches
the 0.85 criterion, and a committed critic-value diagnostic shows the deficit is NOT merely the readout. Across 6
seeds the learned critic value (host-argmax ceiling) scores only **0.556** and the actor-WTA **0.500** — both near
chance (0.333), and on seed 100 the critic is 0.000 while the actor is 1.000 (they disagree; neither is reliable).
The DA-trained value/policy differential stays too small and too heterogeneity-noisy to separate the aligned
utterance robustly. This is a teeth-backed convergence NEGATIVE with the residual mapped: the next need is
value-signal AMPLIFICATION (contrast / divisive normalization), not just a different readout. NO `sim/` edit;
reuse-by-import; additive/default-off.

**Single-seed caution (earned here):** seed 42 alone showed critic-argmax 1.000 vs actor 0.667 — a clean "the
value is right, the readout loses it" story. The 6-seed run REFUTED that headline (critic 0.556, actor 0.500). The
seed-42 result is reported below only as the per-seed example that motivated the diagnostic, NOT as the verdict.

## What was built (and the two v1 blockers it addresses)

<!--derived-->

One spiking bridge, all populations co-resident, the Leg-1 coincidence evaluator FROZEN and re-used:

- `intent[K]` — one-hot communicative goal (world/goal boundary as spikes).
- `utter[K]` — the SPEAKER: K assemblies competing through a shared FS pool in the **GRADED regime**
  (`UTT_FS_W=4`, `FS_UTT_W=4`, tonic `UTT_DRIVE_PA=0`) so the late-window rate TRACKS the intent→utterance
  afferent rather than latching on the first assembly to ignite (v1's hard latch scored 0.167). Winner = highest
  late-window rate after neural lateral inhibition — a NEURAL WTA, **not** `np.argmax` over an imported table.
- `crit[K]` — the SPIKING VALUE CRITIC: `intent[t]→crit[u]` plastic; `rate(crit[u]) = V(intent t, u)`, trained by
  the same actor-critic delta; the action-conditioned baseline.
- `belief[K]` — the LISTENER response (RSA social environment, `build_rsa_bridge`).
- `success[K]` — the FIXED Leg-1 coincidence detector; success = mean rate.

TRAINING is actor-critic with coincidence-contingent, group-scoped DA (volume transmission, never `scope=all`):
eligibility builds from pre×post COACTIVITY (`reward_eligibility_from_coactivity=True`; STDP/Hebbian OFF), the RSA
listener responds → coincidence success `s`, and `delta = REWARD_GAIN·(s − V)` is delivered once as
`current_reward_signal`; the engine converts standing eligibility to Δw. Two additional levers were built against
the convergence failure (see below): **action-localized credit** (`--localize-credit`: a motor-commitment window
that wipes the leaky deliberation eligibility and rebuilds it only on the EXECUTED utterance) and **executed
epsilon-greedy** (the explored action is actually SPOKEN, so every (intent,utterance) pair is sampled — the prior
bias-current exploration rarely flipped the WTA, so losing utterances were never executed and never learned).

## Result (6 seeds: 42 43 44 100 101 102, numpy-CPU)

<!--derived-->

| metric | mean | per-seed | GO gate | reads |
|---|---|---|---|---|
| oracle-weight acc (PRECONDITION: readout weight-controllable) | **1.000** | 1.0 ×6 | ≥ 0.85 | **PASS** — v1 latch scored 0.167 |
| trained choice acc | 0.444 | .667/.333/.333/.333/.667/.333 | ≥ 0.85 | FAIL criterion (but > controls) |
| untrained choice acc (mapping is LEARNED, not wired) | 0.222 | — | ≤ 0.55 | PASS |
| yoked choice acc (DA decoupled from choice — decisive tooth) | 0.167 | 0.0 on 3/6 seeds | ≤ 0.55 | PASS — yoked does not train |
| critic_argmax_acc (DIAGNOSTIC of learned-value separability, 6-seed) | 0.556 | 1.0/.667/.333/0.0/.667/.667 | — | near chance; NOT robust |
| actor_wta_acc (same runs, 6-seed) | 0.500 | .667/0.0/.333/1.0/.667/.333 | — | near chance; disagrees w/ critic |

The Verdict framework prints **UNDEFINED** because it gates GO with a `require(trained ≥ 0.85)`; the instrument
itself FUNCTIONED (the oracle precondition passed, the above-chance floor passed, both controls behaved), so the
scientific reading is a well-posed convergence NEGATIVE, not a fabricated score: contingent-DA training moves the
policy ABOVE both controls (0.444 > untrained 0.222 > yoked 0.167 — and yoked collapses to 0.000 on seeds
42/44/101), i.e. learning happens and is contingent on the actual choice, but it does not reach criterion.

## Why it does not converge — the committed critic-value diagnostic

<!--derived-->

The coincidence-success landscape itself is well-separated and correctly labelled (per-(intent,utterance) success
on seed 42: aligned utterance ≈ 0.046 vs off-target ≈ 0.027–0.033 — the argmax-belief label matches the
success-optimal utterance for every intent, so the failure is NOT a labelling artifact). The `--critic-probe`
committed diagnostic trains the v2b path (localized credit + executed epsilon-greedy) and then reads, per intent,
the actor utterance-WTA winner vs the learned critic value V(intent,u).

**Seed 42 (the per-seed example that motivated the probe):** critic_argmax 1.000 vs actor 0.667 — for the failing
intent the actor rates are near-tied (0.0328/0.0328/0.0322 — heterogeneity breaks the tie to the wrong utterance)
while the critic value separates (0.0433/0.0517/0.0408). That looked like "the value is right, the readout loses
it."

**6-seed reality (the verdict):** critic_argmax **0.556**, actor **0.500** — both near chance (0.333). On seed 100
the actor scores 1.000 while the critic scores 0.000 (they disagree, and the critic is INVERTED). So the learned
value does NOT robustly rank the aligned utterance across seeds; the DA-trained value/policy differential is too
small and too heterogeneity-noisy. The deficit is therefore NOT localised to the actor-WTA readout — the critic
value under-separates too. `critic_argmax_acc` is a HOST-ARGMAX DIAGNOSTIC (explicitly NOT a neural readout and NOT
a shippable choice — a host argmax is a forbidden shortcut for the speaker CHOICE); even that noiseless ceiling
sits near chance, so a neural WTA over the critic populations would not rescue convergence without amplification.

## Honest scope

<!--derived-->

Positive and measured: the oracle-weight precondition is resolved (1.0/6, committed `--oracle-probe`, closing the
v1 provenance gap whose probe was prose only); the read-back is a GENUINE new attempt (contingent-DA three-factor
+ spiking critic + two convergence levers — action-localized credit and executed epsilon-greedy), NOT a re-run of
the v1 WTA-readout; the yoked decisive tooth passes (DA decoupled from the choice does not train the policy — it
collapses to 0.000 on 3/6 seeds). Negative and teeth-backed: trained convergence does not reach the 0.85 criterion
(6-seed 0.444, above both controls but near chance), and the committed critic diagnostic shows the deficit is NOT
localised to the readout — the learned value under-separates too (critic-argmax 0.556, actor 0.500 across 6
seeds). No metric is lifted from a failing arm, and the single-seed critic-argmax 1.000 is explicitly NOT reported
as the verdict (the 6-seed run refuted it). numpy-CPU; NO `sim/` edit; only the reward-modulated three-factor rule
learns (actor `intent→utter` + critic `intent→crit`); the coincidence evaluator + FS competition are frozen.

## Next method — the SAME capability; the residual is a small/noisy learned differential

<!--derived-->

The diagnostic reframes the residual: both the actor-WTA and the learned critic value sit near chance across seeds
(0.500 / 0.556), so the bottleneck is NOT the readout choice alone — the DA-trained value/policy DIFFERENTIAL is
too small (success values 0.027–0.052; gaps ~0.01–0.02) and too heterogeneity-noisy to separate reliably. The
oracle probe proves the graded FS competition IS weight-controllable given an ~8× differential; the open problem
is producing that separation from the reward. Concrete next methods, in order, still no `sim/` edit:
(1) AMPLIFY the value signal before the competition — divisive/contrast normalization across the utterance (or
critic) populations, or a longer reward-integration window, to turn a ~1.2× value gap into a WTA-resolvable one;
(2) sharpen the reward differential itself — a larger `coincidence_gain`/`ITEM` so aligned-vs-misaligned success
separates more (Leg-1 widened ITEM to 80 for exactly this reason), and/or a higher `REWARD_GAIN` with the critic
baseline holding contingency; (3) only then a neural WTA over the amplified critic value populations. The
capability (read success back to shape speaking) is not abandoned — the actor-WTA readout is banked as
reaching-only-controls, and value-signal amplification is next.

## Reproduce

```bash
# ORACLE-WEIGHT acceptance gate (the crux precondition; committed code path, 6 seeds):
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_v2_derisk --oracle-probe \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_v2_oracle_probe.json
# 6-seed training verdict (each seed: oracle probe + contingent train + yoked train):
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_v2_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_v2_summary_6seed.json
# committed CRITIC-VALUE-SEPARABILITY diagnostic (v2b: --localize-credit + executed epsilon-greedy), 6 seeds:
SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_v2_derisk --critic-probe \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_v2b_critic_probe_6seed.json
```
