---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2g-true-hammond-deltap-withhold-baseline-plus-homeostatic-critic
backend: numpy
runner: research/runners/_vocal_gateb_stage2g_hammond_deltap.py
builds-on: 2026-08-06-gateB-stage2f-contingency-gated-exploration-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2g_hammond_deltap/numpy.json
  - research/findings/raw/gateb_stage2g_hammond_deltap/heldout_numpy.json
---

# Gate B Stage 2g: true Hammond ΔP fixes both named residuals (dev-GO 5/6) but the critic normalisation saturates on held-out (NO-GO 4/6)

## Verdict

**STAGE2G_NO_GO at the reliability bar.** Development is a clean earned GO
(`numpy.json`: steer **5/6**, D_contingent−yoked 1.11, reversal P(B) 0→1.0, all
preconditions ok, no NaN). Held-out (730701–706, `heldout_numpy.json`, run as a
PARENT job so it could not orphan) is **4/6** (per-seed
`[T,T,T,F,F,T]`, D_contingent−yoked 0.79, reversal PASS) **with a NaN present** —
so the dev 5/6 OVERFITS and the mechanism does not clear the bar. This is the
same overfit shape as the v6-replay 2-seed GO that dissolved on multiseed; the
dev-only artifact says `STAGE2G_GO` but that is the DEV partition, not a
capability verdict.

## What WORKS (the real advance — both Stage-2f residuals fixed, brain-based)

Stage-2f plateaued at steer 4/6 because its D1−D2 contrast estimated
P(reward|action), not the Hammond ΔP = P(reward|action) − P(reward|no-action),
and a single global `VALUE_GAIN` mis-signed the RPE on heterogeneous seeds.
Stage-2g fixes BOTH, and on dev both target seeds flip to pass:

- **730605** (was D_yoked +0.55, below the gate) → pass. Fix: interleaved
  **no-action / withhold trials** charge the previously-inert `dopamine_S`
  channel into a tonic average-reward integrator (Niv-style), and
  `V(withhold)=gain·[DA_S]` enters the reward-expectation baseline so the DA
  production computes the TRUE Hammond ΔP. Neural (a spiking-rate readout), not a
  host P(reward|·) counter. The `withhold_lesion` control confirms it is
  load-bearing (D_yoked returns).
- **730602** (was D_contingent 0, never exploits) → pass. Fix: a **homeostatic
  critic** — Carandini-Heeger divisive normalisation of the value estimate by the
  pooled striatal baseline — keeps the RPE correctly signed across heterogeneous
  seeds while preserving the locked seeds (730603/730604).

Reversal P(B) 0→1.0, acquisition/expression lesions (Δ0.45 / 0.60), and the
reward-OFF byte-identical guard all still PASS on both partitions.

## Why it fails held-out (VERIFIED against code + artifact — NOT normalisation saturation)

⛔ CORRECTION: an earlier draft of this finding blamed "Carandini-Heeger divisive
normalisation saturation". That was WRONG — traced from a preliminary agent
summary without checking the code. `_value_action` line 190 already floors the
denominator (`max(r0_total, 1.0)`), so it cannot divide by zero. The real cause,
localised in `heldout_numpy.json`:

- The runner defines `target_rate = target_hits / n_acted if n_acted else nan`
  (lines 302, 317). A metric is NaN precisely when the brain emitted **ZERO clean
  actions** in a test block. On **730704** (`baseline_p0=0.0` — a MAXIMALLY-biased
  seed) `contingent_p0_reward1` is NaN → the brain FROZE (n_acted=0) in that block.
- **730705** (`reward_count_reward1=0`) never sampled/was-rewarded on the target
  action → D_contingent=0.

Both held-out failures are the SAME residual: on MAXIMALLY-biased seeds
(`baseline_p0 ∈ {0,1}`) the directed-novelty + uncertainty exploration does not
GUARANTEE the brain samples BOTH actions — it either freezes (NaN) or never tries
the target. This is the identical extreme-bias exploration limit that was Stage-2e's
sole double-failure (730604), now surfacing on 2 held-out seeds. It is a
**BEHAVIOURAL exploration residual, not a numerical/normalisation defect.**

## Next mechanism (no-defer)

Guarantee sampling of BOTH actions on extreme-bias seeds. Options: a neural
**forced-sampling / ε-floor** — a minimum per-action exploration drive from the
novelty/uncertainty signal that does NOT decay while an action remains un-sampled
(count-based `sqrt` novelty already exists; make its floor un-satiable until each
action has ≥K samples) — or a stronger habituation drive that overcomes
`baseline_p0 ∈ {0,1}`. The n_acted=0 case (a frozen brain) should read as UNDEFINED
— itself the signal that exploration failed on that seed. The contingency mechanism
(withhold-ΔP + critic norm) is correct and complete; the last gap is exploration
coverage on the most-biased seeds. Re-validate dev+held-out as one hands-off pool
sweep via `tools/run_and_aggregate`.
