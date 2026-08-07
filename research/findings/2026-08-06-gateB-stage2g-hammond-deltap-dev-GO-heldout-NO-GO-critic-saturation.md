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

## Why it fails held-out (the isolated residual — a numerical defect, not a mechanism gap)

The **Carandini-Heeger divisive normalisation saturates** on low-pooled-baseline
seeds: the denominator (pooled striatal baseline) approaches zero, the normalised
value blows up / goes NaN, and the seed's contingency read collapses. On dev this
bit 730601; on held-out it bites 730704 and 730705 (the two steer failures), and
`NaN present: True` in the artifact. The mechanism that supplies contingency is
correct — the divergence mean stays 0.79 on held-out — but the normalisation lacks
a floor, so a couple of seeds per partition fall into the saturation regime.

## Next mechanism (no-defer — a bounded normalisation, then re-validate on the pool)

Give the divisive normalisation a **biological floor**: a semi-saturating
(Naka-Rushton) denominator `baseline + σ` with a non-zero semi-saturation constant
σ (Heeger's original form already carries it — the current code dropped it), or a
tonic-inhibition floor on the pooled-baseline population so it cannot reach ~0.
This removes the NaN without changing the contingency mechanism, and should carry
the dev 5/6 to held-out. Then re-run dev+held-out as ONE hands-off sweep on the
mini-PC pool (healthy, 36 cores) via `tools/run_and_aggregate` — the parent runs
it, no orphan. The Gate B credit-assignment mechanism is otherwise complete; this
is the last isolated defect between it and a capability GO.
