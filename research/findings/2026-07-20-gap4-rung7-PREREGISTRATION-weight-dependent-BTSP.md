# gap#4 RUNG 7 — PRE-REGISTRATION: weight-dependent bidirectional BTSP (filed BEFORE the run)

**Filed 2026-07-20 before any rung-7 result exists.** Seeds **1400-1405**, never used.

## Why this mechanism, and why it is different from the seven that failed

The literature reframe established that the seven prior mechanisms pursued an objective **biology never achieves**:
real CA1 field spacing is Poisson with a modal gap of ZERO while the potentiation window spans 75-150 cm, so the
signals always collide. Biology resolves the collision not by separating them but by making the **sign of the
weight change depend on the synapse's current weight** — weak potentiate, strong depress — yielding a target shape
independent of the starting state (Milstein 2021: `final vs initial Vm, r = 0.04`).

**PF-5 confirmed that fixed point exists on deployed traces** (starts of 0.3 and 2.0 converge to 1.31 and 1.36;
final maps correlate +0.997; zero synapses at the floor). This pre-registration tests the separate question PF-5
did NOT address: **does that fixed point produce better neighbour-contrast?**

## Parameters — published, not tuned

- `alpha_pot = 0.24`, `alpha_dep = 0.09` — **Milstein's published values**, on the NORMALIZED overlap, verified
  against 8,555,600 deployed per-synapse samples to populate all three zones (68.9 / 9.6 / 21.5%).
- `k_pot = k_dep = 0.02` — **set EQUAL**, so the fixed point `w* = k_pot*q_pot / (k_pot*q_pot + k_dep*q_dep)`
  is determined **purely by the published sigmoid ratio**, with no free parameter to fit.

## PRE-REGISTERED PREDICTIONS

0. **P0 — stage 1 forms:** `map_ok = 1` on >= 5/6. *(Both band attempts and the DoG died here.)*
1. **P1 — adjacent contrast (THE GOAL):** >= **1.60x**, on >= 5/6.
2. **P2 — far contrast retained:** >= 2.0x, on >= 5/6.
3. **P3 — the rule is load-bearing:** the `k_dep = 0` control reproduces the recorded baseline (~1.21x adjacent),
   6/6 — so any change is attributable to the mechanism.
4. **P4 — structural immunity holds in this config too:** no mass pinning at `w_min` (< 5% of synapses), 6/6.

**FALSIFIED if** P1 fails. In that case weight-dependent plasticity, despite being the mechanism biology actually
uses and despite its fixed point being confirmed here, **does not deliver neighbour-contrast in this task** — which
would be a genuinely informative negative, not another mis-implementation.

## The bar, restated

Weight contrast 1.73x currently yields 1.09-1.21x response contrast (transfer eats ~1.5x). P1's 1.60x response
target therefore implies **>= 2.5x adjacent weight contrast**.

## Cap

**One parameterization.** The alphas are published and `k_pot = k_dep` removes the remaining freedom. If P1 fails I
do **not** re-tune `k` or the thresholds — the verdict stands and the next question becomes whether the task itself
(evenly-spaced fields, which the literature says has no empirical basis) is generating the deficit.
