---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/sweetspot/gap4_sweetspot_6seed_aggregate.json
---

# gap#4: the on-bridge SWEET SPOT is LOCATED (forward-representable AND reservoir-fails) — and deep credit CANNOT train the hidden layer even there (6-seed)

**One-line verdict:** the record's central on-bridge blocker was that the two preconditions for a deep-credit test
never co-held — at depth-2 the reservoir carries the task, at depth≥3 the forward collapses. This finding LOCATES
the operating point where they DO co-hold (`n_prop=3`, `hidden=48`, `oracle-epochs=200`): the **oracle fits (mean
0.957)** so the forward is representable, and a **frozen random reservoir fails (mean 0.262)** so credit is
actually needed. **At that sweet spot, neither credit rule trains the hidden layer** — fixed-DFA `train_acc`
0.349, learned-instructive 0.340, both far below oracle 0.957, held-out ≈ the failing reservoir. This is the
first clean 6-seed deep-credit verdict on the production spiking bridge at a *valid* operating point.

Artifact: `research/findings/raw/gap4/sweetspot/gap4_sweetspot_6seed_aggregate.json`.

## The sweet spot + the failure (6 seeds {42,43,44,100,101,102}, w_clip 4000 controlled)

| metric | fixed-DFA | learned microcircuit |
|---|---|---|
| held-out inherit (mean) | 0.275 | 0.235 |
| **train_acc (mean)** | **0.349** | **0.340** |
| beats the failing reservoir 0.262? | 4/6 (by hairs) | 2/6 |
| learned beats fixed? | — | **1/6** |

oracle 0.957 (forward representable), reservoir 0.262 (fails). The `deep_credit_share` field is **pure noise**
here (+1.2 to −2.0 across seeds) because the held-out set is coarse (k=8 classes, ~24 examples) and the metric
divides a hair-sized numerator by a small denominator — **`train_acc` is the reliable signal**, and it is stuck
at ~0.34 for BOTH arms. The credit rule does not fit even the *training* set at the sweet spot.

## What this locates — and what it does NOT yet close

- **NOT the task** (oracle 0.957), **NOT the reservoir** (it fails, 0.262), **NOT weight-blowup** (controlled
  `w_clip=4000`, unlike the earlier `w_clip=None` smoke). The wall is the **credit signal failing to reach /
  organize the spiking hidden layer** — the φ′-vanishing diagnosis (~1600× attenuation over depth,
  `2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md`). Both arms failing
  identically points UPSTREAM of the credit rule (the hidden isn't driven to fit), not to a bad credit signal.
- **The learned microcircuit did NOT beat fixed-DFA here (1/6)** — BUT its self-prediction UNDER-CONVERGED
  (`selfpred_cos` mean 0.292, not → ~1.0): the interneuron never learned to cancel in 150 epochs. So the
  learned-arm result is **provisional, not a hard NO-GO** — a convergence test (epochs 300, `wpi-lr` 0.4,
  `wpi-noise` 0.3) is running before that verdict is banked. This is a METHOD-in-progress, not a capability wall.

## Next (no-defer — the wall names its own surpasses)

1. **[RUNNING]** converge the learned microcircuit (more epochs + faster `wpi-lr`) → does a *converged*
   self-prediction (cos→high) train where the un-converged one didn't?
2. If both arms still fail with credit converged: the wall is φ′-attenuation reaching the hidden — the record's
   **soma-coupling** knob (get the apical credit to the soma) and the **representable-forward coincidence-plateau
   expander** (`2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-...`, Lane C) are the named
   surpasses. Rate-level deep credit is SETTLED (cite, don't re-derive); the residual is precisely on-bridge
   credit-reaching-the-hidden.
