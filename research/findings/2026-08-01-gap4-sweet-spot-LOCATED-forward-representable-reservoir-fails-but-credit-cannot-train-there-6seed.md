---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/sweetspot/gap4_sweetspot_6seed_aggregate.json
  - research/findings/raw/gap4/sweetspot/gap4_sweetspot_convergence_aggregate.json
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
- **The learned microcircuit did NOT beat fixed-DFA here (1/6)** — the initial run left the caveat that its
  self-prediction UNDER-CONVERGED (`selfpred_cos` 0.29). **CONVERGENCE TEST DONE (epochs 300, `wpi-lr` 0.4,
  `wpi-noise` 0.3): the caveat is CLOSED.** The self-prediction now converges (`cos` 0.894, i.e. the interneuron
  DID learn to cancel), and **deep credit STILL fails** — learned held-out 0.247 ≈ reservoir 0.265 ≈ chance,
  train_acc 0.40 (only marginally up from 0.34) still far below oracle 0.963. So a clean, converged instructive
  signal does not clear the wall. This is now a **hard verdict at a valid operating point**: the wall is
  confirmed UPSTREAM of the credit signal — credit cannot organize the spiking hidden layer at depth, regardless
  of teacher quality. (Aggregate: `gap4_sweetspot_convergence_aggregate.json`.)

## Next (no-defer — the wall names its own surpasses)

1. **[DONE]** converge the learned microcircuit → converged (`cos` 0.89) and STILL fails. The wall is upstream,
   not the teacher.
2. **[NEXT — diagnose before tuning]** is the spiking hidden layer even INPUT-SELECTIVE at the sweet spot? If it
   fires identically across inputs (the record's `_gap4_credit_vs_forward_probe` found exactly this — hidden rate
   0, identical across inputs), then no credit rule can shape it and the lever is the **forward drive**
   (`in_current_pA` / `in_bias_pA` / `soma-g` coupling), NOT the credit signal. train_acc stuck at ~0.40 with a
   converged teacher is consistent with a non-selective hidden.
3. Then the named surpasses, in order: strengthen the input→hidden drive so the hidden is input-selective
   (record's strong-forward arc) → **soma-coupling** to get the apical credit to the soma (record's `--soma-g`
   threshold ~100) → the **representable-forward coincidence-plateau expander**
   (`2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-...`). Rate-level deep credit is
   SETTLED (cite, don't re-derive); the residual is precisely on-bridge credit-reaching-a-selective-hidden.
