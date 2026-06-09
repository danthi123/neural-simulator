# Deterministic-nav-faithful de-risk PASS — afferent+critic per-region homeostasis fires the MSN value critic

**Date:** 2026-06-09
**Type:** deterministic-nav-faithful de-risk (CPU, multi-seed). The 6th iteration — and the one that PASSES.
**Predecessors:** the per-region homeostasis protected edit (`89b8d909`, byte-identity-verified both global states); `2026-06-08-navfaithful-derisk-FAIL-homeostasis-confound.md` (critic-only FAIL + the 5 confounds).

## Verdict: **PASS 3/3** — the neural value subtraction works under the strict deterministic nav regime

Per-region homeostasis on **both** the dense `vs_place_context` afferent **and** the `striosome_value` critic
(global homeostasis OFF — the deterministic regime preserved) fires the MSN critic into a useful place-graded
firing range and opens the GABA_B value subtraction, WITHOUT the place code going place-blind. This clears the
boundary the prior five de-risk iterations mapped.

## Result (deterministic regime: global OU/conductance-noise/homeostasis OFF; only per-region masks on vs_place_context + critic; multi-seed 42/43/44, best lead 150 ms)

| Gate | Result | Detail |
|---|---|---|
| (1) V-learned-spatial | **PASS 3/3** | V(near) 0.78→1.30-1.46 Hz, near > far (ratio 1.56–∞) |
| (2) state-specific RPE gap | **PASS 3/3** | gaps 3.19 / 2.35 / 1.39 (all > 1.30); far-burst 42.5 / 39.2 / 41.7 Hz (≫ 10 floor); robust at 200/300/400 ms too |
| (3) location-selective LTP | **PASS 3/3** | w_near 0.20→0.58; w_near/w_far 2.85 / 2.94 / 2.90 (refutes the LTD) |
| (4) actor-not-perturbed | **PASS 3/3** | actor cortex 37.5 Hz with vs without critic, ratio 1.000 |
| **(5) place-selectivity-preserved (NEW)** | **PASS 3/3** | NEAR-ensemble afferent **58.7-61.3 Hz at NEAR vs 0.0 Hz at FAR** — sharply graded, NOT homogenized |
| anti-cheat (a) population code | PASS | Jaccard 0.00 (NEAR ≈38 / FAR ≈42 distinct cells) |
| anti-cheat (b) GABA_B lesion → gap vanishes | PASS 3/3 | zeroing 894 GABA_B synapses → gap 0.96 (pred ≈ unpred) |
| anti-cheat (c) GABA_A-direct → gap fails | PASS 3/3 | critic fires + place-graded but gap 0/3 (the depolarized-SNc wall) |
| anti-cheat (d) regime fidelity | PASS 3/3 | GLOBAL OU/cond-noise/homeostasis all OFF |

## Why it works (and why critic-only didn't)

| Condition (deterministic regime) | critic V(near) | gap robust | verdict |
|---|---|---|---|
| critic-only homeostasis (5th de-risk) | 0.47 / 0.57 / 0.36 Hz | 0/3 | FAIL |
| **afferent+critic homeostasis (this)** | **1.30 / 1.30 / 1.46 Hz** | **3/3 @150 ms** | **PASS** |

The forensic in the 5th-iteration finding predicted this: the global-homeostasis "fix" that made the chain
work was firing the **afferent** harder (lowering its threshold), not just lowering the critic's. Scoping
per-region homeostasis to the afferent region reproduces that faithfully. The place-blindness risk did not
materialize — threshold-homeostasis lowers the afferent's threshold so *driven* cells fire, but a cell with
~0 synaptic drive at FAR still can't cross even a lowered threshold, so place tuning is preserved.

## Fidelity nuance (honest)

This fires the critic via **intrinsic homeostatic plasticity** (a real, deterministic, cell-autonomous
threshold mechanism — Desai 1999; Turrigiano), not the textbook **convergent-excitation up-state** (B.02).
Both are real biology; this is the one the deterministic regime admits. Flagged on the protected edit's
docstring.

## Status + next

The protected per-region homeostasis edit is committed (`89b8d909`, byte-identity-verified). The
deterministic-faithful de-risk PASSES with afferent+critic homeostasis. **Next (overnight):** wire the
validated mechanism into the nav runner — the dense `vs_place_context` afferent (grid-32 tuned, drive-injected
each step) + `vs_place_context → striosome_value` + per-region homeostasis on both + the value-leads-reward
eligibility window + the GABA_B `critic_snc_window` subtraction — cheap-first smoke (critic fires + nav sane)
→ the 6-seed A/B (neural value critic vs the Stage-A host scaffold; acceptance = no nav regression). An honest
nav regression is still a deliverable. Tools: `snc_stageb_critic_probe_navfaithful.py --afferent-homeostasis`.
