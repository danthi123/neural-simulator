---
type: finding
status: contributing
date: 2026-08-01
mechanism: dual-route-morphology
lane: E-language
artifacts:
  - research/findings/raw/dual_route/dual_route_inhib_sweep_aggregate.json
---

# E·Language: the dual-route fix via a SINGLE pool + whole-form→affix inhibition does NOT separate the routes — rule-generalization caps at 0.25 regardless of inhibition — NEGATIVE

<!--derived-->
**One-line verdict:** the named next lever for the morphology NO-GO (the single shared pool couldn't do both rule-
generalization AND blocking — an architectural trade-off, Pinker-Ullman separate systems). This tests a SEPARATE-
route realization on the D-sparse spiking substrate: a dedicated strong PAST→AFFIX pathway (route 1) + the
entrenched stem→whole-form store (route 2) + **whole-form→affix cross-route INHIBITION** (blocking that fires only
for entrenched stems). It does **not** achieve rule generalization. Across an inhibition-strength sweep of **7
points (0.5 … 6.0, seed 42)**, `reg_acc` caps at **0.25** (needs ≥0.90) while blocking holds (irr_acc 0.86–1.0);
**0/7 points pass both gates**. `reg_acc` sits at ~the single-pool baseline (0.19) REGARDLESS of inhibition
strength — so the single-pool + inhibition realization does **not truly separate** the routes. NO `sim/` edit.

Artifact: `research/findings/raw/dual_route/dual_route_inhib_sweep_aggregate.json` (backend numpy/CPU). Runner:
`research/runners/_productive_morphology_dual_route_derisk.py`.

## Result — inhibition-strength sweep, seed 42

<!--derived-->
| inhib-strength | reg_acc (rule, need ≥0.90) | irr_acc (blocking, need ≥0.85) | both? |
|---|---|---|---|
| 0.5 | 0.125 | 1.000 | no |
| 1.0 | 0.125 | 1.000 | no |
| 1.5 | 0.250 | 0.857 | no |
| 2.0 | 0.250 | 0.857 | no |
| 3.0 | 0.125 | 0.857 | no |
| 4.0 | 0.125 | 1.000 | no |
| 6.0 | 0.000 | 1.000 | no |

`reg_acc` never exceeds 0.25; blocking is robust throughout. There is no operating point where the default affix
wins for novel stems.

## Why it fails (the diagnosis, and what it rules out)

<!--derived-->
The hypothesis was that a dedicated PAST→affix route, freed from the shared WTA, would let the affix win for novel
stems, with the whole-form→affix inhibition supplying blocking only for entrenched stems. Two things the sweep
shows: **(1)** blocking works as designed — the inhibition suppresses the affix for irregulars (irr_acc holds even
at low inhib). **(2)** but `reg_acc` stays at the single-pool floor at EVERY inhibition strength, including the
weakest (0.5) and zero-effective (where blocking is still ~1.0). So the affix's failure to win for novel stems is
**not** caused by over-inhibition — it is that, in ONE shared pool, a novel stem's pattern still spuriously
activates whole-form neurons (pattern overlap in the shared recurrent), and those win the readout over the affix,
exactly as in the single-pool runner. The routes are co-located, so they are not actually separated; the
negative-weight inhibition adds blocking but cannot make the affix competitive for novel stems.

## Next (the genuine separation)
The Pinker-Ullman separation must be ARCHITECTURAL, not one pool + inhibition: **genuinely separate pools** — the
PAST→affix rule in its OWN dedicated pool, isolated from the whole-form store, so a novel stem cannot spuriously
retrieve a whole-form, and the affix wins by default; the declarative store in a SECOND pool; and a clean inter-
pool inhibitory projection (whole-form pool → affix neurons) for blocking. Plus the burn-down the build flagged: the
inhibition should be a Dale-compliant di-synaptic interneuron (whole-form → inhibitory interneuron pool → affix),
not the sign-inverted excitatory synapse used here (the engine enforces Dale's law). This finding maps that
cross-route inhibition inside one pool is insufficient — the boundary is the co-location, and the next lever is true
pool separation. A mapped boundary with the mechanism named; no capability abandoned.
