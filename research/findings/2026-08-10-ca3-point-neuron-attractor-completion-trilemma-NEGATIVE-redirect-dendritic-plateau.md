---
type: finding
status: negative
date: 2026-08-10
mechanism: ca3-recurrent-attractor-completion
lane: EPISODIC
seeds: [42, 43, 44]
instrument: the PERMUTED-CUE teeth carry the selectivity call (a wrong partial cue must NOT complete assembly A); completion magnitude vs permuted-cue completion are the two decomposed quantities, swept over weight×density. Controls: untrained (LTP load-bearing) + recurrence-zero (completion is recurrent) + permuted-cue (specificity). GO gate: comp>0.30 AND pattern-selective (perm << comp) at recurrence-gain>0.15.
---

# CA3 recurrent-LTP attractor completion hits the magnitude-vs-specificity TRILEMMA on the point neuron (favorable-case NEGATIVE) — the episodic-recall residual redirects to the dendritic-plateau completion readout (already 6/6)

This ran the weight×density held-out completion sweep the 2026-07-17 refutation
(`2026-07-17-gap5-ca3-recurrents-NOT-silent-transmission-refuted-*`) left explicitly PENDING — the corrected episodic
residual (`2026-08-10-episodic-cortical-cue-recall-completion-6seed-GO-*` at 0.646; NOT a transmission wall — that was
refuted 3×). Result: an honest negative that maps the trilemma and confirms the dendritic redirect.

## The negative (favorable-case, so an upper bound)

<!--derived-->

New runner `research/runners/_gap5_ca3_selective_attractor_sweep_derisk.py` (additive, NO `sim/` edit). To isolate the
ATTRACTOR wall from the known training-collapse confound (2026-07-17: the substrate's rate-Hebbian rule collapses
ca3→ca3 to a uniform ~0.846 fixed point, so a specific attractor never forms via LTP), it HAND-INSTALLS a PERFECT
pattern-selective within-assembly potentiation (the idealized outcome a perfect recurrent LTP would produce; frozen
plasticity), then sweeps weight × density × 3 seeds with the full teeth panel. **Trained recurrent LTP does NOT give
robust pattern-selective completion on the point neuron:** as weight rises, completion magnitude climbs toward ceiling
(~0.85) BUT permuted-cue completion rises FASTER and OVERTAKES it (perm > comp), the cross-assembly cosine collapses
(own ≈ other → all assemblies fuse into one global attractor), and the net self-ignites at rest (no-cue rest firing
0.001→0.12). **The only selective window sits exactly at the magnitude/specificity CROSSOVER, is ~1.3× wide in
weight, and its position is SEED-DEPENDENT — no operating point is GO on all 3 seeds (best 2/3 at d=0.30, W600-800; a
different seed drops at each W).** Controls have teeth every seed: untrained completion 0.00-0.017 (LTP load-bearing),
recurrence-zero ~0.030 pure-drive floor (completion is recurrent). Because PERFECT selective potentiation was
installed, the emergent-LTP point-neuron path can only be WORSE (weight-collapse) — this negative is an UPPER BOUND.

## Why (the trilemma) + the redirect

<!--derived-->

The point soma cannot separate the two horns — magnitude and specificity are anti-correlated across weight (raise W
for completion → perm overtakes; the third horn, silent rest, also breaks). This is the same structural point-soma
limit hit in 2026-07-08/17 across scales. **⇒ REDIRECT: the dendritic-plateau completion readout**
(`2026-07-18-gap5-ca3-functional-completion-CLOSED-6seed-GO-learned-attractor.md`, 5/6 GO / 6/6 MECHANISM, perm=0.000
and nocue=0.000 on EVERY seed) — its bistable silent DOWN-state supplies selectivity DECOUPLED from magnitude, exactly
the horn the point-neuron attractor cannot hold. The episodic-recall residual (0.646 → ceiling) should be lifted via
that dendritic readout, NOT via point-neuron recurrent-attractor strength.

## Instrument-bug catch (the prior runner could not test trained LTP)

<!--derived-->

`_riii_ca3_completion_specificity_derisk.py`'s training loop uses `_run_one_simulation_step()`, which NEVER advances
`current_time_ms` → STDP is INERT (it prints "⛔ STDP IS INERT"), so its "trained" condition was only a higher UNIFORM
built weight (held-out completion 0.000). That runner cannot test trained recurrent LTP at all — the seed/clock class
of silent-failure. The new runner sidesteps it by hand-installing the idealized selective potentiation.

## Honest bounds

<!--derived-->

Tested n_ca3=200 / 3 disjoint assemblies (the closed dendritic finding used 1500-2000; the trilemma is a structural
point-soma property, and larger/overlapping assemblies add crosstalk that only worsens high-W interference — but the
exact 1500-2000 regime was not run here). A 2/3-robust cell exists (d0.30, W600-800) but is a knife-edge needing
6-seed confirmation vs the dendritic path's clean 6/6 — it would not change the redirect. Direct-CA3 drive, plasticity
frozen (isolates recurrent completion; not the full end-to-end loop).

Artifacts: `research/findings/raw/gap5_selective_sweep.json` (weight×density×seed grid),
`research/findings/raw/gap5_fine.json` (fine W-sweep). Reproducer:
`research/runners/_gap5_ca3_selective_attractor_sweep_derisk.py`. NO `sim/` edit. SIM_BACKEND=cupy.
