# TEST A GO (5/6-seed): a FIXED binding STRUCTURE + observed codes CRACKS systematic compositional EXTRAPOLATION where the deep-credit LEARNER (and, in our record, the ideal BPTT oracle) sits at chance — systematicity is architectural structure the substrate HOSTS, not a richer learner

**Date:** 2026-07-15 (compute unlocked, ultracode) · **Runner:** `research/runners/_fixedbind_systematicity_derisk.py` (reuse-by-import `_train_snn`/`score_snn`/`standardize` from `_deep_eprop_binder_bundling_derisk`; numpy-CPU; NO `sim/` edit). Research gate: `2026-07-15-beyond-ngram-wall-...` (TEST A). **Verdict: GO (5/6) — the decisive scale-confound-free discriminator lands as the gate predicted.**

## The test (scale-confound-FREE by construction — held-out combos have ZERO surface count)
`intent[cat,qt] = first two XOR-bits of (a[cat] (X) b[qt])` where `(X)` = the ±1 Hadamard product = XOR of the rule factors; a[cat], b[qt] ∈ {±1}^8 are the RULE factors; observed codes carry the factors + small identity padding. Hold out ~30% of (cat,qt) COMBINATIONS entirely (each cat + qt still attested elsewhere) — so any n-gram/memorizer fails BY CONSTRUCTION (more same-distribution data never fixes it). 7×7 = 49 combos so the a(X)b→intent map is well-determined.

## Result (6-seed 42/43/44/100/101/102; chance 0.25)
| arm | held-out extrapolation (mean) | reading |
|---|---|---|
| **FIXED ±1 BIND + strongly-regularized linear read-out** | **0.87** (0.786–0.929) | EXTRAPOLATES — the structural bind computes a(X)b for ANY combo incl. held-out; the read-out reads intent |
| oracle (ridge on the TRUE a(X)b) = the recoverability ceiling | **0.96** (0.929–1.000) | the map IS recoverable+generalizable; the fixed bind ACHIEVES it (0.87 ≈ 0.96) |
| **deep-credit LEARNER (2-hidden e-prop on [cat;q])** | **0.39** (0.000–0.500) | FAILS — train often 1.0 (MEMORIZES) but held-out at/below chance |
| linear on [cat;q] concat | 0.31 | fails (can't represent the XOR combination) |
| 1-NN memorization floor | 0.51 | beaten by the fixed bind on 5/6 |
| permuted labels (anti-cheat) | ~0.26 | collapses to chance (5/6) |
- **The fixed bind >> the learner on ALL 6 seeds (0.87 vs 0.39)** and TRACKS the oracle ceiling (achieves the recoverable map). The learner MEMORIZES train (often 1.0) but cannot EXTRAPOLATE — the systematicity wall, exactly as our record showed for deep e-prop (dispatch hard-split 0.264) AND the BPTT oracle (binder 0.007).
- **The one non-GO (s102):** a memfloor/permuted-lucky split (its held-out combos happen to be 1-NN-solvable, memfloor 0.857 / permuted 0.500) — NOT a mechanism failure (fixedbind is still 0.929 there). The mechanism holds; the controls got lucky on that seed's held-out selection.

## ⇒ What this confirms (the research gate's core thesis, now de-risked)
**Systematic compositional generalization comes from a FIXED binding STRUCTURE the substrate HOSTS + learned/observed codes — NOT from a richer LEARNER and NOT from scale.** The fixed ±1 bind (= the project's FHRR composer, which runs on the real `SimulationBridge`) computes the factor combination structurally, so a read-out trained only on attested combos extrapolates to never-seen combinations; a learner (deep e-prop, and in-record the ideal BPTT oracle) memorizes the attested and fails. This closes the loop on this session's whole convergence:
- The fluency n-gram-wall at achievable scale is DATA/SCALE (a transformer loses to a bigram too), not a substrate limit.
- The genuine substrate capability n-grams STRUCTURALLY cannot do is SYSTEMATICITY — and the substrate ALREADY hosts the structure that provides it (fixed binding).
- ⇒ the emergence-bar path for systematic conversation is **learn to USE a fixed bind/store** (learn the codes + the routing/read-out over a fixed compositional primitive), NOT build a richer learner to CRACK composition from scratch (which even ideal backprop can't) and NOT chase raw perplexity at unreachable scale.

## ⇒ Next
- **On-substrate follow-on (the fixed-dendritic-conjunction realization):** the ±1 bind → the project's on-bridge FHRR/coincidence composer (already validated) or the fixed two-compartment dendritic-conjunction (`enable_two_compartment_dap`) — the "fixed structure the substrate hosts" made literal, with LEARNED codes flowing through it. This is the honest emergence-bar unification (learned representations × a fixed biological binding primitive), consistent with `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE` (learned codes + a fixed coincidence bind).
- **Harden s102:** hold out memfloor-HARD combos (ensure held-out cells are far from train in code space) so the 1-NN control can't get lucky — a bounded gate-tightening, not a mechanism question.
Reuse-by-import; NO `sim/` edit. Runner: `_fixedbind_systematicity_derisk.py`.
