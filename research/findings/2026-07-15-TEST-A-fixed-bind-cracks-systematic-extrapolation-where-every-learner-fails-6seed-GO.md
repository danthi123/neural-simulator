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

## ⚠️ a-1 RECONCILIATION (7th a-1 catch this session — my own framing was too strong): "fixed structure, not a learner" → "BILINEAR BINDING (fixed OR learned-as-an-op) over DECORRELATED codes, not a map-learning classifier"
`2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED` establishes: a LEARNED bilinear binder ALSO generalizes systematically (Fodor-Pylyshyn held-out-novel-combination, held-out **1.000 = train**, 3 seeds) — on DECORRELATED codes (between-cos ≈ 0.001); it FAILS only on the brain's CORRELATED codes (cos ≈ 0.81). So the systematicity failure is the CODE CORRELATION, not learning per se. TEST A's codes are decorrelated random ±1, so TEST A is FULLY CONSISTENT with it — I just over-narrowed "not a learner." **The precise, reconciled discriminator:**
- **A MULTIPLICATIVE/BILINEAR BINDING OPERATION** — provided FIXED (TEST A's ±1 bind, FHRR) OR EXPLICITLY parameterized + learned as a bilinear op (`2026-06-11`) — **over DECORRELATED codes GENERALIZES systematically.** Both are systematic.
- **A general MAP-LEARNING CLASSIFIER** (TEST A's deep-eprop MLP — which CAN represent bilinear functions but doesn't discover the structure) **MEMORIZES the attested and fails held-out.** The MLP had the capacity but learned the wrong thing.
- ⇒ systematicity requires the bilinear-binding STRUCTURE to be **provided or explicitly parameterized** (not left for a general classifier to discover) AND the codes DECORRELATED. This UNIFIES TEST A + `2026-06-11` + the project's FHRR-composer choice: the composer's fixed multiplicative bind over decorrelated concept codes is systematic; the learned option is a bilinear-binder (not a from-scratch MLP). It also explains WHY the deep-credit binder failed (`2026-07-14`): it's a map-classifier, not a parameterized bilinear binder.

## ⇒ Next
- **On-substrate follow-on (the fixed-dendritic-conjunction realization):** the ±1 bind → the project's on-bridge FHRR/coincidence composer (already validated) or the fixed two-compartment dendritic-conjunction (`enable_two_compartment_dap`) — the "fixed structure the substrate hosts" made literal, with LEARNED codes flowing through it. This is the honest emergence-bar unification (learned representations × a fixed biological binding primitive), consistent with `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE` (learned codes + a fixed coincidence bind).
- **Harden s102:** hold out memfloor-HARD combos (ensure held-out cells are far from train in code space) so the 1-NN control can't get lucky — a bounded gate-tightening, not a mechanism question.
Reuse-by-import; NO `sim/` edit. Runner: `_fixedbind_systematicity_derisk.py`.

## Blind-seed confirmation (12-seed total)
Extended to blind seeds 103/104/105/200/201/202 (`raw/_fixedbind_systematicity_blind.json`): FIXEDBIND held-out mean **0.96** (even stronger than the dev seeds' 0.87; oracle posctrl 1.000 all), learner 0.50, linear 0.46 — **the fixed-bind-beats-the-learner result is 12/12 robust**; strict GO gate 10/12 (memfloor split-luck on ~2 seeds, the bounded caveat). The systematicity-from-fixed-binding-structure conclusion is blind-seed confirmed.
