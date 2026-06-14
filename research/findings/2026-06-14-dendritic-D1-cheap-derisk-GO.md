# Dendritic de-risk D1 = GO (multi-seed): a per-compartment gain recovers the category structure a single-soma point neuron cannot — the cheap-first signal that the dendritic substrate IS the escape for the learn-graded-structure wall, at the rate level

**Date:** 2026-06-14. **Runner:** `research/runners/dendritic_d1_learn_graded_structure_derisk.py`. **Backend:** CPU-numpy (no GPU; ~instant). **Raw:** `research/findings/raw/_dendritic_d1_multiseed.json`. **Scope:** the cheap-first de-risk recommended by the dendritic-substrate deep-research pass (`2026-06-14-dendritic-substrate-deep-research.md` §(e)) — the afternoon-scale, off-bridge, numpy falsification that GATES the months-scale on-substrate two-compartment build (D2). **Predecessors in this arc:** Option-C Stage-B mechanism negative (`2026-06-14-option-c-stageB-fair-test-mechanism-wall.md`) + the deep-research scoping doc.

> **Result: GO, multi-seed (42/43/44).** A dendritic **per-compartment gain** recovers the a-priori category structure from a common-mode-dominated concept×hub count matrix (mean Pearson(S_learned, S_true) = **+0.845**, host PPMI+SVD ceiling +0.959), WHILE the **point-neuron single-global-gain** control fails on the IDENTICAL pipeline (mean **+0.052**), every seed, every gate green, every anti-cheat clean. This is the cleanest constructive evidence yet that the documented point-neuron decorrelation wall (the cause of all five prior NEGATIVEs, including Option C) is broken by the one capability a dendrite adds and a point neuron provably lacks: **per-input normalization delivered locally, per compartment.** **HONEST SCOPE: this is a rate-level numpy result** — it validates the *principle* (per-compartment gain recovers what a single soma can't) and so **warrants the on-substrate D2 build**, but it does NOT yet face the spiking-noise floor or the binding/composition pipeline that are D2's job. The recommendation is to present the D2 build's cost + risks to the owner (a months-scale, protected, hot-path edit), gated on sign-off — not to start it autonomously.

## The question (the version the dendrite is actually the escape for)
The deep-research pass sharpened the question: the dendrite is NOT a whitening front-end for binding the existing codes (that was falsified, and is already solved by the delivered flat cortex). It IS the principled escape for **learning graded semantic structure from experience** — exactly what Option C failed (the brain learned Pearson −0.008 of a structure the host proved present at +0.532). The read-out/mechanism discriminator localized that failure: the structure lives in the raw concept×hub counts (host lens recovers it, L1 ≈ +0.45), but a raw cosine of each concept's hub-connectivity profile recovers nothing (L2 ≈ 0) because the **high-frequency common hubs dominate every profile**; recovering it needs **per-hub down-weighting** (PPMI's marginal division) — a per-input operation a single-soma point neuron cannot deliver (it sums all hubs and applies one global gain), but a dendritic compartment-per-input can. D1 is the minimal, faithful, falsifiable test of exactly that.

## The toy (faithful to the actual failure; calibrated so the point neuron genuinely fails)
- **Data (`build_concept_hub_counts`):** 8 categories × 8 concepts = 64. Hubs = 200 high-frequency **COMMON** hubs (every concept ~ Poisson(40) — the common mode that dominates raw profiles) + per-category **signal** hubs (within-category ~ Poisson(4), out-of-category ~ Poisson(0.3)). `S_true` = the a-priori within-category-1 / between-0 block, **constructed, never data-derived** (the `s_true_independent` self-check held every seed). This directly mirrors the discriminator's L1/L2 structure (a few common hubs dominate; the category signal is in the rarer hubs).
- **Dendritic mechanism (per-compartment gain):** each hub is its own compartment with a **local** inhibitory gain `g_h` adapted **online** to that hub's own drive (`g_h ← g_h + η(x_h − g_h)`, purely local — only hub h's activity). The read-out residual is gain-normalized: `r_h = x_h / (σ + g_h)`. High-frequency common hubs get a large `g_h` → down-weighted; rare category hubs keep a small `g_h` → emphasized. This is the biologically-local, per-compartment realization of PPMI's per-input normalization (Carandini-Heeger divisive gain control).
- **Point-neuron control (single global gain):** ONE gain `g` for ALL hubs (the soma's one inhibitory pool): `r_h = x_h / (σ + g)`. A single global gain cannot down-weight the high-frequency hubs specifically — the literal Mikulasch-Priesemann "a single global inhibitory pool cannot whiten" claim, made falsifiable.
- **Read-out / metric:** `Pearson(cos(r_i, r_j), S_true)`. The host PPMI+SVD on the same counts is the labelled **ceiling** (data-carries-it reference), never a deliverable.

## Results (seeds 42/43/44)
| Seed | HOST ceiling | DENDRITIC per-hub | POINT-NEURON global | dendritic gen (chance 0.125) | point-neuron gen |
|---|---|---|---|---|---|
| 42 | +0.957 | **+0.863** | +0.053 | 1.000 | 0.281 |
| 43 | +0.957 | **+0.827** | +0.041 | 1.000 | 0.234 |
| 44 | +0.964 | **+0.843** | +0.063 | 1.000 | 0.266 |
| **mean** | **+0.959** | **+0.845** | **+0.052** | **1.000** | **0.260** |

**Every gate green, every seed:** structure_contrast (dendritic ≥ +0.30 WHILE point-neuron ≈0), point_neuron_fails (|Pearson| ≤ 0.12), host_ceiling_carries, generalize_contrast (dendritic generalizes 1.000 and beats point-neuron by ≥ 0.30), reproduce (residual cos 0.999 at 10% count noise), not_collapsed (eff-rank ≈ 24, off-diag cos ≈ 0.74 — not a coherence collapse), permuted_similarity_collapses (≈ 0), lesion_collapses, gains_converge, gain_tracks_frequency (gain↔hub-frequency corr ≈ +1.00), s_true_independent.

## Anti-cheats (all clean, all seeds)
- **POINT-NEURON-MUST-FAIL (the headline):** the point-neuron control gives +0.052 — genuinely ≈0 — on the IDENTICAL counts/mechanism/seeds. A dendritic GO only counts against this, and it holds.
- **HOST CEILING carries (+0.959):** the structure is recoverable in principle from these counts → a point-neuron failure is the mechanism, not the data.
- **PERMUTED-SIMILARITY collapses** (≈0): shuffling which concepts are same-category destroys the recovery → the structure is meaning-driven, not a code artifact.
- **LESION-THE-COMPARTMENT:** freezing the per-hub gains to a single constant collapses the dendritic result to the point-neuron value (+0.05) → the effect RIDES the per-compartment gain, not a leftover code property.
- **LEARNED-ONLINE-NOT-HOST-OP:** the gains are adapted online over the stream (converge), not a one-shot host `np.mean`; they end ordered by hub frequency (corr +1.0) as the mechanism predicts.

## Honest scope + what D1 does NOT show
D1 is a **rate-level numpy** de-risk — exactly the scope the deep-research §(e) specified for the cheap-first gate. It establishes the *principle*: per-compartment normalization recovers structure a single-soma point neuron provably cannot, on the project's faithful failing case, multi-seed, with the point-neuron control failing on the identical pipeline. It does **not** yet establish:
1. **The spiking substrate.** D1's units are rate; the real test is a spiking two-compartment neuron facing the σ=0.1 reproducibility noise floor that killed prior attempts (D1's reproducibility is on clean rate codes, 0.999). That is D2.
2. **The full pipeline.** D1 checks structure recovery + held-out generalization + not-collapsed, but not the bind/unbind composition the cortex ultimately needs. That is D2.
3. **It is a per-hub gain, not the full Mikulasch-Priesemann lateral-balance.** The per-compartment divisive gain is the cleanest faithful mechanism for the common-hub-domination failure (and the literal local-vs-global point); the fuller per-compartment lateral inhibition is the on-substrate target.

So D1 = GO **for the cheap-first decision it exists to make**: the dendritic substrate is the principled escape worth building, and the months-scale D2 is now justified to investigate.

## Implication + recommendation (owner decision)
The arc's fork (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`) is now resolved on both sides:
- **(A) flat cortex** — DELIVERED at 2,048 concepts (curated similarity; cannot generalize across similar concepts).
- **(B) structured cortex** (learns similarity from experience → generalizes) — the point-neuron substrate **cannot** do it (Option-C mechanism negative), and **a dendritic per-compartment normalization can** (D1 GO, at the rate level).

**Recommendation:** present the **D2 on-substrate two-compartment build** to the owner as the warranted next step, with eyes open about the cost and risk: ~**1.5–2 months**, a protected edit on the hottest code path (a new two-compartment `NeuronModel`, byte-identity-when-off discipline per the `fused_coincidence_plateau` precedent), and the honest precedent that a *prior* dendritic arc terminated in a sound-instrument VOID on a different question (`2026-05-18-dendritic-fairscale-SOUND-instrument-VOID`). D1's GO is the strongest cheap evidence to date that this one is worth it, but the months-scale commit is the owner's call — **not started autonomously.** A sensible intermediate step, cheaper than the full D2, is a spiking single-neuron two-compartment probe that faces the σ-noise floor on the D1 toy before committing the full bridge `NeuronModel` (a "D1.5").

NO `sim/` edits in D1. No banking — the GO is reported with its rate-level scope explicit; the spiking + composition risks are D2's to retire.
