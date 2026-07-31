---
type: plan
status: live
date: 2026-06-14
---

# Option C fairer de-risk design — a paradigmatic host pre-check that GATES the brain-based test (the logged follow-on to the production build)

**Date:** 2026-06-14. **Status:** DESIGN (read-only pass; the logged FOLLOW-ON to the Option-B production build — NOT a blocker, owner not directed; this informs that decision). **Predecessor:** `research/findings/2026-06-13-option-c-real-cooccurrence-derisk-INCONCLUSIVE.md` (the host-ceiling control fired → the test couldn't fairly judge the mechanism).

## The diagnosed defect (pinned)
The prior host ceiling (`host_ceiling_codes`, `learned_graded_embedding_derisk_probe.py:380`, matrix at :386-393) builds a **target×target, first-order, windowed co-occurrence** matrix — a SYNTAGMATIC measure (scores word pairs by *appearing together*: `red`+`ball`, `ball`+`dog`). The 8×8 taxonomy is PARADIGMATIC (animals substitute for animals). Distributional-semantics literature is explicit: paradigmatic category structure lives in the **second-order** signal — a **target×CONTEXT PPMI matrix** (rows = each target's distribution over a LARGE context vocabulary, NOT restricted to the targets) → truncated SVD → **cosine of the row vectors**. (The controller's own full-context numpy probe already reached Pearson **+0.263**, nn-same-category 0.50, with animals/family/actions clustering — short of 0.5 only on tuning.)

## The fairer design: a two-stage GATED probe
**Stage A (CPU-numpy, ~minutes, GATES the GPU): `build_paradigmatic_host_ceiling`** — target×full-context PPMI (context vocab ~5–10k, stoplist on the highest-frequency function words, optional context-smoothing α=0.75) → truncated SVD (d≈100) → cosine-row paradigmatic similarity → `Pearson(sim, S_true)` + `generalization` via the SAME `structure_recovery`/`run_generalization` the brain side uses (apples-to-apples). **HOST GATE: Pearson ≥ 0.50 AND gen ≥ 0.70** on the (sub-)taxonomy used.
- PASS(full) → run Stage B on the full 8×8.
- PASS(subset) → narrow to the host-recovered categories (Option 2, selected by the per-category recovery report) and run Stage B on that honest sub-taxonomy (scope-tagged).
- FAIL(all) → **NEGATIVE_data_too_syntagmatic** (even the gold-standard second-order measure can't recover it → the cheap-local Option-C question is closed; report + STOP, **zero GPU**).

**Stage B (GPU, ~1–1.5 hr, only if host passes): byte-identical to the prior probe** — `learn_W_homeostatic`(oja) → `divnorm_spreading_readout` → G1 structure-recovery + G2 generalization (A2/A3 collapse) + the HEADLINE permuted-co-occurrence anti-cheat (`permute_corpus` → re-learn → collapse, gated by `_g5_robust`) + beats-random + `W`-distinct-from-counts. `S_true` stays the a-priori taxonomy (the independence assertion fires on both stages — NEVER corpus-derived).

**New host-side anti-cheat:** report the OLD first-order ceiling alongside the new second-order one; the validating signature is **first-order low (~+0.13, syntagmatic) < second-order high (≥0.5, paradigmatic)** — confirms we're measuring substitutability, not scene co-membership.

## Verdicts (the host-passes precondition gates all of them)
- **GO** (G1 ≥0.5 + graded + 2nd-margin ≥+0.10; G2 gen ≥0.7 + A2/A3 collapse; permuted collapses; beats random) → the point-neuron substrate LEARNS graded paradigmatic semantics from real experience → **revisit Option C vs B for production**.
- **BOUNDARY_weak_graded** (permuted collapses + Pearson>0 but gen/2nd-margin marginal) → learns the right structure COARSELY (the biological-learning-strength gap, now cleanly attributable to the mechanism since the host validated the signal) → build on B; the brain-vs-host gap is the deliverable. *Most likely per the Mikulasch-Priesemann point-neuron limit.*
- **NEGATIVE_no_structure** (brain fails G1 WHILE host paradigmatic PASSES) → a CLEAN, biology-translatable mechanism negative: the data demonstrably carries the structure (host proves it) and the point-neuron substrate can't learn it from experience → build on B; the dendritic rewrite is the genuine-distributional path. **The outcome the redesign exists to make obtainable** (the prior test couldn't reach it because the host also failed).
- **NEGATIVE_data_too_syntagmatic** (host fails the correct measure too) → close cheap-local Option-C for free, zero GPU.

## Reuse / cost / scope
- Reuse-by-import: the entire brain-based learn + battery (`learn_W_homeostatic`, `divnorm_spreading_readout`, `structure_recovery`/`architecture_generalization`/`permute_corpus`/`random_gaussian_codes`, `_g5_robust`, `raw_count_matrix`/`offdiag_pearson`) + `build_real_cooccurrence`/`run_seed`/`decide_verdict`/`main` from `option_c_real_cooccurrence_derisk.py`. **ONE new numpy function** (`build_paradigmatic_host_ceiling`) + the gate/narrowing block + the verdict extension. NO `sim/` edits; ~80–120 LOC. Sweep knobs (all numpy, tune ONLY the reference, never the brain codes): `context_vocab_size {3000,5000,10000}`, `svd_dim {50,100,200,300}`, `window {2,5,10}`, `ppmi_alpha {1.0,0.75}`, stoplist on/off.
- **Cost:** Stage A ~2–5 min/setting (~30 CPU-min full sweep); Stage B ~1–1.5 GPU-hr ONLY if host passes. **Best case (host fails) ~30 CPU-min, zero GPU.** Strictly cheaper-in-expectation than the prior attempt (the cheap check now GATES the GPU instead of running after it).
- **Local corpus is NOT the lever** (Option 3 rejected): TinyStories (1.59M tokens) is the largest/cleanest local everyday-concept corpus; tinyshakespeare (archaic, 7× smaller) + distill_corpus (114 KB) carry LESS paradigmatic structure. The defect was the measure, not the corpus.
- **Option 4 (a different brain rule — predictive/successor-representation/Garagnani-Pulvermüller spiking-Hebbian) DEFERRED** until after the host clears AND only if the default rule fails (the "substrate vs rule" disambiguator) — running it now would repeat the original confound.

## Honest framing
The host ceiling is a labelled disambiguator, never a deliverable. The deliverable is always the brain-based result. A `NEGATIVE_no_structure` with host-passes is a first-class scientific deliverable (maps what the point-neuron substrate can't learn from experience). This follow-on does not gate the Option-B production build; it informs whether/when to revisit the brain-faithful learned-similarity path.

**Sources:** arXiv 1906.02479 (second-order co-occurrence of SGNS); Contrasting Syntagmatic vs Paradigmatic in DSMs; Ruder "secret ingredients of word2vec" (PPMI+SVD, α=0.75); word–context matrices; window-size vs category-vs-semantic structure.
