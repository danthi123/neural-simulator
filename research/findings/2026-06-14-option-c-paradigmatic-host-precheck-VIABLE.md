# Option C fairer de-risk — STAGE A (paradigmatic host pre-check): PASS → Option C is VIABLE (the prior INCONCLUSIVE was a measurement defect, not a data limit); a brain-based fair test is warranted

**Date:** 2026-06-14. **Runner:** `research/runners/option_c_paradigmatic_host_precheck.py`. **Backend:** CPU-numpy (no GPU). **Design:** `docs/plans/2026-06-14-option-c-fairer-derisk-design.md`. **Predecessor:** `2026-06-13-option-c-real-cooccurrence-derisk-INCONCLUSIVE.md`. **Scope:** the logged FOLLOW-ON to the Option-B production build — NOT a blocker; this informs whether/when to revisit the brain-faithful learned-similarity path.

> **Result: HOST GATE PASS — Option C is VIABLE.** The prior Option-C de-risk was INCONCLUSIVE because its host "ceiling" control was a target×target FIRST-ORDER (syntagmatic) co-occurrence measure that could not recover the paradigmatic category taxonomy — so the test could not fairly judge the spiking substrate. Replacing it with the correct SECOND-ORDER measure (a target×FULL-context PPMI+SVD = cosine of context-profile rows, the distributional-semantics standard for paradigmatic similarity) and tuning the window to 2 (category structure) + PPMI context-smoothing α=0.75 lifts the host from Pearson **+0.263 → +0.539** (clears the ≥0.50 gate), with **all 8 categories recovered** and nearest-neighbour-same-category **0.859**. ⇒ the paradigmatic structure IS present in the corpus and recoverable by the gold-standard host → a GPU brain-based fair test (Stage B: does the spiking-Hebbian substrate LEARN this host-validated signal?) is now warranted (owner-directed). The cheap-local Option-C question is NOT closed.

## What changed (the measurement defect, pinned + fixed)
The prior host ceiling (`host_ceiling_codes`, `learned_graded_embedding_derisk_probe.py:380`) iterated over scenes built only from the 64 target words → a **target×target, first-order, windowed** matrix = a SYNTAGMATIC measure (binds scene-mates: `red`+`ball`). The 8×8 taxonomy is PARADIGMATIC (substitutability). The fix (`build_target_context_counts` + `ppmi_svd_sim`): a **target×CONTEXT** PPMI matrix (rows = each target's distribution over the top-5,000 context words, NOT restricted to the targets) → truncated SVD → cosine of the row vectors = the standard second-order/paradigmatic measure.

## Sweep result (CPU, ~6 min)
Window × context-vocab × SVD-dim × PPMI-α sweep. The decisive knob is the **window**:

| window | best Pearson | nn-same-category | note |
|---|---|---|---|
| **2** | **+0.539** | **0.859** | category structure — clears the gate |
| 5 | +0.418 | 0.516–0.609 | my prior untuned probe regime (~+0.26 at svd=50) |
| 3 | (between) | | |
| 10 | +0.330 | 0.453–0.484 | too broad — washes out category structure |

**Best: window=2, context=5,000, svd-dim=100, α=0.75 → Pearson +0.539, margin +0.066, nn-same 0.859.** Per-category nearest-neighbour-same rate at the best setting: animals 0.875, food 0.875, body 0.875, family 0.875, actions 0.625, colors **1.000**, places **1.000**, toys 0.750 — every category is recovered (the literature-predicted small-window-for-category-structure effect; large windows capture topical/semantic relatedness instead, hence window=10's collapse).

## What this means
- **The prior INCONCLUSIVE verdict stands corrected as a measurement defect, not a data/mechanism limit.** The host-ceiling control did its job twice over: it first prevented a false "the mechanism can't learn from real text" claim (the original inconclusive), and the redesigned second-order host now shows the signal IS there. (`s_true_independent` held — the taxonomy was never corpus-derived in either measure.)
- **Option C is viable → the brain-based fair test (Stage B) is warranted.** With the host gate cleared, a Stage-B run (the byte-identical spiking-Hebbian learn + divnorm read-out + the full battery, on the window=2 paradigmatic setup) would yield a CLEAN verdict: **GO** (the point-neuron substrate learns graded paradigmatic semantics from real experience → revisit Option C vs the curated Option B for production) / **BOUNDARY_weak_graded** (learns it coarsely — the likely Mikulasch-Priesemann point-neuron outcome) / **NEGATIVE_no_structure** (host passes, substrate fails — a clean, biology-translatable map of what a point-neuron substrate can't learn from experience). None of these were obtainable before, because the host also failed.

## Honest scope + next
- This is the cheap CPU GATE (Stage A) of the fairer design — it decides that a GPU Stage-B run is worth it. It is NOT itself the brain-based result (the host is a labelled disambiguator, never a deliverable).
- **Stage B (the GPU brain-based fair test) is GPU-gated behind the in-flight 32-bridge production deliverable + owner-directed** (Option C is the logged follow-on to the Option-B build, not a blocker). The Stage-B runner is a small extension of `option_c_real_cooccurrence_derisk.py` (the window=2 paradigmatic corpus + the existing learn/read-out/gate battery, with the host pre-gate). Recommendation: after the production deliverable lands, present this PASS + run Stage B (the brain-based fair test) — it is now a clean cheap-first falsification (~1–1.5 GPU-hr) with a genuinely informative outcome whichever way it lands. NO `sim/` edits. No banking — the host pre-check is reported as the gate it is, not as a brain-based result.
