# Vision→concept generalization fidelity — deep-research scoping (the one remaining weak link)

**Date:** 2026-06-16
**Type:** read-only deep-research + catalog-review scoping (the standing "research FIRST at a roadblock" move). NO code edited, NO GPU build run.
**Trigger:** the unified embodied agent (`research/findings/2026-06-16-unified-embodied-agent-stage2-GO.md`) is 6-seed robust on EVERYTHING except one sub-capability: the point-neuron vision→concept generalization keys the **wrong category** at seeds 100/101 (H5 concept-cat accuracy at chance 0.25) while being 0.75 at 42, 0.50 at 44, and a perfect 1.00 at 102. The gen_concept assembly **fires strongly** at the failing seeds (held-out win-fire 1.88 / 1.31), so this is NOT a firing/read-noise problem and NOT a moat problem — it is the per-seed **fidelity** of the generalization itself.

**Scope of the deliverable:** diagnose which of (a)–(d) drives the per-seed swing, from the code + the per-seed data; rank biologically-grounded options to make per-seed fidelity robust; for each, the catalog/Kandel grounding + reusable machinery + expected effect + cost; recommend a cheap-first de-risk with a numeric GATE and its anti-cheats. The deepest fix (the months-scale dendritic rewrite) is explicitly NOT proposed — generalization on the point-neuron substrate is already de-risked (`2026-06-16-generalization-*`); this is a *fidelity* gap, not a mechanism gap.

---

## 0. The stack under the microscope (what actually runs)

The generalization sub-capability is a four-hop chain, all reuse-by-import, NO `sim/` edit:

```
object SHAPE (oriented bar at θ_c on a ring)          build_shape_set()              [_genfrontier_optionB_visual_similarity_derisk.py]
   → real Gabor/V1 front end → V1-complex code         encode_v1 + pool_v1_to_complex (sim.visual_cortex.build_v1_simple_weights)
   → top-K active set = sparse perception drive         vision_to_perception_sets(K=60) [_genfrontier_capstone_vision_to_concept_derisk.py]
   → gen_perception → gen_concept (NMDA) convergence    rate-Hebbian, trained-then-frozen  [_train_merged_convergence, nav_conv_merged_bridge.py]
   → gen_concept SPIKES; category = argmax over         _category_of_concept_spikes (category-MEAN over concept-block spikes)
       per-category mean of the concept-block spikes    [_unified_stage1_merged.py / evaluate_arm_spikes]
```

Fixed scale (`_genfrontier_onsubstrate_convergence_derisk.N_CAT/N_PER_CAT/F`): **N_CAT = 4 categories, N_PER_CAT = 4 exemplars/category, F = 16 concepts.** Category `c`'s visual basis is an oriented bar at `base_theta = (c/4)·π` (so 0°, 45°, 90°, 135°) anchored on a ring at angle `2π·(c/4)`, with per-exemplar jitter of ±7° orientation, ±3 % position, ±8 % length, ±10 % thickness, plus pixel noise (`_render_bar_image`, σ=0.04). The leakage-free split holds out **exactly one exemplar per category** (`held_out = [rng.choice(where(cat_ids==c)) for c in range(4)]`), so each category has **3 train + 1 held-out** exemplars. The held-out exemplar's category is decided by which category's *train* exemplars its top-K active set drives most through the learned convergence.

---

## 1. Diagnosis — it is (a), the held-out/train SPLIT margin, with a structured-confusion contribution

**Verdict: the dominant cause is (a) — a thin per-held-out-exemplar split margin at N_PER_CAT=4 on a ring where adjacent categories are confusable. Evidence is decisive and rules out (b), (c), (d) as the *primary* driver.**

### The decisive contrast (same seeds, two numbers)

From the standalone capstone run (`research/findings/raw/_genfrontier_capstone_vision_to_concept.json`, seeds 42/43/44 — the same machinery the merged bridge imports verbatim):

| seed | Gabor V1-complex code margin (within / between) | active-SET margin | **held-out concept-spike cat-acc** | concept *margin* (same − other) |
|---|---|---|---|---|
| 42 | **+0.755** (0.901 / 0.146) | +0.528 | 0.75 | **+0.066** |
| 43 | **+0.771** (0.930 / 0.159) | +0.553 | 0.50 | **+0.093** |
| 44 | **+0.816** (0.956 / 0.140) | +0.647 | 1.00 | **+0.179** |

And the upstream representation quality (`_genfrontier_optionB_visual_similarity.json`, same seeds): IT-like cluster purity 1.0/0.75/1.0, **RSA-to-pixels 0.99 every seed**, the random-partition null sits 9.3–10.2 SDs below the true margin.

The reading is unambiguous:

1. **The representation is seed-ROBUST and excellent.** The Gabor/V1 code separates the 4 categories with a within/between cosine margin of ~0.76–0.82, std ≈ 0.03 across seeds, RSA-to-pixels 0.99. There is no seed where the *representation* collapses. ⇒ **(b) "the Gabor front end's per-seed discriminability" is FALSIFIED as the primary cause** — its discriminability barely moves seed-to-seed while the outcome swings from chance to perfect.

2. **The convergence fits fine and the assembly spikes strongly.** Concept spikes/cue are 80–171 (and 1.88/1.31 win-fire at the failing merged seeds), flat-distinct collapses to chance, derangement collapses. The convergence is learning the vision-category↔concept-category map. ⇒ **(c) "the rate-Hebbian convergence under-fitting" is NOT primary** — under-fitting would also drop the *train* accuracy and the flat/derangement separation, which hold.

3. **The category READ is a clean category-mean over real spikes** and already discretizes correctly at the good seeds. The read swing tracks the *concept margin*, which is itself thin — so the read is faithfully reporting a thin signal, not adding the noise. ⇒ **(d) "the category read losing discriminability" is NOT primary** (though it is the cheapest place to *buy back* margin — see Option 3).

4. **The signal that DOES swing is the per-held-out-exemplar concept margin: +0.066 / +0.093 / +0.179.** This is the gap between the held-out exemplar's drive into its own category's concept blocks vs the best competing category. It is **razor-thin and 3× variable across seeds**, and it is what flips the argmax. With only 3 train peers per category and 4 held-out decisions per seed, a single held-out exemplar whose jittered bar happens to lie closer (in oriented-edge-column space) to an adjacent category's 3 train exemplars than to its own 3 flips that category's vote — moving the seed's accuracy by 0.25 per flipped exemplar (1.00→0.75→0.50→0.25). That step size is exactly the granularity we observe (0.25/0.50/0.75/1.00).

### The structured-confusion contribution (why some seeds are worse, not random)

The categories are bars at **0°, 45°, 90°, 135°** on a ring. Orientation is *circular*, and the Gabor bank has 8 orientations (22.5° apart). Adjacent categories are 45° apart — only ~2 orientation bins — so a held-out exemplar jittered ±7° (and the bank's RF pooling) puts non-trivial energy into a neighbour's orientation columns. The confusions are therefore **structured** (a category is most confusable with its orientation-adjacent neighbours), not uniform. This is why a *bad* seed's held-out draw (an exemplar near the inter-category boundary) lands wrong, while a *good* seed's draw (an exemplar near its category centre) lands right. The position ring helps separate them, but with only 32×32 retina and a 0.28·size ring radius the position signal is modest.

**Net:** the failure is **small-sample, thin-margin generalization on a confusable 4-class ring** — option (a). The representation (b), the convergence fit (c), and the read mechanism (d) are all healthy; they simply pass through a thin per-exemplar margin that, at N_PER_CAT=4 with structured neighbour-confusion, is below the noise that one held-out draw injects. This is the textbook small-training-set generalization-margin problem, and the biology that fixes it is well-mapped (more exemplars → a population prototype; sharper input separation; a separation stage before binding).

---

## 2. Ranked biology-grounded options to make per-seed fidelity ROBUST

Ranked by **leverage ÷ cost** (cheapest, highest-confidence first). All are reuse-by-import unless noted; none touch the no-confab moat, none require the dendritic rewrite.

### Option 1 (TOP) — more exemplars per category → a population PROTOTYPE (and a deeper held-out margin)
- **What:** raise `N_PER_CAT` from 4 to ~8–12 (and split out 1–2 held-out, leaving 6–10 train peers). The convergence then learns each category from a *prototype distribution* of oriented bars, not 3 samples. More train peers = the held-out exemplar's category vote is averaged over more correct same-category drives, deepening the concept margin and washing out the single-draw structured confusion.
- **Biology grounding:** this is the **IT / ventral-stream object-recognition** principle directly — catalog **E.12 (Ventral "what" stream, Kandel 6e Ch 24 pp ~568–587):** "IT cells fire to specific objects *across viewpoint*"; an IT category cell is tuned to a category *prototype* generalized over many exemplars/views, not memorized from a handful. Categorization in IT/perirhinal is a graded, prototype-and-exemplar code (Kiani 2007 object-category geometry; Op de Beeck / Kriegeskorte RSA — already the project's cited basis for Option B). More exemplars = a denser sampling of the category manifold = the canonical route to robust generalization.
- **Reusable machinery:** `build_shape_set(N_CAT, N_PER_CAT, ...)` already takes `n_exemplars` as a parameter and renders arbitrarily many jittered exemplars; the convergence (`train_convergence`), the concept blocks (`F = N_CAT·N_PER_CAT`), and the read (`_category_of_concept_spikes`) are all parametric in F. The ONLY constants to lift are `N_PER_CAT` (and the perception region must still fit the top-K sets — it does; 2048 V1-complex cells ≫ K=60). On the merged bridge, F sets the gen_concept/gen_fact region sizes (`_generalization_regions_pathways`) — a few hundred more neurons, additive, byte-identity preserved (gen appended last).
- **Expected effect:** the strongest lever on the *root cause*. More train peers per category is the direct statistical fix for a thin held-out margin; standard learning-curve behaviour predicts the per-seed floor rises and its variance shrinks (the 0.25-granular swing softens as each category's vote integrates over more exemplars).
- **Cost:** LOW. A constant change + a slightly larger gen region. Training is a few more co-activation scenes (seconds on GPU). No new mechanism, no `sim/` edit. The cleanest, highest-confidence first move.

### Option 2 — wider category SEPARATION in the stimulus + a more discriminative front-end (de-confuse the ring)
- **What:** two cheap, composable sub-levers: (2a) **spread the categories further apart** — use orientation *and* a distinct shape primitive per category (e.g. bar / cross / L-junction / blob) so categories differ on more than one near-circular axis, removing the adjacent-orientation confusion; (2b) **pool V1→IT more discriminatively** — the current `pool_v1_to_complex` sums over frequency only (phase/frequency invariance), keeping the full 8×16×16 retinotopic code. A coarser spatial pool (a few large IT-like RFs) or an additional orientation-contrast normalization would trade retinotopic detail for cleaner category-level separability (more IT-like, larger RF).
- **Biology grounding:** catalog **E.09 (V1 complex cells, Kandel 6e Ch 23 pp ~561–566)** — "pool simple-cell outputs … spatially invariant within the RF … builds first stage of position invariance"; and **E.12** — the ventral hierarchy *increases RF size and feature complexity* V1→V2→V4→IT, so the category-level code lives at coarser, more pooled stages than the raw V1-complex map. Distinct shape primitives per category is just making the categories *visually* more distinct (the Hubel-Wiesel oriented-bar is one point in a richer feature space).
- **Reusable machinery:** `build_shape_set` / `_render_bar_image` are local helpers — adding a 2nd/3rd shape primitive is a small, self-contained render change (NOT `sim/`). `pool_v1_to_complex` is a runner helper; a coarser pooling variant is a few lines. `build_v1_simple_weights` already accepts `n_orientations`/`n_frequencies`/`n_positions_per_dim` as arguments, so the front end is reconfigurable without `sim/` edits.
- **Expected effect:** directly removes the *structured* part of the confusion (Option 1 removes the *sampling* part). Best combined with Option 1. Sub-lever 2a (distinct primitives) is the higher-leverage half — it makes between-category cosine drop further below within.
- **Cost:** LOW–MEDIUM. The render change is small; the re-pool needs the structure-preservation assert re-checked (the existing `active_set_overlap_margin` gate already guards this). Some risk that a coarser pool *reduces* the within-category overlap too — must be measured, which the existing margin instrumentation does.

### Option 3 — a pattern-SEPARATION stage (DG / Marr-Albus expansion recoding) before the concept assembly
- **What:** insert a sparse high-dimensional expansion between gen_perception and gen_concept: project the 2048 V1-complex code into a *larger* population with strong feedforward inhibition enforcing ~2–5 % activity (a kWTA/sparse layer), then converge *that* onto the concept blocks. Expansion recoding orthogonalizes the near-confusable inputs *before* the category read, so adjacent-orientation categories become linearly separable.
- **Biology grounding:** catalog **D.12 (Pattern separation — DG, Kandel 6e Ch 54 pp 1357–1360):** Marr "expansion recoding" — "divergence onto a larger sparse population *orthogonalizes similar inputs*"; and the cerebellar analogue **F.12/F.13 (granule-layer codon / expansion recoding, Marr 1969 §3.1; Albus 1971 §IV.A):** sparse expansion makes a downstream perceptron-classifier viable that the dense input could not support. This is the canonical biological mechanism for *exactly* the failure mode here (similar inputs that a linear read confuses). The project's own catalog flags DG pattern separation as the orthogonalizer (`feature-catalog.md:5116`).
- **Reusable machinery:** the brain-region framework (`BrainRegion` + `RegionPathway`) already supports a large expansion population + an FS inhibitory pool; the J-cluster "sparse coding via inhibition" motif is the same lateral-inhibition primitive used elsewhere. BUT: a faithful spiking DG layer is a **new region + tuning** (sparsity target, inhibitory drive), and the point-neuron rate-code wall (the documented Mikulasch-Priesemann limit) means the *spiking* sparsification must be driven (it cannot whiten in the analog stage). This is heavier than Options 1–2.
- **Expected effect:** potentially the most *robust* fix (it attacks confusability structurally, not statistically), and the most biologically principled. But it is the highest-variance build — a mistuned sparse layer can *destroy* the within-category overlap (over-separate exemplars of the *same* category, the D.13 separation-vs-completion trade-off: "too much completion → confused episodes; too little → no generalization", Kandel 6e Ch 54 pp 1357, 1360–1361).
- **Cost:** MEDIUM–HIGH. New region + sparsity tuning + a structure-preservation re-validation. Defer behind Options 1–2 unless those plateau.

### Option 4 — a learned read-out / prototype-cleanup on the concept spikes (buy back margin at the read)
- **What:** replace the fixed category-mean argmax with a *learned* category read-out: a small per-category read-out population trained (rate-Hebbian, on the *train* exemplars only) to fire for its category's concept-spike pattern, then category = which read-out population fires most. This is a prototype/template-match cleanup that can weight the discriminative concept dimensions and suppress the confusable ones.
- **Biology grounding:** catalog **E.12** (IT→perirhinal category cells are *learned* read-outs of the object code) + the project's already-validated **NEF thresholded cleanup** (Stewart-Tang-Eliasmith 2011, the Spaun cleanup, used in the composer) — a placed-threshold read discretizes a thin linear margin to a clean winner. The graded-propagation de-risk already builds a downstream NMDA read-out region (`build_propagation_bridge`'s read-out block); making it *category-discriminative* (rather than block-diagonal) is the learned-read-out variant.
- **Reusable machinery:** the read-out region + NMDA integration already exist in `_genfrontier_graded_propagation_derisk.build_propagation_bridge`; the convergence-training loop can train a perception/concept→read-out map on the train set. The familiarity-gate / NEF-cleanup code is reusable.
- **Expected effect:** improves the *read* (d) — turns a +0.066 margin into a cleaner decision *if* the discriminative info is present. But it CANNOT manufacture margin that the representation doesn't carry: if a held-out exemplar genuinely drives the wrong category more (the structured confusion), a learned read trained on train exemplars will also miss it. So it is a margin-*amplifier*, not a margin-*creator* — best as a complement to Option 1/2, not a standalone fix. (The doc's note that "the population-code lever would not fix a wrong generalization" applies here: this helps a thin-but-correct margin, not a wrong one.)
- **Cost:** LOW–MEDIUM. Reuses existing read-out machinery; adds a small train pass. Lower leverage on the root cause than Option 1.

### (Not proposed) Option 5 — the dendritic-substrate rewrite
- Explicitly OUT of scope. Generalization on the point-neuron substrate is already de-risked (`2026-06-16-generalization-*`, CYCLE 88 "decorrelation is a red herring"); this is a *fidelity* gap at N_PER_CAT=4, not a substrate-mechanism gap. The months-scale rewrite is not warranted and not recommended.

---

## 3. Recommended CHEAP-FIRST de-risk

**Probe: "does more exemplars per category deepen the held-out concept margin and lift the per-seed floor?" — Option 1, the highest leverage-÷-cost, as a CPU/numpy or single-GPU-seed sweep that runs BEFORE any merged-bridge build.**

### Why this one first
It attacks the diagnosed root cause (the thin small-sample split margin) with a single constant change, on machinery that is already parametric in `N_PER_CAT`, and it is falsifiable cheaply: if deepening the train set does NOT lift the floor / shrink the variance, the cause is more confusion-structure than sampling (→ pivot to Option 2's distinct-primitive de-confusion), and we learn that for the price of one sweep. It needs no `sim/` edit and no moat change.

### The probe (read-only design; this scoping doc does NOT run it)
A standalone runner (e.g. `research/runners/_genfrontier_vision_fidelity_nper_sweep.py`), reuse-by-import of `build_shape_set`, the Gabor/V1 encode, `vision_to_perception_sets`, and the graded-propagation convergence + spike read — i.e. the EXACT capstone chain, only varying `N_PER_CAT`:

1. For `N_PER_CAT ∈ {4, 6, 8, 12}`, over **the full 6-seed battery 42/43/44/100/101/102** (the seeds that exposed the failure — 100/101 MUST be included), build shapes → Gabor/V1 → top-K → train the convergence on (N_PER_CAT − 1) exemplars/category → hold out **1 per category** (leakage-free, asserted) → read the held-out concept-spike category accuracy + the same-vs-other concept *margin*.
2. Report **per-seed** (never just the mean): the held-out cat-acc, the concept margin, the within/between active-set margin, and the structure-preservation flag, for every (N_PER_CAT, seed).
3. The cheapest first cut can be the **numpy ridge-map analogue** (the convergence's `_genfrontier_crossmodal_unify_derisk` cheap-first lives in numpy and ran on CPU) to get the learning-curve shape in seconds; then confirm the winning N_PER_CAT on a single GPU seed of the real spiking chain. (Two-tier: numpy curve first, one GPU spiking confirmation — minimises GPU time.)

### The numeric GATE
- **GO** if, at some tested N_PER_CAT, **all 6 seeds** reach held-out concept-cat-acc **≥ 0.50** (≥ 2× chance) AND the **6-seed minimum** rises monotonically-ish with N_PER_CAT (a real learning-curve signature, not a fluke), AND the previously-failing seeds **100/101 specifically clear ≥ 0.50** (the failure must be the thing that moves) — with the per-seed concept margin positive on every seed. Target the merged stage-3 bar (the doc's GO seeds sat at 0.75–1.00): a robust config would push the 6-seed *mean* toward ≥ 0.75 with min ≥ 0.50.
- **PARTIAL** if the floor rises but 100/101 still dip below 0.50 at the largest N_PER_CAT — then stack Option 2 (distinct shape primitive per category) on the winning N_PER_CAT and re-gate.
- **NEGATIVE** if more exemplars do NOT lift the per-seed floor (the curve is flat) — this *falsifies* Option 1 as primary, confirms the confusion is structural-not-statistical, and routes directly to Option 2 (de-confuse the ring) / Option 3 (DG separation). An honest negative here is itself the deliverable: it localizes the cause to confusion-structure.

### Anti-cheat controls (mandatory — the project's standing discipline)
1. **Leakage-free split, asserted.** `assert not (set(train) & set(held_out))` exactly as the existing runners do; the held-out exemplar's concept block is NEVER co-activated in training. Reuse the verbatim split (`held_out = [rng.choice(where(cat_ids==c)) for c in range(N_CAT)]`).
2. **Per-seed report, never mean-only.** The whole failure is a per-seed tail (mean 0.542 hid two seeds at chance). The gate is on the 6-seed *minimum* and on 100/101 specifically, not the mean.
3. **NEVER loosen the no-confab moat.** The probe measures generalization *accuracy* only; the abstention gate (`heldout_win_fire` vs `novel_win_fire`, the familiarity contrast) is held fixed at its shipped threshold and the novel-no-category cue must still abstain at every (N_PER_CAT, seed). If a config raises accuracy but breaches the moat, it is a HARD STOP, not a GO. (More exemplars should *help* the moat — a denser category prototype makes a no-category cue *less* familiar — but it must be verified, not assumed.)
4. **Derangement / permuted control.** Re-run the category-derangement arm (co-activate each train exemplar's vision perception with a WRONG-category concept block, the existing `deranged_block` logic) at the winning N_PER_CAT: the held-out cue must land in the WRONG category (margin collapses). This proves the lift is the *learned vision-category↔concept-category correspondence*, not more-neurons-firing-by-chance.
5. **Flat-distinct baseline.** Keep the structure-ablation arm (orthogonal/disjoint perception sets, same sizes, no visual structure) at chance at every N_PER_CAT — confirms the visual structure stays load-bearing as the train set grows (the lift is not the bridge memorizing more disjoint blocks).
6. **Structure-preservation assert.** The `active_set_overlap_margin` within>between gate must stay GREEN at every N_PER_CAT (the top-K conversion must keep preserving the Gabor structure as exemplar count grows).

### What a GO unlocks
A robust `N_PER_CAT` (likely with a touch of Option 2) makes the unified embodied agent's generalization sub-capability 6-seed GO — closing the last weak link, with the integration / moat / nav / compose / conversation / parse already robust. Then the merged-bridge build is a single additive constant change (gen region grows a few hundred neurons, byte-identity preserved), re-validated by the existing `test_merged_rf_composer_coresident` (5/5) + `test_nav_conv_step2b_coresident` (7/7) and a 6-seed stage-3 re-run.

---

## 4. One-paragraph summary

The vision→concept generalization keys the wrong category at some seeds because of **a thin held-out/train split margin (cause a)**, not the representation, the convergence fit, or the read: the Gabor/V1 category code is seed-robust (within/between cosine ~0.76–0.82, RSA-to-pixels 0.99 every seed) and the concept assembly fires strongly, but the per-held-out-exemplar concept margin is razor-thin and 3× variable across seeds (+0.066 / +0.093 / +0.179), so with only **3 train peers per category** on a **4-class orientation ring where adjacent categories are confusable**, one held-out draw near a category boundary flips that category's vote (the observed 0.25-granular accuracy swing). The top biology-grounded fix is **Option 1 — more exemplars per category → a population prototype** (catalog E.12, IT prototype generalization across viewpoint; Kandel 6e Ch 24 pp ~568–587), which directly deepens the held-out margin and washes out the single-draw confusion, is a one-constant reuse-by-import change, and needs no `sim/` edit and no moat change. The recommended cheap-first de-risk is an **`N_PER_CAT ∈ {4,6,8,12}` sweep over the full 6-seed battery (42/43/44/100/101/102, the failing 100/101 included), numpy-curve-first then one GPU spiking confirmation**, gated on **all 6 seeds reaching held-out concept-cat-acc ≥ 0.50 with 100/101 specifically clearing it and the 6-seed minimum rising with N_PER_CAT**, with mandatory anti-cheats: leakage-free split asserted, per-seed (not mean) reporting, the no-confab moat held fixed and never loosened, and the derangement + flat-distinct + structure-preservation controls all green.
