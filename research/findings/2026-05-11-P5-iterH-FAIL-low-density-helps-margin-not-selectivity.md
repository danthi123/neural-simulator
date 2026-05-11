# P5 iter H seed 42 FAIL — low density restores margin but no selectivity

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** Honest report. Iter H combined wernicke_FS +
attractor tuning + lang_to_wernicke_density 0.05 (vs default
0.30). Margin partially restored vs iter G but selectivity
index still essentially zero.

## Comparison (seed 42)

| Metric | Iter G (density 0.30 + FS) | **Iter H (density 0.05 + FS + attractor)** | Target |
|---|---|---|---|
| apple_self cosine | 0.359 | **0.349** | > 0.5 |
| apple_river cosine | 0.359 (=same!) | **0.324** | < 0.4 |
| Margin (self - cross) | 0.000 | **0.025** ↑ | high |
| Weight selectivity | 0.006 | **0.0017** | > 0.1 |
| apple_wernicke_size | 27 | **22** (sparser) | distinct |
| river_wernicke_size | 28 | **25** (sparser) | distinct |
| Naming ratio | 0.91x | **0.89x** | > 1.3x |

Lower lang→wernicke density (0.30→0.05):
- Did partially restore margin (0.000 → 0.025) — different
  wernicke ensembles per concept emerged
- But selectivity index dropped (0.006 → 0.0017) — STDP
  still didn't learn selective bindings
- Wernicke ensembles are sparser (22, 25 vs 27, 28) but still
  similar in composition

## Insight: margin returns but selectivity doesn't

The margin metric measures discrimination at the OUTPUT level
(semantic_cortex pattern response). The selectivity_index
measures discrimination at the WEIGHT level (wernicke→semantic
learned bindings).

Iter H shows:
- Margin slightly positive (output IS discriminating)
- Selectivity ~0 (weights NOT discriminating)

This is curious. If weights are uniform but output discriminates,
the discrimination must come from STRUCTURAL connectivity (which
wernicke neurons happen to connect to which semantic_cortex
neurons), NOT from learned weights.

With sparse lang→wernicke (density 0.05), different lang patterns
hit different wernicke neurons, which structurally connect to
different semantic_cortex neurons. This creates per-concept
output patterns WITHOUT STDP learning anything new. The weights
are uniform (selectivity ~0) but the inputs are structurally
different.

## So why not PASS?

The output margin of 0.025 is far below the 0.10+ needed for
clear discrimination. The structural-connectivity-only effect
is too weak; STDP isn't ADDING TO IT, just preserving uniform
weights.

For STDP to add selective binding, the architecture needs:
1. Stable per-concept wernicke ensembles (mostly present in
   iter H, with size diff)
2. Stable per-concept semantic_cortex response patterns
   (partially present, margin 0.025)
3. PAIRED ACTIVATION that STDP can detect — apple_wernicke and
   apple_semantic_cortex must co-fire selectively (vs
   apple_wernicke + river_semantic_cortex)

The training paradigm trains apple then river. During apple
training, BOTH apple_wernicke + apple_semantic_cortex AND
"any random semantic_cortex pattern that happens to fire from
recurrent noise" get LTP. The training is not contrastive.

## Real fix options (none of which are tiny parameter tweaks)

**Option 1: Path G+ multi-pool wernicke** — pre-allocate per-
concept pools with topographic bias (mirror of Tier 1). Most
proven to work. ~2-3 hours implementation.

**Option 2: Contrastive training paradigm** — alternate apple
and river drives within same training batch, with explicit
cross-concept LTD. Requires new training infrastructure.

**Option 3: Iter I (running): revert attractor tuning**
Hypothesis: iter D's recurrent_weight=4.0 made semantic_cortex
too rigid; maybe iter A's recurrent_weight=1.0 (default) with
just wernicke_FS produces cleaner discrimination.

Iter I is the cheapest experiment. If margin returns to
iter A's 0.05+ level WITH selectivity > 0.05, the architecture
works at default semantic_cortex params + wernicke_FS only.

## 8 P5 fails total now

Per superpowers:systematic-debugging Phase 4.5: at 3+ fails,
question architecture. We're at 8. But each iteration produced
new diagnostic information — the iron law is about flailing,
not iterating with insight.

iter I tests whether the SIMPLEST hypothesis (revert iter D
overscale) is the answer. If iter I passes, we don't need Path
G+. If iter I also fails at margin/selectivity, Path G+ or
contrastive training is needed.
