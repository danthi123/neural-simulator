# EMERGE-39 / toward-semantics — GO (6/6 seeds): the FULLY-ON-SUBSTRATE competitive pooler. The HTM-Spatial-Pooler feature→column permanences LIVE in the bridge's `coincidence_detector` synapse weights and are learned by the committed `sim/` kernel PLUS the one term it structurally lacks — the winner-INACTIVE depression (selectivity). Pins exactly which term a `sim/` kernel edit must add. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge39_onsubstrate_competitive_pooler_derisk.py`; CI guard `tests/test_emerge39_onsubstrate_competitive_pooler.py` (3 tests). Reuse-by-import (`_emerge14` committed kernel + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## Why (the residual EMERGE-38 pinned)
EMERGE-38 validated the competitive-learning MECHANISM (a host HTM Spatial Pooler reaches 0.98 on overlapping categories where a fixed projection gets 0.56), but porting the learning to the committed three-term kernel ALONE degraded to ~0.04. Root cause (measured): the committed `fused_htm_permanence_update` does potentiate(active-feature → WINNER) + depress(active-feature → NON-winner). It gates BOTH terms on `pre_last`, so an *inactive*-presynapse synapse is a no-op — it structurally cannot do the HTM-SP **winner-selectivity** depression (a WINNER column depresses its INACTIVE-feature synapses so it tunes to the features it needs). That is the one missing term.

## The claim (6/6 seeds)
On the 6-overlapping-category task (adjacent share 3/6 features, held-out inheritance, chance 0.17), the HTM-SP permanences live in `cp_connections.data`, learned by the committed kernel (potentiation, `ld=0`) + the added winner-inactive depression (a host op on the same substrate weights) + homeostatic boosting:
- **ON-SUBSTRATE: held-out inheritance 0.94 mean** (0.89/1.00/1.00/0.83/1.00/0.94 across seeds 42/43/44/100/101/102).
- **The added selectivity term is LOAD-BEARING:** potentiation ALONE (no winner-inactive depression) reaches only **0.18** (columns over-potentiate → no discrimination).
- **FIXED (no-learn) projection: 0.56**; **PERMUTED-features 0.14**; **dAP-LESION 0.00**.

## Mechanism
A dense feature→column projection whose permanences are the bridge's synaptic weights (`cp_connections.data`), small random init. Unsupervised loop over the member stream: drive[col] = Σ connected active-feature permanences × homeostatic boost → top-k winners → the committed `fused_htm_permanence_update` (with `ld=0`, potentiation only) raises winners' active-feature permanences → **then the added term** depresses each winner's INACTIVE-feature permanences (so a column that wins for category-0 inputs drops its synapses to features it doesn't need, becoming category-0-selective). The learned codons then drive the inheritance on the spiking bridge (the EMERGE-35 codon→property path).

## Anti-cheats (6/6)
- **NO-SELECTIVITY** (the added winner-inactive term OFF, potentiation only): 0.18 mean — isolates the selectivity depression as the load-bearing term (margin +0.76).
- **FIXED (no-learn)** projection: 0.56 — the untuned baseline; the learned 0.94 beats it (margin +0.38).
- **PERMUTED-features** (input-destruction): 0.14 (below chance) — no discriminative structure to tune to.
- **dAP-LESION** (coincidence off): 0.00.
- 6-seed unanimous GO.

## Significance + what it pins
This de-risks the fully-spiking HTM Spatial Pooler on-substrate and **pins the exact `sim/` kernel edit**: the committed `fused_htm_permanence_update` needs a winner-inactive-depression term `−(1−pre_active)·post_win·λ` (potentiate active→winner, depress inactive→winner). EMERGE-40 makes that term a committed additive fused kernel; here it is a host op on the substrate weights, proving the mechanism first (the disciplined cheap-first-before-`sim/`-edit ladder).

## Honest scope + next
- The permanences are the bridge's synaptic weights (on-substrate storage); the committed `sim/` kernel does the potentiation; the winner-INACTIVE depression is a host op on `cp_connections.data` here (EMERGE-40 = the committed fused kernel). The k-WTA drive read is a top-k over the substrate weights (the spiking FS-WTA lateral-inhibition version is a further rung).
- Next: **EMERGE-40** — the additive `sim/` kernel `fused_htm_winner_inactive_depression` (byte-identical to all existing paths), so both learning terms are committed kernels.

## Artifacts
`research/runners/_emerge39_onsubstrate_competitive_pooler_derisk.py`, `tests/test_emerge39_onsubstrate_competitive_pooler.py`, `research/findings/raw/_emerge39_onsubstrate_competitive_pooler.json`. Prior: `2026-07-02-emerge38-competitive-self-organizing-pooler-GO.md`.
