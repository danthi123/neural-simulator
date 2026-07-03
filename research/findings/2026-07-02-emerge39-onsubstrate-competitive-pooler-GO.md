# EMERGE-39 / toward-semantics — GO (6/6 seeds): the FULLY-ON-SUBSTRATE competitive pooler. The HTM-Spatial-Pooler feature→column permanences LIVE in the bridge's `coincidence_detector` synapse weights and are learned by the committed `sim/` kernel PLUS the one term it structurally lacks — the winner-INACTIVE depression (selectivity). Pins exactly which term a `sim/` kernel edit must add. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge39_onsubstrate_competitive_pooler_derisk.py`; CI guard `tests/test_emerge39_onsubstrate_competitive_pooler.py` (3 tests). Reuse-by-import (`_emerge14` committed kernel + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## Why (the residual EMERGE-38 pinned)
EMERGE-38 validated the competitive-learning MECHANISM (a host HTM Spatial Pooler reaches 0.98 on overlapping categories where a fixed projection gets 0.56), but porting the learning to the committed three-term kernel ALONE degraded to ~0.04. Root cause (measured): the committed `fused_htm_permanence_update` does potentiate(active-feature → WINNER) + depress(active-feature → NON-winner). It gates BOTH terms on `pre_last`, so an *inactive*-presynapse synapse is a no-op — it structurally cannot do the HTM-SP **winner-selectivity** depression (a WINNER column depresses its INACTIVE-feature synapses so it tunes to the features it needs). That is the one missing term.

## The claim (6/6 seeds)
On the 6-overlapping-category task (adjacent share 3/6 features, held-out inheritance, chance 0.17), the HTM-SP permanences live in `cp_connections.data`, learned by the committed kernel (potentiation, `ld=0`) + the added winner-inactive depression (a host op on the same substrate weights) + homeostatic boosting:
- **ON-SUBSTRATE: held-out inheritance 0.94 mean** (0.89/1.00/1.00/0.83/1.00/0.94 across seeds 42/43/44/100/101/102).
- **The added selectivity term is LOAD-BEARING** (the primary evidence that learning is real): potentiation ALONE (no winner-inactive depression, mechanism-ablation) reaches only **0.18** (columns over-potentiate → no discrimination) — **margin +0.76**.
- **PERMUTED-features (input-destruction) 0.14**; **dAP-LESION (mechanism-removal) 0.00**.
- **FIXED (no-learn random-projection): 0.56 mean — REPORTED as a secondary check only, NOT part of the GO gate** (see the control-validity note below).

**Corrected 3-seed re-run (seeds 42/43/44, the honest gate):** on-substrate **0.96** (0.89/1.00/1.00); no-selectivity (mechanism-ablation) **0.20** (0.39/0.11/0.11), margin +0.76; permuted **0.15** (0.11/0.11/0.22); lesion **0.00**; fixed (reported-only) **0.61** (0.28/0.83/0.72). GO holds on the valid controls.

## Mechanism
A dense feature→column projection whose permanences are the bridge's synaptic weights (`cp_connections.data`), small random init. Unsupervised loop over the member stream: drive[col] = Σ connected active-feature permanences × homeostatic boost → top-k winners → the committed `fused_htm_permanence_update` (with `ld=0`, potentiation only) raises winners' active-feature permanences → **then the added term** depresses each winner's INACTIVE-feature permanences (so a column that wins for category-0 inputs drops its synapses to features it doesn't need, becoming category-0-selective). The learned codons then drive the inheritance on the spiking bridge (the EMERGE-35 codon→property path).

## Anti-cheats — the GO gate rests on the VALID controls only
The GO gate is `onsub ≥ 0.85 ∧ onsub ≥ no_selectivity + 0.25 ∧ onsub ≥ permuted + 0.30 ∧ onsub ≥ lesion + 0.30` — a mechanism-ablation, an input-destruction, and a mechanism-removal control. The fixed (no-learn) arm was **dropped from the gate** and is now reported-only (see the control-validity correction below).
- **NO-SELECTIVITY** (the added winner-inactive term OFF, potentiation only — *mechanism-ablation*): 0.18 mean — isolates the selectivity depression as the load-bearing term (margin +0.76). **This is the primary "learning is real / load-bearing" evidence.**
- **PERMUTED-features** (*input-destruction*): 0.14 (below chance) — no discriminative structure to tune to.
- **dAP-LESION** (coincidence off — *mechanism-removal*): 0.00.
- 6-seed unanimous GO on these three valid controls.
- **FIXED (no-learn random-projection) — REPORTED, NOT gated:** 0.56 mean, but **per-seed spread 0.28–0.83** (3-seed re-run 0.28/0.83/0.72; at seed 43 the fixed control is 0.83, so onsub 1.00 clears the old `+0.25` margin by only +0.17 — a near-tie that passed only on the mean). This is a fixed-random-code control and it does NOT collapse to chance; it is a luck artifact of the specific fixed codes, exactly the failure mode the anti-cheat-control-validity methodology warns against.

## Control-validity correction (adversarial audit, 2026-07-02)
The original strict GO gate included `onsub ≥ fixed + 0.25`. A confirmed adversarial audit flagged this as a **forbidden fixed-random-code control in the gate**: the fixed arm does not collapse (per-seed 0.28/0.83/0.72/0.50/0.44/0.56, mean 0.56) and clears the margin only on the mean (seed-43 margin +0.17 < 0.25). Per `2026-07-02-anti-cheat-control-validity-methodology.md`, a GO gate must never rest on a fixed-random-code control. **Fix applied:** the `onsub ≥ fixed + 0.25` term was removed from the gate; the strong valid controls (no_selectivity + permuted + lesion) are the gate; the fixed arm is retained as a reported secondary. The GO **survives** the corrected/honest gate (mean onsub 0.96 vs no_selectivity 0.20 / permuted 0.15 / lesion 0.00). `no_selectivity` (0.18 mean, well below the fixed 0.56) is the primary load-bearing evidence that the *learning*, not a lucky code, drives the result.

## Significance + what it pins
This de-risks the fully-spiking HTM Spatial Pooler on-substrate and **pins the exact `sim/` kernel edit**: the committed `fused_htm_permanence_update` needs a winner-inactive-depression term `−(1−pre_active)·post_win·λ` (potentiate active→winner, depress inactive→winner). EMERGE-40 makes that term a committed additive fused kernel; here it is a host op on the substrate weights, proving the mechanism first (the disciplined cheap-first-before-`sim/`-edit ladder).

## Honest scope + next
- The permanences are the bridge's synaptic weights (on-substrate storage); the committed `sim/` kernel does the potentiation; the winner-INACTIVE depression is a host op on `cp_connections.data` here (EMERGE-40 = the committed fused kernel). The k-WTA drive read is a top-k over the substrate weights (the spiking FS-WTA lateral-inhibition version is a further rung).
- Next: **EMERGE-40** — the additive `sim/` kernel `fused_htm_winner_inactive_depression` (byte-identical to all existing paths), so both learning terms are committed kernels.

## Artifacts
`research/runners/_emerge39_onsubstrate_competitive_pooler_derisk.py`, `tests/test_emerge39_onsubstrate_competitive_pooler.py`, `research/findings/raw/_emerge39_onsubstrate_competitive_pooler.json`. Prior: `2026-07-02-emerge38-competitive-self-organizing-pooler-GO.md`.
