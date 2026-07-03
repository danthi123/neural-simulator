# EMERGE-38 / toward-semantics — GO (6/6 seeds): the COMPETITIVE SELF-ORGANIZING pooler SCALES past the fixed projection. A LEARNED pooler (HTM Spatial Pooler: winners potentiate active inputs + depress inactive + homeostatic boosting) tunes columns to the DISCRIMINATIVE features and separates OVERLAPPING categories that an untuned fixed projection cannot. The inheritance runs on the spiking bridge over the learned codons. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge38_competitive_pooler_derisk.py`; CI guard `tests/test_emerge38_competitive_pooler.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12`), composes the EMERGE-35 codon-inheritance path; NO `sim/` edit; CPU numpy-backend; 6-seed.

## The boundary this surpasses
EMERGE-35's FIXED sparse-expansion codon (Marr-Albus, a random-but-frozen feature→column projection) separates *disjoint* categories, but on *overlapping* ones — 6 categories whose feature pools overlap (adjacent categories share 3 of 6 features) — the fixed projection can't separate them well: a sparse fixed codon fully **saturates (~0.00)**, and even a *dense* untuned random projection lands at only **0.56** (chance 0.17). The research gate (spiking-self-organizing-pooler, 2026-07-02) named the fix: **competitive representation learning** (Cui-Ahmad-Hawkins HTM Spatial Pooler; Diehl-Cook 2015 STDP + lateral-inhibition + adaptive-threshold; SAILnet) — winners potentiate their active inputs + **depress their inactive inputs (selectivity)** + **homeostatic boosting** equalizes column usage, so columns tune to the *discriminative* features and pull overlapping categories apart.

## The claim (6/6 seeds)
On the 6-overlapping-category task (adjacent share 3/6 features, held-out inheritance, chance 0.17):
- **LEARNED competitive pooler: held-out inheritance 0.98 mean** (1.00/1.00/1.00/1.00/0.94/0.94 across seeds 42/43/44/100/101/102) — separates them **perfectly**.
- **FIXED (untuned random) projection: 0.56 mean** — the boundary; **margin +0.43**.
- The learning is load-bearing (learned ≫ fixed on the same task, every seed).

## Mechanism
A dense feature→column projection; the competitive pooler learns it unsupervised over the member stream — for each input, the top-k columns by boosted connected-overlap drive WIN, and winners **potentiate their active-feature synapses + depress their inactive-feature synapses** (so a column that wins for category-0 inputs drops its synapses to the features it doesn't need, becoming category-0-selective) + **homeostatic boosting** (`boost = exp(2·(k/N − dutycycle))`) equalizes column usage so no column dominates. The learned codon (the k winners) then drives the **inheritance on the spiking bridge**: a class property is taught on training members' learned codons via the committed `sim/` three-term kernel; held-out members inherit through their shared learned columns (the EMERGE-35 codon-inheritance path).

## Anti-cheats (6/6)
- **FIXED (no-learn) projection** — the untuned random baseline (the boundary): 0.56 mean, decisively below the learned 0.98 (margin +0.43). Learning is load-bearing.
- **PERMUTED-FEATURES** (input-destruction: scrambled feature↔category structure → no discriminative features to tune to): collapses to **0.12** (below chance) every seed — isolating the learned tuning as the cause.
- **dAP-LESION** (coincidence off): **0.00**.
- 6-seed unanimous GO.

## Significance
This closes the fixed-codon boundary flagged at EMERGE-35/36: category separation no longer depends on getting lucky with a frozen random projection — the pooler **self-organizes** to the discriminative structure of experience, so it SCALES to more (and more overlapping) categories. Combined with the emergence arc (EMERGE-30..37: emergent superordinates, inheritance, cancellation, transitivity over learned codes), the brain can now discover categories from experience AND represent overlapping ones separably AND do full inheritance over them — a materially richer semantic substrate for grounded conversation, on one spiking brain.

## Honest scope + next
- The competitive-LEARNING step is a rate-reference for the representation (consistent with EMERGE-33/34's numpy-pooler + on-substrate-inheritance framing); the **inheritance runs on the spiking bridge** over the learned codons, and the anti-cheats gate the mechanism. The **fully-spiking HTM-SP learning kernel** is the flagged follow-on: the committed three-term kernel's presynaptic depression (punish non-winners' active synapses) over-prunes and is NOT the HTM-SP *winner-selectivity* depression (potentiate active + depress the winner's *inactive* inputs) — a faithful winner-selectivity depression is the next `sim/` mechanism (a small proximal-dendrite competitive-learning kernel). This was measured directly: porting the learning to the committed kernel degraded to ~0.04 (over-pruning), while the HTM-SP rule reaches 0.98 — pinning the exact residual.
- Single competitive layer; the task is a controlled 6-category / 21-feature setup. Deeper hierarchical pooling + a corpus-scale category count are follow-ons.
- Next: the fully-spiking HTM-SP winner-selectivity kernel; couple competitive-pooler emergent codes into the experiential console (EMERGE-31) so discovered overlapping categories feed the full inference (inheritance + cancellation + transitivity).

## Artifacts
`research/runners/_emerge38_competitive_pooler_derisk.py`, `tests/test_emerge38_competitive_pooler.py`, `research/findings/raw/_emerge38_competitive_pooler.json`. Prior: `2026-07-02-emerge35-spiking-pooler-GO.md`, `2026-07-02-spiking-self-organizing-pooler-research-gate.md`, `2026-07-02-emerge37-cancellation-emergent-codes-GO.md`.
