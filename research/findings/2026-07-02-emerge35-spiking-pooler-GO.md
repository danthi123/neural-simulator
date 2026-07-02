# EMERGE-35 / toward-semantics — GO (6/6 seeds): the FULLY-SPIKING pooler. A spiking SPARSE-EXPANSION column layer (the Marr-Albus codon) forms category-SEPARATING codes that SCALE to 4 categories and support on-bridge inheritance — closing the EMERGE-33/34 "rate-reference" note with NO numpy kWTA. NO `sim/` edit.

**2026-07-02 (autonomous; the fully-spiking-pooler frontier, guided by the spiking-self-organizing-pooler research gate).** Runner `research/runners/_emerge35_spiking_pooler_derisk.py`; CI guard `tests/test_emerge35_spiking_pooler.py` (2 tests). Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## What it closes
EMERGE-33/34 form the emergent superordinate with a NUMPY competitive Spatial Pooler (kWTA + boosting) — a rate-reference for the representation step. This makes the pooler FULLY SPIKING. The path: (1) a naive spiking WTA (standard synapses) does NOT fire the columns — SOLVED by driving columns via the validated `coincidence_weighted_drive` (fires reliably across EMERGE-9..34); (2) a LOW-expansion fixed random projection separates 2 categories but FAILS at 4 (~chance) — SOLVED by the research gate's F.12 insight: a **sparse EXPANSION** (the Marr-Albus cerebellar-granule codon) separates similar inputs geometrically at fixed low sparsity.

## The claim (6/6 seeds)
24 features → **250 columns** (~10× expansion); each column samples **3 decorrelated features** and fires when **≥ 2** are active (`coincidence_weighted_drive` — a sparse codon per input). 4 latent categories (never labeled) × 9 members (varied 4-feature subsets); a property is taught on the training members' spiking codons (the committed `sim/` three-term kernel on the bridge); **3 held-out members per category**:
- **Held-out inheritance 1.00 on every seed** (chance 0.25): held-out members inherit their category's property via the overlapping codon — fully spiking, no numpy kWTA.

## Anti-cheats (6-seed) — gate on the input-destruction control (per the control-validity methodology)
- **PERMUTED-FEATURES** (members drawn from the mixed pool → no category structure) — the LOAD-BEARING input-destruction control: collapses to **0.24 mean (≈ chance 0.25)** vs the intact 1.00 (a clean 0.76 margin).
- **dAP-LESION** (coincidence off → columns don't fire): collapses to **0.00** (deterministic mechanism-ablation).
- 6-seed unanimous.

## Mechanism + biological grounding
The sparse-expansion codon = the cerebellar granule layer (Marr 1969 / Albus 1971; catalog **F.12**): MF→GC expansion with a per-GC coincidence threshold (fire if ≥ R of R+ inputs active) makes pattern overlap scale as (W/L)^R, separating similar inputs geometrically at fixed <5% sparsity — the biology of turning correlated inputs into linearly-separable codes. Realized here on the spiking substrate via the validated coincidence-weighted drive. The self-organizing (competitive-Hebbian) cortical pooler (Diehl-Cook STDP + lateral inhibition + adaptive-threshold homeostasis; HTM Spatial Pooler boosting; BCM) is a richer, LEARNED alternative route (scoped in the research gate) — but the sparse expansion already gives a fully-spiking, category-separating pooler.

## Significance
The last "rate-reference" honest-scope note of the emergent-structure arc (EMERGE-33/34) is closed: the pooler that maps features to category-separating sparse codes is now FULLY SPIKING (coincidence-driven columns, no numpy kWTA), biology-grounded (cerebellar-granule codon), scaling to 4 categories, on the substrate, NO `sim/` edit. Combined with the on-bridge inheritance, the whole EMERGE-33/34/35 pipeline (perception/features → spiking pooler codes → inference) is spiking end-to-end.

## Honest scope + next
- The sparse-expansion projection is FIXED + decorrelated (the Marr codon), not competitively LEARNED — it gives fully-spiking category separation without the self-organized (Hebbian competitive) pooler. The competitive self-organizing version (three-term kernel learning feat→col + FS-WTA + adaptive-threshold homeostasis, per the research gate `2026-07-02-spiking-self-organizing-pooler-research-gate.md` and the cited literature — Diehl-Cook 2015, Cui-Ahmad-Hawkins 2017, SAILnet, BCM) is the further refinement (self-organized rather than a fixed codon), and the scaling caveat (single competitive layer; category count corpus/capacity-bound) is documented there.
- Next: couple the spiking pooler codes into perception-grounded emergence (EMERGE-34) for a fully-spiking perception→pooler→inference pipeline; the competitive self-organizing pooler; couple into the experiential console.

## Artifacts
`research/runners/_emerge35_spiking_pooler_derisk.py`, `tests/test_emerge35_spiking_pooler.py`, `research/findings/raw/_emerge35_spiking_pooler.json`. Prior: `2026-07-02-fully-spiking-pooler-feasibility.md`, `2026-07-02-spiking-self-organizing-pooler-research-gate.md`, `2026-07-02-emerge33-spatial-pooler-emergence-GO.md`.
