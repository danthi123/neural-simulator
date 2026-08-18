---
type: finding
status: go
date: 2026-08-17
mechanism: wave1-banking
---
## second-order metacognition monitor (learned_acc) — 6-seed GO

**Set:** `_second_order_metacog_monitor_derisk`, `confidence_read=learned_acc`, seeds 42/43/44/100/101/102, numpy, 160 trials/seed.

**Result: GO, 6/6 seeds.** A slow-NMDA `meta_schema` region reads the brain's own first-order 2AFC WTA competition and emits a graded spiking confidence that predicts first-order correctness in the type-2 SDT currency (Maniscalco & Lau). mean type2_AUC=0.841 (chance 0.50), mean meta-d'=2.72, mean M-ratio=1.77, mean type1_acc=0.753 (all in the [0.60,0.90] operating window). Per-seed: type2_AUC>=0.767, meta-d'>0, M-ratio>=1.11. <!--derived--> (means/per-seed minima over the 6 cited runs)

**Why it's load-bearing (not a falling metric):** learned monitor fit on a SEPARATE calibration block (seed+100003, 96 trials), scored on held-out 160 — no in-sample leak. Meta-lesion (sever monitor access) collapses type2_AUC->0.50 / meta-d'->0 on all 6 while d' & accuracy are UNCHANGED (the type-1/type-2 dissociation that defines metacognition). Permuted-confidence collapses on all 6. Within-class type2_AUC>chance (min 0.706) — tracks correctness, not stimulus. <!--derived--> (0.706 is the per-seed minimum over the cited runs)

Banked artifacts (this branch): `research/findings/raw/metacog/metacog_learnedacc_s*.json` (+.prov.json), seeds 42/43/44/100/101/102.

**Honest residual:** "second-order" = type-2 (judgment about the decision), NOT metacog-of-metacog. M-ratio>1 in 5/6 (supra-ideal, up to 2.45) — likely small-sample + 20-feature richness, so "near the ideal" is slightly overstated; the collapses license the claim, not the M-ratio magnitude. The monitor's WEIGHTS are a host logistic fit (confidence is spiking, the accuracy-mapping is host-computed) — a functional de-risk, not synaptically self-organized or production-integrated. Functional correlate only; no phenomenal claim.
