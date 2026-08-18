---
type: finding
status: partial
date: 2026-08-17
mechanism: wave1-banking
---
## self-schema-metacog-6 — wave-1 verdict: PARTIAL / BOUNDARY (3/6 seeds GO)

> **⚠️ SUPERSEDED (2026-08-18) → this PARTIAL is the DYNAMIC-POOL config; a stronger operating point is a GO 6/6.**
> The wave-2 config `--learned-report-steps 80 --response1-tonic-pa 200` on the same runner reaches **verdict: GO,
> n_go: 6/6** (committed `dc3978263`, on main): aggregate `research/findings/raw/lanes/metacog/metacog_self_schema_
> dynamic_integration_6seed_report80_resp1plus200_fanout_aggregate.json` — mean self type-2 AUC 0.769, meta-d′ 2.18,
> self-vs-meta spearman 0.95, and all four controls (domain-dissociation, meta-lesion, permuted-confidence,
> self-read-lesion) collapse on ALL 6 seeds. The 3/6 below is a real boundary of the dynamic-pool op-point, NOT of the
> faculty; the faculty is GO at the report80/resp1+200 point. No further compute owed.

**Runner:** `_laneC_self_schema_metacog_integration_derisk` (numpy, 160 trials/seed).
**Claim under test:** a self_schema confidence pool reads a dynamic meta/aPFC (Fleming-Daw type-2 SDT) confidence population through fixed on-substrate synapses, yielding a metacognitive self-report.

**Result.** The runner's own 8-component GO gate passes on 3/6 seeds — s42, s44, s100 GO; s43, s101, s102 NO-GO. Per the runner's aggregation (`GO` iff all seeds GO), the 6-seed verdict is **PARTIAL**. It does NOT meet the >=5/6 bar, so it is not a GO.

**What is load-bearing (controls hold 6/6, not just headline seeds):**
- meta-lesion → self type-2 AUC to 0.50, meta-d'→0 (meta_d_attributable≈1.0) on every seed
- self-read lesion → self AUC 0.50 on every seed
- permuted confidence↔trial pairing → self AUC ≈0.50 on every seed
- domain dissociation 6/6 (drive_offset_by_class = 0 → not a class-drive artifact)

So the self-schema report's metacognitive signal genuinely flows through the routed synapses; it is not a static readout or a trial-order leak.

**Failure modes (why not GO):**
- s101: type-1 accuracy 0.48 (at chance) → window control correctly voids a degenerate M-ratio=4.8. Task-difficulty failure, not routing.
- s43: all metrics pass but self~meta Spearman 0.60 < 0.75 (tracking-fidelity miss).
- s102: genuine routing loss — self AUC 0.62 (<0.65) and M-ratio 0.50 (<0.6), while source meta pool read AUC 0.74 (meta-d' 1.86); the self read was lossier than its source.

**Residual / honesty.** Metacognitive integration is real and non-shortcut, but robust at the strict functional bar only 3/6. Explicitly a functional correlate — `honest_scope` disclaims subjective experience and production abstain/hedge is unchanged. A wave-2 follow-up (`lanes/metacog/…report80_resp1plus200…` + fanout_aggregate) likely re-attacks the failing seeds; its verdict is out of scope here.

Banked artifacts (this branch): `research/findings/raw/laneC/self_schema_metacog_s*.json` (+.prov.json), seeds 42/43/44/100/101/102.
