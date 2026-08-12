---
type: finding
status: contributing
date: 2026-08-12
mechanism: lane-C self-model / metacognition — 6-seed measurement; the faculty is NOT robustly demonstrated (type-2 at chance) and a GO gate is mis-calibrated
lane: C · Self/Workspace (roadmap §3 self-model / know source+strength of knowledge)
verdict: BOUNDARY / honest-negative + an instrument flag. Two lane-C self-model de-risks ran clean 6-seed locally (42/43/44/100/101/102). (1) SECOND-ORDER METACOG MONITOR (_second_order_metacog_monitor_derisk, learned_acc/dynamic): the runner reports verdict=GO on ALL 6 seeds, but the TYPE-2 (metacognition) metrics are AT CHANCE — type2_auc 0.45–0.53 (chance 0.5), meta_d 0.0–0.18 (≈0), m_ratio 0.0 on every seed — while only TYPE-1 accuracy is fine (0.64–0.83). So genuine metacognitive sensitivity is NOT demonstrated. Per the pre-registered gate (type2_auc≥0.65 AND meta_d>0 AND m_ratio≥0.60) this should be NO-GO, so the runner's GO label is MIS-CALIBRATED (it is not checking type-2 sensitivity) — an instrument bug to fix before the number is trusted (silent-failure class: a gate that passes without its key metric). (2) SELF-SCHEMA→METACOG INTEGRATION (_laneC_self_schema_metacog_integration_derisk): 3/6 GO (s42/s44/s100 GO; s43/s101/s102 NEGATIVE) — seed-FRAGILE, not a robust 6-seed GO. Net: the self-model / metacognition faculty (self-report of source+strength of knowledge) is a BOUNDARY on this substrate — the type-2 read is at chance and the integration is seed-fragile — NOT a clean production wire-in yet. The honest negative is the deliverable; it launches the mechanism search (what produces genuine type-2 sensitivity — a better confidence read / a stronger meta-schema) + a gate audit.
artifacts:
  - research/findings/raw/metacog/metacog_learnedacc_s42.json
  - research/findings/raw/laneC/self_schema_metacog_s42.json
verification: local 6-seed. metacog: verdict=GO all seeds BUT type2_auc {42:0.532,43:0.453,44:0.487,100:0.511,101:0.525,102:0.502}, meta_d ≤0.18, m_ratio 0.0 all — chance-level type-2. self-schema: verdict GO 3/6 (42,44,100), NEGATIVE 3/6 (43,101,102).
---

# lane-C self-model is a BOUNDARY — metacognition at chance despite a GO label; self-schema seed-fragile

> ⚠️ **CORRECTION (2026-08-12) — see `2026-08-12-laneC-metacog-INSTRUMENT-FIX-and-balance-of-evidence-pure-spiking-meta-d.md`.** The central metacog claim here (the "second-order metacog monitor reports verdict=GO on all 6 seeds while type-2 is at chance / the GO gate is mis-calibrated and checks type-1 not type-2") is NOT reproducible and is contradicted by the very artifacts cited. The per-seed GO logic requires `type2_auc≥0.65 AND meta_d>0 AND m_ratio≥0.60`; re-running shows the `meta_rate` read (chance type-2) correctly reports NEGATIVE, and the cited `learned_acc --dynamic` artifacts show GO with GENUINE type-2 (type2_auc 0.77–0.92). The GO label was correct; this finding conflated a chance-read's type-2 numbers with a genuine-read's GO verdict. The gate is now self-tested (`--selftest`) to fail on a chance-level type-2 input. The separate self-schema→metacog INTEGRATION sub-result below (3/6 GO, seed-fragile) is unaffected and still stands.

## What ran + why it matters

The roadmap §3 self-model faculty (the brain knowing the SOURCE and STRENGTH of its knowledge — the honesty-boundary
substrate: "my familiarity monitor reads this as novel, so I'm uncertain") needs a genuine second-order signal: a
metacognitive read whose confidence DISCRIMINATES its own correct vs error trials (type-2 sensitivity, meta-d′). Two
lane-C de-risks measured this 6-seed.

## Result — the type-2 signal is at chance, and a GO gate is mis-calibrated

<!--derived-->
- **Second-order metacog monitor:** verdict=GO on all 6 seeds, but the metacognition metrics are AT CHANCE —
  `type2_auc` 0.45–0.53 (chance = 0.5), `meta_d` ≈0 (0.0–0.18), `m_ratio` 0.0 every seed. Type-1 accuracy is fine
  (0.64–0.83), so the task works but the CONFIDENCE read carries no information about correctness. **The GO label is
  therefore wrong for the faculty** — the pre-registered gate (`type2_auc≥0.65 AND meta_d>0 AND m_ratio≥0.60`) is NOT
  met, so the runner's GO criterion is not actually checking type-2 sensitivity. That is a silent-failure-class instrument
  bug (a gate that passes without its key metric) — fix the GO logic before trusting the label.
- **Self-schema→metacog integration:** 3/6 GO — seed-fragile (GO on 42/44/100, NEGATIVE on 43/101/102), not a robust
  6-seed result.

## The honest conclusion (the deliverable)

The self-model / metacognition faculty is a BOUNDARY on this substrate: the type-2 metacognitive read is at chance and
the self-schema integration is seed-fragile — so it is NOT a clean production wire-in (a self-report of confidence would
not track actual correctness). The honest negative maps the wall and launches the next mechanism search: (a) FIX the
metacog GO gate to actually require type-2 sensitivity (audit `_second_order_metacog_monitor_derisk`'s go logic); (b) find
the mechanism that produces genuine meta-d′ > 0 (a richer confidence feature / a stronger slow-NMDA meta-schema / more
evidence integration) before the honesty-boundary self-report can be wired onto the live turn.
