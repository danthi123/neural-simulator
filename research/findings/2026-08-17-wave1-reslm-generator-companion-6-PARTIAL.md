---
type: finding
status: partial
date: 2026-08-17
mechanism: wave1-banking
---
## reslm-generator-companion-6 — learned-W_in companion to the fixed reservoir generator (6-seed) — PARTIAL (rate-GO / spike-BOUNDARY)

**Result.** The RESLM generator's companion process = a plastic input projection W_in learned alongside the fixed spiking reservoir, tested as learn-W_in vs fixed-random-W_in on held-out next-class generalization (class-structured Markov task, one synonym/class held out). 6 seeds (42/43/44/100/101/102), noise=0.

learn_heldout = 0.900 in every config (chance 0.167). In the STRONG-confound regime (raw/_reslm_generalize_confound.json) the companion is 6-seed robust: fixed collapses to ~0.322, margin +0.578, positive 12/12. In the WEAK-confound regime (raw/_reslm_generalize_6seed.json, the named file) a fixed random projection already generalizes (JL), so margin is 0.0 in 6/8 configs (max +0.122) — near-null on its own. <!--derived--> (0.322/0.578/0.122 are cross-config aggregates/margins; per-config rows in the cited artifacts)

**Controls that make the confound-regime result load-bearing.** noise=0 deterministic (structural, not noise artifact — this fixed a prior mis-instrument); held-out synonyms force generalization not lookup; learn_train==fixed_train==0.888 (equal training fit, so the held-out gap is not capacity/fitting); chance floor 0.167. The on-bridge sibling adds input-lesion->chance, label-scramble->chance, dw_rec==0, no weight transport.

**Residual (why PARTIAL, not GO).** (1) Rate (numpy) synthetic instrument, not the generator producing language. (2) The margin is regime-dependent — zero benefit when JL already separates inputs. (3) Like-for-like-verified sibling finding: the mechanism realizes on spikes but the generalization BENEFIT does NOT transfer rate->spike (BDSP coarseness boundary, spiking margin 0.00). Production is spiking, so the standing is a boundary, not a production GO.

Files: raw/_reslm_generalize_6seed.json (named), raw/_reslm_generalize_confound.json (load-bearing). No .prov.json sidecars exist. Runner: research/runners/_reslm_generalize_rate_check.py.

Note: the `_reslm_generalize_6seed.json` raw JSON is a historical pre-gate artifact (records no backend/device) retained at its origin path (cited above); the load-bearing `_reslm_generalize_confound.json` is on main and this finding carries the verdict evidence.
