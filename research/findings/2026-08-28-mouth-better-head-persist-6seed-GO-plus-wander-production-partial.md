---
type: finding
status: positive
date: 2026-08-28
verdict: TWO queued GPU verdicts harvested. (1) MOUTH better-head PERSIST is 6/6 GO — the e-prop-learned WKV read-out head (the learn-runner's better head) recovers the copied head at ratio mean 0.9273 (min 0.8906) across all 6 seeds, anti-cheat collapses 6/6, and all 6 per-seed heads persisted to disk (the {seed}-templated save fixed the prior single-seed-only bug). This is the durable brain-native mouth head for the crutch-burndown. (2) GENERATIVE-WANDER at PRODUCTION scale (n_ca3=2000, 6 seeds) is a QUALIFIED confirmation: genuine emergent attractor FORMATION holds 6/6 (emergent=True, genuine_formation=True every seed), but the blend-balance / novelty quality is MIXED across seeds (blend_balance_min ranges 0.0-0.5) — the mechanism scales but its blend quality is a tuning residual, not a clean GO.
mechanism: harvest of the queued eprop-learn 6-seed persist + the generative-wander production-scale (n_ca3=2000) 6-seed verify
lane: e-mouth-fluency
artifacts:
  - research/findings/raw/_persist_eprop_head_scope/eprop_learn_persist_6seed.json
  - research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json
runner: research/runners/_wkv_mouth_readout_eprop_learn_derisk.py
---

# Harvest: mouth better-head persist 6/6 GO (ratio 0.9273) + generative-wander production-scale qualified (formation 6/6, blend mixed)

## (1) Mouth better-head persist — 6/6 GO

Artifact: `research/findings/raw/_persist_eprop_head_scope/eprop_learn_persist_6seed.json` (cupy, 6 seeds, `--save-w-hat` per-seed).

- **`go_count = 6` of 6, `go_5of6 = True`.**
- **`sub_recov_ratio_mean = 0.9273`, `sub_recov_ratio_min = 0.8906`** — every seed recovers >0.89 of the copied head (the better learn-runner head).
- **`anticheats_collapse_count = 6`** (shuffle collapses on all seeds — genuinely substrate).
- All 6 per-seed heads persisted: `wkv_eprop_learned_head_0p94_s{42,43,44,100,101,102}.npz` (the `{seed}`-templated save path fixed the prior bug where only the last seed persisted). These are the durable brain-native mouth heads, loadable via `BRAIN_WKV_MOUTH_LEARNED_HEAD` per seed.

## (2) Generative-wander at production scale — qualified (formation holds, blend mixed)

Artifact: `research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json` (cupy, 6 seeds, n_ca3=2000, `--emergent`).

- **`emergent = True` 6/6, `genuine_formation = True` 6/6** — the spontaneous attractor formation (DG-selected, BTSP-formed membership) holds at production scale on every seed.
- **`blend_balance_min` = `[0.0, 0.5, 0.0, 0.444, 0.04, 0.0]`, `novelty_max_overlap` = `[0.0, 0.552, 0.143, 0.458, 0.36, 0.16]`** — the blend-balance + novelty quality is INCONSISTENT across seeds (2 seeds well-balanced, the rest low). So the wander mechanism SCALES (genuine formation) but its blend quality is a tuning residual.

## What this settles + next

The mouth crutch-burndown gets its durable per-seed head (0.9273, 6/6). The continuous-life wander is CONFIRMED to form genuine emergent attractors at production scale, but the blend/novelty quality needs tuning before an idle-loop production-default (the blend-balance floor is the lever — 3/6 seeds sit near 0). Neither is a wall; the wander blend-quality is the named next rung for the continuous-life idle-loop.
