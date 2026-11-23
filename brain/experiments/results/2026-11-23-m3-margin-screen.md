# Capability: Episodic recall (fetch selection) — M3 margin-gated gain, zero-cost screen
# Record: brain/research/2026-11-23-biggest-defect-read.md (candidate M3); no model loaded, no life run
# Expected-fail: n/a (screen on banked rows). DEV numbers, 3 seeds, NOT a claim
## Verdict: DOES NOT SEPARATE (a verdict on these recorded fields at n=67, not on margin-gating as a method)

Data: `_scratch/stage2/read/lifeKP_s319-321.json`. Rows = the recall-voice prereg §2 told rows
(`brain/research/2026-11-23-recall-voice-dev-prereg.md`): `res["probes"]`, arm intact / hippocampus-off, line trunk, no
tag, nights 5-15, item in the ledger with `telling` truthy, `ok` not None; paired by (seed, night, row index, item,
question `you`). n = 223; intact right 123/223, hippocampus-off right 138/223 (the read's numbers). Hurt (intact wrong,
off right) 41, helped (intact right, off wrong) 26, same 156. Per seed hurt/helped: s319 6/10, s320 14/11, s321 21/5.

Recorded fetch fields: the `fetch` dict (idx, sim_raw, margin_raw, margin_w, raw_winner, weighted_only_winner,
confirmed) exists ONLY on `intact-frozen` rows (the M-use/freeze probe), not on `intact` rows, which carry only
`recalled`, `fetched`, `right_episode`, `wrong_fetch`. So each intact row is joined to its intact-frozen twin by the same
key (night, row index, item, question); all 67 hurt+helped rows have a twin with `fetch`. The twin's recalled text
equals the intact row's on 54/67 (hurt 35/41, helped 19/26), so the twin is a proxy, not always the same fetch.

AUROC (hurt scored higher; 0.5 = none), all 67 hurt+helped rows:
| field | s319 (6v10) | s320 (14v11) | s321 (21v5) | pooled (41v26) |
|---|---|---|---|---|
| sim_raw | 0.22 | 0.66 | 0.45 | 0.42 |
| margin_raw | 0.35 | 0.61 | 0.44 | 0.44 |
| margin_w | 0.40 | 0.57 | 0.50 | 0.43 |

Reading: no field reaches AUROC >= 0.70 in the same direction on 3/3 seeds; the direction flips between seeds and every
pooled value is within 0.08 of 0.5. Fetches that hurt and fetches that helped look the same on every recorded
selection statistic, so a margin/sim gate on `reinstate` gain would not tell them apart by these fields. By the intact
row's own fields, hurt rows are 8 right episode / 28 wrong episode / 5 no fetch, helped 6 / 16 / 4: the right episode
is fetched on 8 hurt rows, where no selection-confidence gate can help. Caveats: n per seed 16/25/26 (AUROC SE about
0.12-0.15, so only a large effect could show); fields come from the frozen twin; unrecorded quantities (the
strength-weighted score of the winner, winner identity vs margin) are not in the rows. Not tested: a gate using
right_episode (an oracle label, not a statistic).
Script: `brain/experiments/results/2026-11-23-m3-margin-screen/m3b.py`. No frozen file touched.
