# D2-PR DEV read (prereg 2026-11-21-replay-priority-prereg.md, section 3)
Seeds 319/320/321 (DEV). B = lifeS1F, arms PR-l (`loss`) and PR-x (`strength_x_loss`). All values read from the JSONs by `d2pr_bars.py` (brain.eval.life_leak, life_echo, life_table, life_bars). Bars are evaluated per seed and pooled as the prereg states.

## 3.1 Engagement (nights 5 and 10; permanent share and top-6 share lower than B on 3/3 seeds)

**Comparison used.** B's night records have no `replay_priority` block (no per-episode weights), and its `replay_share` lumps the permanent exchanges into class `other`, so B's permanent share and top-6 share cannot be read from the B JSON. Each PR night record carries `strength_permanent_share` / `strength_top6_share`: the strength-only weighting (exactly B's rule) computed on the SAME night store, inside the same night. That within-night reference is what B would have replayed from that state, so the paired comparison is actual (PR weighting) vs that reference. It is exact only to the degree the PR life's store at that night equals B's: check below that the B `replay_share` class values equal the PR record's (identical = same store and same strengths).

| arm | seed | night | permanent share | strength-only ref | top-6 share | strength-only ref | max share | ref | lower perm? | lower top6? |
|---|---|---|---|---|---|---|---|---|---|---|
| PR-l | 319 | 5 | 0.036 | 0.298 | 0.436 | 0.436 | 0.087 | 0.104 | True | True |
| PR-l | 319 | 10 | 0.019 | 0.248 | 0.303 | 0.410 | 0.072 | 0.117 | True | True |
| PR-l | 320 | 5 | 0.046 | 0.270 | 0.426 | 0.450 | 0.085 | 0.099 | True | True |
| PR-l | 320 | 10 | 0.021 | 0.234 | 0.295 | 0.402 | 0.078 | 0.106 | True | True |
| PR-l | 321 | 5 | 0.029 | 0.290 | 0.596 | 0.439 | 0.144 | 0.101 | True | False |
| PR-l | 321 | 10 | 0.028 | 0.244 | 0.321 | 0.409 | 0.099 | 0.100 | True | True |
| PR-x | 319 | 5 | 0.086 | 0.298 | 0.598 | 0.436 | 0.121 | 0.104 | True | False |
| PR-x | 319 | 10 | 0.092 | 0.248 | 0.517 | 0.410 | 0.169 | 0.117 | True | False |
| PR-x | 320 | 5 | 0.160 | 0.270 | 0.488 | 0.450 | 0.104 | 0.099 | True | False |
| PR-x | 320 | 10 | 0.109 | 0.234 | 0.439 | 0.402 | 0.120 | 0.106 | True | False |
| PR-x | 321 | 5 | 0.105 | 0.290 | 0.537 | 0.439 | 0.124 | 0.101 | True | False |
| PR-x | 321 | 10 | 0.148 | 0.244 | 0.495 | 0.409 | 0.141 | 0.100 | True | False |

Reference sanity (does the strength-only reference match across the two arms, and B's class replay_share match the PR record's?):

| seed | night | ref perm PR-l / PR-x | ref top6 PR-l / PR-x | B replay_share == PR-l record | == PR-x record |
|---|---|---|---|---|---|
| 319 | 5 | 0.2981 / 0.2981 | 0.4363 / 0.4363 | True | True |
| 319 | 10 | 0.2483 / 0.2483 | 0.4098 / 0.4098 | True | True |
| 320 | 5 | 0.2698 / 0.2698 | 0.4504 / 0.4504 | True | True |
| 320 | 10 | 0.2337 / 0.2337 | 0.4017 / 0.4017 | True | True |
| 321 | 5 | 0.2897 / 0.2897 | 0.4394 / 0.4394 | True | True |
| 321 | 10 | 0.2436 / 0.2436 | 0.4093 / 0.4093 | True | True |

Class-level cross-check against B (B `replay_share` normalised by life_table.replay_class_shares vs the PR arm's actual `shares_by_class`; B has no permanent class, it sits inside `other`):

| arm | seed | night | class: B share -> arm share |
|---|---|---|---|
| PR-l | 319 | 5 | AB 0.111->0.208; D 0.083->0.091; E 0.058->0.105; F 0.175->0.269; H 0.057->0.041; T 0.048->0.017; other 0.467->0.233; (arm permanent 0.036) |
| PR-l | 319 | 10 | AB 0.053->0.089; D 0.163->0.102; E 0.058->0.060; F 0.198->0.302; H 0.027->0.028; T 0.055->0.081; other 0.446->0.320; (arm permanent 0.019) |
| PR-l | 320 | 5 | AB 0.113->0.134; D 0.113->0.103; E 0.063->0.096; F 0.137->0.245; H 0.098->0.112; T 0.046->0.051; other 0.431->0.213; (arm permanent 0.046) |
| PR-l | 320 | 10 | AB 0.056->0.101; D 0.143->0.085; E 0.065->0.064; F 0.188->0.384; H 0.057->0.039; T 0.054->0.039; other 0.438->0.268; (arm permanent 0.021) |
| PR-l | 321 | 5 | AB 0.118->0.131; D 0.038->0.102; E 0.064->0.124; F 0.140->0.339; H 0.100->0.043; T 0.047->0.017; other 0.494->0.215; (arm permanent 0.029) |
| PR-l | 321 | 10 | AB 0.056->0.099; D 0.052->0.128; E 0.060->0.055; F 0.193->0.364; H 0.086->0.044; T 0.054->0.046; other 0.499->0.236; (arm permanent 0.028) |
| PR-x | 319 | 5 | AB 0.111->0.080; D 0.083->0.143; E 0.058->0.104; F 0.175->0.281; H 0.057->0.051; T 0.048->0.019; other 0.467->0.237; (arm permanent 0.086) |
| PR-x | 319 | 10 | AB 0.053->0.066; D 0.163->0.262; E 0.058->0.037; F 0.198->0.321; H 0.027->0.026; T 0.055->0.021; other 0.446->0.176; (arm permanent 0.092) |
| PR-x | 320 | 5 | AB 0.113->0.063; D 0.113->0.139; E 0.063->0.117; F 0.137->0.240; H 0.098->0.065; T 0.046->0.013; other 0.431->0.204; (arm permanent 0.160) |
| PR-x | 320 | 10 | AB 0.056->0.058; D 0.143->0.146; E 0.065->0.046; F 0.188->0.358; H 0.057->0.050; T 0.054->0.042; other 0.438->0.192; (arm permanent 0.109) |
| PR-x | 321 | 5 | AB 0.118->0.084; D 0.038->0.085; E 0.064->0.130; F 0.140->0.263; H 0.100->0.070; T 0.047->0.050; other 0.494->0.213; (arm permanent 0.105) |
| PR-x | 321 | 10 | AB 0.056->0.054; D 0.052->0.172; E 0.060->0.036; F 0.193->0.277; H 0.086->0.074; T 0.054->0.019; other 0.499->0.219; (arm permanent 0.148) |

**3.1 PR-l: FAIL, so the arm is UNDEFINED per prereg 3.1** (5/6 seed-nights both shares lower; the bar needs 3/3 seeds on each of nights 5 and 10).
**3.1 PR-x: FAIL, so the arm is UNDEFINED per prereg 3.1** (0/6 seed-nights both shares lower; the bar needs 3/3 seeds on each of nights 5 and 10).

## 3.2 Primary: stock self-reply leak (life_leak, nights 10 and 15, intact + hippocampus-off pooled per seed)

| arm | seed | leaks / n (nights 10+15) | share | night 10 | night 15 |
|---|---|---|---|---|---|
| B | 319 | 39/156 | 0.250 | 27/82 | 12/74 |
| B | 320 | 54/152 | 0.355 | 28/82 | 26/70 |
| B | 321 | 42/152 | 0.276 | 24/82 | 18/70 |
| PR-l | 319 | 17/156 | 0.109 | 17/82 | 0/74 |
| PR-l | 320 | 18/152 | 0.118 | 17/82 | 1/70 |
| PR-l | 321 | 41/152 | 0.270 | 10/82 | 31/70 |
| PR-x | 319 | 22/156 | 0.141 | 1/82 | 21/74 |
| PR-x | 320 | 17/152 | 0.112 | 15/82 | 2/70 |
| PR-x | 321 | 35/152 | 0.230 | 21/82 | 14/70 |

**3.2 PR-l: PASS**. Below B per seed: {319: True, 320: True, 321: True}. Pooled 76/460 = 0.1652 vs B 135/460 = 0.2935; relative reduction 43.7% (needs >= 30%).
**3.2 PR-x: PASS**. Below B per seed: {319: True, 320: True, 321: True}. Pooled 74/460 = 0.1609 vs B 135/460 = 0.2935; relative reduction 45.2% (needs >= 30%).

## 3.3 Guards (any fail = KILL)

### Guard H: hippocampus-off H recall, nights 5-15

Scoring: per hippocampus-off trunk probe night, `summary.H_right` (count of kind-H rows with `ok` true) out of the night's H rows (6), so the value is already 'of 6'; the per-seed value is the mean over probe nights 5-15 (nights 5-15 that have a hippocampus-off probe, all 11). Bar: arm mean >= B mean - 0.5 on every seed.

| seed | B mean /6 | PR-l mean /6 | diff | PR-x mean /6 | diff |
|---|---|---|---|---|---|
| 319 | 2.545 | 2.545 | +0.000 | 2.182 | -0.364 |
| 320 | 1.818 | 3.182 | +1.364 | 2.455 | +0.636 |
| 321 | 1.818 | 1.455 | -0.364 | 1.636 | -0.182 |

Per night (B / PR-l / PR-x, of 6):

| seed | n5 | n6 | n7 | n8 | n9 | n10 | n11 | n12 | n13 | n14 | n15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 5/5/3 | 6/0/4 | 2/0/3 | 2/4/1 | 2/1/2 | 2/3/2 | 2/4/2 | 2/2/2 | 2/3/1 | 3/3/2 | 0/3/2 |
| 320 | 2/2/5 | 2/2/2 | 1/5/2 | 2/5/2 | 1/3/4 | 2/3/3 | 2/3/3 | 2/4/1 | 2/4/1 | 2/2/2 | 2/2/2 |
| 321 | 3/2/5 | 2/0/1 | 2/3/2 | 2/1/1 | 1/1/2 | 1/2/1 | 2/2/2 | 1/2/1 | 2/1/1 | 2/0/1 | 2/2/1 |

### Guard direct: hippocampus-off recall of day-1-2 facts

Definition (as life_table.extra_rows): hippocampus-off trunk battery rows of kind H or rotating whose ledger item has min(told_days) <= 2 (day-1-2 facts; all six H items are told on day 1, the dan/nova facts have no telling but are H rows), scored with `ok`. Rate pooled over probe nights 5-15 per seed, converted to /6 as rate x 6. Bar: arm >= B - 1 (of 6) on every seed. Nights 1-15 shown beside.

| seed | B k/n (rate x6) | PR-l | diff /6 | PR-x | diff /6 | nights 1-15: B / PR-l / PR-x (rate x6) |
|---|---|---|---|---|---|---|
| 319 | 30/74 (2.43) | 31/74 (2.51) | +0.08 | 26/74 (2.11) | -0.32 | 46/103 (2.68) / 52/103 (3.03) / 51/103 (2.97) |
| 320 | 20/73 (1.64) | 40/73 (3.29) | +1.64 | 30/73 (2.47) | +0.82 | 47/102 (2.76) / 52/102 (3.06) / 47/102 (2.76) |
| 321 | 21/76 (1.66) | 19/76 (1.50) | -0.16 | 19/76 (1.50) | -0.16 | 36/105 (2.06) / 41/105 (2.34) / 40/105 (2.29) |

### Guard collapse (life_bars.language: collapse = most-repeated reply share >= 0.4 on any trunk probe night of the arm)

| probe arm | B seeds-ever (max share per seed) | PR-l | PR-x |
|---|---|---|---|
| both-off | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) | 0 (0.06, 0.06, 0.06) |
| consolidated-off | 1 (0.50, 0.06, 0.06) | 1 (0.50, 0.06, 0.06) | 1 (0.50, 0.06, 0.06) |
| hippocampus-off | 2 (0.45, 0.55, 0.30) | 1 (0.30, 0.50, 0.30) | 3 (0.40, 0.50, 0.55) |
| intact | 2 (0.40, 0.40, 0.30) | 0 (0.30, 0.25, 0.30) | 2 (0.30, 0.75, 0.50) |
| intact-frozen | 2 (0.40, 0.55, 0.30) | 0 (0.30, 0.25, 0.35) | 2 (0.30, 0.75, 0.40) |

Collapse 'not worse' = for every probe arm the number of collapsed seeds is <= B's (strict reading; PR-x hippocampus-off 3 seeds vs B 2 is the failing cell, though LG-COL's own <=3/6 rule still passes). Also LG-COL / KILL-10 as life_bars reports: B: LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds); PR-l: LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds); PR-x: LG-COL DEV (3 seeds), KILL-10 DEV (3 seeds)

### Guard language KL, night 15 (bar: <= 1.25 x B per seed)

| seed | B KL n15 | PR-l KL | ratio | PR-x KL | ratio | NLL on/off n15 B | PR-l | PR-x |
|---|---|---|---|---|---|---|---|---|
| 319 | 2.9333 | 1.9118 | 0.652 | 2.2596 | 0.770 | 4.747/1.931 | 3.731/1.931 | 3.968/1.931 |
| 320 | 2.3791 | 2.3616 | 0.993 | 2.0892 | 0.878 | 4.107/1.931 | 4.186/1.931 | 3.841/1.931 |
| 321 | 2.8580 | 1.8590 | 0.650 | 1.8687 | 0.654 | 4.674/1.931 | 3.633/1.931 | 3.561/1.931 |

Whole-transcript read (3 sonnet readers plus refuters): NOT computed here; it needs the blind readers (not in the JSONs). UNDEFINED in this script; the guards below are the four numeric ones.

- PR-l: H-off PASS {319: True, 320: True, 321: True}; direct d1-2 PASS {319: True, 320: True, 321: True}; collapse PASS; KL n15 PASS {319: True, 320: True, 321: True}
- PR-x: H-off PASS {319: True, 320: True, 321: True}; direct d1-2 PASS {319: True, 320: True, 321: True}; collapse FAIL; KL n15 PASS {319: True, 320: True, 321: True}

## 3.4 Secondary (report only)

### Hippocampus-off recall of once-told E and F facts, nights 5-15

(a) battery rows (kinds H / rotating / neardup, `ok`) whose ledger class is E or F, hippocampus-off trunk; (b) cue-ladder direct cue C0 rows (hippocampus-off, tracks E and F, all nights), `ok`.

| seed | cls | B battery k/n | PR-l | PR-x | B cue C0 | PR-l | PR-x |
|---|---|---|---|---|---|---|---|
| 319 | E | 1/4 | 2/4 | 2/4 | 0/6 | 2/6 | 1/6 |
| 319 | F | 0/10 | 2/10 | 2/10 | 3/18 | 5/18 | 5/18 |
| 320 | E | 1/6 | 2/6 | 0/6 | 2/6 | 2/6 | 2/6 |
| 320 | F | 3/10 | 3/10 | 2/10 | 5/18 | 2/18 | 9/18 |
| 321 | E | 2/3 | 1/3 | 0/3 | 3/6 | 3/6 | 1/6 |
| 321 | F | 1/10 | 3/10 | 4/10 | 4/18 | 3/18 | 3/18 |
| pooled | E | 4/13 (0.308) | 5/13 (0.385) | 2/13 (0.154) | 5/18 (0.278) | 7/18 (0.389) | 4/18 (0.222) |
| pooled | F | 4/30 (0.133) | 8/30 (0.267) | 8/30 (0.267) | 12/54 (0.222) | 10/54 (0.185) | 17/54 (0.315) |

### Wrong fetches and hub top-1 (hippocampal read-outs; PR should not change them)

wrong fetch = life_bars._h_wrong (kind-H rows with wrong_fetch, all arms and probes), plus the intact-arm wrong/src sum over nights 5-15 (life_table.seed_rows); hub top-1 = life_table.extra_rows live_top1 (share of the most fetched episode among live fetches) mean and max over nights 5-15.

| seed | arm | _h_wrong wf / H rows | intact wf/src (n5-15) | hub top-1 mean (max) n5-15 |
|---|---|---|---|---|
| 319 | B | 167/318 | 71/102 | 0.595 (1.000) |
| 319 | PR-l | 167/318 | 71/102 | 0.573 (1.000) |
| 319 | PR-x | 167/318 | 71/102 | 0.595 (1.000) |
| 320 | B | 105/318 | 48/92 | 0.591 (1.000) |
| 320 | PR-l | 105/318 | 48/92 | 0.621 (1.000) |
| 320 | PR-x | 102/318 | 48/92 | 0.562 (1.000) |
| 321 | B | 97/318 | 48/95 | 0.430 (0.750) |
| 321 | PR-l | 97/318 | 48/95 | 0.416 (0.670) |
| 321 | PR-x | 96/318 | 48/95 | 0.405 (0.670) |

### Interview AUROC and invented-memory rate

UNDEFINED from the results JSON: life_bars does not report them; they come from `life_interview_read` over blind-reader claims on the interview units (blind_PRl / blind_PRx / blind_S1F directories, reader output not part of the life JSONs). Not computed here.

### Adapter norm (night 15, and x night 1)

| seed | B n15 (x n1) | PR-l | PR-x | n5 B / PR-l / PR-x | n10 B / PR-l / PR-x |
|---|---|---|---|---|---|
| 319 | 660.5 (11.94x) | 586.1 (11.19x) | 630.1 (10.04x) | 263 / 225 / 253 | 488 / 420 / 455 |
| 320 | 674.7 (13.20x) | 617.0 (11.06x) | 617.2 (11.45x) | 287 / 282 / 242 | 495 / 442 / 443 |
| 321 | 692.6 (13.30x) | 588.4 (11.53x) | 571.3 (11.05x) | 302 / 187 / 217 | 530 / 394 / 405 |

### First-person echo (life_echo, report only, not a bar; nights 10+15, echo counted on told-item battery rows not scored right)

| seed | arm | probe arm | n | ok | echo | echo with key | echo share |
|---|---|---|---|---|---|---|---|
| 319 | B | intact | 16 | 2 | 5 | 4 | 0.312 |
| 319 | B | hippocampus-off | 16 | 1 | 5 | 4 | 0.312 |
| 319 | PR-l | intact | 16 | 8 | 2 | 2 | 0.125 |
| 319 | PR-l | hippocampus-off | 16 | 7 | 3 | 3 | 0.188 |
| 319 | PR-x | intact | 16 | 1 | 10 | 9 | 0.625 |
| 319 | PR-x | hippocampus-off | 16 | 1 | 7 | 6 | 0.438 |
| 320 | B | intact | 16 | 1 | 5 | 4 | 0.312 |
| 320 | B | hippocampus-off | 16 | 1 | 5 | 4 | 0.312 |
| 320 | PR-l | intact | 16 | 7 | 0 | 0 | 0.000 |
| 320 | PR-l | hippocampus-off | 16 | 6 | 2 | 1 | 0.125 |
| 320 | PR-x | intact | 16 | 2 | 5 | 5 | 0.312 |
| 320 | PR-x | hippocampus-off | 16 | 1 | 6 | 5 | 0.375 |
| 321 | B | intact | 16 | 2 | 6 | 6 | 0.375 |
| 321 | B | hippocampus-off | 16 | 3 | 7 | 6 | 0.438 |
| 321 | PR-l | intact | 16 | 5 | 3 | 3 | 0.188 |
| 321 | PR-l | hippocampus-off | 16 | 3 | 2 | 2 | 0.125 |
| 321 | PR-x | intact | 16 | 3 | 8 | 6 | 0.500 |
| 321 | PR-x | hippocampus-off | 16 | 2 | 8 | 8 | 0.500 |
| pooled | B | intact | 48 | 5 | 16 | 14 | 0.333 |
| pooled | B | hippocampus-off | 48 | 5 | 17 | 14 | 0.354 |
| pooled | PR-l | intact | 48 | 20 | 5 | 5 | 0.104 |
| pooled | PR-l | hippocampus-off | 48 | 16 | 7 | 6 | 0.146 |
| pooled | PR-x | intact | 48 | 6 | 23 | 20 | 0.479 |
| pooled | PR-x | hippocampus-off | 48 | 4 | 21 | 19 | 0.438 |

## Bar summary

| bar | PR-l | PR-x |
|---|---|---|
| 3.1 engagement | **FAIL, arm UNDEFINED per prereg 3.1** (perm and top-6 below strength-only ref on 5/6 seed-nights) | **FAIL, arm UNDEFINED per prereg 3.1** (perm and top-6 below strength-only ref on 0/6 seed-nights) |
| 3.2 leak | **PASS** (below B on 3/3 seeds; pooled 76/460=0.165 vs B 135/460=0.293, reduction 43.7% (bar >=30%)) | **PASS** (below B on 3/3 seeds; pooled 74/460=0.161 vs B 135/460=0.293, reduction 45.2% (bar >=30%)) |
| 3.3 H-off | **PASS** ({319: True, 320: True, 321: True}) | **PASS** ({319: True, 320: True, 321: True}) |
| 3.3 direct d1-2 | **PASS** ({319: True, 320: True, 321: True}) | **PASS** ({319: True, 320: True, 321: True}) |
| 3.3 collapse | **PASS** (seeds-ever per probe arm <= B) | **FAIL** (seeds-ever per probe arm <= B) |
| 3.3 KL n15 | **PASS** ({319: True, 320: True, 321: True}) | **PASS** ({319: True, 320: True, 321: True}) |
| 3.3 transcript read | **UNDEFINED** (needs blind readers, not in JSON) | **UNDEFINED** (needs blind readers, not in JSON) |

- PR-l: engagement FAIL, arm UNDEFINED per prereg 3.1; leak PASS; guards PASS (numeric guards; transcript read UNDEFINED). Prereg section 4 outcome by these numbers: engagement bar failed: arm UNDEFINED per prereg 3.1 (leak/guards shown for information).
- PR-x: engagement FAIL, arm UNDEFINED per prereg 3.1; leak PASS; guards FAIL. Prereg section 4 outcome by these numbers: KILLED (a numeric guard fails).
