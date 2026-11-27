# Life bars (3 seeds: [7, 7, 7]) - DEV, thresholds as written, counts only

_Permutation tests: 10000 perms, numpy default_rng(20261121), one-sided. Verdict label is `DEV (n seeds)` when n<6; `core` is the threshold result (5/6 applied as ceil(5n/6)) shown for logic checking only._

## Verdicts

| id | bar | verdict | core | pooled | p |
|---|---|---|---|---|---|
| P0 | Forgetting precondition P0-v2 (age ladder, hippocampus-off, LOW items) | DEV (3 seeds) UNDEFINED | UNDEFINED | {seeds_passing: 0, defined_seeds: 0, need: 1, sensitivity_age_offset0_passing: n/a (ladder), perm_stat: -, mean_matched_night_drop: -} | - |
| FM-v2 | False memory FM-v2 (never-told ladder questions, any told key, STOCK out) | DEV (3 seeds) UNDEFINED | UNDEFINED | {seeds_passing: 0, defined: 0, need: 1, mean_fm_v2: {hippocampus-off: -, intact: -, both-off: -}, mean_v1_style: {hippocampus-off: -, intact: -, both-off: -}} | - |
| SAV-G | Savings G_null_sub night 1 | DEV (3 seeds) UNDEFINED | UNDEFINED | - | - |
| SAV-dR | Savings dR_null_sub night 1 | DEV (3 seeds) UNDEFINED | UNDEFINED | - | - |
| CUE-R | R_cue at +1 (intact) and +3 (hippocampus-off), E+F pooled | DEV (3 seeds) | FAIL | {intact +1: {R_cue: 0.000, n_items: 2, seeds_positive: 0}, hippocampus-off +3: {R_cue: -, n_items: 0, seeds_positive: 0}} | {intact +1: 1.000, hippocampus-off +3: -} |
| CUE-D | direct (C0) minus uncued (C4), pooled, age <= 3 nights, headline arms | DEV (3 seeds) | PASS | {diff: 1.000, n_items: 2, C0: 1.000, C4: 0.000} | - |
| CUE-H | H has 0 wrong answers on every probe, every seed | DEV (3 seeds) UNDEFINED | UNDEFINED | {seeds_clean: 0, intact_not_right: 2, total_wrong_fetch: 17, total_H_rows: 60} | - |
| CUE-HL | HIGH-use >= LOW-use recall at matched age (hippocampus-off) | DEV (3 seeds) | PASS | {seeds_ok: 1, defined: 1, intact_arm: {seeds_ok: 1, defined: 1}} | - |
| TRF-T4 | T_life > 0 at k=4 (intact copy) | DEV (3 seeds) UNDEFINED | UNDEFINED | {mean_T: -, perm_stat: -, seeds_positive: 0, need: 1, mean_dT: -} | - |
| TRF-K | T increases with k (k=4 above k=1, pooled, intact) | DEV (3 seeds) UNDEFINED | UNDEFINED | {mean_T_k1: -, mean_T_k2: -, mean_T_k4: -} | - |
| TRF-B0 | both-off T about 0 at k=4 | DEV (3 seeds) UNDEFINED | UNDEFINED | {mean_T: -, se: -} | - |
| TRF-N0 | L-nosleep T about 0 at k=4 | DEV (3 seeds) UNDEFINED | UNDEFINED | - | - |
| TRF-C | consolidated-off reduces T at k=4 | DEV (3 seeds) UNDEFINED | UNDEFINED | {mean_reduction: -, seeds_reduced: 0, defined: 0, mean_T_consolidated_off: -} | - |
| LG-KL | night-15 KL <= 2x night-1 KL (per seed) | DEV (3 seeds) UNDEFINED | UNDEFINED | {seeds_ok: 0, defined: 0} | - |
| LG-COL | collapse on no more than 3/6 seeds, per arm | DEV (3 seeds) | PASS | {both-off: {seeds_ever: 0, seeds_at_last_probe: 0}, consolidated-off: {seeds_ever: 0, seeds_at_last_probe: 0}, hippocampus-off: {seeds_ever: 0, seeds_at_last_probe: 0}, intact: {seeds_ever: 0, seeds_at_last_probe: 0}, intact-frozen: {seeds_ever: 0, seeds_at_last_probe: 0}} | - |
| KILL-10 | section-10 per-seed kill (collapse on 2 consecutive probe nights, as amended by 13) | DEV (3 seeds) | PASS | {triggered: []} | - |

### P0 Forgetting precondition P0-v2 (age ladder, hippocampus-off, LOW items)

- definition: per seed: item-weighted ladder recall(age 1: 12 ladder + 8 matched-night controls) - recall(age>=10) >= 0.15 AND mean log-odds(age>=10) < log-odds(age 1) (span rows) AND recall(age>=10) - slot-matched never-told floor (nights 11-15, per question) >= 0.10; VALID if >=5/6 seeds
- registered bar: VALID if >=5/6 seeds pass; else savings UNDEFINED (design 2026-11-22 section 3)
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: [{'seed': 7, 'instrument': 'age-ladder', 'recall_age1': 1.0, 'n_items_age1': 2, 'recall_age10p': None, 'n_items_age10p': 0, 'recall_controls': None, 'n_controls': 0, 'matched_night_drop': None, 'logodds_age1': 5.2872, 'n_lo_age1': 10, 'logodds_age10p': None, 'n_lo_age10p': 0, 'drop': None, 'floor_margin': None, 'slot_floor': None, 'floor_questions': 0, 'floor_rows': 0, 'H_off': 0.9167, 'item_means_age1': {'f07': 1.0, 'f56': 1.0}, 'item_means_age10p': {}, 'clauses': None, 'pass': None}, {'seed': 7, 'instrument': 'age-ladder', 'recall_age1': 1.0, 'n_items_age1': 2, 'recall_age10p': None, 'n_items_age10p': 0, 'recall_controls': None, 'n_controls': 0, 'matched_night_drop': None, 'logodds_age1': 4.7521, 'n_lo_age1': 10, 'logodds_age10p': None, 'n_lo_age10p': 0, 'drop': None, 'floor_margin': None, 'slot_floor': None, 'floor_questions': 0, 'floor_rows': 0, 'H_off': 1.0, 'item_means_age1': {'f07': 1.0, 'f56': 1.0}, 'item_means_age10p': {}, 'clauses': None, 'pass': None}, {'seed': 7, 'instrument': 'age-ladder', 'recall_age1': 1.0, 'n_items_age1': 2, 'recall_age10p': None, 'n_items_age10p': 0, 'recall_controls': None, 'n_controls': 0, 'matched_night_drop': None, 'logodds_age1': 4.903, 'n_lo_age1': 10, 'logodds_age10p': None, 'n_lo_age10p': 0, 'drop': None, 'floor_margin': None, 'slot_floor': None, 'floor_questions': 0, 'floor_rows': 0, 'H_off': 0.9167, 'item_means_age1': {'f07': 1.0, 'f56': 1.0}, 'item_means_age10p': {}, 'clauses': None, 'pass': None}]
- pooled: {seeds_passing: 0, defined_seeds: 0, need: 1, sensitivity_age_offset0_passing: n/a (ladder), perm_stat: -, mean_matched_night_drop: -}; p: -
- note: a seed with an empty age bin is UNDEFINED and counts as not passing; the pooled permutation is reported beside the 5/6 count

### FM-v2 False memory FM-v2 (never-told ladder questions, any told key, STOCK out)

- definition: per question then averaged: hippocampus-off nights 11-15, intact and both-off night 15; per seed intact <= both-off + 0.10 AND hippocampus-off <= both-off + 0.10; >=5/6 seeds
- registered bar: PASS if >=5/6 seeds; its own verdict, never gates savings (design 2026-11-22 section 3b)
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: [{'seed': 7, 'hippocampus-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {2: 0.4167}}, 'intact': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'both-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'pass': None}, {'seed': 7, 'hippocampus-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {2: 0.0}}, 'intact': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'both-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'pass': None}, {'seed': 7, 'hippocampus-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {2: 0.0833}}, 'intact': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'both-off': {'fm_v2': None, 'v1_style': None, 'questions': 0, 'rows': 0, 'by_night': {}}, 'pass': None}]
- pooled: {seeds_passing: 0, defined: 0, need: 1, mean_fm_v2: {hippocampus-off: -, intact: -, both-off: -}, mean_v1_style: {hippocampus-off: -, intact: -, both-off: -}}; p: -
- both-off / null arm: {both-off: -}
- note: v1_style = the Stage-1 rate (any told key, STOCK included) on the same rows

### SAV-G Savings G_null_sub night 1

- definition: needs --pruned-a records
- registered bar: see design 5/13
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {}
- pooled: -; p: -
- note: no pruned_a savings section in any life

### SAV-dR Savings dR_null_sub night 1

- definition: needs --pruned-a records
- registered bar: see design 5/13
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {}
- pooled: -; p: -
- note: no pruned_a savings section in any life

### CUE-R R_cue at +1 (intact) and +3 (hippocampus-off), E+F pooled

- definition: R_cue = P(C2) - max(P(C3), P(C4)); pooled item-level permutation p<0.05 and R_cue>0 at each of +1 and +3 (per-seed direction count beside it; pooled permutation is primary per 13)
- registered bar: R_cue>0 at +1 and +3 (pooled p<0.05 primary, 13); per-seed 5/6 reported
- verdict: **DEV (3 seeds)** (core FAIL)
- per seed: {intact +1: {7: 0.000}, hippocampus-off +3: {}}
- pooled: {intact +1: {R_cue: 0.000, n_items: 2, seeds_positive: 0}, hippocampus-off +3: {R_cue: -, n_items: 0, seeds_positive: 0}}; p: {intact +1: 1.000, hippocampus-off +3: -}
- both-off / null arm: {both-off +1: {R_cue: 0.000, n_items: 2}, both-off +3: {R_cue: -, n_items: 0}}

### CUE-D direct (C0) minus uncued (C4), pooled, age <= 3 nights, headline arms

- definition: mean C0 right - mean C4 right over items in (intact +1) and (hippocampus-off +3) >= 0.4
- registered bar: pooled >= 0.4
- verdict: **DEV (3 seeds)** (core PASS)
- per seed: {7: 1.000}
- pooled: {diff: 1.000, n_items: 2, C0: 1.000, C4: 0.000}; p: -

### CUE-H H has 0 wrong answers on every probe, every seed

- definition: wrong = battery row kind 'H' not RIGHT (strict scorer ok False; reader wins when run), intact arm, every non-tagged probe of every line; per-arm not-right and wrong_fetch (retrieval, not an answer) reported
- registered bar: 0 wrong on every probe on 6/6 seeds (any wrong H answer = fail)
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {7: {wrong_fetch: 17, n_H_rows: 60, not_right: 28, not_right_names_other_key: 4, wrong_fetch_by_arm: {intact: 7, hippocampus-off: 0, consolidated-off: 7, both-off: 0, intact-frozen: 3}, not_right_by_arm: {intact: 2, hippocampus-off: 1, consolidated-off: 12, both-off: 12, intact-frozen: 1}, H_n_by_arm: {intact: 12, hippocampus-off: 12, consolidated-off: 12, both-off: 12, intact-frozen: 12}}}
- pooled: {seeds_clean: 0, intact_not_right: 2, total_wrong_fetch: 17, total_H_rows: 60}; p: -

### CUE-HL HIGH-use >= LOW-use recall at matched age (hippocampus-off)

- definition: per seed: LOW-weighted mean over exactly matched ages of recall(H) - recall(LOW) >= 0; H age counted from its last told-or-re-mentioned day (schedule 'remention' turns), LOW = battery rotating/told-neardup + cue C0 rows
- registered bar: HIGH>=LOW at matched age on >=5/6 seeds
- verdict: **DEV (3 seeds)** (core PASS)
- per seed: {7: {diff: 0.250, matched_ages: 1, n_LOW: 4}}
- pooled: {seeds_ok: 1, defined: 1, intact_arm: {seeds_ok: 1, defined: 1}}; p: -
- note: an undefined seed counts as not passing

### TRF-T4 T_life > 0 at k=4 (intact copy)

- definition: T = s*(mean_sys - mean_home of lp(X)-lp(Y) over the never-told incident questions) read the morning both domains have 4 incidents told; >0 on >=5/6 seeds AND pooled permutation (domain labels permuted within seed, 10000 perms) p<0.05. RAW T (the registered quantity); dT = T - T0 is shown in the table
- registered bar: T>0 at k=4 on >=5/6 seeds (+ common-rule pooled p<0.05, 13)
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {7: -}
- pooled: {mean_T: -, perm_stat: -, seeds_positive: 0, need: 1, mean_dT: -}; p: -
- both-off / null arm: {both_off_T_k4: {7: -}}

### TRF-K T increases with k (k=4 above k=1, pooled, intact)

- definition: mean over seeds of T(k=4) > mean of T(k=1)
- registered bar: k=4 above k=1 pooled
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {k1: {7: -}, k2: {7: -}, k4: {7: -}}
- pooled: {mean_T_k1: -, mean_T_k2: -, mean_T_k4: -}; p: -

### TRF-B0 both-off T about 0 at k=4

- definition: |mean T| <= 2 SE across seeds (n>=3); no numeric threshold is registered for 'about 0', this rule is mine
- registered bar: both-off T about 0
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {7: -}
- pooled: {mean_T: -, se: -}; p: -

### TRF-N0 L-nosleep T about 0 at k=4

- definition: needs a nosleep line with transfer records
- registered bar: L-nosleep T about 0
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {}
- pooled: -; p: -
- note: no L-nosleep transfer records in these lives

### TRF-C consolidated-off reduces T at k=4

- definition: T_intact - T_consolidated-off at the consolidated-off k=4 night (same night) > 0 on >=5/6 seeds and mean > 0 (no permutation test is registered for this clause)
- registered bar: consolidated-off must reduce T
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {7: -}
- pooled: {mean_reduction: -, seeds_reduced: 0, defined: 0, mean_T_consolidated_off: -}; p: -

### LG-KL night-15 KL <= 2x night-1 KL (per seed)

- definition: KL of adapters-on vs pretrained on the frozen held-out set (nights[].lang_guard.kl); every seed must satisfy it; UNDEFINED if any seed lacks the lang_guard record at night 1 or 15
- registered bar: night-15 KL <= 2x night 1 (never a kill)
- verdict: **DEV (3 seeds) UNDEFINED** (core UNDEFINED)
- per seed: {7: {kl_n1: 0.174, kl_n15: -, ratio: -, nll_on_n1: 2.000, nll_off_n1: 1.931, nll_on_n15: -, nll_off_n15: -, kl_by_night: {1: 0.174, 2: 0.293}}}
- pooled: {seeds_ok: 0, defined: 0}; p: -
- note: reading a seed without lang_guard records as UNDEFINED, not a pass

### LG-COL collapse on no more than 3/6 seeds, per arm

- definition: a seed collapses in an arm if the most-repeated reply share is >=0.4 on ANY probe night of that arm (trunk probes); every arm must have <=3 collapsed seeds (counts only when n<6)
- registered bar: collapse on <=3/6 seeds per arm
- verdict: **DEV (3 seeds)** (core PASS)
- per seed: {both-off: {7: False}, consolidated-off: {7: False}, hippocampus-off: {7: False}, intact: {7: False}, intact-frozen: {7: False}}
- pooled: {both-off: {seeds_ever: 0, seeds_at_last_probe: 0}, consolidated-off: {seeds_ever: 0, seeds_at_last_probe: 0}, hippocampus-off: {seeds_ever: 0, seeds_at_last_probe: 0}, intact: {seeds_ever: 0, seeds_at_last_probe: 0}, intact-frozen: {seeds_ever: 0, seeds_at_last_probe: 0}}; p: -

### KILL-10 section-10 per-seed kill (collapse on 2 consecutive probe nights, as amended by 13)

- definition: intact-arm collapse share >=0.4 on two adjacent probe nights ends that seed's life; norm and KL clauses are dropped by 13
- registered bar: no seed ended
- verdict: **DEV (3 seeds)** (core PASS)
- per seed: {7: {seed: 7, triggered: False, night: -, first_night: -}}
- pooled: {triggered: []}; p: -
- note: would have ended: none

## P0 detail: LOW recall by age (hippocampus-off, pooled over seeds; age = nights slept) - age-ladder rows (P0-v2)

| age | recall | n |
|---|---|---|
| 1 | 1.000 | 6 |

Fits (report only): -

Offset-0 sensitivity: n/a (ladder). Drop label: none.

### v1 observation set (battery + cue-C0 + ladder, deduplicated by (item, night), ladder row kept)

| age | recall | n |
|---|---|---|
| 1 | 0.833 | 12 |

## Savings detail per branch night

| branch night | seed | G_null_sub | excl | G raw | G both-off | dR | dR_null_sub | recall A/B | trunk G_null_sub |
|---|---|---|---|---|---|---|---|---|---|

## Transfer table (T raw, T0, dT = T - T0, night)

| seed | cell | T | T0 | dT | night |
|---|---|---|---|---|---|

## Collapse by arm

| arm | seeds ever >=0.4 | seeds at last probe | max share per seed |
|---|---|---|---|
| both-off | 0 | 0 | {7: 0.056} |
| consolidated-off | 0 | 0 | {7: 0.056} |
| hippocampus-off | 0 | 0 | {7: 0.056} |
| intact | 0 | 0 | {7: 0.056} |
| intact-frozen | 0 | 0 | {7: 0.056} |

## Adapter norm per night (report only; x night 1)

| seed | 1 | 2 |
|---|---|---|
| 7 | 50.890 (1.000x) | 107.566 (2.114x) |
