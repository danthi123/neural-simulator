# Life bars (6 seeds: [42, 43, 44, 100, 101, 102])

_Permutation tests: 10000 perms, numpy default_rng(20261121), one-sided. Verdict label is `DEV (n seeds)` when n<6; `core` is the threshold result (5/6 applied as ceil(5n/6)) shown for logic checking only._

## Verdicts

| id | bar | verdict | core | pooled | p |
|---|---|---|---|---|---|
| P0 | Forgetting precondition (hippocampus-off, LOW items) | FAIL | FAIL | {seeds_passing: 0, defined_seeds: 6, need: 5, sensitivity_age_offset0_passing: 0} | - |
| SAV-G | G_null_sub, branch night 1 (all 8 A vs 8 B, hippocampus-off minus both-off) | UNDEFINED | UNDEFINED | {mean_G_null_sub: 0.669, perm_stat: 0.669, seeds_positive: 4, need: 5} | 0.053 |
| SAV-dR | dR_null_sub at branch night 1 (recall A minus B, minus both-off dR), of 8 | UNDEFINED | UNDEFINED | {mean_dR_null_sub: 0.167, seeds_ge_1.5: 2} | - |
| CUE-R | R_cue at +1 (intact) and +3 (hippocampus-off), E+F pooled | FAIL | FAIL | {intact +1: {R_cue: 0.046, n_items: 65, seeds_positive: 3}, hippocampus-off +3: {R_cue: 0.031, n_items: 64, seeds_positive: 3}} | {intact +1: 0.057, hippocampus-off +3: 0.114} |
| CUE-D | direct (C0) minus uncued (C4), pooled, age <= 3 nights, headline arms | FAIL | FAIL | {diff: 0.233, n_items: 129, C0: 0.233, C4: 0.000} | - |
| CUE-H | H has 0 wrong answers on every probe, every seed | FAIL | FAIL | {seeds_clean: 0, intact_not_right: 311, total_wrong_fetch: 822, total_H_rows: 1908} | - |
| CUE-HL | HIGH-use >= LOW-use recall at matched age (hippocampus-off) | PASS | PASS | {seeds_ok: 6, defined: 6, intact_arm: {seeds_ok: 6, defined: 6}} | - |
| TRF-T4 | T_life > 0 at k=4 (intact copy) | FAIL | FAIL | {mean_T: 0.242, perm_stat: 0.242, seeds_positive: 4, need: 5, mean_dT: 0.591} | 0.044 |
| TRF-K | T increases with k (k=4 above k=1, pooled, intact) | FAIL | FAIL | {mean_T_k1: 0.242, mean_T_k2: 0.626, mean_T_k4: 0.242} | - |
| TRF-B0 | both-off T about 0 at k=4 | PASS | PASS | {mean_T: -0.349, se: 0.441} | - |
| TRF-N0 | L-nosleep T about 0 at k=4 | UNDEFINED | UNDEFINED | - | - |
| TRF-C | consolidated-off reduces T at k=4 | PASS | PASS | {mean_reduction: 0.588, seeds_reduced: 6, defined: 6, mean_T_consolidated_off: -0.334} | - |
| LG-KL | night-15 KL <= 2x night-1 KL (per seed) | FAIL | FAIL | {seeds_ok: 0, defined: 6} | - |
| LG-COL | collapse on no more than 3/6 seeds, per arm | PASS | PASS | {both-off: {seeds_ever: 0, seeds_at_last_probe: 0}, consolidated-off: {seeds_ever: 0, seeds_at_last_probe: 0}, hippocampus-off: {seeds_ever: 2, seeds_at_last_probe: 1}, intact: {seeds_ever: 3, seeds_at_last_probe: 1}, intact-frozen: {seeds_ever: 2, seeds_at_last_probe: 1}} | - |
| KILL-10 | section-10 per-seed kill (collapse on 2 consecutive probe nights, as amended by 13) | FAIL | FAIL | {triggered: [44]} | - |

### P0 Forgetting precondition (hippocampus-off, LOW items)

- definition: per seed: LOW recall(age>=10 nights) <= recall(age 1) - 0.15 AND mean log-odds(age>=10) < log-odds(age 1) AND recall(age>=10) >= never-told false rate + 0.10; VALID if >=5/6 seeds
- registered bar: VALID if >=5/6 seeds pass; else savings UNDEFINED
- verdict: **FAIL** (core FAIL)
- per seed: [{'seed': 42, 'recall_age1': 0.4, 'n_age1': 15, 'recall_age10p': 0.6667, 'n_age10p': 3, 'logodds_age1': 3.1121, 'n_lo_age1': 48, 'logodds_age10p': 2.7588, 'n_lo_age10p': 46, 'drop': -0.2667, 'floor_margin': -0.0889, 'never_false': 0.7556, 'never_n': 45, 'H_off': 0.3556, 'clauses': {'drop>=0.15': False, 'logodds_lower': True, 'above_false+0.10': False}, 'pass': False}, {'seed': 43, 'recall_age1': 0.3333, 'n_age1': 15, 'recall_age10p': 0.25, 'n_age10p': 4, 'logodds_age1': 2.1152, 'n_lo_age1': 48, 'logodds_age10p': 3.3308, 'n_lo_age10p': 46, 'drop': 0.0833, 'floor_margin': -0.15, 'never_false': 0.4, 'never_n': 45, 'H_off': 0.5667, 'clauses': {'drop>=0.15': False, 'logodds_lower': False, 'above_false+0.10': False}, 'pass': False}, {'seed': 44, 'recall_age1': 0.2, 'n_age1': 15, 'recall_age10p': 0.0, 'n_age10p': 2, 'logodds_age1': 3.092, 'n_lo_age1': 48, 'logodds_age10p': 2.6382, 'n_lo_age10p': 46, 'drop': 0.2, 'floor_margin': -0.6444, 'never_false': 0.6444, 'never_n': 45, 'H_off': 0.3778, 'clauses': {'drop>=0.15': True, 'logodds_lower': True, 'above_false+0.10': False}, 'pass': False}, {'seed': 100, 'recall_age1': 0.375, 'n_age1': 16, 'recall_age10p': 0.0, 'n_age10p': 3, 'logodds_age1': 3.4468, 'n_lo_age1': 48, 'logodds_age10p': 3.1723, 'n_lo_age10p': 46, 'drop': 0.375, 'floor_margin': -0.7333, 'never_false': 0.7333, 'never_n': 45, 'H_off': 0.5222, 'clauses': {'drop>=0.15': True, 'logodds_lower': True, 'above_false+0.10': False}, 'pass': False}, {'seed': 101, 'recall_age1': 0.0714, 'n_age1': 14, 'recall_age10p': 0.0, 'n_age10p': 4, 'logodds_age1': 2.5702, 'n_lo_age1': 48, 'logodds_age10p': 3.4529, 'n_lo_age10p': 46, 'drop': 0.0714, 'floor_margin': -0.5556, 'never_false': 0.5556, 'never_n': 45, 'H_off': 0.3333, 'clauses': {'drop>=0.15': False, 'logodds_lower': False, 'above_false+0.10': False}, 'pass': False}, {'seed': 102, 'recall_age1': 0.3333, 'n_age1': 15, 'recall_age10p': 0.0, 'n_age10p': 4, 'logodds_age1': 2.0576, 'n_lo_age1': 48, 'logodds_age10p': 2.8292, 'n_lo_age10p': 46, 'drop': 0.3333, 'floor_margin': -0.4222, 'never_false': 0.4222, 'never_n': 45, 'H_off': 0.3889, 'clauses': {'drop>=0.15': True, 'logodds_lower': False, 'above_false+0.10': False}, 'pass': False}]
- pooled: {seeds_passing: 0, defined_seeds: 6, need: 5, sensitivity_age_offset0_passing: 0}; p: -
- note: a seed with an empty age bin is UNDEFINED and counts as not passing

### SAV-G G_null_sub, branch night 1 (all 8 A vs 8 B, hippocampus-off minus both-off)

- definition: G_null_sub = mean_A(lo_off - lo_bothoff) - mean_B(same); G>0 on >=5/6 seeds AND pooled item-level permutation (A/B labels permuted within seed, 10000 perms, one-sided) p<0.05
- registered bar: G>0 on >=5/6 seeds and pooled p<0.05 (design 5 as amended by 13); valid only if P0 VALID
- verdict: **UNDEFINED** (core UNDEFINED)
- per seed: {42: 0.840, 43: 0.481, 44: 0.888, 100: 1.944, 101: -0.072, 102: -0.064}
- pooled: {mean_G_null_sub: 0.669, perm_stat: 0.669, seeds_positive: 4, need: 5}; p: 0.053
- both-off / null arm: {G_both_off_raw_mean: 0.150, note: both-off raw G is not a bar after 13 (item difficulty); reported beside}
- threshold result ignoring the P0 gate: FAIL
- note: UNDEFINED because P0 is not VALID (P0: FAIL); numbers shown are what the threshold would give: raw G mean 0.8191, excl mean 0.7663

### SAV-dR dR_null_sub at branch night 1 (recall A minus B, minus both-off dR), of 8

- definition: mean over seeds of per-seed dR_null_sub >= +1.5 (of 8)
- registered bar: recall difference at night 1 >= +1.5 of 8 pooled
- verdict: **UNDEFINED** (core UNDEFINED)
- per seed: {42: 2, 43: -2, 44: 2, 100: 1, 101: -1, 102: -1}
- pooled: {mean_dR_null_sub: 0.167, seeds_ge_1.5: 2}; p: -
- threshold result ignoring the P0 gate: FAIL
- note: UNDEFINED because P0 is not VALID (P0: FAIL); numbers shown are what the threshold would give:

### CUE-R R_cue at +1 (intact) and +3 (hippocampus-off), E+F pooled

- definition: R_cue = P(C2) - max(P(C3), P(C4)); pooled item-level permutation p<0.05 and R_cue>0 at each of +1 and +3 (per-seed direction count beside it; pooled permutation is primary per 13)
- registered bar: R_cue>0 at +1 and +3 (pooled p<0.05 primary, 13); per-seed 5/6 reported
- verdict: **FAIL** (core FAIL)
- per seed: {intact +1: {42: 0.091, 43: 0.182, 44: -0.091, 100: 0.000, 101: 0.091, 102: 0.000}, hippocampus-off +3: {42: 0.000, 43: 0.000, 44: 0.100, 100: 0.091, 101: -0.100, 102: 0.091}}
- pooled: {intact +1: {R_cue: 0.046, n_items: 65, seeds_positive: 3}, hippocampus-off +3: {R_cue: 0.031, n_items: 64, seeds_positive: 3}}; p: {intact +1: 0.057, hippocampus-off +3: 0.114}
- both-off / null arm: {both-off +1: {R_cue: 0.000, n_items: 16}, both-off +3: {R_cue: 0.000, n_items: 16}}

### CUE-D direct (C0) minus uncued (C4), pooled, age <= 3 nights, headline arms

- definition: mean C0 right - mean C4 right over items in (intact +1) and (hippocampus-off +3) >= 0.4
- registered bar: pooled >= 0.4
- verdict: **FAIL** (core FAIL)
- per seed: {42: 0.227, 43: 0.318, 44: 0.286, 100: 0.191, 101: 0.191, 102: 0.182}
- pooled: {diff: 0.233, n_items: 129, C0: 0.233, C4: 0.000}; p: -

### CUE-H H has 0 wrong answers on every probe, every seed

- definition: wrong = battery row kind 'H' not RIGHT (strict scorer ok False; reader wins when run), intact arm, every non-tagged probe of every line; per-arm not-right and wrong_fetch (retrieval, not an answer) reported
- registered bar: 0 wrong on every probe on 6/6 seeds (any wrong H answer = fail)
- verdict: **FAIL** (core FAIL)
- per seed: {42: {wrong_fetch: 137, n_H_rows: 318, not_right: 218, not_right_names_other_key: 164, wrong_fetch_by_arm: {intact: 65, hippocampus-off: 0, consolidated-off: 17, both-off: 0, intact-frozen: 55}, not_right_by_arm: {intact: 58, hippocampus-off: 58, consolidated-off: 23, both-off: 24, intact-frozen: 55}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}, 43: {wrong_fetch: 141, n_H_rows: 318, not_right: 165, not_right_names_other_key: 115, wrong_fetch_by_arm: {intact: 65, hippocampus-off: 0, consolidated-off: 19, both-off: 0, intact-frozen: 57}, not_right_by_arm: {intact: 41, hippocampus-off: 39, consolidated-off: 21, both-off: 24, intact-frozen: 40}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}, 44: {wrong_fetch: 131, n_H_rows: 318, not_right: 221, not_right_names_other_key: 161, wrong_fetch_by_arm: {intact: 61, hippocampus-off: 0, consolidated-off: 16, both-off: 0, intact-frozen: 54}, not_right_by_arm: {intact: 59, hippocampus-off: 56, consolidated-off: 24, both-off: 24, intact-frozen: 58}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}, 100: {wrong_fetch: 140, n_H_rows: 318, not_right: 178, not_right_names_other_key: 118, wrong_fetch_by_arm: {intact: 65, hippocampus-off: 0, consolidated-off: 17, both-off: 0, intact-frozen: 58}, not_right_by_arm: {intact: 44, hippocampus-off: 43, consolidated-off: 22, both-off: 24, intact-frozen: 45}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}, 101: {wrong_fetch: 123, n_H_rows: 318, not_right: 218, not_right_names_other_key: 145, wrong_fetch_by_arm: {intact: 55, hippocampus-off: 0, consolidated-off: 15, both-off: 0, intact-frozen: 53}, not_right_by_arm: {intact: 55, hippocampus-off: 60, consolidated-off: 24, both-off: 24, intact-frozen: 55}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}, 102: {wrong_fetch: 150, n_H_rows: 318, not_right: 207, not_right_names_other_key: 157, wrong_fetch_by_arm: {intact: 70, hippocampus-off: 0, consolidated-off: 20, both-off: 0, intact-frozen: 60}, not_right_by_arm: {intact: 54, hippocampus-off: 55, consolidated-off: 23, both-off: 24, intact-frozen: 51}, H_n_by_arm: {intact: 90, hippocampus-off: 90, consolidated-off: 24, both-off: 24, intact-frozen: 90}}}
- pooled: {seeds_clean: 0, intact_not_right: 311, total_wrong_fetch: 822, total_H_rows: 1908}; p: -

### CUE-HL HIGH-use >= LOW-use recall at matched age (hippocampus-off)

- definition: per seed: LOW-weighted mean over exactly matched ages of recall(H) - recall(LOW) >= 0; H age counted from its last told-or-re-mentioned day (schedule 'remention' turns), LOW = battery rotating/told-neardup + cue C0 rows
- registered bar: HIGH>=LOW at matched age on >=5/6 seeds
- verdict: **PASS** (core PASS)
- per seed: {42: {diff: 0.021, matched_ages: 9, n_LOW: 55}, 43: {diff: 0.203, matched_ages: 8, n_LOW: 56}, 44: {diff: 0.175, matched_ages: 9, n_LOW: 59}, 100: {diff: 0.213, matched_ages: 9, n_LOW: 55}, 101: {diff: 0.205, matched_ages: 8, n_LOW: 55}, 102: {diff: 0.097, matched_ages: 8, n_LOW: 55}}
- pooled: {seeds_ok: 6, defined: 6, intact_arm: {seeds_ok: 6, defined: 6}}; p: -
- note: an undefined seed counts as not passing

### TRF-T4 T_life > 0 at k=4 (intact copy)

- definition: T = s*(mean_sys - mean_home of lp(X)-lp(Y) over the never-told incident questions) read the morning both domains have 4 incidents told; >0 on >=5/6 seeds AND pooled permutation (domain labels permuted within seed, 10000 perms) p<0.05. RAW T (the registered quantity); dT = T - T0 is shown in the table
- registered bar: T>0 at k=4 on >=5/6 seeds (+ common-rule pooled p<0.05, 13)
- verdict: **FAIL** (core FAIL)
- per seed: {42: 0.251, 43: 1.119, 44: 0.050, 100: -0.066, 101: 0.580, 102: -0.482}
- pooled: {mean_T: 0.242, perm_stat: 0.242, seeds_positive: 4, need: 5, mean_dT: 0.591}; p: 0.044
- both-off / null arm: {both_off_T_k4: {42: -1.046, 43: 1.046, 44: -1.046, 100: -1.046, 101: 1.046, 102: -1.046}}

### TRF-K T increases with k (k=4 above k=1, pooled, intact)

- definition: mean over seeds of T(k=4) > mean of T(k=1)
- registered bar: k=4 above k=1 pooled
- verdict: **FAIL** (core FAIL)
- per seed: {k1: {42: -0.723, 43: 1.240, 44: -0.426, 100: -0.196, 101: 1.100, 102: 0.458}, k2: {42: -0.183, 43: 2.003, 44: 0.265, 100: 1.015, 101: 0.676, 102: -0.023}, k4: {42: 0.251, 43: 1.119, 44: 0.050, 100: -0.066, 101: 0.580, 102: -0.482}}
- pooled: {mean_T_k1: 0.242, mean_T_k2: 0.626, mean_T_k4: 0.242}; p: -

### TRF-B0 both-off T about 0 at k=4

- definition: |mean T| <= 2 SE across seeds (n>=3); no numeric threshold is registered for 'about 0', this rule is mine
- registered bar: both-off T about 0
- verdict: **PASS** (core PASS)
- per seed: {42: -1.046, 43: 1.046, 44: -1.046, 100: -1.046, 101: 1.046, 102: -1.046}
- pooled: {mean_T: -0.349, se: 0.441}; p: -

### TRF-N0 L-nosleep T about 0 at k=4

- definition: needs a nosleep line with transfer records
- registered bar: L-nosleep T about 0
- verdict: **UNDEFINED** (core UNDEFINED)
- per seed: {}
- pooled: -; p: -
- note: no L-nosleep transfer records in these lives

### TRF-C consolidated-off reduces T at k=4

- definition: T_intact - T_consolidated-off at the consolidated-off k=4 night (same night) > 0 on >=5/6 seeds and mean > 0 (no permutation test is registered for this clause)
- registered bar: consolidated-off must reduce T
- verdict: **PASS** (core PASS)
- per seed: {42: 1.089, 43: 0.545, 44: 0.759, 100: 0.354, 101: 0.623, 102: 0.156}
- pooled: {mean_reduction: 0.588, seeds_reduced: 6, defined: 6, mean_T_consolidated_off: -0.334}; p: -

### LG-KL night-15 KL <= 2x night-1 KL (per seed)

- definition: KL of adapters-on vs pretrained on the frozen held-out set (nights[].lang_guard.kl); every seed must satisfy it; UNDEFINED if any seed lacks the lang_guard record at night 1 or 15
- registered bar: night-15 KL <= 2x night 1 (never a kill)
- verdict: **FAIL** (core FAIL)
- per seed: {42: {kl_n1: 0.164, kl_n15: 2.095, ratio: 12.813, nll_on_n1: 2.010, nll_off_n1: 1.931, nll_on_n15: 3.811, nll_off_n15: 1.931, kl_by_night: {1: 0.164, 2: 0.401, 3: 0.563, 4: 0.670, 5: 0.781, 6: 0.921, 7: 1.020, 8: 1.350, 9: 1.410, 10: 1.562, 11: 1.784, 12: 1.946, 13: 2.018, 14: 2.393, 15: 2.095}}, 43: {kl_n1: 0.141, kl_n15: 2.260, ratio: 16.078, nll_on_n1: 2.000, nll_off_n1: 1.931, nll_on_n15: 4.066, nll_off_n15: 1.931, kl_by_night: {1: 0.141, 2: 0.352, 3: 0.470, 4: 0.686, 5: 0.798, 6: 1.042, 7: 1.163, 8: 1.480, 9: 1.648, 10: 1.725, 11: 1.755, 12: 2.111, 13: 2.094, 14: 2.470, 15: 2.260}}, 44: {kl_n1: 0.135, kl_n15: 2.313, ratio: 17.159, nll_on_n1: 2.002, nll_off_n1: 1.931, nll_on_n15: 3.983, nll_off_n15: 1.931, kl_by_night: {1: 0.135, 2: 0.330, 3: 0.526, 4: 0.614, 5: 0.922, 6: 1.029, 7: 1.367, 8: 1.233, 9: 1.627, 10: 1.566, 11: 1.673, 12: 1.595, 13: 1.865, 14: 1.958, 15: 2.313}}, 100: {kl_n1: 0.172, kl_n15: 2.414, ratio: 13.997, nll_on_n1: 2.018, nll_off_n1: 1.931, nll_on_n15: 4.195, nll_off_n15: 1.931, kl_by_night: {1: 0.172, 2: 0.352, 3: 0.704, 4: 0.756, 5: 1.187, 6: 1.175, 7: 1.256, 8: 1.452, 9: 1.641, 10: 1.644, 11: 1.916, 12: 1.921, 13: 2.079, 14: 2.210, 15: 2.414}}, 101: {kl_n1: 0.141, kl_n15: 2.324, ratio: 16.526, nll_on_n1: 1.989, nll_off_n1: 1.931, nll_on_n15: 3.997, nll_off_n15: 1.931, kl_by_night: {1: 0.141, 2: 0.330, 3: 0.511, 4: 0.862, 5: 1.071, 6: 1.251, 7: 1.527, 8: 1.498, 9: 1.864, 10: 1.789, 11: 2.288, 12: 1.846, 13: 2.015, 14: 2.352, 15: 2.324}}, 102: {kl_n1: 0.170, kl_n15: 2.412, ratio: 14.181, nll_on_n1: 2.024, nll_off_n1: 1.931, nll_on_n15: 4.251, nll_off_n15: 1.931, kl_by_night: {1: 0.170, 2: 0.439, 3: 0.405, 4: 0.679, 5: 0.922, 6: 1.022, 7: 1.190, 8: 1.380, 9: 1.528, 10: 1.595, 11: 1.812, 12: 1.861, 13: 2.002, 14: 2.355, 15: 2.412}}}
- pooled: {seeds_ok: 0, defined: 6}; p: -

### LG-COL collapse on no more than 3/6 seeds, per arm

- definition: a seed collapses in an arm if the most-repeated reply share is >=0.4 on ANY probe night of that arm (trunk probes); every arm must have <=3 collapsed seeds (counts only when n<6)
- registered bar: collapse on <=3/6 seeds per arm
- verdict: **PASS** (core PASS)
- per seed: {both-off: {42: False, 43: False, 44: False, 100: False, 101: False, 102: False}, consolidated-off: {42: False, 43: False, 44: False, 100: False, 101: False, 102: False}, hippocampus-off: {42: False, 43: False, 44: True, 100: False, 101: True, 102: False}, intact: {42: False, 43: True, 44: True, 100: False, 101: True, 102: False}, intact-frozen: {42: False, 43: False, 44: True, 100: False, 101: True, 102: False}}
- pooled: {both-off: {seeds_ever: 0, seeds_at_last_probe: 0}, consolidated-off: {seeds_ever: 0, seeds_at_last_probe: 0}, hippocampus-off: {seeds_ever: 2, seeds_at_last_probe: 1}, intact: {seeds_ever: 3, seeds_at_last_probe: 1}, intact-frozen: {seeds_ever: 2, seeds_at_last_probe: 1}}; p: -

### KILL-10 section-10 per-seed kill (collapse on 2 consecutive probe nights, as amended by 13)

- definition: intact-arm collapse share >=0.4 on two adjacent probe nights ends that seed's life; norm and KL clauses are dropped by 13
- registered bar: no seed ended
- verdict: **FAIL** (core FAIL)
- per seed: {42: {seed: 42, triggered: False, night: -, first_night: -}, 43: {seed: 43, triggered: False, night: -, first_night: -}, 44: {seed: 44, triggered: True, night: 15, first_night: 14}, 100: {seed: 100, triggered: False, night: -, first_night: -}, 101: {seed: 101, triggered: False, night: -, first_night: -}, 102: {seed: 102, triggered: False, night: -, first_night: -}}
- pooled: {triggered: [44]}; p: -
- note: would have ended: s44 at night 15

## P0 detail: LOW recall by age (hippocampus-off, pooled over seeds; age = nights slept)

| age | recall | n |
|---|---|---|
| 1 | 0.289 | 90 |
| 2 | 0.450 | 20 |
| 3 | 0.273 | 88 |
| 4 | 0.071 | 14 |
| 5 | 0.267 | 15 |
| 6 | 0.214 | 14 |
| 7 | 0.149 | 74 |
| 8 | 0.200 | 10 |
| 9 | 0.300 | 10 |
| 10 | 0.000 | 1 |
| 11 | 0.500 | 2 |
| 12 | 0.000 | 3 |
| 13 | 0.000 | 2 |
| 14 | 0.167 | 12 |

Fits (report only): {bins_used: 11, power: {c: 0.315, rate: 0.270, R2: 0.285}, exp: {c: 0.303, rate: 0.065, R2: 0.304}}

## Savings detail per branch night

| branch night | seed | G_null_sub | excl | G raw | G both-off | dR | dR_null_sub | recall A/B | trunk G_null_sub |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 0.840 | 1.089 | 1.078 | 0.238 | 2 | 2 | 2/0 | 0.846 |
| 1 | 43 | 0.481 | 0.360 | 0.636 | 0.155 | -2 | -2 | 1/3 | -0.487 |
| 1 | 44 | 0.888 | 1.367 | 0.045 | -0.843 | 2 | 2 | 2/0 | 0.079 |
| 1 | 100 | 1.944 | 2.174 | 5.256 | 3.312 | 1 | 1 | 1/0 | 1.499 |
| 1 | 101 | -0.072 | -0.387 | -2.349 | -2.277 | -1 | -1 | 0/1 | 0.999 |
| 1 | 102 | -0.064 | -0.004 | 0.248 | 0.313 | -1 | -1 | 1/2 | 0.378 |
| 1 | mean | 0.669 | 0.766 | 0.819 | 0.150 | - | 0.167 | - | 0.552 | (p 0.053) |
| 2 | 42 | 0.976 | 1.597 | 1.213 | 0.238 | 0 | 0 | 1/1 | -0.493 |
| 2 | 43 | 1.045 | 0.796 | 1.200 | 0.155 | 0 | 0 | 1/1 | -0.409 |
| 2 | 44 | 0.909 | 1.171 | 0.066 | -0.843 | 1 | 1 | 2/1 | 0.030 |
| 2 | 100 | 4.486 | 4.951 | 7.798 | 3.312 | 2 | 2 | 2/0 | 1.662 |
| 2 | 101 | 0.711 | 0.313 | -1.567 | -2.277 | 1 | 1 | 1/0 | -0.005 |
| 2 | 102 | 1.360 | 1.799 | 1.672 | 0.313 | 1 | 1 | 1/0 | 0.276 |
| 2 | mean | 1.581 | 1.771 | 1.730 | 0.150 | - | 0.833 | - | 0.177 | (p 0.004) |
| 3 | 42 | 0.377 | 0.958 | 0.615 | 0.238 | 0 | 0 | 1/1 | -1.184 |
| 3 | 43 | 0.694 | 0.532 | 0.849 | 0.155 | 1 | 1 | 1/0 | 0.056 |
| 3 | 44 | 1.302 | 1.804 | 0.459 | -0.843 | 0 | 0 | 1/1 | 0.766 |
| 3 | 100 | 0.293 | 0.419 | 3.605 | 3.312 | -1 | -1 | 1/2 | 0.668 |
| 3 | 101 | 0.949 | 0.694 | -1.329 | -2.277 | 1 | 1 | 1/0 | 0.127 |
| 3 | 102 | -0.074 | 0.173 | 0.238 | 0.313 | 2 | 2 | 2/0 | 0.747 |
| 3 | mean | 0.590 | 0.763 | 0.740 | 0.150 | - | 0.500 | - | 0.197 | (p 0.125) |

## Transfer table (T raw, T0, dT = T - T0, night)

| seed | cell | T | T0 | dT | night |
|---|---|---|---|---|---|
| 42 | intact@k1 | -0.723 | -1.046 | 0.323 | 3 |
| 42 | intact@k2 | -0.183 | -1.046 | 0.863 | 6 |
| 42 | intact@k4 | 0.251 | -1.046 | 1.296 | 13 |
| 42 | hippocampus-off@k1 | -0.839 | -1.046 | 0.206 | 3 |
| 42 | hippocampus-off@k2 | -0.210 | -1.046 | 0.835 | 6 |
| 42 | hippocampus-off@k4 | 0.177 | -1.046 | 1.223 | 13 |
| 42 | consolidated-off@k4 | -0.986 | -1.046 | 0.059 | 15 |
| 42 | both-off@k4 | -1.046 | -1.046 | 0.000 | 15 |
| 43 | intact@k1 | 1.240 | 1.046 | 0.194 | 3 |
| 43 | intact@k2 | 2.003 | 1.046 | 0.958 | 6 |
| 43 | intact@k4 | 1.119 | 1.046 | 0.074 | 13 |
| 43 | hippocampus-off@k1 | 1.211 | 1.046 | 0.165 | 3 |
| 43 | hippocampus-off@k2 | 2.033 | 1.046 | 0.987 | 6 |
| 43 | hippocampus-off@k4 | 1.067 | 1.046 | 0.022 | 13 |
| 43 | consolidated-off@k4 | 1.016 | 1.046 | -0.029 | 15 |
| 43 | both-off@k4 | 1.046 | 1.046 | 0.000 | 15 |
| 44 | intact@k1 | -0.426 | -1.046 | 0.620 | 3 |
| 44 | intact@k2 | 0.265 | -1.046 | 1.311 | 6 |
| 44 | intact@k4 | 0.050 | -1.046 | 1.096 | 13 |
| 44 | hippocampus-off@k1 | -0.475 | -1.046 | 0.571 | 3 |
| 44 | hippocampus-off@k2 | 0.246 | -1.046 | 1.291 | 6 |
| 44 | hippocampus-off@k4 | -0.012 | -1.046 | 1.033 | 13 |
| 44 | consolidated-off@k4 | -1.009 | -1.046 | 0.037 | 15 |
| 44 | both-off@k4 | -1.046 | -1.046 | 0.000 | 15 |
| 100 | intact@k1 | -0.196 | -1.046 | 0.850 | 3 |
| 100 | intact@k2 | 1.015 | -1.046 | 2.061 | 6 |
| 100 | intact@k4 | -0.066 | -1.046 | 0.979 | 13 |
| 100 | hippocampus-off@k1 | -0.268 | -1.046 | 0.778 | 3 |
| 100 | hippocampus-off@k2 | 0.976 | -1.046 | 2.022 | 6 |
| 100 | hippocampus-off@k4 | -0.040 | -1.046 | 1.006 | 13 |
| 100 | consolidated-off@k4 | -1.026 | -1.046 | 0.019 | 15 |
| 100 | both-off@k4 | -1.046 | -1.046 | 0.000 | 15 |
| 101 | intact@k1 | 1.100 | 1.046 | 0.054 | 3 |
| 101 | intact@k2 | 0.676 | 1.046 | -0.370 | 6 |
| 101 | intact@k4 | 0.580 | 1.046 | -0.466 | 13 |
| 101 | hippocampus-off@k1 | 1.147 | 1.046 | 0.101 | 3 |
| 101 | hippocampus-off@k2 | 0.685 | 1.046 | -0.361 | 6 |
| 101 | hippocampus-off@k4 | 0.627 | 1.046 | -0.418 | 13 |
| 101 | consolidated-off@k4 | 0.921 | 1.046 | -0.125 | 15 |
| 101 | both-off@k4 | 1.046 | 1.046 | 0.000 | 15 |
| 102 | intact@k1 | 0.458 | -1.046 | 1.504 | 3 |
| 102 | intact@k2 | -0.023 | -1.046 | 1.023 | 6 |
| 102 | intact@k4 | -0.482 | -1.046 | 0.563 | 13 |
| 102 | hippocampus-off@k1 | 0.323 | -1.046 | 1.369 | 3 |
| 102 | hippocampus-off@k2 | -0.062 | -1.046 | 0.984 | 6 |
| 102 | hippocampus-off@k4 | -0.494 | -1.046 | 0.552 | 13 |
| 102 | consolidated-off@k4 | -0.920 | -1.046 | 0.125 | 15 |
| 102 | both-off@k4 | -1.046 | -1.046 | 0.000 | 15 |

## Collapse by arm

| arm | seeds ever >=0.4 | seeds at last probe | max share per seed |
|---|---|---|---|
| both-off | 0 | 0 | {42: 0.056, 43: 0.056, 44: 0.056, 100: 0.056, 101: 0.056, 102: 0.056} |
| consolidated-off | 0 | 0 | {42: 0.056, 43: 0.056, 44: 0.111, 100: 0.056, 101: 0.056, 102: 0.111} |
| hippocampus-off | 2 | 1 | {42: 0.250, 43: 0.200, 44: 0.450, 100: 0.350, 101: 0.550, 102: 0.250} |
| intact | 3 | 1 | {42: 0.250, 43: 0.450, 44: 0.550, 100: 0.250, 101: 0.450, 102: 0.200} |
| intact-frozen | 2 | 1 | {42: 0.250, 43: 0.250, 44: 0.550, 100: 0.250, 101: 0.450, 102: 0.200} |

## Adapter norm per night (report only; x night 1)

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 64.797 (1.000x) | 138.422 (2.136x) | 178.266 (2.751x) | 216.902 (3.347x) | 277.028 (4.275x) | 325.852 (5.029x) | 373.599 (5.766x) | 412.835 (6.371x) | 443.588 (6.846x) | 484.813 (7.482x) | 514.327 (7.938x) | 552.821 (8.532x) | 593.216 (9.155x) | 623.223 (9.618x) | 651.084 (10.048x) |
| 43 | 70.988 (1.000x) | 105.517 (1.486x) | 137.582 (1.938x) | 210.754 (2.969x) | 259.061 (3.649x) | 323.150 (4.552x) | 366.898 (5.168x) | 420.419 (5.922x) | 460.352 (6.485x) | 502.054 (7.072x) | 533.948 (7.522x) | 571.942 (8.057x) | 602.588 (8.489x) | 640.829 (9.027x) | 665.855 (9.380x) |
| 44 | 53.367 (1.000x) | 89.698 (1.681x) | 129.914 (2.434x) | 205.622 (3.853x) | 256.462 (4.806x) | 317.469 (5.949x) | 344.043 (6.447x) | 385.915 (7.231x) | 431.925 (8.094x) | 456.409 (8.552x) | 488.104 (9.146x) | 520.416 (9.752x) | 563.977 (10.568x) | 596.763 (11.182x) | 626.914 (11.747x) |
| 100 | 59.054 (1.000x) | 104.181 (1.764x) | 205.587 (3.481x) | 241.031 (4.082x) | 330.650 (5.599x) | 361.188 (6.116x) | 387.514 (6.562x) | 413.349 (7.000x) | 474.351 (8.033x) | 511.181 (8.656x) | 544.002 (9.212x) | 570.599 (9.662x) | 605.948 (10.261x) | 639.899 (10.836x) | 667.581 (11.305x) |
| 101 | 54.848 (1.000x) | 99.900 (1.821x) | 173.077 (3.156x) | 231.886 (4.228x) | 274.264 (5.000x) | 328.037 (5.981x) | 374.247 (6.823x) | 415.169 (7.569x) | 456.698 (8.327x) | 490.505 (8.943x) | 524.748 (9.567x) | 547.298 (9.979x) | 587.402 (10.710x) | 629.791 (11.482x) | 653.327 (11.912x) |
| 102 | 59.639 (1.000x) | 107.732 (1.806x) | 161.417 (2.707x) | 242.611 (4.068x) | 303.573 (5.090x) | 327.008 (5.483x) | 381.784 (6.402x) | 407.519 (6.833x) | 453.283 (7.601x) | 496.121 (8.319x) | 530.756 (8.899x) | 566.080 (9.492x) | 593.954 (9.959x) | 639.266 (10.719x) | 668.267 (11.205x) |
