# D2-PR2 PR-a (tempered loss replay, alpha 0.6) DEV read, 1.5B, s319-321: numeric bars PASS (3.1, 3.2, 3.3 numeric); verdict PENDING the 3.3 blind transcript read

Prereg: brain/research/2026-11-22-replay-priority-2-prereg.md (s3-4); parent brain/research/2026-11-21-replay-priority-prereg.md.
Arm PR-a = Brain(replay_priority="loss_a") (brain_kw in JSON). B = lifeS1F, reference PR-l = lifePRl (results in
brain/experiments/results/2026-11-22-d2pr-dev-read-1.5b.md). Script: _scratch/stage2/read/d2pr2_bars.py (a sed copy of
d2pr_bars.py, arms relabelled), full tables in _scratch/stage2/read/d2pr2_bars.md. Numbers read from JSONs via brain.eval
life_leak / life_bars / life_table / life_echo. No model loaded.
Where ran: all 9 lives device=cpu fp32. s319 node-dl4g243, s320 node-dl5d243, s321 node-mxl1462drl; B, PR-a and PR-l on the
SAME node per seed (the closed PR-x confound does not apply).

## 3.1 Engagement (perm share AND top-6 below within-night strength-only ref, nights 5+10, 3/3): PASS 6/6 seed-nights
| seed | night | perm PR-a (ref) | top-6 PR-a (ref) | max share (ref) | PR-l top-6 |
|---|---|---|---|---|---|
| 319 | 5 | 0.044 (0.298) | 0.383 (0.436) | 0.071 (0.104) | 0.436 |
| 319 | 10 | 0.033 (0.248) | 0.223 (0.410) | 0.052 (0.117) | 0.303 |
| 320 | 5 | 0.050 (0.270) | 0.393 (0.450) | 0.077 (0.099) | 0.426 |
| 320 | 10 | 0.033 (0.234) | 0.224 (0.402) | 0.049 (0.106) | 0.295 |
| 321 | 5 | 0.055 (0.290) | 0.349 (0.439) | 0.066 (0.101) | 0.596 (fail) |
| 321 | 10 | 0.028 (0.244) | 0.222 (0.409) | 0.048 (0.100) | 0.321 |
Ref identical across PR-a and PR-l records and B's class replay_share equals the PR-a record on all 6 seed-nights.

## 3.2 Primary leak, nights 10+15 pooled (intact + hippocampus-off): PASS
| seed | B | PR-a | PR-l (ref) |
|---|---|---|---|
| 319 | 39/156 = 0.250 (n10 27/82, n15 12/74) | 13/156 = 0.083 (9/82, 4/74) | 17/156 = 0.109 |
| 320 | 54/152 = 0.355 (28/82, 26/70) | 13/152 = 0.086 (4/82, 9/70) | 18/152 = 0.118 |
| 321 | 42/152 = 0.276 (24/82, 18/70) | 22/152 = 0.145 (21/82, 1/70) | 41/152 = 0.270 |
Pooled PR-a 48/460 = 0.104 vs B 135/460 = 0.294: -64.4% (bar >=30%, below B 3/3). PR-l was -43.7%.
Report only, nights 1-15 paired (45 seed-nights, sign-flip, 200k draws): PR-a mean share diff -0.092, p=0.0002, lower on
7/10/12 nights (s319/320/321; ties at 0 not counted). PR-l -0.098, p<1e-5.

## 3.3 Guards, numeric (all PASS)
| guard | s319 | s320 | s321 |
|---|---|---|---|
| H hippo-off n5-15 mean /6: B / PR-a (diff, bar >= -0.5) | 2.545 / 3.818 (+1.27) | 1.818 / 2.273 (+0.45) | 1.818 / 3.909 (+2.09) |
| direct hippo-off d1-2 facts n5-15, rate x6: B / PR-a (diff, bar >= -1) | 2.43 / 3.73 (+1.30) | 1.64 / 2.47 (+0.82) | 1.66 / 3.79 (+2.13) |
| KL n15: B / PR-a (ratio, bar <= 1.25) | 2.933 / 2.221 (0.757) | 2.379 / 2.172 (0.913) | 2.858 / 2.684 (0.939) |
| NLL on/off n15: B / PR-a | 4.747/1.931 / 3.992/1.931 | 4.107/1.931 / 3.832/1.931 | 4.674/1.931 / 4.406/1.931 |
Collapse seeds-ever (>=0.4 top-reply share), B vs PR-a: both-off 0/0; consolidated-off 1/1; hippocampus-off 2/1 (PR-a max
shares 0.20/0.35/0.50); intact 2/1; intact-frozen 2/1. PR-a <= B in every arm: PASS. Note PR-a s321 reaches 0.50 on intact
and hippo-off (B s321 0.30), but it is 1 seed against B's 2.
Direct d1-2 nights 1-4 (not a bar, derived as nights 1-15 minus 5-15): B/PR-a k of 29 per seed: s319 16/18, s320 27/21,
s321 15/12. s320 PR-a is 1.24/6 below B on nights 1-4 (it would exceed 1/6 if that window were a bar).
Transcript blind read (life_read.py, 2 sonnet readers, week 3): PENDING. Needs the reader runs on lifePRa and lifeS1F
(canaries; new-defect-class check: identity loss, name overwrite, marker leak, stuck tail absent in B). Not run here.

## 3.4 Report only
- Echo (life_echo, n10+15) pooled intact/hippo-off: B 16/48=0.333, 17/48=0.354; PR-a 7/48=0.146, 4/48=0.083; PR-l 0.104, 0.146.
  Per seed intact PR-a 0.062/0.312/0.062 (B 0.312/0.312/0.375); s320 intact 5/16 is as high as B.
- Hippo-off once-told recall n5-15 battery k/n: E B 4/13, PR-a 3/13; F B 4/30, PR-a 14/30. Cue C0: E B 5/18, PR-a 7/18; F B 12/54, PR-a 22/54.
- Wrong fetches 167/105/97 (B) = PR-a on all seeds; hub top-1 mean B/PR-a: 0.595/0.582, 0.591/0.614, 0.430/0.416 (unchanged, as expected).
- Adapter norm n15 B/PR-a: 660.5/600.8, 674.7/590.5, 692.6/648.5 (grows more slowly, as predicted).
- Interview AUROC / invented-memory rate and reader repetition counts: PENDING (blind-reader outputs, none for PR-a yet).

## Verdict per prereg s4
3.1 PASS, 3.2 PASS, numeric 3.3 PASS on 3/3 seeds. The prereg outcome "passes 3.1, 3.2 and 3.3" (registered six-seed test on a
FRESH TEST split, default only on that result) is PROVISIONAL: the final 3.3 clause (blind whole-transcript read) is not yet run.
If the read finds a new defect class absent in B: KILLED. Not NARROWED, not UNDEFINED, not KILLED on any numeric bar.
CORRECTION (refuter 2, REFUTED the "no numeric bar points to KILLED" wording): KILL-10 (design s10 as amended s13; life_bars.kill_check)
FIRES for PR-a s321 at night 2 (first night 1): intact collapse share 0.50 on n1 and 0.45 on n2, a stuck stock self-reply
"You're Dan, my friend." (9/20 intact replies on n1, on unrelated questions). No B or PR-l life triggers it; B s321 never exceeds 0.30.
Per seed, s321 PR-a is worse than B on intact, hippocampus-off and intact-frozen (0.50 vs 0.30); the seeds-ever collapse count passes only
because it pools different seeds. This is a candidate "stuck tail absent in B" (a named KILLED defect class) and must be judged by the
blind read. Code sha is not recorded in the life JSONs (B finished before PR-a was built; same node per seed). tools/where_ran.py returns
unknown for these JSONs (it does not read the top-level host field); hosts above come from the JSON fields.
Caveats: DEV, 3 seeds, one life each; per-night leak is noisy (s321 n10 21/82 vs n15 1/70; PR-a s321 is -47% there against
-64% pooled); no code sha in the JSONs; B carries no per-episode weights, so engagement is vs the in-night strength-only ref.
Run needed: life_read.py blind readers (2 sonnet, week 3) on lifePRa_s319/320/321 vs lifeS1F (reader runs are Claude calls, not local compute).
