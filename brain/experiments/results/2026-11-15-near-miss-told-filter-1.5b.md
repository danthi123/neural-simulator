# Result: M4b, the frame guard minus told-topic cues, against false recall after sleep, 1.5B: NOT A LEAD (six registered seeds, matched halves; and the CPU seeds)

**Correction (2026-11-15, the skeptic, before landing).** The M4b job file gave s100/s101/s102 the wrong slept half
(even/odd/even; the banked off arm and M4 used odd/even/odd). On those three seeds the guard arm trained and was scored
on a different split of the 12 facts than its off arm, so the registered comparison below is VOID on them and the
registered verdict is not final. They are rerun with the matched halves (`jobs/2026-11-15-aws-near-miss-ft-fix.txt`,
registered code unchanged). s42/s43/s44 and all six CPU seeds are matched.
**Rerun done** (AWS L4, batch 20261115-1202, 3 jobs, rc 0; slept odd/even/odd, recorded in each JSON). The table
below is the matched six-seed comparison; the unmatched files are kept in `void_unmatched_halves/`. The verdict did not
change (NOT A LEAD); the numbers did a little.
Also found: `told_topic` keeps the "User:" prefix in the told word set, so cues about "my user ..." were dropped as
told (s101, s102: 2 cues); "name" and "birthday" match the seeded self episodes, not the six slept facts. In the matched
runs FT dropped 2-3 cues per seed (15 in all), a smaller ablation than the prereg's upper bound of 16.

Prereg: `brain/research/2026-11-15-near-miss-told-filter-1.5b.md` (N0-N6, addendum for seeds 200-205, all written
before any run). Runs: AWS g6.xlarge (L4, `device cuda`), batch 20261115-0937, 7 jobs (nmoff_s42 rerun + nmft x 6),
all rc 0. Off arm: the banked `nmoff_s*` of batch 20261114-2031 (`results/2026-11-14-near-miss-guard-1.5b/`).
Report-only seeds 200-205: pool40/41/42 (`device cpu`), against the banked CPU `nmoff_s20*` (in `extra/`).
All files re-scored (`brain.experiments.rescore`) before `near_miss_table <dir> --on nmft`; tables in `table.txt` and
`extra/table.txt`.

## Registered six seeds, arm s (S0 key, gain 0.5), of 12 unless said

| | told | slept told | new told | para | false | self (3) | invented (never-told) | abstain, told rows |
|---|---|---|---|---|---|---|---|---|
| guard off | 7.17 | 4.67 | 2.50 | 4.67 | 6.83 | 3.00 | 5.00 | 0.00 |
| M4 guard (11-14) | 6.00 | 3.50 | 2.50 | 3.17 | 3.83 | 3.00 | | |
| M4b guard, told topics dropped | 6.33 | 3.50 | 2.83 | 3.17 | 5.67 | 3.00 | 5.33 | 0.67 |

Per-seed false off/on: 11/6, 4/2, 10/10, 5/1, 7/11, 4/4 (lower on 3/6; s101 rises by 4).

- N0 holds: the nmoff_s42 rerun reproduces the banked file on all 39 rows (0 reply differences in arms c, s, r). FT
  kept 16-18 of 18-20 generated cues per seed.
- N1 FAILS: 5.67 > 3.42; lower on 3/6.
- N2 holds: 6.33 >= 6.09.
- N3 FAILS: para 3.17 < 3.67.
- N4 holds: self 3/3 on every seed.
- N5 holds: invented 5.33 <= 6.00.
- N6 FAILS: told-row abstentions 0.67 > 0.50.
- **VERDICT (registered): NOT A LEAD.**

Reported, not criteria: arm c told 4.83 -> 4.17, false 6.83 -> 4.00, invented 5.17 -> 6.67; arm r told 8.00 -> 7.00,
false 6.67 -> 4.33.

Report-only seeds 200-205 (CPU, arm s): told 6.50 -> 6.50, para 3.33 -> 3.67, false 8.83 -> 6.33, invented
3.00 -> 4.50, told-row abstentions 0 -> 1.33. Also NOT A LEAD (N1, N5, N6 fail).

## What it says (matched six seeds; fresh sonnet skeptic: NARROWED, numbers reproduce, cue count fixed)

- Host: `tools/where_ran.py` reports unknown (no host stamp in the JSONs); AWS L4 rests on the batch logs.

- Dropping the told-topic cues gave back most of M4's false-recall gain (M4 3.83 -> M4b 5.67, off 6.83) and only a
  little of its told cost (6.00 -> 6.33). Slept told is the same as under M4 (3.50); the told total is a bit higher
  only because new (day-2) told rose.
- The prereg's own branch applies: "If the told cost stays the same, it is not carried by the told-topic cues: the
  guard and the told bindings compete in one rank-8 adapter (the capacity risk named in M4)". The told-topic cues
  were part of what made M4 work, not the cause of its cost.
- Paraphrase cost is the same as M4 (-1.5), and told rows are still refused sometimes (0.67 per seed).
- Lever count on this defect: 9. A research query ran before the next one (`tools/deep_research.sh`); it predicts a
  content cause, not capacity, which the rank diagnosis tests.

## Next

The capacity diagnosis named in M3 and M4/M4b: `brain/research/2026-11-15-rank-diagnosis-1.5b.md` (R0-R3, rank 2 and
rank 32), running on AWS.
