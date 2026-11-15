# Result: near-miss / frame guard in sleep (M4) against false recall after sleep, 1.5B: NOT A LEAD

Prereg: `brain/research/2026-11-14-near-miss-guard-1.5b.md` (criteria N0-N4, Amendments 1-2 before any test seed,
Amendment 3 a post-verdict diagnostic). Runs: AWS g6.xlarge (L4, `device cuda`), batch 20261114-2031, 12 jobs, all
rc 0 (runner rc fix verified). Report-only seeds 200-205: pool40/41/42 (`device cpu`). Table: `table.txt` here
(`python -m brain.experiments.near_miss_table <rescored dir> <rescored 11-14 tellmix dir>`). The raw and re-scored
`ok` fields are identical on all 24 files.

## Registered six seeds (42/43/44/100/101/102), arm s (S0 key, gain 0.5), of 12 unless said

| | told | slept told | new told | para | false (never-told answered with a told fact) | self (3) |
|---|---|---|---|---|---|---|
| guard off | 7.17 | 4.67 | 2.50 | 4.67 | 6.83 | 3.00 |
| guard on | 6.00 | 3.50 | 2.50 | 3.17 | 3.83 | 3.00 |

- N0 holds: guard-off reproduces the 11-14 tellmix told and false per seed exactly (arms s and r). The guard kept
  18-19 cues per seed (`near_miss_log`). Day-1 replies, before any night, are identical off vs on in 6/6 sessions.
- N1 FAILS: 3.83 > 0.5 x 6.83 = 3.42. It is lower on 5/6 seeds (per-seed drops 6/2/3/4/0/3; no single seed carries it).
- N2 FAILS: 6.00 < 0.85 x 7.17 = 6.09 (36 vs 43 told summed; it misses by one answer).
- N3 FAILS: para 3.17 < 4.67 - 1 (4 of 6 seeds lower).
- N4 holds: self 3/3 on every seed.
- **VERDICT (registered): NOT A LEAD.** N3 alone settles it; the LEAD false bar (-2 on >= 5/6) holds.

Reported, not criteria:
- Arm c (hippocampus off): slept told 4.67 -> 2.83, false 6.83 -> 2.33.
- Arm r: told 8.00 -> 6.00, false 6.67 -> 2.50.
- Seeds 200-205 (CPU, arm s): told 6.50 -> 6.33, para 3.33 -> 3.67, false 8.83 -> 5.50 (lower on 6/6). That is LEAD by
  the table, but it is report-only, and the code sha and model file on the pool are not recorded.

## What the skeptics found (four fresh sonnet refuters, one lens each; all NARROWED, none overturned the verdict)

- **Arithmetic.** The arithmetic is correct and matches the prereg. Raw and re-scored files give the same verdict.
- **Half the "fix" is substitution, not honesty.** "False" counts only a told fact intruding; an invented answer scores
  clean. Arm s, registered seeds: 18 fewer intrusions, of which 8 became abstentions ("I'm sorry, but I don't have
  information about your horse.") and 10 became a fresh invented answer ("Mr. Smith", "San Francisco"). The guard
  lowers told-fact intrusion; it does not lower confabulation overall.
- **Answer rate falls.** `tools/answer_rate.py` cannot read this layout, so abstentions were counted by phrase.
  - Arm s, registered seeds: told 0 -> 7 of 72, para 4 -> 8, never 1 -> 9. Abstentions rose on 5/6 seeds and fell on none.
  - Arm s, seeds 200-205: told 0 -> 9 and para 1 -> 11.
  - Arm c: 5 -> 38 of 234.
  - This is FLIP-HURTS-ANSWERS: any flip of this guard would need a ruling. That is moot at NOT A LEAD.
- **Partly selective, not proven.** False recall fell proportionally more than slept told recall in every arm.
  - Arm s, registered: slept told kept 0.75, false kept 0.56.
  - Arm c: 0.61 vs 0.34.
  - Seeds 200-205, arm s: 0.96 vs 0.62.
  - Across 12 seeds the per-seed drops are uncorrelated (r = -0.08 in arm s), which a uniformly weaker night would not give.
  - Still, the record has no matched weaker-night control. Amendment 3 runs one (below).
- **Cue leak is minor.** The topic filter kept synonyms of probe topics ("Who is my mom?", "Who is my dad?", "What is my
  school like?"). They account for at most 1-2 of the 18 dropped intrusions, and every never-told question falls
  (aunt 5 -> 2, cousin 5 -> 2, parrot 2 -> 0). Extend the filter (mom/mum/dad/school/pet/vacation) in any rerun.
- **Told-topic cues explain part of the cost.** "Who is my best friend?", grandmother and son cues hold told topics at
  their pretrained answer, which fits the told/para loss (the prereg named this risk).
- **Where it ran.** `tools/where_ran.py`: "24 file(s): unknown=24". The brain/ lane stamps no host or sha. The
  registered files are `device cuda` (AWS L4, peak about 11-13 GB); the extras are `device cpu`. The probe reads
  nothing from `research/findings/raw/`.

## Correction to the record

`results/2026-11-14-sall-contrast-mixed-1.5b.md` said the aunt -> Oslo lure "fires on 12/12 runs". The reference JSONs
show aunt -> Oslo on 2 of 6 seeds (s44, s101) in every arm, i.e. 4 of 12 runs per arm. What holds on 12/12 runs is "at
least one never-told question answered with a told fact". Oslo leaks into uncle, aunt and cousin on 4 runs each.
Corrected there.

## What it means

- Method verdict: **M4 as built (guard the pretrained cortex's own "my X" questions in sleep) is NOT A LEAD.** It moves
  false recall the most of any lever so far (-3.0 of 12, lower on 11 of 12 seeds), but it does so by pulling the "my
  X" frame back toward the pretrained cortex. Told facts and paraphrases pay for it, and about half the removed
  intrusions become abstentions or new inventions.
- Lever count on this defect: 7. The capability (not saying a told fact where nothing was told) stays open.
- Next:
  1. Read Amendment 3, the matched weaker-night control (seeds 200-205, pool, running 11-15). If the guard is
     selective, the next method keeps it and removes the told-topic cost: drop cues that name a told topic, the
     mirror of the test-hygiene filter (a brain-side rule, its own scaffold row). If it is not selective, false recall
     is a night-strength trade-off, and the next method is M5 (a learned metamemory monitor trained on sleep
     self-tests, `brain/research/2026-11-03-confabulation-next-methods.md`).
  2. Measure confabulation, not only intrusion: count invented answers as false in the next prereg.
