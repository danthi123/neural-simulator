# Stage-0 recount (host only), 2026-11-27

Script: `recount.py` (same directory). Input: `_scratch/stage2/read/lifeKP_s319|s320|s321.json`, `lived` entries with
`line == "trunk"` only (prunedA excluded), rows of kind `sameday` / `remention`, self-name items (dan, nova) excluded
(4 per seed, 12 total), empty replies dropped. "Names the key" = reply matches a ledger `keys` entry at a word start
(case-insensitive). Hedge = not sure / don't know / I think / I bet / maybe / perhaps / probably / do you know.
First-token class from the reply's first word: You/Your(You're), My/I(I'm, I've), else terse/other.
"Pass frame-blind filter" = names the key (hedges included, voice not consulted).

## Days 1-14 (the design window)
| seed | answered | names key | You/Your | My/I | terse/other | hedged (of names) | unhedged | pass filter |
|---|---|---|---|---|---|---|---|---|
| 319 | 18 | 16 | 6 | 4 | 6 | 2 | 14 | 16 |
| 320 | 18 | 9 | 1 | 2 | 6 | 1 | 8 | 9 |
| 321 | 18 | 11 | 3 | 1 | 7 | 0 | 11 | 11 |
| sum | 54 | 36 | 10 | 7 | 19 | 3 | 33 | 36 |

## Days 1-15 (reconciles with the earlier peek)
Sum: 57 answered, 38 name the key (You/Your 10, My/I 9, terse/other 19), 3 hedged, 38 pass. By seed: 319 19/16, 320 19/10, 321 19/12.
The 38 and the 10/19/9 split reproduce the earlier peek exactly; its answered total (63, "21 per life") is not
reproduced (57 here, 19 per life, self-name excluded; 23 per life if self-name rows are kept).

## Stage-1 floor (>= 8 confirmed per seed)
Met on all three seeds on the frame-blind filter (16 / 9 / 11), but seed 320 clears it by 1 only. Caveats: the filter is
frame-blind, so a reply that names the key in the wrong frame or wrong context counts (for example 321 "I'm Cornelius",
320 "Linden Road is a beautiful road"); a blind hand-read (not done here) may drop some. Seed 320 has 9 of which 6 are
terse/other and only 1 You/Your: the user-voice signal there is thin. Counts are host-only, unregistered.
