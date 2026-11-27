# Stage-0 hand-read (blind), 54 answered trunk exchanges, days 1-14, self-name items excluded
Labels written (handread_labels.json) before the key was opened; reply text only was read. After opening the key, every L/N/U reply was checked against the ledger keys: no fact mismatches.
Boundary calls: x032 ("It was a gift from your grandmother's house...") counted LISTENER-CORRECT (addresses the user with "your", though the first word is "It"); x021 and x042 (echo commentary on the asked-about name/road) OTHER; x018 ("You are a digital brain" as a job) WRONG-FACT; x043 ("I'm Cornelius") WRONG-FRAME.

| seed | n | LISTENER | NEUTRAL | USER-VOICE | HEDGED | WRONG-FACT | WRONG-FRAME | OTHER | L only | L+N |
|---|---|---|---|---|---|---|---|---|---|---|
| 319 | 18 | 6 | 6 | 2 | 3 | 1 | 0 | 0 | 6 | 12 |
| 320 | 18 | 1 | 4 | 2 | 8 | 0 | 0 | 3 | 1 | 5 |
| 321 | 18 | 4 | 5 | 0 | 6 | 0 | 1 | 2 | 4 | 9 |

Floor of 8 per seed:
- LISTENER-CORRECT only: 319 no (6), 320 no (1), 321 no (4). Fails on all seeds.
- LISTENER + NEUTRAL: 319 yes (12), 320 no (5), 321 yes (9). Fails on seed 320.
Caveat: sample is 18 per seed, one hand-reader, so the L-only/L+N counts could move by 1-2 on boundary calls (x032, x021, x042).
