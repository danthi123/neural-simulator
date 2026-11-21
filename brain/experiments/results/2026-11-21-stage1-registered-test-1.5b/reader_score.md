# Blind whole-transcript read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [42, 43, 44, 100, 101, 102]; weeks [1, 3]; lived context: all. Stage-0/DEV bars are report-only.

## Instrument validity (design section 7)

| bar | value | mark |
|---|---|---|
| reader1: canary recall strict >= 5/6 | 6/6 | PASS |
| reader1: canary recall loose >= 5/6 | 6/6 | PASS |
| reader1: decoys flagged and rated at or below both-off | keyword-stuffer: flagged True, pairwise both-off; generic: flagged True, pairwise both-off; refuse: flagged True, pairwise both-off | PASS |
| reader1: duplicate spread (MDE, defects) | 10 | n/a |
| reader2: canary recall strict >= 5/6 | 5/6 | PASS |
| reader2: canary recall loose >= 5/6 | 6/6 | PASS |
| reader2: decoys flagged and rated at or below both-off | keyword-stuffer: flagged True, pairwise both-off; generic: flagged True, pairwise both-off; refuse: flagged True, pairwise both-off | PASS |
| reader2: duplicate spread (MDE, defects) | 10 | n/a |
| intact preferred to both-off, last week, >= 4/6 seeds (pairs1; scaled to 4/6; a seed counts only when both A/B orders name intact = consistent) | 0/6; inconsistent (tie) 0/6; all pairs consistent 15/15 | FAIL |
| week-3 defects per turn not above week 1, >= 4/6 seeds (reader1; scaled to 4/6; within the duplicate rate spread = not distinguishable) | 3/6; within MDE 3/6 | FAIL |

Cohen kappa (n=47 units): any defect 1.0; coherence 0.8719 (mean abs diff 0.064)

| type | kappa | Spearman (per-unit count) |
|---|---|---|
| invented | 1.0 | 0.8789 |
| wrong | 0.7415 | 0.7152 |
| wrong-episode | 0.4695 | 0.59 |
| perspective | 1.0 | 0.742 |
| intrusion | 0.4108 | 0.5554 |
| off-topic | 0.5377 | 0.4558 |
| repetition-or-collapse | 1.0 | 0.9006 |
| asks-known | 0.8162 | 0.8686 |
| experience-claim | 0.6713 | 0.6625 |

Spearman of per-unit defect counts (n=36 real units, canaries and decoys excluded): total 0.5819, battery 0.8783 (bar to show, report-only: total >= 0.6 -> FAIL)

Pairwise kappa: 0.9333

## reader1

missing reads: 0; MDE (defects) 10, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w1 | 6 | 208 | 56 | 2.0 | 23 | 30 | 6 | 80 | 5 | 26 | 22 | 4 | 12 |
| both-off|w3 | 6 | 226 | 45 | 1.333 | 50 | 18 | 5 | 29 | 4 | 58 | 38 | 0 | 24 |
| hippocampus-off|w1 | 6 | 268 | 98 | 2.0 | 24 | 5 | 25 | 125 | 11 | 46 | 19 | 6 | 7 |
| hippocampus-off|w3 | 6 | 303 | 106 | 1.0 | 45 | 14 | 13 | 48 | 13 | 109 | 46 | 0 | 15 |
| intact|w1 | 6 | 262 | 96 | 2.0 | 23 | 6 | 21 | 131 | 6 | 37 | 22 | 8 | 8 |
| intact|w3 | 6 | 292 | 106 | 1.167 | 36 | 19 | 23 | 52 | 10 | 78 | 38 | 0 | 36 |

| canary | id | strict | loose |
|---|---|---|---|
| collapse | T4 | True | True |
| invented | T6 | True | True |
| perspective | T18 | True | True |
| pet-named | T20 | True | True |
| experience | T21 | True | True |
| wrong-episode | T41 | True | True |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| keyword-stuffer | T9 | 66 | 1 | 49 | 1 | True |
| generic | T10 | 34 | 1 | 38 | 1 | True |
| refuse | T30 | 3 | 1 | 27 | 2 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T40/T36 | 0 | 0.0 | 0 | 0 |
| T23/T44 | 10 | 0.13889999999999997 | 1 | 0 |

## reader2

missing reads: 0; MDE (defects) 10, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w1 | 6 | 261 | 54 | 2.0 | 24 | 30 | 2 | 118 | 6 | 43 | 21 | 5 | 12 |
| both-off|w3 | 6 | 216 | 46 | 1.333 | 53 | 21 | 8 | 31 | 4 | 40 | 34 | 0 | 25 |
| hippocampus-off|w1 | 6 | 264 | 99 | 1.833 | 26 | 11 | 15 | 126 | 11 | 36 | 23 | 8 | 8 |
| hippocampus-off|w3 | 6 | 284 | 105 | 1.0 | 49 | 17 | 14 | 48 | 7 | 58 | 46 | 0 | 45 |
| intact|w1 | 6 | 299 | 99 | 2.0 | 25 | 6 | 26 | 135 | 13 | 60 | 22 | 6 | 6 |
| intact|w3 | 6 | 301 | 109 | 1.0 | 47 | 24 | 20 | 47 | 13 | 64 | 37 | 0 | 49 |

| canary | id | strict | loose |
|---|---|---|---|
| collapse | T4 | True | True |
| invented | T6 | True | True |
| perspective | T18 | True | True |
| pet-named | T20 | False | True |
| experience | T21 | True | True |
| wrong-episode | T41 | True | True |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| keyword-stuffer | T9 | 80 | 1 | 49 | 1 | True |
| generic | T10 | 75 | 1 | 33 | 1 | True |
| refuse | T30 | 1 | 1 | 24 | 2 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T40/T36 | 0 | 0.0 | 0 | 0 |
| T23/T44 | 10 | 0.13880000000000003 | 0 | 0 |

## pairs1

intact vs both-off (last week) winners per seed: {'42': 'both-off', '43': 'both-off', '44': 'both-off', '100': 'both-off', '101': 'both-off', '102': 'both-off'}
last week vs first week (intact): {'42': 'intact-w1', '43': 'intact-w1', '44': 'intact-w1', '100': 'intact-w1', '101': 'intact-w1', '102': 'intact-w1'}
decoys vs both-off: {'T9': 'both-off', 'T10': 'both-off', 'T30': 'both-off'}
pairs consistent over both A/B orders: 15/15 (consistency rate 1.0; same-position rate 0.0)
inconsistent pairs: []

## pairs2

intact vs both-off (last week) winners per seed: {'42': 'both-off', '43': 'both-off', '44': 'both-off', '100': 'tie', '101': 'both-off', '102': 'both-off'}
last week vs first week (intact): {'42': 'intact-w1', '43': 'intact-w1', '44': 'intact-w1', '100': 'intact-w1', '101': 'intact-w1', '102': 'intact-w1'}
decoys vs both-off: {'T9': 'both-off', 'T10': 'both-off', 'T30': 'both-off'}
pairs consistent over both A/B orders: 14/15 (consistency rate 0.933; same-position rate 0.067)
inconsistent pairs: [('intact-vs-bothoff', 100, ['both-off', 'intact'])]

## RIGHT = strict ok AND reader verdict yes (section 5)

strict vs reader-1 disagreement rate 0.476; reader kappa 0.9463; reader disagreements for adjudication: 7

| arm / cut | n | strict ok | verdict yes | RIGHT | strict-reader disagree | veto (strict ok, verdict no) |
|---|---|---|---|---|---|---|
| both-off|H | 72 | 0 | 40 | 0 | 0.556 | 0 |
| both-off|all | 98 | 0 | 51 | 0 | 0.52 | 0 |
| both-off|neardup | 2 | 0 | 1 | 0 | 0.5 | 0 |
| both-off|rotating | 24 | 0 | 10 | 0 | 0.417 | 0 |
| both-off|w1 | 48 | 0 | 25 | 0 | 0.521 | 0 |
| both-off|w3 | 50 | 0 | 26 | 0 | 0.52 | 0 |
| hippocampus-off|H | 72 | 29 | 57 | 27 | 0.444 | 2 |
| hippocampus-off|all | 98 | 32 | 73 | 30 | 0.459 | 2 |
| hippocampus-off|neardup | 2 | 0 | 0 | 0 | 0.0 | 0 |
| hippocampus-off|rotating | 24 | 3 | 16 | 3 | 0.542 | 0 |
| hippocampus-off|w1 | 48 | 18 | 40 | 17 | 0.5 | 1 |
| hippocampus-off|w3 | 50 | 14 | 33 | 13 | 0.42 | 1 |
| intact|H | 72 | 30 | 57 | 29 | 0.403 | 1 |
| intact|all | 98 | 33 | 75 | 32 | 0.449 | 1 |
| intact|neardup | 2 | 0 | 1 | 0 | 0.5 | 0 |
| intact|rotating | 24 | 3 | 17 | 3 | 0.583 | 0 |
| intact|w1 | 48 | 19 | 43 | 19 | 0.5 | 0 |
| intact|w3 | 50 | 14 | 32 | 13 | 0.4 | 1 |

| control | expect | reader 1 | reader 2 |
|---|---|---|---|
| answer | yes | True | True |
| answer | yes | True | True |
| subject | no | True | True |
| refuse | no | True | True |
| echo | no | True | True |
| answer | yes | True | True |
