# Blind whole-transcript read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [319, 320, 321]; weeks [3]; lived context: all. Stage-0/DEV bars are report-only.

## Instrument validity (design section 7)

| bar | value | mark |
|---|---|---|
| reader1: canary recall strict >= 5/6 | 6/6 | PASS |
| reader1: canary recall loose >= 5/6 | 6/6 | PASS |
| reader1: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader1: duplicate spread (MDE, defects) | 3 | n/a |
| reader2: canary recall strict >= 5/6 | 6/6 | PASS |
| reader2: canary recall loose >= 5/6 | 6/6 | PASS |
| reader2: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader2: duplicate spread (MDE, defects) | 4 | n/a |
| week-3 defects per turn not above week 3, >= 4/6 seeds (reader1; scaled to 2/3; within the duplicate rate spread = not distinguishable) | 3/3; within MDE 3/3 | PASS |

Cohen kappa (n=20 units): any defect 1.0; coherence 0.4595 (mean abs diff 0.1)

| type | kappa | Spearman (per-unit count) |
|---|---|---|
| invented | 1.0 | 0.875 |
| wrong | 0.375 | 0.8509 |
| wrong-episode | 0.7619 | 0.2014 |
| perspective | 1.0 | 0.589 |
| intrusion | 0.5714 | -0.4912 |
| off-topic | 1.0 | 0.4757 |
| repetition-or-collapse | 1.0 | 0.3348 |
| asks-known | 0.4828 | 0.7559 |
| experience-claim | 0.4898 | 0.4824 |

Spearman of per-unit defect counts (n=9 real units, canaries and decoys excluded): total 0.5443, battery 0.8939 (bar to show, report-only: total >= 0.6 -> FAIL)

## reader1

missing reads: 0; MDE (defects) 3, coherence 1

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 72 | 6 | 1.0 | 5 | 6 | 4 | 20 | 5 | 17 | 9 | 1 | 5 |
| hippocampus-off|w3 | 3 | 108 | 46 | 1.0 | 22 | 6 | 8 | 16 | 4 | 33 | 12 | 1 | 6 |
| intact|w3 | 3 | 127 | 47 | 1.0 | 21 | 7 | 8 | 25 | 5 | 36 | 21 | 1 | 3 |

| canary | id | strict | loose |
|---|---|---|---|
| pet-named | T2 | True | True |
| invented | T4 | True | True |
| experience | T5 | True | True |
| perspective | T11 | True | True |
| collapse | T17 | True | True |
| wrong-episode | T18 | True | True |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| refuse | T1 | 1 | 1 | 22 | 1 | True |
| keyword-stuffer | T13 | 37 | 1 | 28 | 1 | True |
| generic | T15 | 17 | 1 | 22 | 1 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 2 | 0.032200000000000006 | 1 | 0 |
| T12/T10 | 3 | 0.0484 | 1 | 1 |

## reader2

missing reads: 0; MDE (defects) 4, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 64 | 4 | 1.667 | 5 | 5 | 4 | 16 | 3 | 16 | 11 | 0 | 4 |
| hippocampus-off|w3 | 3 | 122 | 45 | 1.0 | 18 | 7 | 3 | 32 | 5 | 39 | 15 | 1 | 2 |
| intact|w3 | 3 | 104 | 42 | 1.0 | 18 | 7 | 5 | 18 | 6 | 32 | 15 | 1 | 2 |

| canary | id | strict | loose |
|---|---|---|---|
| pet-named | T2 | True | True |
| invented | T4 | True | True |
| experience | T5 | True | True |
| perspective | T11 | True | True |
| collapse | T17 | True | True |
| wrong-episode | T18 | True | True |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| refuse | T1 | 1 | 1 | 20 | 1 | True |
| keyword-stuffer | T13 | 65 | 1 | 20 | 2 | True |
| generic | T15 | 19 | 1 | 24 | 2 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 4 | 0.0645 | 3 | 0 |
| T12/T10 | 4 | 0.0645 | 2 | 0 |
