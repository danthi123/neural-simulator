# Blind whole-transcript read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [319, 320, 321]; weeks [3]; lived context: all. Stage-0/DEV bars are report-only.

## Instrument validity (design section 7)

| bar | value | mark |
|---|---|---|
| reader1: canary recall strict >= 5/6 | 5/6 | PASS |
| reader1: canary recall loose >= 5/6 | 5/6 | PASS |
| reader1: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader1: duplicate spread (MDE, defects) | 3 | n/a |
| reader2: canary recall strict >= 5/6 | 6/6 | PASS |
| reader2: canary recall loose >= 5/6 | 6/6 | PASS |
| reader2: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader2: duplicate spread (MDE, defects) | 10 | n/a |
| week-3 defects per turn not above week 3, >= 4/6 seeds (reader1; scaled to 2/3; within the duplicate rate spread = not distinguishable) | 3/3; within MDE 3/3 | PASS |

Cohen kappa (n=20 units): any defect 1.0; coherence 0.3478 (mean abs diff 0.15)

| type | kappa | Spearman (per-unit count) |
|---|---|---|
| invented | 1.0 | 0.8716 |
| wrong | 0.2708 | 0.4633 |
| wrong-episode | 0.6923 | 0.821 |
| perspective | 1.0 | 0.2422 |
| intrusion | 0.2941 | -0.3015 |
| off-topic | 0.0 | 0.5983 |
| repetition-or-collapse | 1.0 | 0.5782 |
| asks-known | -0.0714 | None |
| experience-claim | 0.4792 | 0.3062 |

Spearman of per-unit defect counts (n=9 real units, canaries and decoys excluded): total 0.6261, battery 0.698 (bar to show, report-only: total >= 0.6 -> PASS)

## reader1

missing reads: 0; MDE (defects) 3, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 77 | 16 | 1.333 | 16 | 6 | 3 | 21 | 1 | 14 | 13 | 0 | 3 |
| hippocampus-off|w3 | 3 | 107 | 50 | 1.0 | 24 | 2 | 10 | 16 | 3 | 31 | 17 | 0 | 4 |
| intact|w3 | 3 | 94 | 48 | 1.0 | 23 | 5 | 6 | 18 | 7 | 22 | 10 | 1 | 2 |

| canary | id | strict | loose |
|---|---|---|---|
| pet-named | T2 | True | True |
| invented | T4 | True | True |
| experience | T5 | True | True |
| perspective | T11 | False | False |
| collapse | T17 | True | True |
| wrong-episode | T18 | True | True |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| refuse | T1 | 6 | 1 | 30 | 1 | True |
| keyword-stuffer | T13 | 65 | 1 | 30 | 1 | True |
| generic | T15 | 20 | 1 | 17 | 2 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 3 | 0.0484 | 0 | 0 |
| T12/T10 | 1 | 0.016199999999999992 | 1 | 0 |

## reader2

missing reads: 0; MDE (defects) 10, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 79 | 19 | 1.667 | 17 | 4 | 4 | 20 | 0 | 18 | 13 | 0 | 3 |
| hippocampus-off|w3 | 3 | 98 | 47 | 1.0 | 22 | 3 | 8 | 22 | 0 | 33 | 10 | 0 | 0 |
| intact|w3 | 3 | 110 | 50 | 1.0 | 22 | 4 | 5 | 30 | 1 | 32 | 16 | 0 | 0 |

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
| refuse | T1 | 1 | 1 | 28 | 1 | True |
| keyword-stuffer | T13 | 29 | 1 | 36 | 2 | True |
| generic | T15 | 34 | 1 | 15 | 2 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 10 | 0.1613 | 5 | 0 |
| T12/T10 | 1 | 0.016199999999999992 | 0 | 0 |
