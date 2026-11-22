# Blind whole-transcript read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [319, 320, 321]; weeks [3]; lived context: all. Stage-0/DEV bars are report-only.

## Instrument validity (design section 7)

| bar | value | mark |
|---|---|---|
| reader1: canary recall strict >= 5/6 | 6/6 | PASS |
| reader1: canary recall loose >= 5/6 | 6/6 | PASS |
| reader1: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader1: duplicate spread (MDE, defects) | 6 | n/a |
| reader2: canary recall strict >= 5/6 | 5/6 | PASS |
| reader2: canary recall loose >= 5/6 | 5/6 | PASS |
| reader2: decoys flagged and rated at or below both-off | refuse: flagged True, pairwise None; keyword-stuffer: flagged True, pairwise None; generic: flagged True, pairwise None | FAIL |
| reader2: duplicate spread (MDE, defects) | 4 | n/a |
| week-3 defects per turn not above week 3, >= 4/6 seeds (reader1; scaled to 2/3; within the duplicate rate spread = not distinguishable) | 3/3; within MDE 3/3 | PASS |

Cohen kappa (n=20 units): any defect 1.0; coherence 0.0 (mean abs diff 0.05)

| type | kappa | Spearman (per-unit count) |
|---|---|---|
| invented | 1.0 | 0.8326 |
| wrong | 0.8276 | 0.0094 |
| wrong-episode | 0.0 | -0.0452 |
| perspective | 1.0 | 0.6271 |
| intrusion | 0.5604 | 0.7305 |
| off-topic | 0.0 | 0.4104 |
| repetition-or-collapse | 1.0 | 0.8745 |
| asks-known | 0.3182 | -0.125 |
| experience-claim | 0.5833 | 0.1193 |

Spearman of per-unit defect counts (n=9 real units, canaries and decoys excluded): total 0.9167, battery 0.8966 (bar to show, report-only: total >= 0.6 -> PASS)

## reader1

missing reads: 0; MDE (defects) 6, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 84 | 8 | 1.0 | 8 | 8 | 0 | 17 | 5 | 31 | 10 | 1 | 4 |
| hippocampus-off|w3 | 3 | 107 | 39 | 1.0 | 19 | 5 | 1 | 25 | 7 | 37 | 12 | 0 | 1 |
| intact|w3 | 3 | 109 | 40 | 1.0 | 18 | 5 | 7 | 23 | 9 | 31 | 13 | 0 | 3 |

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
| refuse | T1 | 5 | 1 | 21 | 1 | True |
| keyword-stuffer | T13 | 35 | 1 | 27 | 1 | True |
| generic | T15 | 63 | 1 | 36 | 1 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 0 | 0.0 | 0 | 0 |
| T12/T10 | 6 | 0.0968 | 2 | 0 |

## reader2

missing reads: 0; MDE (defects) 4, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w3 | 3 | 83 | 7 | 1.333 | 9 | 2 | 2 | 18 | 5 | 24 | 11 | 2 | 10 |
| hippocampus-off|w3 | 3 | 101 | 37 | 1.0 | 20 | 6 | 6 | 24 | 6 | 20 | 15 | 0 | 4 |
| intact|w3 | 3 | 111 | 42 | 1.0 | 21 | 5 | 1 | 17 | 12 | 38 | 15 | 0 | 2 |

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
| refuse | T1 | 1 | 1 | 25 | 2 | True |
| keyword-stuffer | T13 | 27 | 1 | 22 | 1 | True |
| generic | T15 | 19 | 1 | 36 | 1 | True |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T0/T7 | 4 | 0.0645 | 2 | 0 |
| T12/T10 | 2 | 0.032299999999999995 | 1 | 0 |
