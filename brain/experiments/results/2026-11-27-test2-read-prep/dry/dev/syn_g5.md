# Blind whole-transcript read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [319, 320, 321]; weeks [1, 3]; lived context: all. Stage-0/DEV bars are report-only.

## Instrument validity (design section 7)

| bar | value | mark |
|---|---|---|
| reader1: canary recall strict >= 5/6 | 0/6 | FAIL |
| reader1: canary recall loose >= 5/6 | 0/6 | FAIL |
| reader1: decoys flagged and rated at or below both-off | refuse: flagged False, pairwise tie; generic: flagged False, pairwise tie; keyword-stuffer: flagged False, pairwise tie | FAIL |
| reader1: duplicate spread (MDE, defects) | 0 | n/a |
| reader2: canary recall strict >= 5/6 | 0/6 | FAIL |
| reader2: canary recall loose >= 5/6 | 0/6 | FAIL |
| reader2: decoys flagged and rated at or below both-off | refuse: flagged False, pairwise tie; generic: flagged False, pairwise tie; keyword-stuffer: flagged False, pairwise tie | FAIL |
| reader2: duplicate spread (MDE, defects) | 0 | n/a |
| intact preferred to both-off, last week, >= 4/6 seeds (pairs1; scaled to 2/3; a seed counts only when both A/B orders name intact = consistent) | 0/3; inconsistent (tie) 3/3; all pairs consistent 0/9 | FAIL |
| week-3 defects per turn not above week 1, >= 4/6 seeds (reader1; scaled to 2/3; within the duplicate rate spread = not distinguishable) | 3/3; within MDE 3/3 | PASS |

Cohen kappa (n=29 units): any defect 1.0; coherence 1.0 (mean abs diff 0.0)

| type | kappa | Spearman (per-unit count) |
|---|---|---|
| invented | 1.0 | None |
| wrong | 1.0 | None |
| wrong-episode | 1.0 | None |
| perspective | 1.0 | None |
| intrusion | 1.0 | None |
| off-topic | 1.0 | None |
| repetition-or-collapse | 1.0 | None |
| asks-known | 1.0 | None |
| experience-claim | 1.0 | None |

Spearman of per-unit defect counts (n=18 real units, canaries and decoys excluded): total None, battery None (bar to show, report-only: total >= 0.6 -> FAIL)

Pairwise kappa: 1.0

## reader1

missing reads: 0; MDE (defects) 0, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| both-off|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hippocampus-off|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hippocampus-off|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| intact|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| intact|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| canary | id | strict | loose |
|---|---|---|---|
| collapse | T5 | False | False |
| pet-named | T8 | False | False |
| invented | T13 | False | False |
| wrong-episode | T15 | False | False |
| perspective | T20 | False | False |
| experience | T23 | False | False |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| refuse | T6 | 0 | 3 | 0 | 3 | False |
| generic | T12 | 0 | 3 | 0 | 3 | False |
| keyword-stuffer | T17 | 0 | 3 | 0 | 3 | False |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T19/T9 | 0 | 0.0 | 0 | 0 |
| T1/T14 | 0 | 0.0 | 0 | 0 |

## reader2

missing reads: 0; MDE (defects) 0, coherence 0

| arm, week | units | defects | battery defects | coherence | invented | wrong | wrong-episode | perspective | intrusion | off-topic | repetition-or-collapse | asks-known | experience-claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both-off|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| both-off|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hippocampus-off|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hippocampus-off|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| intact|w1 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| intact|w3 | 3 | 0 | 0 | 3.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| canary | id | strict | loose |
|---|---|---|---|
| collapse | T5 | False | False |
| pet-named | T8 | False | False |
| invented | T13 | False | False |
| wrong-episode | T15 | False | False |
| perspective | T20 | False | False |
| experience | T23 | False | False |

| decoy | id | defects | coherence | both-off defects | both-off coherence | flagged |
|---|---|---|---|---|---|---|
| refuse | T6 | 0 | 3 | 0 | 3 | False |
| generic | T12 | 0 | 3 | 0 | 3 | False |
| keyword-stuffer | T17 | 0 | 3 | 0 | 3 | False |

| duplicate pair | defect spread | per-turn spread | battery spread | coherence spread |
|---|---|---|---|---|
| T19/T9 | 0 | 0.0 | 0 | 0 |
| T1/T14 | 0 | 0.0 | 0 | 0 |

## pairs1

intact vs both-off (last week) winners per seed: {'319': 'tie', '320': 'tie', '321': 'tie'}
last week vs first week (intact): {'319': 'tie', '320': 'tie', '321': 'tie'}
decoys vs both-off: {'T17': 'tie', 'T12': 'tie', 'T6': 'tie'}
pairs consistent over both A/B orders: 0/9 (consistency rate 0.0; same-position rate 1.0)
inconsistent pairs: [('w3-vs-w1', 319, ['intact-w1', 'intact-w3']), ('intact-vs-bothoff', 319, ['both-off', 'intact']), ('w3-vs-w1', 320, ['intact-w3', 'intact-w1']), ('intact-vs-bothoff', 320, ['both-off', 'intact']), ('w3-vs-w1', 321, ['intact-w3', 'intact-w1']), ('intact-vs-bothoff', 321, ['intact', 'both-off']), ('decoy-vs-bothoff', 319, ['both-off', 'decoy:keyword-stuffer']), ('decoy-vs-bothoff', 320, ['both-off', 'decoy:generic']), ('decoy-vs-bothoff', 321, ['both-off', 'decoy:refuse'])]

## pairs2

intact vs both-off (last week) winners per seed: {'319': 'tie', '320': 'tie', '321': 'tie'}
last week vs first week (intact): {'319': 'tie', '320': 'tie', '321': 'tie'}
decoys vs both-off: {'T17': 'tie', 'T12': 'tie', 'T6': 'tie'}
pairs consistent over both A/B orders: 0/9 (consistency rate 0.0; same-position rate 1.0)
inconsistent pairs: [('w3-vs-w1', 319, ['intact-w1', 'intact-w3']), ('intact-vs-bothoff', 319, ['both-off', 'intact']), ('w3-vs-w1', 320, ['intact-w3', 'intact-w1']), ('intact-vs-bothoff', 320, ['both-off', 'intact']), ('w3-vs-w1', 321, ['intact-w3', 'intact-w1']), ('intact-vs-bothoff', 321, ['intact', 'both-off']), ('decoy-vs-bothoff', 319, ['both-off', 'decoy:keyword-stuffer']), ('decoy-vs-bothoff', 320, ['both-off', 'decoy:generic']), ('decoy-vs-bothoff', 321, ['both-off', 'decoy:refuse'])]

## RIGHT = strict ok AND reader verdict yes (section 5)

strict vs reader-1 disagreement rate 0.703; reader kappa 1.0; reader disagreements for adjudication: 0

| arm / cut | n | strict ok | verdict yes | RIGHT | strict-reader disagree | veto (strict ok, verdict no) |
|---|---|---|---|---|---|---|
| both-off|H | 36 | 0 | 36 | 0 | 1.0 | 0 |
| both-off|all | 55 | 0 | 55 | 0 | 1.0 | 0 |
| both-off|neardup | 7 | 0 | 7 | 0 | 1.0 | 0 |
| both-off|rotating | 12 | 0 | 12 | 0 | 1.0 | 0 |
| both-off|w1 | 25 | 0 | 25 | 0 | 1.0 | 0 |
| both-off|w3 | 30 | 0 | 30 | 0 | 1.0 | 0 |
| hippocampus-off|H | 36 | 21 | 36 | 21 | 0.417 | 0 |
| hippocampus-off|all | 55 | 25 | 55 | 25 | 0.545 | 0 |
| hippocampus-off|neardup | 7 | 0 | 7 | 0 | 1.0 | 0 |
| hippocampus-off|rotating | 12 | 4 | 12 | 4 | 0.667 | 0 |
| hippocampus-off|w1 | 25 | 16 | 25 | 16 | 0.36 | 0 |
| hippocampus-off|w3 | 30 | 9 | 30 | 9 | 0.7 | 0 |
| intact|H | 36 | 19 | 36 | 19 | 0.472 | 0 |
| intact|all | 55 | 24 | 55 | 24 | 0.564 | 0 |
| intact|neardup | 7 | 0 | 7 | 0 | 1.0 | 0 |
| intact|rotating | 12 | 5 | 12 | 5 | 0.583 | 0 |
| intact|w1 | 25 | 17 | 25 | 17 | 0.32 | 0 |
| intact|w3 | 30 | 7 | 30 | 7 | 0.767 | 0 |

| control | expect | reader 1 | reader 2 |
|---|---|---|---|
| subject | no | False | False |
| echo | no | False | False |
| refuse | no | False | False |
| answer | yes | True | True |
| answer | yes | True | True |
| answer | yes | True | True |
