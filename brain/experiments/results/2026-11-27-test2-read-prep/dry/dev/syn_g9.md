# G9 cross-arm pairwise read (read-spec-2026-11-21, spec sha256 6cf3690a68e1)

Seeds [319, 320, 321]; week 3; lived context all; user turns identical arm vs B: {'319': True, '320': True, '321': True}.

**G9: UNDEFINED** (arm preferred or tie on >= 4/6 seeds (scaled ceil(4n/6)), per reader; two readers, both valid (else UNDEFINED); every reader valid and passing; readers: 2).

| reader | valid | complete | reads valid / bad / dup id / missing | arm / B / tie | arm or tie | need | G9 | decoys not above both-off | decoys strictly below both-off | same-position non-dup (cap) | dups tie (report-only) | canary copy not preferred | canary source preferred (report-only) | order consistency (all / cross) | same-position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pairs1 | False | True | 30 / 0 / 0 / 0 | 0 / 0 / 3 | 3/3 | 2 | n/a | 3/3 | 0/3 (need 2) | 1.0 (0.333) | 3/3 | 6/6 (need 5 of 6 planted) | 0/6 | 0.0 / 0/3 | 1.0 |
| pairs2 | False | True | 30 / 0 / 0 / 0 | 0 / 0 / 3 | 3/3 | 2 | n/a | 3/3 | 0/3 (need 2) | 1.0 (0.333) | 3/3 | 6/6 (need 5 of 6 planted) | 0/6 | 0.0 / 0/3 | 1.0 |

## pairs1 per seed

| seed | winner | orders | decoy | decoy vs both-off winner | dup of | dup raw choices |
|---|---|---|---|---|---|---|
| 319 | tie | ['arm', 'B'] | keyword-stuffer | tie | arm | ['A', 'A'] |
| 320 | tie | ['B', 'arm'] | generic | tie | B | ['A', 'A'] |
| 321 | tie | ['B', 'arm'] | refuse | tie | arm | ['A', 'A'] |

| canary | seed | planted on | winner |
|---|---|---|---|
| pet-named | 319 | arm | tie |
| invented | 320 | arm | tie |
| experience | 321 | arm | tie |
| wrong-episode | 321 | B | tie |
| perspective | 321 | B | tie |
| collapse | 321 | arm | tie |

## pairs2 per seed

| seed | winner | orders | decoy | decoy vs both-off winner | dup of | dup raw choices |
|---|---|---|---|---|---|---|
| 319 | tie | ['B', 'arm'] | keyword-stuffer | tie | arm | ['B', 'B'] |
| 320 | tie | ['arm', 'B'] | generic | tie | B | ['B', 'B'] |
| 321 | tie | ['arm', 'B'] | refuse | tie | arm | ['B', 'B'] |

| canary | seed | planted on | winner |
|---|---|---|---|
| pet-named | 319 | arm | tie |
| invented | 320 | arm | tie |
| experience | 321 | arm | tie |
| wrong-episode | 321 | B | tie |
| perspective | 321 | B | tie |
| collapse | 321 | arm | tie |

Readers combined (report-only; a winner only when both readers agree): {'319': 'tie', '320': 'tie', '321': 'tie'}

Pairwise kappa: 0.0
