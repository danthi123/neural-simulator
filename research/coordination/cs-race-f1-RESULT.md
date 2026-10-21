# [cs-race-f1] RESULT — F1 feasibility of the cue-addressed race: NO-GO on the pre-registered grid

PROVISIONAL (agent-filed; Claude verifies). Filed 2026-10-21 by the driver.

**Verdict: NO-GO under K1 (grid exhausted).** No row of the pre-registered
F1 grid passes clauses (a)–(i) on 8/8 dev seeds × 10 draws. The kill
criteria were committed in the probe before its first run (`3fb9155cc`), so
this NO-GO is banked with the packet's named next carrier: **a recurrent
persistent mention trace.**

preconditions:

- probe `research/runners/cs_race_f1.py` at revision `3fb9155cc`; K1–K6 written into the probe before its first run
- backend numpy, device cpu; dev seeds 1–8 (none are gate seeds); draws=10; organ only — shared `idle_replay_organ.py` unedited
- local runs, each shard under the 10-minute compute-placement bound; all 10 shards rc=0
- 360 clause records total; zero error/incomplete clauses (K2 clean — every fail below is behavior, not instrument failure)
- artifacts `research/findings/raw/_cs_race/f1/row{0,1,2,3,4}/seed*.json` (10 files, all present)

## Clause × grid-row table (pass count / 8 seeds)

```
row  grid (NE, U, tau_f)     a  b  c  d  e  f  g  h  i   all8
0    (16, 0.15,  50 ms)      0  0  8  8  0  0  2  0  8   NO
1    (16, 0.30, 150 ms)      0  0  8  8  0  0  2  0  8   NO
2    (16, 0.45, 300 ms)      0  0  8  8  0  0  0  0  8   NO
3    (24, 0.30, 150 ms)      0  0  8  8  0  0  0  0  8   NO
4    (24, 0.45, 300 ms)      0  0  8  8  0  0  0  0  8   NO
```

`row_all_pass_8_seeds` is `false` in all ten JSONs.

## Clause legend and reading

Clauses follow the packet's (a)–(i): (a) recency cue wins ≥8/10; (b) a
mention on the context turn flips the winner ≥8/10; (c) half-cue commits
nothing; (d) a cue with no matching engram commits nothing; (e) selective
steer — monotone, early ≥8/10, late ≤1/10, non-selective ≤1/10; (f) reversed
slot order keeps the content winner ≥9/10 non-tie; (g) pre-cue window all 5
turns, STP recorded per turn, winner never wins an unrelated/half cue; (h)
near ties keep both position reads inside [2, 8]/10; (i) both lesions
deterministic across 10 draws.

Reading: address-by-content fails on every row for the race's core claims —
(a) recency, (b) mention, (e) steer, (f) reversal, (h) near-tie band — 0/8
seeds at all five (U, tau_f, assembly-size) settings, while (c), (d) and (i)
pass 8/8 everywhere and (g) reaches at best 2/8. Stronger or slower
facilitation and larger assemblies do not move the failing set, so within
this grid the cue-addressed race as specified does not get feasibility.

## Notes

- Step 0 (default-regime two-patient route) is a separate probe; its first
  pool canary caught a probe-side bug (verdict written before `decide()`) —
  fixed at `72503d198`, re-canary next.
- No bar was read, none exists for this probe; no scorer or default changed.
