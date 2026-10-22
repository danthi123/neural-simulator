# [cs-race-f1b] RESULT — step-1b rerun: reference-resolved NO-GO on rows 3–4, UNDEFINED on rows 0–2

> **AMENDMENT (2026-10-22, RULING cs-race-f1g-0-1022):** clause (b) and every mention-ownership reading in this file are UNDEFINED (carrier unengaged: 0 mention spikes at MENT_PA 40 pA, sub-rheobase for the mention slots), NOT a null; clause (a) (recency) and the controls stand; clause (c) used MENT_MAX_PA 90 and fired.

PROVISIONAL (agent-filed; Claude verifies). Filed 2026-10-21 by the driver,
executing RULING `cs-race-f1-void-1021b` (the `cs-race-f1-RESULT` blanket
NO-GO stays VOIDED; this file supersedes its reading).

**Verdict.** Reference-resolved rows 3–4: **NO-GO under K1 (real)** — the
positive control is perfect (10/10 commits × 8 dev seeds) yet no row passes
all nine clauses on 8/8 seeds; (a) 0/8 and (b) 0/8 fail hardest. Rows 0–2:
**UNDEFINED, not a negative** — the control failed at seeds 3/4/5, so the
reference is unresolved exactly where the ruling forbids a NO-GO claim.

preconditions:

- probe `research/runners/cs_race_f1.py` at `research/cs-race` `d627d5188` (build) + `1ef63850a` (attribution store order fix, smoke rc=0); K1–K6 written before the first run at `3fb9155cc`, unchanged
- backend numpy, device cpu; dev seeds 1–8 (no gate seeds); draws=10; shared `idle_replay_organ.py` unedited (probe-local operating point only)
- rerun #2: rows 0–4 × both 4-seed shards, all 10 invocations rc=0, 1655 s; each invocation under the 10-minute compute-placement bound
- 360 clause records, zero clause errors (K2 clean); per-draw `engram_counts`, `gate_counts`, `first_gate_step`, `slot_counts`, `latch_pre_cue`, `cue_spikes`, `v_mV`, `winner` saved in every JSON
- positive control (ruling item d): one engram written alone — encode `grass`, full cue `yak/eat`, no competitor — bar ≥9/10 commits per dev seed
- instrument-failure line filed in `research/FAILURE_LOG.md` (`abadb1394`)
- artifacts `research/findings/raw/_cs_race/f1b/row{0..4}/seed*.json` + their own `declare_output` sidecars (20 files, commit `d7d31fcf0`); the sidecar `runner` field records the copy script that called `declare_output` (it stamps caller argv), probe revision is the one named above
- the voided grid at `research/findings/raw/_cs_race/f1/` is UNTOUCHED: replacing existing result files is an evidence rewrite the gate reserves for Claude (NEEDS `cs-race-f1b-evidence` — copy `f1b/` over `f1/` or ratify `f1b/` as canonical)

## Positive control (the reference)

```
row   per-seed commits (s1..s8)            control
0     9 10 3 7 1 10 10 9                  FAIL (s3,s4,s5)
1     10 10 6 9 2 10 10 9                 FAIL (s3,s5)
2     10 10 6 8 2 10 10 9                 FAIL (s3,s4,s5)
3     10 10 10 10 10 10 10 10             PASS
4     10 10 10 10 10 10 10 10             PASS
```

## Clause × grid-row table (pass count / 8 seeds)

```
row  grid (NE, U, tau_f)     a  b  c  d  e  f  g  h  i   row verdict
0    (16, 0.15,  50 ms)      0  0  7  8  0  6  8  1  2   UNDEFINED
1    (16, 0.30, 150 ms)      1  0  7  8  0  6  8  2  2   UNDEFINED
2    (16, 0.45, 300 ms)      1  0  7  8  0  6  8  2  2   UNDEFINED
3    (24, 0.30, 150 ms)      0  0  4  8  0  4  8  3  4   NO-GO (real)
4    (24, 0.45, 300 ms)      0  0  4  8  0  4  8  3  4   NO-GO (real)
```

Wins summed over 8 seeds (the bar is ≥8 wins **per seed**): rows 3/4
`a` = 9 and 8 of 80 possible draws, `b` = 7 and 7 — i.e. (b) averages
under one win per seed. `attributable_to` = 1.0: the b-vs-i1 gap is
fully the mention gate (i1 lesion wins = 0), so the gate moves the race
but nowhere near its bar.

## Reading

1. The instrument is now trustworthy where its reference passes: rows 3–4
   ran a perfect control and still fail (a)/(b)/(e)/(f)/(h)/(i) → K1 NO-GO
   is real on those rows, with the packet's named next carrier unchanged
   (recurrent persistent mention trace).
2. The control itself is seed-dependent at NE=16: s3/s4/s5 latch 1–8/10
   even alone, while s1/s2/s6/s7/s8 latch 9–10/10; NE=24 rows are stable
   10/10. Reference stability at NE=16 is the next open question before
   any row-0/1/2 claim can carry a verdict (design/Claude's call, not
   re-tuned here).
3. Per-draw counts now exist for every draw, so the next diagnosis (which
   stage drops the latch on s3/s4/s5) is a data read, not a rebuild.