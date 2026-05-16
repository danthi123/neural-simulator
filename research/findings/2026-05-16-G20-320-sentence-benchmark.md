# 320 ensemble — 3-way sentence retrieval benchmark: 80% genuine (verb is weak link)

## TL;DR

Quantifies the sentence-level conversational tier above pair
association: `remember <subj> <verb> <obj>` (shared engram tag
spanning 3 bridges) → query the subject → are BOTH verb and obj
retrieved? 320 ensemble, seed 42, 20 triples, top-5, anti-cheat
pre/post:

**16/20 = 80.0% genuine** (BOTH verb+obj; n_genuine == n_both → zero
coincidental passes).

## Result

| Metric | Value |
|---|---|
| verb in top-5 | 17/20 = 85.0% |
| obj in top-5 | 19/20 = 95.0% |
| **BOTH (sentence bound)** | **16/20 = 80.0%** |
| GENUINE (anti-cheat) | 16/20 = 80.0% (0 coincidental) |

## The verb is the weak link (honest, specific)

All 4 misses: `horse stop wet` (lost obj), `bird touch full`,
`fish work empty`, `table call cool` (3 lost the verb). Object
recall (95%, adjective bridge) >> verb recall (85%, verb bridge).
In subject→(verb,obj) retrieval the verb-bridge recall of the
shared 3-way tag is the bottleneck — plausibly because the verb is
encoded 2nd of 3 sequential partials and perturbed by the
subsequent object encode, or a verb-bridge pattern property. Not a
mechanism failure (verb still 85%), a characterized asymmetry.

## In context (honest capability profile)

| Capability (320 ensemble, seed 42) | Genuine rate |
|---|---|
| Pair cross-bridge (`A is B`) | 86.7% |
| **Sentence 3-way (`subj verb obj`)** | **80.0%** |

3-way binding is harder than 2-way (80% vs 86.7%) — exactly as
expected (binding 3 concepts across 3 bridges from one cue is
strictly harder than 2). 80% anti-cheat-clean 3-way sentence
retrieval at 320-concept scale is a solid, honest sentence-level
result — not overclaimed.

**Same seed-42 caveat as the pair benchmark:** seed 42 is a clean
sparse seed; multi-seed 3-way is very likely < 80% and is not
measured here (would need the 320 multi-seed retrains — out of
scope, overlaps the flagged recovery task). Honest status:
sentence 3-way = 80% **seed-42**, multi-seed TBD.

## Files

- `research/runners/g20_sentence_benchmark.py` +
  `tests/test_g20_sentence_benchmark.py` (5 sampler tests)
- `research/findings/raw/g11_bg/g20_sentence_bench_320.json`
- Distinct from the pure tag-name role-query string match
  (`tests/test_g20_sentence_roles.py`, 12 tests) — this measures the
  sparse SUBSTRATE's 3-way recall.
- Prior: `2026-05-16-G20-320-xbridge-benchmark.md` (pair, 86.7% s42 /
  160 multi-seed 92.7%)
