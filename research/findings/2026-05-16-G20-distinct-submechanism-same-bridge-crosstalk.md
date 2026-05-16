# The distinct cross-bridge sub-mechanism: same-bridge-as-query cross-talk (characterized)

## TL;DR

The remediation closure named an open sub-mechanism: concepts that
self-win yet still lose cross-bridge (the ~5/6 of misses NOT fixed by
under-recall remediation). Pure analysis of all 10 cross-bridge
misses across both 320 benchmarks (zero GPU) characterizes it:

- **50% (5/10): SAME-AS-QUERY(A) cross-talk** — the winning
  distractor is a concept from the **query word's own bridge**.
- **40% (4/10): THIRD-bridge** spurious win (neither A's nor B's
  bridge).
- **10% (1/10): SAME-AS-TARGET(B)** near-miss (right bridge, wrong
  concept).

The dominant (plurality) component is same-bridge-as-query
cross-talk. It points to a **second artifact-safe, query-time-only
lever** (no retrain, no pattern change).

## Evidence (n=10, both 320 cross-bridge benchmarks)

| A → B | won by | distractor origin |
|---|---|---|
| one[Efn] → touch[Bvb] | it[Efn] | SAME-AS-QUERY |
| hit[Bvb] → every[Efn] | look[Bvb] | SAME-AS-QUERY |
| hit[Bvb] → dark[Cadj] | hold[Bvb] | SAME-AS-QUERY |
| ear[Anoun] → catch[Bvb] | arm[Anoun] | SAME-AS-QUERY |
| do[Efn] → light[Cadj] | yes[Efn] | SAME-AS-QUERY |
| stand[Bvb] → is[Efn] | always[Dsp] | THIRD |
| narrow[Cadj] → under[Dsp] | feel[Bvb] | THIRD |
| wait[Bvb] → thanks[Efn] | tomorrow[Dsp] | THIRD |
| thin[Cadj] → but[Efn] | smell[Bvb] | THIRD |
| head[Anoun] → catch[Bvb] | carry[Bvb] | SAME-AS-TARGET |

## Mechanism

`g20_multibridge.query_concept` searches every bridge's tags
containing the query word, recalls each, and aggregates by **max
firing rate across bridges**. Querying A stimulates A's own
well-trained per-concept tag (+ the new partial A_B tag) in A's
bridge. A's home-bridge concepts are strongly trained and fire hard;
the cross-bridge B signal comes only from a **partial** A_B encode
in B's bridge. So A-bridge concepts enjoy a "home advantage" in the
raw-rate competition and can out-fire the legitimate cross-bridge
target. That is exactly the 50% SAME-AS-QUERY pattern.

The 40% THIRD-bridge component is a separate, weaker effect
(spurious strong responders elsewhere) — not explained by the
home-advantage mechanism; likely the residual index/dynamical noise.

## Actionable (artifact-safe, query-time-only) lever — NOT yet implemented

The home-advantage is a **scoring artifact of un-normalized
cross-bridge max-rate aggregation**, not an architecture defect.
Candidate fix (query-time only, no retrain, no pattern/artifact
change — analogous safety profile to the under-recall remediation):

- **Per-bridge rate normalization** before aggregation (z-score or
  divide by that bridge's tag-baseline), so a high-baseline home
  bridge cannot dominate purely by baseline activation; OR
- **Down-weight / exclude same-bridge-as-query candidates** when the
  query is explicitly cross-bridge (the answer to "what is A?" via a
  cross-bridge `A_B` tag should privilege the partner bridge).

This is a `query_concept` change only — testable exactly like the
remediation was (re-run the cross-bridge benchmark, measure). It is
**not implemented here** (characterization finding); it is the clear
next plan→implement→test step.

## Honest scope

- n = 10 misses. 50% SAME-AS-QUERY is the **plurality, not a
  majority of all behavior** — 40% THIRD-bridge is a substantial
  distinct secondary component a single re-weighting lever will NOT
  address. So the lever is expected to recover ~half the residual
  misses at best, not all. Stated explicitly to avoid overclaiming a
  silver bullet (the under-recall lever was likewise modest +3.3pp;
  expect similar order).
- Pure analysis of committed benchmark JSON; zero GPU.

## Files

- analysis inline (committed JSON: `g20_xbridge_bench_320.json`,
  `g20_xbridge_bench_320_keepall_baseline.json`)
- Continues: `2026-05-16-G20-remediation-end-to-end-CLOSURE.md`
  (which named this sub-mechanism open). Next: implement + test the
  query-time aggregation re-weighting lever.
