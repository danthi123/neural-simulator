# Two stacked artifact-safe levers: +6.7pp combined (80.0→86.7, seed 42) — thread closure

## TL;DR

The same-bridge-crosstalk characterization derived a second
artifact-safe lever (query-time aggregation re-weighting). Tested
3 modes on the remediated 320 ensemble (keep-all, seed 42, 30 pairs):

| Aggregation | genuine cross-bridge | vs max |
|---|---|---|
| max (baseline) | 25/30 = 83.3% | — |
| perbridge_norm | 20/30 = 66.7% | **−16.7pp (NEGATIVE)** |
| samebridge_downweight ×0.4 | 26/30 = 86.7% | **+3.3pp** |

**Combined two artifact-safe levers (controlled, keep-all, seed 42):**
original **80.0%** → +under-recall remediation **83.3%** →
+samebridge_downweight **86.7%** = **+6.7pp**, entirely post-hoc /
query-time (no retrain, no `generate_sparse_patterns` change,
validated artifact preserved).

## Findings (honest)

- **perbridge_norm is NEGATIVE (−16.7pp).** The home-advantage
  diagnosis was correct, but blunt per-bridge 0-1 normalization
  over-corrects: it discards the firing-magnitude information that
  legitimately distinguishes a strong correct cross-bridge retrieval
  from weak noise. Cleanly rejected — documented so the dedicated
  session does not revisit it.
- **samebridge_downweight (×0.4) is a modest positive (+3.3pp).**
  Down-weighting candidates from the query word's home bridge
  recovers ~1 of the residual same-bridge-crosstalk misses — exactly
  the bounded order predicted (50% of misses are same-bridge; a
  blunt factor recovers a fraction of those, not all).
- **Stacked levers compose:** remediation (+3.3pp) and
  samebridge_downweight (+3.3pp) are independent and additive here
  (+6.7pp combined). Notably the combined keep-all 86.7% (idx-12
  INCLUDED) qualitatively matches what the ensemble previously
  achieved only by EXCLUDING idx-12 — i.e. the two safe levers
  recover the cost of the known weak index without touching the
  pattern set.

## Honest scope / caveats

- **Seed 42, n=30.** Each +3.3pp = +1 pair / 30; combined +6.7pp =
  +2 pairs. Wide CIs at this n. The DIRECTION is mechanism-grounded
  and consistent (remediation fixes under-recall; downweight fixes
  same-bridge home-advantage), but the exact magnitudes are NOT
  generalizable point estimates — multi-seed confirmation is the
  dedicated session's job.
- ×0.4 was a first guess, untuned; not claimed optimal.
- 40% of misses are THIRD-bridge spurious (a distinct effect neither
  lever targets) — so ~86.7% keep-all is a realistic near-ceiling
  for these two safe levers at seed 42, not a path to ~100%.

## Net (honest)

Two **independent, composable, fully artifact-safe** levers
(post-hoc capture-quality gate + query-time samebridge-downweight)
deliver a combined **+6.7pp** (80.0→86.7, keep-all, seed 42),
no retrain / no pattern change / validated artifact preserved —
a legitimate production recipe addition with honestly-bounded,
mechanism-grounded, modest benefit. One candidate lever
(perbridge_norm) cleanly falsified. This closes the
same-bridge-crosstalk thread with a *validated improvement*, not
just characterization.

## Files

- `research/runners/g20_xbridge_benchmark.py` (`--aggregation`)
- `g20_xbridge_agg_{max,perbridge_norm,samebridge_downweight}.json`
- Closes: `2026-05-16-G20-distinct-submechanism-same-bridge-crosstalk.md`.
  Open (dedicated session): multi-seed confirmation; the residual
  ~40% third-bridge sub-effect; ×factor tuning.
