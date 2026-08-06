---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-episode-pattern-overlap + v6-fixed-GABA-A-competition
runner: research/runners/_laneC_source_monitor_overlap_sweep.py
artifacts:
  - research/findings/raw/source_monitor_overlap_sweep/calibration_650_651.json
  - research/findings/raw/source_monitor_overlap_sweep/development_652_653_654.json
---

# Episode-pattern OVERLAP makes the weakest-source margin criterion SATISFIABLE (rival burden > 0), but the fixed symmetric GABA-A competition FAILS it — it is rich-get-richer, so it degrades the weakest source. Honest NO-GO.

## The protocol fix worked: a real recall-time rival burden now exists

`make_overlapping_episode_patterns` gives the three pure-source episode patterns
(seen/heard/self) a shared core of `round(overlap*12)` cells (mixed + unseen stay
disjoint). This is the WORLD constructing the patterns — the same host boundary as
`make_episode_patterns`. With overlap 0.2/0.4 the shared cells, learned to BOTH
their own source and a rival, drive rivals during a source's own recall:
`min_rival_burden_off` rises from 0.00 (disjoint) to 0.11–0.49 across all 5 seeds.
So the v9/instrument wall (`rival_burden = 0` ⇒ criterion unsatisfiable) is removed.

## The fixed instrument stays honest under overlap

The zero-learned-weight control (no experience; recall order under the v6 per-recall
`reset_dynamical_state`) yields `strict=False` at EVERY overlap level on EVERY seed
(`control_zero_weight_strict=false`, all rows). No stepping-history artifact
manufactures a margin; the improvement, where it appears, is real.

## The verdict: competition moves the WINNER, degrades the WEAKEST — NO-GO

`weakest_source_margin_strictly_improved` (min(M) > min(L)) at overlap>0:

| overlap | 650 | 651 | 652 | 653 | 654 | strict-true |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.0 | F | F | F | F | F | 0/5 (rival_burden=0, min M==L) |
| 0.2 | F | F | F | T | F | 1/5 |
| 0.4 | F | F | F | T | F | 1/5 |

1/5 seeds is a NO-GO, and the single seed that passes (653) never clears the frozen
0.15 floor (its min margin M = +0.005) and shows the wrong source dominating the
weakest recall at overlap 0.4.

## Why (mechanism-level, quantified): symmetric lateral GABA-A is rich-get-richer

Numbers below are 3-decimal reads of the cited `calibration_650_651.json`.

Each source-memory pool drives its own FS interneuron, which inhibits the rivals.
This amplifies whoever is already ahead. For the STRONGEST source it works as
intended — seed 650, overlap 0.2, seen: rival burden 0.277→0.220 and margin <!--derived-->
0.037→0.063 (+0.026). But for the WEAKEST source the sign flips: the rivals win the <!--derived-->
competition and inhibit the target, so competition INCREASES the weak source's rival
burden — seed 650, overlap 0.2, self_generated: rival burden 0.326→0.416, margin <!--derived-->
0.046→−0.009, and a rival now dominates (`dominant_source_correct=false`). The <!--derived-->
weakest source's margin `min(M) < min(L)` on 4/5 seeds at both overlaps.

## The binding constraint is DIRECTION, not magnitude

It is not "not enough suppression" and not the floor. Competition demonstrably
reduces rival firing — for the winner. The weak source cannot be helped by a rule
whose inhibition is proportional to the WINNER's drive, because the weak source is
the loser. min(M) is driven MORE negative; even the best case (seed 653, overlap
0.2, +0.005) is ~30× below the 0.15 floor.

## Next mechanism (do NOT start this turn)

Attack the weak source's OWN gain / make inhibition fair, not the (rich-get-richer)
lateral rule: (1) a BCM sliding-threshold / metaplastic selectivity rule on each
source's episode→source recall synapses that RAISES the weakest source's own firing
gain (increase the weak source; do not equalise rates as v8 did); or (2) a
self-normalising / feedback-normalised inhibition where each pool's inhibition
scales with the RIVAL drive it receives rather than the winner's output, so the
strong pool cannot bury the weak one. Re-run this overlap sweep as the instrument.

## Provenance

Runner-side only (`_laneC_source_monitor_overlap_sweep.py`, reusing the v6 fixed
gate + `_source_margin` verbatim); no `sim/` edit; no frozen criterion loosened
(the disjointness precondition is replaced by a controlled, reported overlap — the
protocol fix named by the instrument finding). NumPy backend, deterministic.
Artifacts: `research/findings/raw/source_monitor_overlap_sweep/calibration_650_651.json`
and `research/findings/raw/source_monitor_overlap_sweep/development_652_653_654.json`,
each with a `.prov.json` sidecar.
