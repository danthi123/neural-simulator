---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-winner-fatigue-setpoint-sepbias-drives-6of6-assembly-disjointness
lane: memory
integration_faculty: d5-live-consolidation
---

# D5 pattern-separation set-point: sep_bias=1000 drives 6/6 disjoint+healthy assemblies (knob 1 closed)

**Board #73/#71 — knob 1 of the learn-through-use default-on flip.** The read-side D5 crosstalk was already CLOSED
(finding 2026-08-21-d5-crosstalk-isolated-read-CLOSED: the mV "shift" was period-2 read noise a snapshot-isolated read
removes entirely; on DISJOINT assembly membership the neighbour shift is exactly 0 by construction — `withinA[A] ∩
withinA[B] = {}` ⇒ consolidating A cannot touch B's read-path). That left ONE operating-point residual before default-on:
the winner-fatigue set-point reached full disjointness on only ~3/6 seeds at the prototype `sep_bias=500`. This finding
sweeps the set-point and pins the operating point that makes it 6/6.

## Mechanism

`sep_bias` is a per-CA3 hyperpolarizing WINNER-FATIGUE bias (pA) applied to already-fired units during assembly
formation — a homeostatic set-point (Turrigiano synaptic scaling) that holds sparsity while DRIVING disjointness
(Leutgeb 2007 / Bakker 2008 dentate pattern separation: distinct inputs are decorrelated into non-overlapping sparse
codes). Higher bias fatigues repeat-winners harder, pushing overlapping assemblies apart — but too high risks emptying
or densifying an assembly, so the operating point must be the SMALLEST bias that reaches full disjointness while every
assembly stays healthy (non-empty, non-dense).

## Result — the sep_bias → disjointness curve (Layer A, 6 seeds 42/43/44/100/101/102)

`research/findings/raw/_d5_sepbias_sweep/sb{500,750,1000,1250,1500,2000}.json`, per-seed `on_max_shared` (0 = fully
disjoint) + `on_not_dense`/`on_min_size` (healthy):

| sep_bias | disjoint (max_shared==0) | healthy | note |
|---|---|---|---|
| 500  | 3/6 <!--derived: count of seeds with on_max_shared==0 in sb500.json--> | 6/6 | prototype; s44/s100/s102 retain a residual shared connection |
| 750  | 5/6 <!--derived: sb750.json--> | 6/6 | only s44 residual |
| **1000** | **6/6** <!--derived: sb1000.json, every seed on_max_shared==0--> | **6/6** | **the threshold — full disjointness, all assemblies healthy** |
| 1250 | 6/6 <!--derived: sb1250.json--> | 6/6 | holds |
| 1500 | 6/6 <!--derived: sb1500.json--> | 6/6 | holds |
| 2000 | 6/6 <!--derived: sb2000.json--> | 6/6 | holds (no collapse up to 2000) |

At **sep_bias=1000** every seed's `on_max_shared` is 0 and every assembly is healthy (`on_not_dense=True`,
`on_min_size` 8–20) — the smallest bias that closes 6/6 disjointness. It holds monotonically to 2000 with no dense
collapse, so 1000 is a robust operating point, not a knife-edge.

## What this closes

By the crosstalk-CLOSED structural result (disjoint membership ⇒ the consolidated memory's read-path shares NO
connection with a neighbour ⇒ neighbour shift = 0), **6/6 disjoint at sep_bias=1000 ⇒ 6/6 zero A→B crosstalk at the
production encode.** This knob-1 claim (6/6 disjoint) does NOT depend on any further run — it is carried entirely by the
per-seed `on_max_shared=0` in `research/findings/raw/_d5_sepbias_sweep/sb1000.json` (every seed 42/43/44/100/101/102 has
`on_max_shared=0` and `on_not_dense=true`) plus that structural result. The `on_overlaps[].jaccard=0.0` in that artifact
is the disjointness SUCCESS signal (zero assembly overlap), not an uninterpretable metric floor. A full-verdict Layer-B confirm at sep_bias=1000
(all 6 crosstalk-seeds, store_te=20 → consol_te=40) is running as belt-and-suspenders corroboration of the empirical
neighbour shift → ~0; its artifact will land alongside. Knob 1 (the pattern-separation set-point) of the two D5
learn-through-use default-on blockers is therefore CLOSED.

## Remaining before D5 learn-through-use default-on

Knob 2: the **rise-to-6/6 read-window** — the deterministic saturating-tail residual the stabilized-read finding
disentangled from the (now-removed) period-2 read noise (a plateau-depth read that saturates non-monotonically near the
top, NOT noise). That is a bounded/soft read or a pre-saturation read-window fix, not a substrate wall. NO `sim/` edit;
this set-point is ADDITIVE + default-off; the binary moat gate is unchanged (the surfaced strength is a faithful spiking
read, not a phenomenal claim).
