---
type: finding
status: contributing
date: 2026-08-26
mechanism: longitudinal-continuous-life-develop-loop
lane: continuous-substrate
seeds: [910, 911, 912, 913, 920, 921, 922, 923, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040]
---

# The continuous-life longitudinal loop is GO across 43 seeds to 120 simulated days

## Claim
The brain's between-turn CONTINUOUS LIFE — the loop that lives, learns through use, and retains across simulated
days — runs GO on the GPU-scale substrate over a long timescale, on every seed tested. This is the owner's #1
continuous-substrate priority (make the brain alive between messages) demonstrated at soak length: the substrate
LEARNS through daily use and RETAINS what it learned, with day N measurably different from day 0.

## Result (the four-day window, free compute)
Aggregate (gate-passing, per-seed source artifacts listed inside):
`research/findings/raw/_harvest_2026_08_26/continuous_life_43seed_agg.json`.
`research/runners/_longitudinal_develop_loop_gpu` ran fresh-seed loops each cycle:
- 8 seeds at **120 simulated days** (`longi_nd120_s*.json`) — all `go: true`.
- 35 seeds at **60 simulated days** (`longi_nd60_s*.json`) — all `go: true`.

On every one of the 43 seeds: `loop_closed`, `lineage_persisted`, `day0_vs_dayN_differs`, and `moat_clean` are
all true. Recall accuracy mean 1.0 and retention accuracy mean 1.0 on every seed. The real-learning signal —
corr(M,C) between what the stream-cortex learned and the ground-truth code — sits around 0.85 across seeds (120d
mean ~0.85, min ~0.84; 60d mean ~0.86), i.e. the daily development is REAL code-learning, not decoding artifact.
Day-over-day development is visible in the trends (vocab, concepts-heard, facts all grow across the run).

The persistent-living-loop soak corroborates aliveness over long continuation
(`persistent_living_seg6000_c*.json`, 38/39 GO across continuation segments) — self-directed survival (band occupancy,
energy floor), persistence across reset, and drive-neural correlation hold across nearly every continuation
segment.

## Instrument + control
- Instrument for "learns through use": corr(M,C), the correlation of the stream-cortex weights with the true
  fact code — a direct read of whether daily use changed the substrate toward the data.
- Anti-cheat control: a FROZEN-brain arm (`frozen_anticheat_ok`) confirms the recall does not come from a
  static decoder; `day0_vs_dayN_differs` confirms the state actually developed; the 0-false-accept moat
  controls for confabulation (retention is genuine recall, not permissiveness).

## Caveat (STDP-inert fix, honest bookkeeping)
Loop processes started BEFORE `d8e817bd1` (the WM-buffer STDP clock fix, now on main via the four-day merge) ran
with the conversation working-memory buffer's STDP silently inert (delta_t==0). Those runs are NOT corrupt — the
substrate learning they measure is the StreamCortex Hebbian (the corr(M,C) signal) + the composer fact-store, both
unaffected; only the WM-buffer STDP was dead there. Loops launched after the fix are STDP-faithful (applied-count
544k-575k/turn). The WM-attractor synapses are non-plastic by design, so genuine WM-attractor learn-through-use
STDP is a de-risked follow-on, not covered by this soak.

## Next (no-defer)
This is the runner-level GO for the continuous substrate. The production default-on flip of the continuous engine
already landed separately (ledger `continuous-state-engine`); the next rung is the ongoing continuous-flip soak +
merging the wander SELECTION onto the recall composer's own bridge (one-brain organ merge) and making the idle-tick
scheduler a neural default-mode oscillation rather than a host timer.
