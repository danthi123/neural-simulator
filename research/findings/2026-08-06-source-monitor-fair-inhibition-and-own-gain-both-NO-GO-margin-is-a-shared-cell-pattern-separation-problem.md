---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-self-normalised-fair-inhibition + own-gain-ceiling
runner: research/runners/_laneC_source_monitor_fair_inhibition_sweep.py
artifacts:
  - research/findings/raw/source_monitor_fair_inhibition/fair_canonical_650-654.json
  - research/findings/raw/source_monitor_fair_inhibition/fair_shunting_e65_650-654.json
  - research/findings/raw/source_monitor_fair_inhibition/own_gain_ceiling_650-654.json
---

# Self-normalised "fair" inhibition does NOT reverse the rich-get-richer direction in a spiking substrate, and the own-gain alternative is inert — the weakest-source margin is a SHARED-CELL pattern-separation problem, not a competition or gain problem. Both named fixes are NO-GO.

## What was built (both named fixes from the overlap NO-GO's contract)

The overlap NO-GO (⛔ not; the still-valid `2026-08-06-source-monitor-episode-overlap-...-NO-GO`)
isolated the binding constraint as DIRECTION: v6's per-source interneurons inhibit
only their RIVALS (self-excluded), so inhibition scales with the winner and the
weak source is buried. Fix 1 (primary): a SHARED normalization interneuron pool
driven by ALL source-memory pools that feeds GABA-A back to ALL of them INCLUDING
each pool's own drive — canonical self-inclusive divisive normalization (a spiking
PV-basket pooled-inhibition circuit, not a host rate op). Fix 2 (alternative): BCM
/ intrinsic own-gain on the weak source. Instrument, overlap patterns, and every
frozen criterion are v6-verbatim; the zero-weight control reads `strict=False` on
all rows (instrument honest).

## Fix 1 result: fair inhibition makes the weakest margin MORE negative — NO-GO

`weakest_source_strictly_improved` (min(M) > min(L)), canonical shared pool
(n=12, w=3, E_i=-75 mV; `fair_canonical_650-654.json`):

| overlap | 650 | 651 | 652 | 653 | 654 | strict-true |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.2 | F | T | F | F | F | 1/5 |
| 0.4 | F | T | F | F | F | 1/5 |

On the three DEVELOPMENT seeds it is 0/6, and it never clears the 0.15 floor.
Shunting inhibition (E_i=-65 mV, `fair_shunting_e65_650-654.json`) is 1/10. Worse
than v6's rich-get-richer (1/5) — fix 1 REGRESSES the metric it was built to fix.

## Why (mechanism-level, quantified): spiking threshold + GABA burst/release is anti-divisive

Divisive normalization's margin-compression (divide every rate by a common
denominator → a negative margin becomes LESS negative) needs a GRADED rate code.
It does not survive spiking thresholds + fast GABA kinetics. Two artifacts:
(a) overlap 0, seed 650, "seen" recall — OFF `{seen .178, heard 0, self 0}`
(clean); fair-ON `{seen .144, heard .258, self 0}`: blanket GABA-A rebound-fires
an otherwise-SILENT rival above the driven target. (b) overlap 0.4, seed 650,
"seen" recall — OFF `{seen .183, heard .230, self .199}`; fair-ON `{seen .168,
heard .32, self .352}`: release-rebound AMPLIFIES the strongly-driven rivals
(self .199→.352) while the near-threshold target is pushed down. Equal blanket
inhibition near threshold is a hard winner-take-all, not soft division.

## Fix 2 result (ceiling): own-gain ENGAGES but CANNOT clear the floor — saturation

`own_gain_ceiling_650-654.json` (v6 circuit, competition OFF, each source's OWN
episode→source recall synapses scaled ×0.5–8 — the exact synapses BCM would
potentiate, a host oracle). `lever_engaged=True` on all 10 rows (the target's own
rate responds — note this required scaling the WEIGHTS: the firing-threshold array
is a spike-DETECTION set-point that does NOT change Izhikevich v-peak spiking, so
a threshold lever reads a false zero). The best oracle own-gain improves min-margin
by only a few hundredths (best min-margin +0.0525 at overlap 0.2, +0.0175 at 0.4)
and NEVER clears 0.15 on any of the 10 rows. Why it saturates: scaling the target's
episode→source weights up to ×8 first raises then slightly LOWERS its own rate
(stronger drive → more spike adaptation), a refractory/adaptation ceiling, while the
rival keeps firing on the shared cells. Own-gain cannot outrun a rival it cannot
suppress — a BCM rule earns the same ceiling. The deficit is rival co-firing on the
SHARED cells.

## The isolated binding constraint + the next mechanism

Neither suppressing rivals at recall (fair inhibition rebounds) nor boosting the
target (saturated) can work, because the margin is set at ENCODING: symmetric
Hebbian learning lets each shared overlap cell potentiate EQUALLY to every source
it co-activated with, so at recall the shared cells drive rivals at the same
refractory ceiling as the target. The next mechanism is COMPETITIVE learning at
encoding — heterosynaptic LTD / outgoing-weight conservation on the episode→source
synapses so each shared cell COMMITS its output to one source (or dentate-style
pattern separation that sparsifies the overlap before it is learned). Attack the
shared cell's LEARNED fan-out, not recall-time competition or target gain.

## Provenance

Runner-side only (`_laneC_source_monitor_fair_inhibition_sweep.py`), reusing the
v6 fixed instrument + overlap patterns + `_source_margin` verbatim; no `sim/`
edit (fair wiring uses `RegionPathway` + `BrainRegion.syn_reversal_potential_i_override`;
own-gain scales the target pool's episode→source CSR weights as an explicit host
oracle). No frozen criterion loosened. NumPy backend, deterministic. Artifacts +
`.prov.json` sidecars:
`research/findings/raw/source_monitor_fair_inhibition/fair_canonical_650-654.json`,
`research/findings/raw/source_monitor_fair_inhibition/fair_shunting_e65_650-654.json`,
`research/findings/raw/source_monitor_fair_inhibition/own_gain_ceiling_650-654.json`.
