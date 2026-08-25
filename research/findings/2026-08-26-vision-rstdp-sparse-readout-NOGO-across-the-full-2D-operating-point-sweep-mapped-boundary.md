---
type: finding
status: contributing
date: 2026-08-26
mechanism: vision-rstdp-sparse-object-readout
lane: vision
seeds: [42, 43, 44, 100, 101, 102]
---

# Vision: the R-STDP sparse "which-object" readout is NO-GO across the full 2D operating-point sweep (board #75)

## Claim
A learned sparse R-STDP readout does NOT recover a load-bearing object ("which") decode off the spiking vision
path anywhere in the swept operating range. The negative is not one bad setting — the whole 2D sweep of readout
width x training length reads NO-GO. This MAPS the boundary: the next rung needs a DIFFERENT readout mechanism,
not another point in this space.

## Result
`research/runners/_vision_rstdp_readout_derisk`, 6 seeds per cell, over the operating grid
n_s2 in {24,32,48,64,96,128,192,256} x epochs in {30,60,100} (14 cells): aggregate
`research/findings/raw/_harvest_2026_08_26/vision_rstdp_2dsweep_agg.json` (per-cell sources `vrstdp_*_6seed.json`).
Every cell -> `overall_verdict: RSTDP-READOUT-NOGO`, with
`per_seed_capability_go` false on all 6 seeds. The learned spiking-WTA held decode does not clear the config
NO-GO floor (0.34; object chance 0.25) and does not beat the random-readout control — i.e. learning is not
load-bearing on the sparse spiking readout at any width or training length in range.

## Instrument + control
- Instrument: held-set object decode via a spiking soft-WTA over per-class spike-sums, after R-STDP training.
- Control (the discriminating one): the RANDOM-readout arm. The de-risk was framed to REQUIRE learned >> random
  (learning load-bearing) because on RATE, random == learned (the #72 result). The sweep confirms the hoped
  spiking reframe fails: learned does not beat random on spikes across the grid.

## Why it fails / next mechanism (no-defer)
The mechanism note points at the cause: the distributed random S2 code is quantization-fragile, so a sparse
discriminative R-STDP readout has no stable margin to learn against. This is a verdict on the METHOD (sparse
R-STDP readout off a random distributed code), not on the capability. Next rungs, each a different readout: (1) a
learned DENSE population readout (supervised local objective) rather than sparse R-STDP; (2) shape the S2 code so
it is not quantization-fragile (learned/decorrelating S2 rather than random templates) BEFORE asking a spiking
readout to separate it; (3) a credit path other than reward-modulated STDP for the readout weights. The mapped
operating point (14 cells) is the deliverable that scopes rung selection.
