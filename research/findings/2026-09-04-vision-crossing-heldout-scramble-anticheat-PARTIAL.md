---
type: finding
status: partial
claim_check: measured
date: 2026-09-04
mechanism: vision configural-binding crossing — held-out-position + scramble-null anti-cheats (open Q1)
lane: D·Perception
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json
verdict: >
  The flat width-matched ELM crossing (n_s2=1152, no binding) is REAL but WEAKER than the earlier interleaved
  split showed. Under the two anti-cheats open-Q1 named — a contiguous held-out-position block split (genuine
  spatial extrapolation, never bracketed by trained neighbours) AND a scramble-null on the LEARNED spiking-WTA
  readout itself — the run returns `LINDISCRIM-READOUT-PARTIAL-beat3/6-lb6/6`: learning is load-bearing on ALL 6
  seeds (learned >> random spiking-WTA), but it clears the NO-GO floor on only 3/6 seeds (down from 5/6 on the
  interleaved/interpolation split). So the crossing is NOT a pure ELM-overfit (learning genuinely carries it), but
  its held-out-POSITION generalization is the real residual — half the seeds do not extrapolate to unseen
  positions. A verdict on the METHOD's robustness, not the capability.
---

# Vision crossing under the harder anti-cheats: real but not position-robust

## What ran
`_vision_lindiscrim_readout_derisk.py` (the build-ahead-added `--heldout-position` + `--scramble-null` flags), the
flat width-matched control arm (`--conj-bind none --n-s2 1152 --ridge 0.5`), 6 seeds, on the mini-PC pool (numpy).
Result: `research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json`,
overall verdict `LINDISCRIM-READOUT-PARTIAL-beat3/6-lb6/6`.

## The two numbers
- **learning load-bearing: 6/6** — the learned signed-discriminant spiking-WTA readout beats its random-weight
  twin on every seed. The crossing is a genuine learned effect, not a fixed-projection artifact.
- **beats the NO-GO floor: 3/6** — under the contiguous held-out-position block split (train on the first half of
  position indices, test on the unseen second half) the readout clears the floor on only 3 of 6 seeds, versus 5/6
  on the original interleaved split the earlier finding used.

## Reading it honestly
Open Q1 asked whether the 5/6 crossing was real or an ELM-overfit at high width. The answer is BETWEEN: not an
overfit (learning is load-bearing 6/6), but not robust to genuine spatial extrapolation either (3/6 on held-out
positions). The interleaved split flattered the result — trained neighbours bracketed every test position, so it
measured interpolation, not extrapolation. The scramble-null passing (the learned readout falls to chance on
pixel-scrambled held images) confirms the readout is using real structure, not a leak.

## Next (no-defer: the residual, quantified)
The residual is held-out-POSITION generalization on 3/6 seeds, not the whole crossing. Two concrete levers:
characterize WHICH seeds fail and whether it tracks a covariate (S2 bank overlap with the held positions), and
test whether the configural-binding arm (`--conj-bind fixed`, which cleared 6/6 under the old anti-cheats) holds
up better than the flat ELM under held-out-position — i.e. whether binding buys genuine position-invariance the
flat pool lacks. That companion arm was named in the runner's GO-gate but not run here.
