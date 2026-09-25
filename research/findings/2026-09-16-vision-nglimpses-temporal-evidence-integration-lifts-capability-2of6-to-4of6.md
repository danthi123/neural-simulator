---
type: finding
status: contributing
date: 2026-09-16
mechanism: D-vision satdiv readout — n_glimpses (temporal evidence integration over independent LIF glimpses)
lane: perception (Vision)
seeds: [42, 43, 44, 100, 101, 102]
verdict: The n_glimpses lever (the satdiv finding's own named next-lever for the spike-quantization gap) genuinely
  lifts vision configural-binding capability. At the GO cell (sigma8/scale760/ridge1.0), per-seed capability_go
  rises 2/6 (n_glimpses=2 default) -> 3/6 (3,4) -> 4/6 (6,8), peaking at n_glimpses=6 (per-seed held-out accuracies
  well above the 0.25 chance floor), then plateauing by 8. Averaging the C2 read over more independent LIF glimpses
  reduces the spike-quantization
  noise that capped the learned spiking-WTA read below the rate-ceiling. A further genuine step OFF the borderline
  (capability_go doubled), but STILL PARTIAL — it does not reach the runner's 5/6 beats-floor capability bar, let
  alone 6/6. 0-Claude-token pool compute.
runner: research/runners/_vision_lindiscrim_readout_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/satdiv_refine_sig8_sc760_r1p0_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim3_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim4_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim8_6seed.json
external: NO-EXTERNAL-NEEDED -- sweeps an EXISTING runner flag (--n-glimpses, temporal evidence integration,
  Reynolds-Heeger evidence-accumulation framing already banked in the D-vision arc); no new mechanism or claim.
builds_on:
  - research/findings/2026-09-16-pool-harvest-vision-satdiv-ridge-lifts-off-borderline-touchpointa-multiseed.md
---

# Vision n_glimpses (temporal evidence integration) lifts configural-binding capability 2/6 -> 4/6 (still partial)

Harvest of the pool batch staged while the owner games (remote mini-PC CPU, 0 Claude tokens). The 2026-09-16 satdiv
finding named the next lever explicitly: the GO cell (sigma8/scale760/ridge1.0) sat at capability_go 2/6 with a
spike-quantization gap (the LEARNED spiking-WTA held read ~0.50 vs the rate-ceiling ~0.62), and `--n-glimpses`
(temporal evidence integration — averaging the C2 per-template MAX-over-locations spike code over G independent LIF
glimpses) is the runner's OWN named mechanism for that gap, never swept (all ~100+ lane artifacts used the
n_glimpses=2 default). Swept 6-seed: n_glimpses {3,4,6,8} at the GO cell.

## Result — a real, saturating lever

(counts + accuracies read from `by_code.count.summary.per_seed_capability_go` and
`.per_seed[].decode.LEARNED_spkwta_held` in `research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim4_6seed.json`
and its nglim3/6/8 siblings + the `satdiv_refine_sig8_sc760_r1p0_6seed.json` baseline; each restated number below
carries its own `<!--derived-->` mark.)

| n_glimpses | capability_go | mean held-out acc | beats-floor(5/6)+load-bearing |
|-----------:|:-------------:|:-----------------:|:-----------------------------:|
| 2 (default)| 2/6           | 0.500             | GO |
| 3          | 3/6           | 0.526             | GO |
| 4          | 3/6           | 0.530             | GO | <!--derived-->
| 6          | **4/6**       | **0.556**         | GO | <!--derived-->
| 8          | 4/6           | 0.538             | GO | <!--derived-->

- **The lever is real and monotonic up to a plateau.** More temporal evidence -> cleaner spiking readout ->
  more seeds clear the strict per-seed capability bar (2/6 -> 4/6). It peaks at n_glimpses=6 (mean held 0.556) and <!--derived-->
  plateaus/slightly regresses by 8 — the evidence-integration benefit saturates, consistent with diminishing returns
  from averaging more independent samples of the same underlying spike code.
- **Mechanistically** this is exactly the spike-quantization gap the satdiv finding predicted: the rate-ceiling
  (~0.62) is the noise-free readout, and each glimpse is a noisy spike-count sample of it; averaging G glimpses drives
  the learned spiking-WTA read toward that ceiling (0.50 -> 0.556 by G=6). The `task_go_5of6_beat_and_lb` (clears the <!--derived-->
  0.34 config-C NO-GO floor + learning load-bearing) holds at every G.

## Honest scope

This is a de-risk-level advance on the D-vision satdiv READOUT arc, not a production-faculty status change (so no
ledger/board row — the anti-noise line). capability_go is now 4/6 — genuinely off the borderline and doubled from
baseline, but it does NOT reach the 5/6 beats-floor capability bar or full 6/6, so vision configural binding is NOT
solved. The lever has saturated at this operating point (G=6); the next lever is elsewhere (the ~0.62 rate-ceiling
itself, i.e. a richer S2/C2 code, not more glimpses). Functional read-outs only.
