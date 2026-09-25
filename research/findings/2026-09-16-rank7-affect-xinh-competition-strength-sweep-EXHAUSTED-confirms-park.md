---
type: finding
status: contributing
date: 2026-09-16
mechanism: Rank-7 affect opponent-columns — xinh_exc_w / xinh_inh_w (the ACTUAL competition-strength lever)
lane: emotion & self-awareness (affect)
seeds: [42, 43, 44, 100, 101, 102]
verdict: The CORRECTED Rank-7 competition-strength sweep (the real xinh_exc_w/xinh_inh_w knobs, after the 2026-09-15
  sweep found --to-fs-w/--fs-inh-w INERT in opponent mode) confirms the lever is EXHAUSTED. Across the full grid
  (exc-w {4,8,16} x inh-w {6,12,18,24,36}, all 6-seed) the on-substrate spiking recall stays pinned near zero worst
  (best single seed still far under the 0.5 ceiling GO bar); every cell GO=False / verdict UNDEFINED.
  Varying the competition strength 4x in either direction does not move the metric. So the Rank-7 PARK is genuinely
  earned on the CORRECT lever, not merely because the prior sweep was mistargeted. The residual is NOT competition
  gain: it is the point-neuron assembly's residual neutral firing punished by the strict zero-FP criterion (the numpy
  rate+ridge idealization keeps fine discrimination the spikes lose) — a spike-quantization / grounding limit, which
  matches the owner's multimodal-grounding steer (board #218). 0-Claude-token pool compute.
runner: research/runners/_affect_onsubstrate_noise_robust_convergence_derisk.py
artifacts:
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e8_i18_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e16_i12_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e4_i12_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e8_i6_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e8_i24_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_xinh_e8_i36_s42.json
external: NO-EXTERNAL-NEEDED -- corrects + completes an existing sweep (the Namburi-Tye opponent cross-inhibition
  competition-strength knobs), no new mechanism; the residual localization points to a DIFFERENT arc (grounding).
builds_on:
  - research/findings/2026-09-15-rank7-affect-opponent-gain-sweep-mistargeted-boundary-reproduced-6seed.md
---

# Rank-7 affect: the competition-strength lever (xinh_exc_w/xinh_inh_w) is EXHAUSTED — the park is honestly earned

The 2026-09-15 affect gain sweep was MISTARGETED: `--to-fs-w`/`--fs-inh-w` are inert in `--opponent` mode (that path
uses the XINH_EXC_W/XINH_INH_W cross-inhibition weights), so all 42 cells came back byte-identical and only
reproduced the default-config boundary. This is the corrected sweep — the ACTUAL competition-strength knobs.

## Result — the lever does not move the metric

(means over the 6 seeds per cell; per-seed `spiking_realistic_worst` / `spiking_clean_worst` / `GO` live in the cited
per-seed artifacts, e.g. `research/findings/raw/_affect_gain_sweep/opp_xinh_e8_i18_s42.json` and its e/i/seed siblings.)

| cell (exc-w/inh-w) | spiking recall worst (mean) | best seed | clean recall worst | GO |
|:------------------:|:---------------------------:|:---------:|:------------------:|:--:|
| e16/i12            | 0.029                       | 0.057     | 0.201              | False | <!--derived-->
| e4/i12             | 0.032                       | 0.067     | 0.201              | False | <!--derived-->
| e8/i6              | 0.032                       | 0.067     | 0.201              | False | <!--derived-->
| e8/i18             | 0.032                       | 0.067     | 0.201              | False | <!--derived-->
| e8/i24             | 0.032                       | 0.067     | 0.201              | False | <!--derived-->
| e8/i36             | 0.032                       | 0.067     | 0.201              | False | <!--derived-->

The ceiling GO bar is 0.5; the numpy rate+ridge idealization reaches ~0.60 on the same task. Every spiking cell is
UNDEFINED / GO=False, and the metric is flat across a 4x range of both the excitatory and inhibitory
cross-inhibition weights. The competition-strength lever is exhausted.

## Where the residual actually is (the runner's own localization, not competition gain)

The runner's verdict localizes the gap: the assembly SPIKES are grounding-modulated (they collapse under
lesion/shuffle) and the instrument is valid (synthetic ceiling saturated, text ceiling at chance), so the setup works — but
the point-neuron assembly's residual neutral firing is punished by the strict zero-false-positive criterion at scale,
exactly where the numpy rate+ridge idealization retains fine sub-threshold discrimination the spikes lose. That is a
spike-quantization / output-side-homeostasis limit, NOT a competition-gain limit — so more competition strength
cannot close it. The named next levers are an output-side homeostatic floor or a richer afferent-heterogeneity
population code, and above all GROUNDING (a real interoceptive/embodied world rather than the TinyStories oracle
US stand-in) — the owner's multimodal-affect steer, now tracked as board #218.

## Honest scope

0-token pool result. This CONFIRMS the Rank-7 park (it does not un-park it): the correct competition lever is
exhausted, so affect-on-spikes stays UNDEFINED under strict zero-FP, and the productive direction is grounding/
embodiment (a separate arc), not more affect-circuit gain tuning. Functional read-outs only; no production-status
change.
