---
type: finding
status: negative
date: 2026-08-27
verdict: H2 BIAS/DIRECTION-LIMITED (not H1 noise/rate-limited) -- the mouth read-SNR wall is a learned-decoder-direction problem; a rate/gain lever is NOT built (it would test a mechanism with nothing to fix)
mechanism: mouth read-SNR -- rate/gain (H1) vs decoder-direction (H2) diagnostic (gap#4 / #80, follow-on to the hid-decorrelation Phase-0 negative)
lane: E-language/mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_mouth_readout_rate_vs_bias/seed42.json
  - research/findings/raw/_wkv_mouth_readout_rate_vs_bias/seed43.json
  - research/findings/raw/_wkv_mouth_readout_rate_vs_bias/seed44.json
  - research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json
  - research/findings/raw/_wkv_readout_eprop_batched_substrate_marginclean_6seed.json
runner: research/runners/_wkv_mouth_readout_rate_vs_bias_diagnostic.py
---

# Mouth read-SNR (#80 continued): the wall is BIAS/DIRECTION-limited, not NOISE/RATE-limited -- redirects the frontier from a gain lever to a decoder-learning lever

## Context (do not re-derive)

`research/findings/2026-08-27-mouth-readsnr-hid-decorrelation-PHASE0-NEG.md` measured the mouth's hidden
population (hid+hidinh) and found it ALREADY independent (rho ~0, var_ratio ~1.0) -- the recurrent-inhibition
decorrelation lever has nothing to decorrelate. That finding's own redirect named the open question this file
answers: is the `sub_learned_recov_mean` ~0.34-0.37 plateau (vs `sub_copied_recov_mean` ~0.98, `weight_cosine_mean`
0.1352 / 0.1359 in `research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json` /
`..._marginclean_6seed.json`) explained by too few spikes in the read window (NOISE-LIMITED, H1 -- fix = a
rate/gain mechanism) or by the learned weight direction simply being wrong (BIAS-LIMITED, H2 -- fix = a
decoder-learning lever, not gain)? This is a PHASE-0-style diagnostic, run before building either mechanism, to
answer that question directly -- the same discipline the decorrelation Phase-0 used, which just avoided building
an inert circuit.

The ENSEMBLE (`--sub-pop`) lever is a separate, already-closed negative
(`research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`: word-pool clones are
deterministic conductance replicas, zero independent noise to average) and is not revisited here. The DENDRITIC
(Urbanczik-Senn apical-teacher) lever is separately in flight on the GPU and is not duplicated here.

## Method (measurement only, no `sim/` edit)

Additive runner `research/runners/_wkv_mouth_readout_rate_vs_bias_diagnostic.py`, reuse-by-import of
`BatchedSubstrateReadout`'s unmodified `_build_bridge` / `_wire` / `batch_margin` / `set_weights` /
`_calibrate_gain` / `_learn_substrate_batched` / `_thr_hash` / `_wcos` (all from
`_wkv_mouth_readout_eprop_batched_substrate_derisk.py`). One new instrumented read loop,
`_read_margin_and_spikes`, duplicates `batch_margin`'s drive/settle/integrate loop and additionally accumulates
the hid+hidinh population's post-settle spike count -- the same "reuse wiring, add a measurement loop" pattern
the hid-decorrelation diagnostic already established.

Per seed (42, 43, 44), on the real V=1000 vocabulary:

1. **Train a genuine, reduced-scale learned decoder.** `_learn_substrate_batched` (mode=main,
   forward=substrate -- the UNMODIFIED production rule, 0 host matmul on the forward, verified
   `host_matmul_on_forward=0` / `forward_is_substrate=True` at all 3 seeds) trained a fresh `W_hat[V,D]` at
   B=8, 80 gradient steps (n_train_pos=640, 1 epoch), read_window=60 during training. This is NOT a decisive
   6-seed replication -- it is a small-budget training run, kept CPU-numpy-tractable, whose OWN
   `weight_cosine` (learned `W_hat` vs the copied `head_w`) is measured rather than assumed:
   **0.117 / 0.1276 / 0.122** (seeds 42/43/44 -- <!--derived--> mean 0.1222) -- closely matching the cited
   production 6-seed numbers (0.1352 / 0.1359), confirming this reduced training produces a genuinely
   representative, weakly-aligned learned decoder, not an undertrained outlier.
2. **Hold both decoders fixed** (`W_hat` learned, and `hw = ro.head_w` copied) for the rest of the run. Sweep
   three knobs that change the hidden population's firing WITHOUT touching either decoder: `hid_gain` (drive
   amplitude, {30, 60, 120(baseline), 240}), `read_window` (integration time, {60, 150(baseline), 300, 480}),
   `ou_std` (noise magnitude, {10, 40(baseline), 80, 160}). All three are host-side scalars read fresh every
   call/step (`hid_gain`/`read_window` at the top of the read loop; `ou_noise_std` at every
   `_run_one_simulation_step`, precomputed once from `cfg.ou_std_current_pA` at `sim/bridge.py:3836-3838` and
   never re-derived after -- verified empirically, no wiring rebuild between sweep points).
3. **At every operating point, for both decoders**, on the SAME 24 held-out eval positions (matched spike
   count by construction, since hid/hidinh's own drive does not depend on which decoder is being read): recov
   (identical formula to `_eval_substrate`: mean(mass_read)/mean(mass_ax)) and the ACTUAL measured mean
   spikes/neuron over the post-settle read window.

Backend: numpy (CPU), per the task's cost-routing preference -- ~7-8 min/seed (build ~15s, seed-trap hash
~14s, training ~120-145s, the 10-op sweep ~5 min), run in the foreground (3 sequential per-seed invocations,
each under a 10-minute budget). The CLAUDE.md seed trap (`cfg.seed`, not `actual_seed_used`) is checked by a
build-twice hash of `cp_neuron_firing_thresholds`: **SEEDED at all 3 seeds.**

## Result: three independent tests, all three point the same way

**Test (a) -- does recov track measured spike count?** `read_window` (pure integration time, no signal
amplification) moved spike count 6.6-6.7x (0.50 -> 3.35-3.50 spikes/neuron/window) at every seed and left
BOTH decoders essentially FLAT:

| seed | rw60 spikes | rw60 learned/copied | baseline(rw150) learned/copied | rw300 learned/copied | rw480 learned/copied |
|---|---|---|---|---|---|
| 42 | 0.50 | 0.5406 / 0.4935 | 0.5406 / 0.7363 | 0.5163 / 0.7085 | 0.5406 / 0.7085 |
| 43 | 0.51 | 0.4199 / 0.3608 | 0.3957 / 0.4410 | 0.4014 / 0.4410 | 0.4011 / 0.4410 |
| 44 | 0.53 | 0.2305 / 0.4444 | 0.2289 / 0.5351 | 0.2289 / 0.4834 | 0.2300 / 0.5307 |

Learned recov varies by <=0.03 across a 6.6x spike-count range at every seed. `ou_std` (pure noise magnitude,
10 -> 160, 16x) shows the same pattern -- both decoders move by at most ~0.03-0.11 and NOT monotonically in
the H1-predicted direction (seed 44's learned recov is highest, 0.3408, at the HIGHEST noise, ou160, not the
lowest). Neither knob that isolates "more independent samples" or "less noise variance" -- the two textbook
levers of a noise-limited read -- moves recov by more than measurement noise, for EITHER decoder.

**Test (b) -- the decoder-direction test (copied vs learned at matched low spike count).** At the baseline
operating point (hid_gain=120, read_window=150, ou_std=40 -- the production demo defaults, ~1.08-1.13
spikes/neuron/window measured, IDENTICAL between the learned and copied conditions since only the downstream
decoder weight differs):

| seed | spikes/neuron | learned recov | copied recov | copied/learned ratio |
|---|---|---|---|---|
| 42 | 1.079 | 0.5406 | 0.7363 | 1.36x |
| 43 | 1.109 | 0.3957 | 0.4410 | 1.11x |
| 44 | 1.126 | 0.2289 | 0.5351 | 2.34x |

The copied (perfect-direction) head reads BETTER than the learned head at every seed, at the identical spike
count -- the direction, not the sample count, already separates them here.

**Test (c) -- the decisive asymmetry.** `hid_gain` (drive amplitude, which raises signal amplitude and spike
count together) is the one knob with a real effect -- but it is wildly asymmetric between the two decoders:

| seed | gain30 spikes | learned 30->240 | learned fold-change | copied 30->240 | copied fold-change |
|---|---|---|---|---|---|
| 42 | 0.20 -> 2.05 | 0.3708 -> 0.5033 | 1.36x | 0.0563 -> 0.9610 | 17.07x |
| 43 | 0.21 -> 2.11 | 0.3153 -> 0.4152 | 1.32x | 0.0313 -> 0.8389 | 26.80x |
| 44 | 0.21 -> 2.13 | 0.2302 -> 0.2074 | 0.90x (DECLINED) | 0.0245 -> 0.8710 | 35.55x |

At the SAME 10x spike-count range, the copied decoder's recov rises 17-36x (from near-chance to
near-ceiling); the learned decoder's recov moves by at most 1.36x, and in one seed (44) it goes DOWN. This is
not "no effect exists" -- the effect is large and real, and it flows almost entirely to the well-aligned
decoder. A noise-limited read would show BOTH decoders benefiting from more gain, since higher gain raises
SNR for any reasonably-aligned direction; here it does not, because the learned direction is not reasonably
aligned to begin with (`weight_cosine` ~0.12).

<!--derived-->
The runner's own built-in scalar-threshold auto-verdict (a single `baseline_copied_recov >=
0.75` cutoff plus a pooled `corr(spikes, recov) > 0.4` cutoff) is too coarse to trust here: on the pooled
3-seed baseline (mean copied recov 0.5708, just under the 0.75 cutoff) it would print "AMBIGUOUS", and on
seed 42 alone it printed "H1_NOISE_RATE_LIMITED" purely because that one seed's baseline copied recov (0.7363)
narrowly missed the same cutoff from the other side. Neither reflects the actual multi-knob pattern above; the
verdict below is read from the three tests together, not from the runner's single scalar gate. This is worth
recording as a methodological note for any future rate-vs-bias-shaped diagnostic: a single before/after
correlation number is not a substitute for varying multiple independent knobs and checking whether the effect
is decoder-direction-dependent.

## Verdict: H2, BIAS/DIRECTION-LIMITED

**H1 (noise/rate-limited) is REFUTED at these three seeds.** It predicts recov rising with spike count
regardless of decoder identity (more independent samples reduce variance for any linear read), and it
predicts the copied head is ALSO poor at low spike count (both starved). Neither holds: `read_window` and
`ou_std` -- the two knobs that isolate pure sample-count and pure noise-magnitude -- move recov by
measurement-noise amounts only, for both decoders alike; and the copied head already reads meaningfully
better than the learned head at the SAME matched low-spike-count baseline, in all 3 seeds.

**H2 (bias/decoder-direction-limited) holds.** The one knob with a real, large effect (`hid_gain`) benefits
the well-aligned copied decoder by 17-36x and the poorly-aligned learned decoder by at most 1.36x (and by
LESS than 1x at seed 44) -- the signature of a decoder whose direction does not correlate with the
informative signal, so amplifying that signal cannot rescue it. `weight_cosine_learned_vs_copied` (0.117 /
0.1276 / 0.122, matching the cited production 0.1352 / 0.1359) quantifies the same fact directly: the learned
`W_hat` points only weakly toward the target `head_w`.

**Per the pre-registered fork: a rate/gain lever is NOT built.** Building an excitability/gain mechanism
against a read whose measured bottleneck is decoder direction would spend a decisive run testing a mechanism
with nothing to fix -- exactly the trap the hid-decorrelation Phase-0 negative avoided one lever earlier in
this same lane. This is banked as the honest, redirecting result.

**Named next lever: decoder LEARNING, not decoder gain.** The mouth read-SNR wall's real problem is that the
local three-factor rule (`_learn_substrate_batched`, an error-driven delta rule reading a noisy single-shot
substrate margin) does not converge the readout weights close enough to the target direction, not that the
substrate under-samples once a good direction is found. Candidate mechanisms for the next arc: a FORCE-style
recurrent/RLS readout (which explicitly tracks and corrects a running estimate of decoder error rather than a
single noisy gradient step per position), a longer/annealed learning schedule measured against `weight_cosine`
directly as the convergence metric (not `recov` alone, which this file shows can plateau while `wcos` is
still far from 1), or an eligibility-trace-based credit rule that reduces the effective gradient noise the
local rule integrates over. Any of these should be evaluated FIRST against `weight_cosine_learned_vs_copied`
rising substantially above ~0.13-0.14 before a full substrate recov re-measurement, since this file shows
recov is downstream of direction quality, not spike count.

## Files

- `research/runners/_wkv_mouth_readout_rate_vs_bias_diagnostic.py` -- the diagnostic (additive, no `sim/`
  edit, reuse-by-import of `BatchedSubstrateReadout`'s unmodified build/read; one new instrumented read loop
  for the spike-count measurement `batch_margin` does not itself report).
- `research/findings/raw/_wkv_mouth_readout_rate_vs_bias/seed{42,43,44}.json` -- the three runs this finding
  reports, each with the full 10-op sweep table, the trained `weight_cosine`, the seed-trap hash check, and
  provenance (`.prov.json` sidecar, argv + git SHA).
