---
type: finding
status: negative
date: 2026-08-27
verdict: Schuessler aligned/oblique READOUT-MAGNITUDE knob is a 3-seed NULL here (sweeping initial OR sustained ||W|| does not move weight_cosine); and the SAME full-scale run corrects the candidate-C finding's reduced-scale remark -- at production scale the ideal-map forward reaches wcos ~0.40 / hostlin ~0.86 vs the substrate's 0.1359 / 0.3705, so the read DOES cost the endpoint (via a ||W||->cap runaway), it just is not fixable by magnitude or per-word calibration
mechanism: mouth read-SNR decoder direction -- readout weight magnitude (aligned/oblique regime) as the control knob (gap#4 / #80)
lane: E-language-mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_readout_decoder_direction/hostproxy_mag_s42.json
  - research/findings/raw/_wkv_readout_decoder_direction/hostproxy_mag_s4344.json
  - research/findings/raw/_wkv_readout_decoder_direction/initscale_s42.json
  - research/findings/raw/_wkv_readout_eprop_batched_substrate_marginclean_6seed.json
runner: research/runners/_wkv_mouth_readout_hostproxy_magnitude_sweep.py
---

# Mouth read-SNR (#80): the aligned/oblique READOUT-MAGNITUDE knob is a NULL here -- and the full-scale test corrects the substrate-vs-ideal endpoint picture

Artifact: `research/findings/raw/_wkv_readout_decoder_direction/hostproxy_mag_s42.json` (plus `hostproxy_mag_s4344.json`, `initscale_s42.json`, and `_wkv_readout_eprop_batched_substrate_marginclean_6seed.json` as the full-scale substrate anchor).

## External source (live literature, coordinator-supplied)

Schuessler et al. 2023, eLife, "Aligned and oblique dynamics in recurrent neural networks" (https://elifesciences.org/articles/93060): a linear-readout network trains into an ALIGNED regime (dynamics along the readout, high readout weight_cosine) or an OBLIQUE one (dynamics oblique to it, low weight_cosine); the proposed control knob is the READOUT WEIGHT MAGNITUDE (small -> oblique, large -> aligned). Our learned decoder's weight_cosine 0.1359 IS oblique -- so the magnitude knob is the warranted cheap lever to test.

## Test: sweep the magnitude knob at FULL data scale (host_proxy is matmul-only)

The host-proxy forward (`W@h + head_b`, the exact map whose per-step gradient the decoder-direction finding proved the substrate reproduces to cos ~0.99) needs NO substrate reads, so it trains to convergence at the production data volume (9600 train positions, 8 epochs) in ~3 min/seed on CPU. Sweep (i) the SUSTAINED magnitude (w_target cap {40,120,400,1200} and uncapped) and (ii) the INITIAL magnitude (init_scale {0.3,1,3,10}). Seeds 42/43/44.

## Result: magnitude does NOT move the direction -- a clean null

**Sustained magnitude is inert** because the objective's natural converged norm is BELOW the cap: final `||W||` = 23.79/23.96/23.87 (seeds 42/43/44), so w_target {40..1200..uncapped} give BIT-IDENTICAL weight_cosine per seed (0.3956 / 0.3969 / 0.4003) -- the cap never binds, and removing it changes nothing. **Initial magnitude is also inert**: across init_scale 0.3x->10x the endpoint wcos is flat and, if anything, DROPS at 10x (seed 42: 0.396 -> 0.3716; final `||W||` still ~24-25), because a large random init partially persists under a short-of-infinite budget. `aligned_signal` 0/3. The reduced-scale SUBSTRATE init sweep (`initscale_s42`) agrees: wcos 0.0736 (1x) -> 0.0339 (10x), the WRONG direction.

<!--derived-->
The likely reason the knob does not transfer: Schuessler's degeneracy is between an RNN's internal dynamics and its readout, trained end-to-end; here the readout is a single trained linear map on a FIXED feature, and the softmax-onehot objective + weight decay pin `||W||` to ~24 regardless of init/cap, so there is no magnitude degree of freedom to exploit.

## The same run corrects the candidate-C finding's reduced-scale remark

`2026-08-27-mouth-readsnr-decoder-direction-perword-calib-NOGO-...md` reported (at REDUCED scale) host_proxy ceiling 0.083 ~= substrate 0.078 and inferred "the read is not the wcos bottleneck". At FULL scale that inference is wrong: host_proxy reaches weight_cosine **~0.398** / hostlinear_recov **~0.86** (seeds 42/43/44) while the full-scale SUBSTRATE forward reaches only `weight_cosine_mean` **0.1359** / `hostlinear_recov_mean` **0.3705** (`marginclean_6seed`, `w_hat_norm` **40.0** -- it HIT the cap). So the read DOES cost the endpoint (~0.40 -> ~0.136 wcos; ~0.86 -> ~0.37 recov). The candidate-C NO-GO still stands (per-word calibration cannot help a per-step gradient that is already cos ~0.99 ideal), but the read gap is real -- it just is not a per-word-affine or a magnitude effect. <!--derived-->

## Verdict + redirect

**Magnitude knob: NULL (refuted, 3 seeds).** Banked. **The real, measured gap is the substrate ENDPOINT vs the ideal map** (wcos 0.136 vs 0.40; recov 0.37 vs 0.86), with a clean fingerprint: the substrate runs `||W||` to the cap (40) while the ideal converges at ~24. <!--derived--> This matches the runner's own note -- the graded read UNDER-reads, so the softmax never gets confident, the error persists, and `||W||` runs away in a slightly-oblique direction. So the ~0.99-aligned per-step gradient still lands oblique because the OPERATING POINT (confidence / norm) is wrong, not the gradient direction and not the magnitude regime.

**Next levers (operating-point / objective, not magnitude, not per-word read calibration):** (a) fix the substrate's softmax CONFIDENCE (read temperature / gain so `||W||` converges near the ideal ~24 instead of running to the cap) -- a cheap scalar test; (b) a FORCE/RLS or per-output regression objective that does not depend on softmax confidence; (c) the already-staged DENDRITIC decisive (per-unit teacher -- an objective change) on gpu_queue. The objective/data itself also caps the IDEAL map at wcos ~0.40 (hostlin ~0.86, already near the copied-head 0.98 recov), so for the mouth's FUNCTION the recov gap 0.37 -> 0.86 is the win to chase.

## Files

- `research/runners/_wkv_mouth_readout_hostproxy_magnitude_sweep.py` -- the full-scale host_proxy magnitude sweep (matmul-only, no bridge).
- `research/runners/_wkv_mouth_readout_init_scale_sweep_derisk.py` -- the reduced-scale substrate+host_proxy init/w_target sweep.
