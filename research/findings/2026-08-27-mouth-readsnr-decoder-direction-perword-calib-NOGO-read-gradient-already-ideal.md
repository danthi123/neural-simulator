---
type: finding
status: negative
date: 2026-08-27
verdict: candidate C (per-word read calibration) 6-seed NO-GO -- the substrate read is NOT mis-directing the learning gradient (it is already cos ~0.99 aligned with the ideal-map gradient), so a read fix has nothing to correct; the low learned weight_cosine is an OBJECTIVE/data limit, not a read artifact -- redirects the frontier from read fixes to the credit/objective
mechanism: mouth read-SNR decoder direction -- per-word (per-readout-neuron) affine read calibration vs the global-gain rule (gap#4 / #80, follow-on to the rate-vs-bias H2 diagnostic)
lane: E-language-mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_readout_decoder_direction/gradalign_s424344.json
  - research/findings/raw/_wkv_readout_decoder_direction/gradalign_s100101102.json
  - research/findings/raw/_wkv_readout_decoder_direction/diag3arm_s42.json
  - research/findings/raw/_wkv_readout_decoder_direction/calib_endpoint_s42.json
runner: research/runners/_wkv_mouth_readout_gradalign_derisk.py
---

# Mouth read-SNR (#80): the learned decoder direction is bad, but NOT because the read distorts it -- candidate C (per-word calibration) is a 6-seed NO-GO; redirect to the objective

Artifact: `research/findings/raw/_wkv_readout_decoder_direction/gradalign_s424344.json` (plus `gradalign_s100101102.json`, `diag3arm_s42.json`, `calib_endpoint_s42.json`).

## The question (do not re-derive)

`2026-08-27-mouth-readsnr-rate-vs-bias-H2-bias-limited.md` proved the wall is DIRECTION-limited: the substrate-forward-learned `W_hat` has `weight_cosine` ~0.135 to the copied head, and more spikes / less noise on a FIXED decoder do nothing. The named next lever was decoder LEARNING. This file asks WHY the learning rule lands on a mis-aligned direction, then tests the warranted read fix. <!--derived-->


## Three instruments (all reuse-by-import, NO sim/ edit)

1. **Read-noise + averaging** (`diag3arm_s42`, `_..._decoder_direction_diagnostic.py`): the substrate read is barely noisy (per-read CV 0.0799; a single read correlates 0.9979 with a 6-read average). Averaging K=3 reads per gradient step leaves `weight_cosine` FLAT (avgK 0.0788 vs substrate 0.0781) -- read noise is not the lever.
2. **Gradient alignment** (6 seeds, `_..._gradalign_derisk.py`): the decisive test. A delta-rule's fixed point is set by its gradient direction, so if the substrate-forward gradient matches the ideal-map (host-proxy `W@h+head_b`) gradient, the read cannot be what mis-aligns the endpoint. At representative weights (init, 0.25/0.5/1.0 x head_w), over 10 batches, it compares the softmax-onehot gradient under three reads of the SAME positions.
3. **Endpoint** (`calib_endpoint_s42`, `_..._perword_calib_derisk.py`): candidate C trained head-to-head vs the global-gain lesion + the host-proxy ceiling at matched hypers.

## Candidate C: per-word (per-readout-neuron) affine read calibration

Give each word-pool its own read slope `a[v]`, offset `c[v]` (a per-postsynaptic-neuron intrinsic gain/threshold homeostasis; the production rule uses ONE global `gain` scalar), measured once per seed with RANDOM probes (independent of head_w -> no target leak). Learning forward: `corrected=(margin_sub-c)/a` then softmax-onehot. Default-OFF; byte-identical to the lesion when `a:=gain,c:=0`.

## Result: the read gradient is already near-ideal -> candidate C has nothing to fix

**Gradient alignment, 6 seeds** (means over the six per-seed cos values in the two gradalign JSONs; per-half summaries carry `cos_sub_host_mean` / `cos_calib_host_mean` / `cos_realign_gain_mean`). `cos(g_substrate, g_ideal)` mean **0.9928** across seeds (per-W: init ~1.0, 0.25x 0.9998, 0.5x 0.995, head_w ~0.975). The GLOBAL-gain substrate read already produces an essentially ideal gradient. Per-word calibration does NOT improve it: `cos(g_calib, g_ideal)` mean **0.9868** -- a realignment of **-0.006** (slightly WORSE, never better), at every seed. Byte-identical-off verified at all 6 seeds (`cos(g_lesion, g_substrate)=1.000`). GO 0/6. <!--derived-->


**Endpoint (reduced scale).** calib `weight_cosine` **0.0791** vs lesion **0.0779** (x1.015, no lift); the host-proxy CEILING is **0.0828** -- i.e. even the EXACT-map forward barely beats the substrate at matched batched hypers, so the read is not the wcos bottleneck. (Reduced CPU scale caps the absolute wcos; the ~0.51 host-proxy wcos from 2026-08-14 came with the full 40k-position budget.)

## Verdict + redirect

**Candidate C is a characterized 6-seed NO-GO**, and it refutes its own premise: the substrate read is NOT mis-directing learning -- its per-step gradient is cos ~0.99 aligned with the ideal linear map, so a per-word (or any) read calibration has nothing to correct. The residual mis-alignment (largest at head_w, cos ~0.975) is per-word-INDEPENDENT and small. <!--derived--> The low learned `weight_cosine` is therefore a property of the softmax-onehot classification OBJECTIVE + limited data + base-rate prior -- SHARED by the exact-map host-proxy forward -- not a substrate-read artifact.

**Next lever = the credit/objective, NOT the read.** (a) A FORCE/RLS recurrent readout that tracks + corrects a running decoder-error estimate (far more sample-efficient at recovering a direction than the single-step softmax delta). (b) A direction/regression objective (match the teacher's continuous per-output target, e.g. MSE toward the teacher-projected logits) rather than argmax cross-entropy, which under-constrains direction. (c) The already-built DENDRITIC lever is itself an objective change (per-unit sigmoid teacher vs cross-unit softmax) -- I STAGED its decisive 6-seed GPU run: `research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic` (queue depth 1). Note: it keeps the same near-ideal basal read, so it can only help via the objective route, not a read fix.

## Files

- `research/runners/_wkv_mouth_readout_gradalign_derisk.py` -- the decisive gradient-alignment test.
- `research/runners/_wkv_mouth_readout_perword_calib_derisk.py` -- candidate C (per-word calibration) + endpoint arms.
- `research/runners/_wkv_mouth_readout_decoder_direction_diagnostic.py` -- the read-noise / averaging diagnostic.
