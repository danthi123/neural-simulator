# JOINT-training GO (6-seed, transport-free): input-dependent SELECTIVITY helps a CO-TRAINED recurrent generator carry deep context — the trained reservoir does NOT absorb it, and the joint model marginally edges the n-gram floor at the deep tail (where the frozen coupling could not)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_joint_selssm_eprop_generator_derisk.py` · CI `tests/test_reslm_joint_selssm_eprop_generator.py` (3/3) · raw `research/findings/raw/_jointssm/`. numpy; NO `sim/` edit.
**Status:** ✅ GO 6/6. The decisive mission test the frozen coupling's adversarial-verify reframed.

## Why (the reframe from the frozen coupling)

The adversarial-verify of the FROZEN coupling (`-COUPLE-selssm-into-eprop-generator-GO-5of6.md`, CORRECTION block) showed that adding the selective channel to a *frozen* e-prop reservoir is mostly a shallow readout FIX (the frozen reservoir already carries the context) + a modest deep-selective residual, and does not beat the n-gram floor at the deep tail. The honest mission question — *does input-dependent selectivity help a TRAINED recurrent generator carry deep context?* — lives in the JOINT setting: co-train the reservoir W_rec AND a selective channel TOGETHER and compare to the reservoir-alone e-prop generator.

## Setup (single variable = the co-trained selective channel; everything transport-free)

Per token: reservoir `h_t=(1−a)h_{t-1}+a·tanh(W_rec h_{t-1}+W_in x_t+b)`; selective `c_t=λ_t c_{t-1}+(1−λ_t)inj`, `λ_t=σ(w·E[tok]+b_g)`; read-out `p=softmax(W_out[h;c])`. **Both** learning signals are FIXED-RANDOM-FEEDBACK broadcasts of the read-out error (reservoir `Bh@err`, gate `Bc@err`) — NO BPTT, NO weight transport (the transport-free discipline the frozen coupling's verify established). Arms: **joint_eprop** (reservoir e-prop, read-out over `h_t` only) · **joint_eprop_sel** (co-trained selective channel, read-out over `[h_t,c_t]`) · **joint_eprop_fix** (λ FIXED = a co-trained input-INDEPENDENT accumulator; isolates the SELECTIVITY). TinyStories V=200, deep d≥4.

## Result — 6-seed GO (deep d≥4)

| seed | joint_eprop | sel | fix | bigram | sel_gain | fix_gain | GO |
|---|---|---|---|---|---|---|---|
| 42 | 4.012 | 3.319 | 3.711 | 3.384 | +0.693 | +0.300 | GO |
| 43 | 3.971 | 3.395 | 3.717 | 3.498 | +0.576 | +0.254 | GO |
| 44 | 3.838 | 3.272 | 3.591 | 3.358 | +0.566 | +0.247 | GO |
| 100 | 3.995 | 3.341 | 3.668 | 3.444 | +0.654 | +0.327 | GO |
| 101 | 3.980 | 3.382 | 3.743 | 3.452 | +0.598 | +0.237 | GO |
| 102 | 3.937 | 3.283 | 3.630 | 3.428 | +0.654 | +0.307 | GO |

- **sel_gain > 0 on 6/6** (mean **+0.624**) — even with the reservoir CO-TRAINED (a stronger baseline than the frozen coupling), adding the co-trained selective channel robustly lowers deep-context CE.
- **sel_gain > fix_gain on 6/6** (mean +0.624 vs +0.279, ~2×) — the input-dependent SELECTIVITY is load-bearing: the trained reservoir does NOT absorb it, and it beats a co-trained fixed (input-independent) accumulator.

## By-depth (6-seed) — selectivity persists at the deep tail, and the joint model edges the bigram there

| depth | eprop | sel | fix | bigram | sel<eprop | sel<fix | sel−bigram |
|---|---|---|---|---|---|---|---|
| 1 | 5.085 | 3.881 | 4.504 | 3.503 | +1.204 | +0.624 | +0.377 |
| 2 | 4.903 | 3.583 | 4.212 | 3.605 | +1.320 | +0.629 | −0.023 |
| 3 | 4.494 | 3.315 | 3.890 | 3.407 | +1.179 | +0.574 | −0.092 |
| 4-5 | 4.158 | 3.318 | 3.746 | 3.497 | +0.840 | +0.428 | −0.179 |
| 6-9 | 3.902 | 3.356 | 3.671 | 3.411 | +0.545 | +0.314 | −0.055 |
| 10-99 | 3.688 | 3.300 | 3.556 | 3.330 | +0.388 | +0.256 | **−0.030** |

- **`sel<fix` (clean input-dependent selectivity benefit) is positive at EVERY depth including the deep tail** (+0.256 at d≥10) — a genuine, decaying-but-persistent deep-selective benefit, and at the deep tail it is STRONGER than the frozen coupling's (+0.19). Co-training does not let the reservoir absorb the selective function at any depth.
- **`sel−bigram` is negative from d2 through the deep tail** (incl. **−0.030 at d≥10**) — in the JOINT setting the selective generator marginally EDGES the memoryless bigram floor even at the deep tail, where the frozen coupling was worse-than-bigram (+0.04). A real (if small) improvement from co-training.

## ⇒ honest verdict (with the frozen coupling's lessons applied)

Input-dependent selectivity, trained FULLY transport-free (no BPTT, random feedback) and CO-TRAINED with the reservoir, robustly improves the emergent recurrent generator at every context depth (6/6) and the input-dependent form beats a co-trained fixed accumulator (6/6) — so the selectivity is a genuine addition the trained reservoir does not subsume. As in the frozen case, the benefit is largest at shallow depth (a representation/readout improvement) and decays with depth, BUT here it persists more strongly at the deep tail (+0.26) and the joint model marginally beats the n-gram floor at the deep tail (−0.030), which the frozen coupling could not. **Honest scope (no overclaim):** the deep-tail edge over the bigram is marginal (−0.03) at this tractable scale (the CEILING/reservoir-scale regime bounds how far the deep tail can go without 50–200× more scale); the claim is "co-trained selectivity genuinely helps the recurrent generator carry deep context and marginally clears the n-gram floor at the deep tail," NOT a large deep-tail win.

## Next
- The mission-central ON-BRIDGE (spiking) realization: the reservoir is spiking (EMERGE-82) and the selective channel is on-bridge (`cp_ssm_state`, Rung 4b-iii-b) — wire the co-trained selective channel into the spiking generator (both learned transport-free).
- Scale (does the deep-tail edge widen with V/data — the Rung-3 trajectory direction).
- raw `research/findings/raw/_jointssm/seed*.json`.
