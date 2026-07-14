# Deep-credit-on-spikes POPULATION-CODING de-risk — runner built (no `sim/` edit), smoke INCONCLUSIVE (under-trained) with a RED FLAG: the read-SNR fidelity does NOT rise with pool size K (flat ~0.28 at K=1/8/16), contra the population-coding hypothesis's core prediction

**Date:** 2026-07-14 (overnight parallel arc)
**Runner:** `research/runners/_onbridge_deep_credit_population_derisk.py` (subagent-built, controller-verified: no `sim/` edit, real `_pool`/`_broadcast` block-mean read, K=1==single-neuron baseline). Reuse-by-import of `OnBridgeBDSPNet`.
**Status:** honest inconclusive-at-smoke — the runner is a good deliverable; the smoke needs proper scale to decide, AND a red flag on the population-coding hypothesis surfaced.

## What ran
The `2026-07-07-onbridge-spiking-deep-credit-training-research-gate.md` named POPULATION CODING (K neurons/logical unit, pooled event rate) as the fix for the on-bridge spiking deep-credit "does-not-train" boundary (diagnosed there as a single-neuron finite-spike-Poisson-noise wall, CV≈1.0-1.4; pooling K should cut CV by √K). This de-risk implements it (`--pool-k` block-mean read + broadcast credit + the §1d `read_snr_corr` diagnostic) and sweeps K∈{1,8,16}, 3 seeds, CPU smoke (H=40, ~30 epochs).

## Result — two honest reads

1. **Training: K=1/8/16 all FAIL to clearly train at smoke scale** (best-arm inherit 0.19-0.37 vs chance 0.333, no clear K-lift; K=1 reproduces the does-not-train boundary as intended). The runner's own verdict: the net is UNDER-TRAINED at H40/ep30 — the contrast isn't readable at this smoke; needs more epochs / wider H / GPU. So training is inconclusive (not a clean negative — the net barely trains at any K).

2. **RED FLAG — the read-SNR fidelity is FLAT across K:** `corr(pooled E, soma_rate)` mean = **+0.289 (K=1) → +0.291 (K=8) → +0.277 (K=16)** — it does NOT rise with pool size. The research gate (§1d) predicted this correlation should RISE toward the rate reference as K grows (the direct fingerprint of the √K population lift). It does not. ⇒ pooling K neurons is NOT averaging out independent noise as the Poisson-CV model assumed.

## ⇒ a0 ROOT CAUSE (confirmed by reading the substrate) — the pooled neurons are CORRELATED because the drive is deterministic

Reading `OnBridgeBDSPNet.__init__` + the drive setup: each neuron is driven by a **deterministic CONSTANT tonic current** (`tonic_h_pA=450`, `tonic_o_pA=500` — `drive[slice] = tonic_h_pA`, identical for every neuron in a slice), and the config does **NOT enable OU-process or conductance noise** (grep: no `cfg.enable_ou_process`/`enable_conductance_noise`). So the K neurons of a logical unit receive IDENTICAL drive + IDENTICAL synaptic input from the shared upstream unit — decorrelated ONLY by the bridge's small Izhikevich parameter heterogeneity. Nearly-lockstep firing ⇒ **there is almost no INDEPENDENT noise to average**, so pooling gives ~no √K gain → the flat read-SNR. The research gate's Poisson-CV model (which predicted √K) assumed INDEPENDENT single-neuron noise; the substrate's deterministic-tonic-drive spiking is highly CORRELATED across the pool, so the population fix as-is cannot apply.

## ⇒ the a0-informed FIX (biology-grounded) + the cheap test

The brain's cortical neurons fire irregularly because they receive INDEPENDENT high-conductance background synaptic bombardment (Destexhe-Rudolph 2003 "the high-conductance state"; catalog independent-Poisson-background) — that independent noise is exactly what makes population averaging (catalog E.03) work trial-by-trial. The runner's deterministic tonic drive is a SHORTCUT that removes it, correlating the pool. **Fix: give each neuron INDEPENDENT background noise (enable the OU process with a per-neuron seed, or add an independent per-neuron randn background current) → the K neurons decorrelate → their independent noise averages with pooling → read-SNR rises with K → the net can train.** This is testable CHEAPLY with a FORWARD-ONLY diagnostic (no full training): measure the pairwise correlation among a unit's K neurons + `corr(pooled E, soma_rate)` vs K, WITH vs WITHOUT the independent noise. Expected: WITHOUT → high pairwise corr + flat read-SNR (the current result); WITH independent noise → low pairwise corr + read-SNR RISING with K. (Building this forward-only decorrelation de-risk next — it decides the population hypothesis cheaply.)

## ⇒ (earlier) honest diagnosis (a0: read the substrate) + next

The most likely cause of the flat read-SNR (a0 read of the runner): the hidden/output neurons are driven by a per-neuron TONIC depolarizing background (`OnBridgeBDSPNet` drive array); if the K neurons of a logical unit share that drive with NO independent per-neuron noise, they fire near-IDENTICALLY (correlated) — and pooling correlated units gives NO √K gain (√K only averages INDEPENDENT noise). If so, the read-SNR wall on this substrate is NOT the finite-spike-*Poisson* noise the gate assumed; the spiking is more regular/correlated than Poisson, so population pooling (as-is) cannot fix it. **Caveat:** the diagnostic is measured on an UNDER-TRAINED net, so it is not yet clean — a proper-scale run is needed before concluding.

**Next (queued; needs the GPU, currently on the selective-SSM validated run):** (1) a properly-trained-scale run (wider H, more epochs, GPU) so the read-SNR + training contrast is readable; (2) if the flat-read-SNR holds → verify the pool's noise independence (does adding INDEPENDENT per-neuron OU/conductance noise, or diverse thresholds, DECORRELATE the pool so √K kicks in?) — the honest fix if the neurons are correlated; (3) if pooling genuinely can't lift it, that REFRAMES the deep-credit-on-spikes boundary away from "single-neuron Poisson noise" toward the actual bottleneck (correlated spiking / drive-mapping / low-pass window). Honest negative = first-class deliverable (maps the residual).

## Files
- `research/runners/_onbridge_deep_credit_population_derisk.py`; raw `raw/_onbridge_kfair_K{1,8,16}_s{42,43,44}.json` (the K-sweep + read-SNR diagnostic). Follows `2026-07-07-onbridge-spiking-deep-credit-training-research-gate.md`.
