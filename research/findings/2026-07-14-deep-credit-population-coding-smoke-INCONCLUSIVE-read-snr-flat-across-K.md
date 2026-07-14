# Deep-credit-on-spikes POPULATION-CODING lever — REFUTED by direct forward-only measurement: pooling ALREADY works (the pool is already decorrelated, √K gain already present) but does NOT lift the across-input read-SNR → the boundary is REPRESENTATIONAL / credit-STRUCTURE, not population read-variance

## ⚠️⚠️ SECOND CORRECTION (2026-07-14, the 6-seed firming) — the "population lever REFUTED" conclusion below was PREMATURE + config-dependent; the honest state is INCONCLUSIVE

A 6-seed forward-only re-run (`raw/_decorr/decorr_seed*.json`) revealed the read-SNR-vs-K behavior is **CONFIG-DEPENDENT**, so the single-config "refuted" below is withdrawn:
- **ROBUST (6/6, keep):** the pool is ALREADY DECORRELATED (within-unit corr ~0.03) and INDEPENDENT OU noise does NOT further decorrelate it (0.034→0.011) — so my a0 "correlated pool from deterministic drive" root cause IS refuted, robustly. OU is on by default (config.py:120).
- **WITHDRAWN — "the read-SNR is flat across K → population lever refuted":** that was ONE config (the first probe: tonic 450, read-SNR K1→16 0.46→0.39 flat). The 6-seed config (tonic 560 / apical 2000 / ff_w_init 4.5) shows the read-SNR RISING with K (0.118→0.339, 6/6) — i.e. pooling DOES lift the read-SNR here. So the population read-SNR benefit is CONFIG-DEPENDENT, NOT cleanly absent. The √K pooling gain is present in both (K=16: 3.5 vs √16=4.0).
- **⇒ HONEST STATE: INCONCLUSIVE (not refuted).** The a0 correlated-pool root cause is refuted; the population-coding hypothesis is neither cleanly confirmed nor refuted by these forward-only smokes — the read-SNR benefit depends on the operating point. **The decisive test is the END-TO-END TRAINING test at a CONTROLLED config where the read-SNR demonstrably rises with K** (does pooling then let the net TRAIN, where K=1 does not?) — forward-only read-SNR is only a proxy. I over-concluded from a single config; the multi-seed firming caught it. (This is a substantial controlled-training de-risk, not another forward-only smoke — queued, not overnight-quick.)

---

## ⚠️ (first correction — the "REFUTED/representational" reframe, now itself SCOPED by the 6-seed config-dependence above)

## ⚠️ CORRECTION + DECISIVE REFRAME (2026-07-14 — a forward-only decorrelation probe refuted BOTH the population hypothesis AND my a0 root-cause; direct measurement, `research/runners/_onbridge_deep_credit_decorrelation_derisk.py` + raw `_onbridge_deep_credit_decorrelation.json`)

My a0 root-cause below ("the pool is correlated because the drive is deterministic with no independent noise") is **WRONG**, corrected by reading the substrate MORE carefully + measuring directly:
- **`CoreSimConfig()` defaults `enable_ou_process=True, ou_std_current_pA=100.0`** (config.py:120-122, "enabled by default for biological realism") — and `OnBridgeBDSPNet` inherits it (does not disable it). So each neuron ALREADY has INDEPENDENT OU background noise (the per-step `randn(n_neurons)` draw). My earlier a0 grepped the RUNNER for `cfg.enable_ou_process` (not set there) and wrongly concluded OU was off — it is on by the CONFIG DEFAULT. (Lesson: read the config default, not just the runner — a too-shallow a0.)
- **Direct measurement (forward-only, K×noise):** at the committed default the within-unit pairwise correlation of the K pool neurons is **~0.03 (already decorrelated, NOT lockstep)** — each hidden neuron has its own random feedforward weights + no membrane reset between passes; and the pooling gain `single_CV/pooled_CV` is **already ~√K** (K=16: 3.56 vs √16=4.0). ⇒ pooling ALREADY averages independent variance; population read-variance is NOT the bottleneck.
- **Yet the across-input read-SNR does NOT rise with K** (corr(pooled E, clean-rate) K=1→16: 0.46→0.39, flat/falling), and **adding MORE OU noise HURTS it** (K=16: 0.39→0.035 at σ=100). ⇒ the flat-read-SNR residual is NOT trial-to-trial read variance (which pooling already cuts) — it is **across-input SIGNAL fidelity = a REPRESENTATIONAL / credit-STRUCTURE limit.**

**⇒ DECISIVE REFRAME: the population-coding lever (the `2026-07-07-onbridge-spiking-deep-credit-training-research-gate.md` fix) is REFUTED as the fix.** Pooling already works and doesn't help; independent noise doesn't help (hurts). The deep-credit-on-spikes boundary is NOT population read-SNR — the indicated lever is the **microcircuit clean-error CREDIT CHANNEL / the hidden REPRESENTATION itself**, not population coding. **Do NOT pursue the population/independent-noise lever end-to-end.** Honest caveats (cheap follow-ups): the read-SNR reference (membrane-derived soma proxy) is itself OU-contaminated → a noise-independent reference would sharpen it; 1-seed forward-only → a quick multi-seed forward confirm firms the refutation. This is the read-the-substrate-and-MEASURE discipline correcting a too-hasty a0 hypothesis — an honest negative that re-maps the boundary correctly.

---

### (Original write-up below — the RED-FLAG framing is superseded by the CORRECTION above; the runner + the flat-read-SNR data remain accurate, but the "correlated pool" root cause is refuted.)

# Deep-credit-on-spikes POPULATION-CODING de-risk — runner built, smoke INCONCLUSIVE with a RED FLAG [ROOT CAUSE RETRACTED — see CORRECTION: the pool is already decorrelated; the residual is representational, not read-variance]

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
