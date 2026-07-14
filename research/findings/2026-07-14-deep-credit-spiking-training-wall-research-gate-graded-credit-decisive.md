# Spiking deep-credit training wall — deep-research SURPASS gate + adversarial verify → the #1 mechanism is an ALREADY-IMPLEMENTED graded-credit flag (`enable_bdsp_graded_credit`); the capacity sweep confirms the binary read trains at NO cheap config; the decisive graded-vs-binary A/B is running

**Date:** 2026-07-14 (overnight)
**Status:** research gate DONE (9-agent workflow, adversarially verified) + capacity sweep DONE (credit-direction wall confirmed) + the decisive single-variable A/B (graded vs binary credit) RUNNING 6-seed. NO `sim/` edit (the graded mechanism is the committed additive/default-off `enable_bdsp_graded_credit`; the runner-side wiring is a default-False param threaded through `_run_arm`/`OnBridgeBDSPNet`).

## The wall (established, our own record)
The depth-2 two-compartment SPIKING net on ONE `SimulationBridge` (`OnBridgeBDSPNet`, committed `enable_bdsp` burst-dependent plasticity) does NOT train the real depth-required compositional-inheritance task at CPU-smoke scale — every deep-credit arm BELOW chance 0.333, 6/6 (`2026-07-07-deep-credit-onbridge-spiking-6seed-does-not-train-at-cheap-scale.md`), while the backprop ORACLE reaches 1.0 (task learnable) and the SAME rule trains the SAME task at the numpy RATE reference (0.69). So: mechanism ports to spikes at XOR, trains the real task at rate, fails on spikes at cheap scale.

## The deep-research SURPASS gate (9-agent workflow: 5 readers → synthesis → 3 adversarial skeptics)
- **Isolated residual (quantified):** a SINGLE-NEURON readout-VARIANCE wall. The postsynaptic credit factor is read as the MEASURED burst rate `cp_bdsp_B` — a finite-spike stochastic SAMPLE with CV≈1.0–1.4 at the operating event rate (n_eff≈8 vs a required ~150–500) — so forward activation, logits, AND descending credit are all noise-dominated per example; weights move (~7013) but in NOISE directions. NOT the pre-spike Mikulasch-Priesemann whitening wall → it is in the CHEAP regime.
- **Reframe (the surpass):** every working deep-spiking trainer (surrogate-BPTT, DECOLLE, SuperSpike, e-prop) reads the credit's local factor from a GRADED, low-CV signal (σ′(v−θ) = "how close to threshold"), NOT a 1-bit spike/burst count, and averages over an ENSEMBLE. Burstprop/FA LACK this graded read — exactly our failure.
- **Ranked #1 (CPU-cheap, biology-faithful):** read the credit factor from the graded low-variance signal instead of the 1-bit burst.
- **ADVERSARIAL VERIFY refuted all top-3 AS-FRAMED (the discipline working):** the #1 skeptic made a **code-confirmed catch** — the proposed "replace `cp_bdsp_E` with σ′(v−θ)" *mislocates* the read (`cp_bdsp_E` is the PRESYNAPTIC eligibility = Payeur's Ẽ_pre, not the credit gate). The correctly-located, **ALREADY-IMPLEMENTED** graded surrogate is the committed flag **`enable_bdsp_graded_credit`** (bridge.py:7275, additive/default-off, added 2026-07-12): it swaps the POSTSYNAPTIC credit factor from the sampled burst `cp_bdsp_B` to the graded EXPECTATION `cp_bdsp_E * cp_bdsp_P` (event-rate × burst-probability) via the kernel identity `B − Pbar·E == E·(P − Pbar)` — the low-variance clean-error credit that can carry the fine per-synapse sign the sampled B cannot (the on-bridge M2.6 realization). It does NOT touch the presynaptic eligibility → correctly located. (The #2/#3 skeptics likewise sharpened those bets: score the n_eff ladder against the BACKPROP-ORACLE credit not rate-FA; isolate DECOLLE with the identical binary read first.)

## Capacity sweep — the binary read trains at NO cheap config (a0 disambiguation, DONE)
More epochs (50→100→150→200) + wider H (24→48) at the favorable config: `deep_train` stays **0.15–0.225 ≈ chance 0.333** (0/2 every config) despite massive weight movement (ff up to 9117). ⇒ NOT under-training — the binary-burst credit moves weights in non-solving directions at every cheap config. This is the rock-solid baseline: the graded A/B is now a CLEAN disambiguator, not confounded by under-optimization.

## The decisive test (RUNNING 6-seed) — `_onbridge_deep_credit_graded_derisk.py`
Single variable = `graded_credit`. At the EXACT config where binary K=1 fails (tonic_h 560/tonic_o 620/apical 2000/ff_w_init 4.5, H=24, ep=50, K=1, plain-FA, the semantic-inheritance task), run TWO arms differing ONLY in the credit factor: **binary** (`cp_bdsp_B`, the does-not-train baseline) vs **graded** (`cp_bdsp_E*cp_bdsp_P`). Plus anti-cheats on the graded arm: permuted-label → ~chance (no leakage); apical-lesion → collapse to floor (apical credit load-bearing); oracle 1.0 (task learnable).
- **GO gate:** graded TRAINS (inherit > floor+0.03 AND > chance+0.03) where binary FAILS, ≥4/6, permuted ~chance, lesion collapses ⇒ the readout-VARIANCE wall is surpassed by the low-variance graded credit → the emergence engine's central on-spike training wall is CHEAP-fixable + biology-faithful.
- **NEGATIVE gate:** graded also fails ⇒ the wall is credit-STRUCTURE (the FA direction itself), not read-variance → next: oracle-aligned credit / node-perturbation-done-right / GPU scale.
- Either way the graded A/B **disambiguates read-variance vs credit-structure** — the still-open sub-question the whole deep-credit arc has been circling.

## ▶ RESULT (6-seed, DECISIVE) — the read-variance hypothesis is REFUTED; the wall is CREDIT-STRUCTURE

| arm | trains_at_all | mean inherit_heldout | mean deep_train |
|---|---|---|---|
| BINARY (`cp_bdsp_B`, sampled burst) | **0/6** | 0.228 | ~chance |
| GRADED (`cp_bdsp_E*cp_bdsp_P`, low-variance) | **0/6** (the lone 1/6 is a degenerate seed-101 instance, oracle only 0.370) | 0.204 | ~chance |

- **Graded does NOT rescue** — mean graded_inh 0.204 is if anything *below* binary 0.228, both below chance 0.333, on the config where the binary read fails at every cheap capacity (ep→200, H→48). Anti-cheats clean: graded permuted 0.228 (~chance, no leakage), lesion 0.222 (apical credit not spuriously carrying the answer), oracle 1.0 (task learnable).
- **⇒ the residual is NOT single-neuron read variance.** Lowering the credit-factor variance (the graded expectation `E·P` replacing the sampled `B`, same expected credit via `B−Pbar·E ≡ E·(P−Pbar)`) leaves training at chance. Combined with the capacity sweep (not under-training) and the earlier Node-Perturbation-frozen-on-the-small-net result, the wall is the **CREDIT DIRECTION / STRUCTURE of feedback-alignment at depth on the point-neuron substrate**, biting every credit rule — NOT the readout noise the prior verdict emphasized. This is NEW information that sharpens the 2026-07-13 "readout-noise/SNR wall" verdict: reducing the read noise (graded) does not help ⇒ it is not (only) read noise.
- **Honest scope:** the graded A/B was at K=1 (single neuron), which is off-spec for burstprop's ensemble `p=b/e` (BurstCCN used ~500 neurons/unit) — so the ENSEMBLE-SCALE form (population K=128–256 on GPU) is a distinct, still-untested lever, as is DECOLLE (per-layer LOCAL credit that DELETES the fragile multi-hop FA chain entirely). Both are the research-gate-ranked next mechanisms for a credit-STRUCTURE wall, now launched.

## NEXT (running, the credit-STRUCTURE mechanisms — research-gate #3 DECOLLE + #6 ensemble-scale)
1. **DECOLLE-ize the depth (CPU, the sharpened single-variable form):** per-layer fixed-random LOCAL readout + local target, IDENTICAL binary read — change ONLY the credit STRUCTURE (delete the multi-hop cross-layer FA chain). If per-layer local credit trains where the deep FA chain fails ⇒ the wall IS the depth of the FA credit chain.
2. **Ensemble-scale burstprop (GPU, the definitional form):** K=128–256 neurons/logical unit so `p=b/e` is a genuine ensemble statistic (as burstprop was validated). Does the definitional ensemble form train where the off-spec K=1/8 fails.

## Files
- Research gate workflow: `spiking-deep-credit-training-wall-research-gate` (journal in the run's transcript dir).
- Capacity sweep: `research/findings/raw/_dccap/cap_*_s*.json`.
- Decisive A/B: `research/runners/_onbridge_deep_credit_graded_derisk.py`; raw `research/findings/raw/_dcgraded/graded_ab_s*.json`. Threading (additive, default-False): `_semantic_inheritance_onbridge_spiking_derisk.py` (`graded_credit` param on `OnBridgeBDSPNet` + `_run_arm`).
