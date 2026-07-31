---
type: plan
status: live
date: 2026-07-07
---

# D1 build spec — spiking Burst-Dependent Plasticity (BDSP / microcircuit) on the two-compartment substrate

> The cheapest-first de-risk of the deep-lever research gate (`research/findings/2026-07-07-deep-lever-research-gate-spiking-deep-credit.md`). From the D0 read (BurstCCN repo + Payeur 2021 + Stuck-Naud 2024 + Sacramento-Senn 2018, verified). Goal: reproduce the confirmed EMERGE-1 rate-scale **depth-2** result **on spikes**. **Primary rule = microcircuit** (EMERGE-5c: noise-robust); Burstprop = cheaper first arm.

## The rule (single-phase, fixed-random-feedback, transport-free — NO settling loop)
**Burstprop / BDSP** (Payeur 2021, DOI 10.1038/s41593-021-00857-x):
```
dw_ij = eta * [ B_i - Pbar_i * E_i ] * Etilde_j   =   eta * E_i * (P_i - Pbar_i) * Etilde_j
```
- `E_i` = event rate (isolated spike OR first-of-burst) = the FEEDFORWARD message.
- `B_i` = burst rate = `P_i · E_i` (burst = 2nd spike with ISI < θ_burst).
- `P_i = sigmoid(β · v_apical,i)` = burst probability, set by the APICAL/top-down input = the credit channel.
- `Pbar_i` = slow EMA of `P_i` (init `P0`) = the SINGLE-PHASE constant baseline (no teach/no-teach phase switch).
- `Etilde_j` = presynaptic eligibility trace. `P0` = rest burst prob (~0.03 spiking); at rest `v_apical=0 ⇒ P≈Pbar ⇒ dw≈0` (the physical no-spurious-learning moat).
Three-factor, fully LOCAL; apical sets LTP/LTD sign WITHOUT changing `E` (the multiplexing invariant). Feedback = separate fixed-random matrices (BurstCCN `weight_Y`/`weight_Q`), NOT transposes (no weight transport).

**Microcircuit variant (PRIMARY, Sacramento-Senn 2018 + Urbanczik-Senn 2014)** — an SST-like interneuron cancels the predictable top-down so the apical carries ERROR not raw teaching (noise-robust). Local Urbanczik-Senn rules (M2.6/2.7/2.8): `dW = eta*(phi(u)-phi(v_hat))*r_pre`; self-predicting fixed point `W^PI* = -W^PP_{k,k+1}` ⇒ apical reduces to `W^PP_{k,k+1}·(phi(u_{k+1})-phi(u^I_k))` = the backprop δ, **closed form, no relaxation loop**.

## Machinery mapping (REUSE — most of it exists)
| Ingredient | Reuse (file:line) | New? |
|---|---|---|
| Apical compartment `v_apical` (leaky, plateau, electrotonic coupling) | `bridge.py:279` alloc, `:6460-6485` ODE; `config.py:213-217` (`enable_two_compartment_dap`, apical_tau/R/g_couple) | Delta: route top-down feedback into `cp_v_apical`, read `P=σ(β·v_apical)` |
| Apical transfer / plateau | `fused_graded_dendritic_plateau` `kernels.py:281-330` | reuse |
| Event/burst STP demux | `fused_stp_decay_recovery` `kernels.py:333-345` | reuse state |
| Presynaptic eligibility `Etilde` | `cp_eligibility_trace` `bridge.py:736`, `fused_eligibility_trace_decay` `kernels.py:453` | reuse verbatim |
| Per-synapse plasticity gate / freeze | `cp_plasticity_rate_gain` `bridge.py:2564,2885`, `set_plasticity_gate` | reuse |
| Fixed-random apical feedback `Y` (no transport) | `RegionPathway(plastic=False, ...)` `regions.py:250-316`, `inject_explicit_wiring` `bridge.py:2196` | reuse (separate seed, frozen; assert `Y≠W/Wᵀ`) |
| Co-resident special dynamics via mask | RF `neuron_mask` pattern `bridge.py:5646-5837` | reuse pattern → `_burst_neuron_mask` (default None = byte-identical) |
| Rate ORACLE | `DendriticMLP` `dendritic_mlp.py:58-95`; `MicrocircuitMLP` (`_emerge3`); task `make_task` (`_emerge1`) | reuse as numerical ceiling (microcircuit 0.978 / burstprop 0.796) |
| **`fused_bdsp` / burst-detector / burst-prob plasticity** | — | **ABSENT — the un-started brick (build it, additive default-off)** |

## Red-flag check (D0): CLEAR — no settling loop (build the closed-form/single-phase variant), no weight transport (fixed-random `Y`). No reframe needed.

## Build ladder (cheap-first)
- **Stage A (go/no-go, single neuron, CPU, hours):** a single spiking two-compartment neuron reproduces event/burst MULTIPLEXING. GO: (i) `E` tracks basal drive, ~invariant to apical (feedforward uncorrupted); (ii) `P` monotone in apical, `P≈P0` at `v_apical=0`; (iii) `E`/`B` separable <5% cross-talk. (NOTE: EMERGE-4 already GO'd single-neuron multiplexing R²=0.936 — Stage A may reduce to re-confirming on the exact D1 config before building the net.)
- **Stage B (decisive, GPU multi-seed):** `N_BITS=10 → 384 → 384 → 2` two-compartment spiking pyramidal layers on ONE bridge; fixed-random apical feedback; BDSP (microcircuit primary, Burstprop comparison arm) on the feedforward synapses. Reuse the EMERGE-1 task/splits/seeds VERBATIM. Match-to-rate: within tolerance of the rate oracle.

## New bricks (all ADDITIVE, DEFAULT-OFF, byte-identical when unused)
1. Two-compartment spiking pyramidal: reuse `cp_v_apical` + apical ODE; route top-down into apical (apical controls bursting, does NOT force somatic spikes).
2. Burst detector: per-neuron `last_spike_step` + burst flag; low-pass → `E`, `B`; `P=B/max(E,ε)`; `Pbar=EMA(P)` init `P0`. (~40-60 lines + EMA buffers)
3. `fused_bdsp_update`: `dw = eta·Etilde_j·(B_i − Pbar_i·E_i)`, gated by `cp_plasticity_rate_gain`, beside `fused_stdp_weight_update`. (~30-50 lines) [Microcircuit arm: the 3 Urbanczik-Senn rules + interneuron population — larger, the decided noise-robust rule.]
4. Fixed-random apical feedback `RegionPathway(plastic=False)` from layer l+1 → layer l apical; separate seed; frozen.
5. `_burst_neuron_mask` restricting the 2nd-compartment state + BDSP to the burst slice (default None = byte-identical).

## Pre-registered GO (multi-seed 42/43/44)
Spiking held-out **≥0.75 AND > spiking-vanilla-FA + 0.10 AND > apical-lesion floor + 0.10**; level-1 XOR probe **≥0.70**; train-vs-heldout gap SHRINKS vs FA. Decisive within-net contrast: same net/seed/init/feedback, only the rule differs.

## Pre-registered anti-cheats (each must hold)
1. **Fixed-vs-learned feedback** — `Y` fixed-random; byte-check `Y` never written after init, never equals a forward `W/Wᵀ` (no transport).
2. **Permuted-error/label** — shuffle `y` → held-out ~chance (generalization, not leakage).
3. **Wrong-sign apical** — negate `(P−Pbar)`/`B`/`Y` → held-out ≤ chance+0.05 (anti-learns; the burst-coded sign IS the credit).
4. **Apical-lesion** — `Y=0` / force `v_apical=0` → `P≡P0`, no credit → collapses to no-credit floor, probe ~0.5.
5. **No-teaching null (P0 moat)** — target detached → `dw≈0`, weights ~unchanged, held-out ~chance.
6. **Oracle ceiling** — fenced backprop ≥0.80 held-out (else INCONCLUSIVE).
7. **Memorization floor** — single-layer / apical-lesion arm = the point-neuron no-credit floor.

## Sources
Payeur et al. 2021 Nat Neurosci (10.1038/s41593-021-00857-x); Greedy et al. 2022 NeurIPS BurstCCN (arXiv 2206.11769, `github.com/neuralml/BurstCCN`); Naud-Sprekeler 2018 PNAS; Stuck-Naud 2024 (10.1088/2634-4386/adb511); Sacramento et al. 2018 NeurIPS (arXiv 1810.11393); Urbanczik-Senn 2014 Neuron. Project: `2026-07-01-burst-multiplexed-dendritic-credit-assignment-spec.md`, `-spiking-burst-substrate-scoping.md`, `2026-07-02-emerge5c-microcircuit-noise-robust-GO.md`, `sim/dendritic_plasticity.py::urbanczik_senn_update`.
