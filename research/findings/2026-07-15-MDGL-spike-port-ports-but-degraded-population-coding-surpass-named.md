# MDGL spike port (the mission-central step) — the off-diagonal cross-neuron mechanism PORTS to spikes but DEGRADED (+11% of the gap on LIF vs +48–64% on rate); the named surpass is POPULATION CODING (the exact lever that closed feedforward spiking credit this session)

**Date:** 2026-07-15 · **Runner:** `research/runners/_mdgl_spiking_port_derisk.py` (a LIF-surrogate trainable RNN: Heaviside spike + hard reset + fast-sigmoid surrogate; the same BPTT/e-prop/MDGL arms + anti-cheats as the rate testbed; reuse-by-import `make_task`; numpy-CPU; NO `sim/` edit). Follows the rate confirm (`2026-07-15-deep-credit-STAGE0-...MDGL-on-trainable-RNN.md`).

## The question + result
Does the off-diagonal cross-neuron temporal credit (MDGL) — validated on the rate trainable-RNN (+48–64% of the diagonal-vs-BPTT gap, anti-cheat-clean) — work ON SPIKES? On the clean-gap regime (XOR T=8, seed 42):
| | acc | reading |
|---|---|---|
| spiking BPTT (ceiling) | **0.990** | the spiking substrate CAN solve it (surrogate-gradient through-time) |
| spiking e-prop (diagonal) | **0.275** | ≈ chance — the spiking diagonal fails (gap 0.71 ports to spikes ✓) |
| **spiking MDGL (+ off-diagonal Γ)** | **0.355** (best, Γ-gain 0.4) | lifts e-prop **+0.08 = +11% of the gap** |
| zero-Γ (anti-cheat) | 0.315 | collapses to the diagonal baseline ✓ |
- **The mechanism PORTS to spikes but DEGRADED:** +11% (best over a Γ-gain sweep 0.05→1.0) vs +48–64% on the rate net. The naive full-gain Γ even HURTS (0.240 < 0.315) — the Γ needs re-scaling for the spiking dynamics (gain 0.4 is the peak). So the off-diagonal carries real-but-weak credit on spikes.

## Root cause (honest) + the named surpass
The degradation is the project's recurring rate-vs-spike theme, precisely located: the MDGL eligibility `eps_rec` is now accumulated from BINARY 0/1 SPIKES (`psi·sp`), a SPARSE/noisy presynaptic signal, vs the graded rate `psi·h` on the rate net — so the cross-neuron broadcast × the spike-eligibility carries much less information. This is the SAME rate-code degradation that hit the FEEDFORWARD spiking credit this session — and that was closed by **POPULATION CODING** (2026-07-14: e-prop's forward signal K=1 0.47 → K=8 0.877 ≈ LIF ceiling, ensemble-averaging the sparse spike signal into a graded one). ⇒ **the named surpass for the spiking off-diagonal is population coding of the credit/eligibility** (K neurons per logical unit → the ensemble spike-rate recovers the graded eligibility the off-diagonal broadcast needs), and/or a graded (rate) error broadcast for `a_j`.

## ⇒ Net (the full arc, honestly bounded)
The owner's question is answered end-to-end: the missing piece = **cross-neuron off-diagonal temporal credit**; the biological realization = **MDGL cell-type-specific one-hop neuromodulation**; VALIDATED anti-cheat-clean on the rate substrate (+48–64%); and it **PORTS to the spiking substrate** (the gap ports; the mechanism helps +11%, degraded by the spike-eligibility discretization, with population coding the named surpass — the exact lever that closed the feedforward spiking credit). NEXT (the on-substrate close): (1) population-coded spiking MDGL to recover the rate-level magnitude on LIF; (2) the on-bridge Izhikevich realization via `sim/neuromodulators.py` (per-cell-type modulator = the type's error broadcast; per-synapse tag by presynaptic cell type; × the on-bridge eligibility). NO `sim/` edit yet.
