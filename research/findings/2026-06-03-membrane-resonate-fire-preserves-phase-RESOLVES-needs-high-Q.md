# Membrane-level resonate-and-fire preserves the phase readout — RESOLVES (needs a high-Q resonator) — 2026-06-03

**One line:** The one membrane-level gap the spiking-STDP finding left open — does a *genuine spiking
membrane* (input spikes → integration → output spike, under leak + spike timing), not the steady-state
readout, fire at the right phase? — RESOLVES: a resonate-and-fire membrane reproduces the learned phase map
at **1.00** (no leak), and the result carries a **biology-translatable insight** — phase coding requires a
**high-Q, low-leak resonator** (intrinsic resonant currents like Ih), and a naive integrate-and-fire fails.

This closes the membrane-level rung flagged in
`2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md`: the readout/cleanup were confirmed in the
rf substrate, but not the input-spike → membrane-integration → output-spike step under realistic dynamics.

## Model

A complex membrane `z` per output neuron rotates one carrier cycle (`e^{-iω}` per step); each active input
neuron delivers a real synaptic kick to `z` at the timestep of its phase; the neuron fires at its **resonant
phase** `arg(z)` — the time a resonate-and-fire neuron emits (Izhikevich 2001 / Frady-Sommer 2019). Weights
are the validated real-weight STDP map.

## Result (`research/findings/raw/_membrane_resonate_fire_phase_probe.py`, N=8, D=128, T=360, 5 seeds)

| membrane leak / step | resonate-and-fire retrieval |
|---|---|
| 0.0 (perfect resonator) | **1.00** |
| 0.005 | 0.82 |
| 0.02 | 0.33 |
| **naive integrate-and-fire (first threshold crossing), no leak** | **0.03 (chance 0.12)** |

**Verdict: RESOLVES** — the membrane preserves the phase readout at 1.00 with a high-Q resonator, and the
naive integrate-and-fire control retrieves at chance.

## Two findings, both honest

**1. The full spiking pipeline is validated end-to-end.** Input phase-coded spikes, integrated through the
learned real weights by a resonate-and-fire membrane, produce an output spike at the right phase (1.00).
Combined with the prior results (STDP learns the weights; the rf-substrate cleanup; bind/unbind composition),
the *entire* path — learn → encode → integrate → spike → read out → compose — works in genuine spikes.

**2. Phase coding needs a high-Q (low-leak) resonator — a biology-translatable constraint.** Leak weights
recent kicks more than early ones, biasing the resultant phase, so it degrades the readout (0.82 at
leak=0.005, 0.33 at 0.02). `leak=0.005/step` over T=360 is a membrane time constant of order ~70 ms — long
for a regular-spiking cell. The brain does not achieve sustained theta-cycle phase integration with a long
*passive* time constant; it uses **intrinsic resonant currents** (Ih and friends) that make the neuron a
high-Q oscillator. The project's Hodgkin-Huxley presets already carry Ih (`fused_hh_h_current_update`), so
this is not a missing mechanism — it is a *prediction* the substrate makes about *which* neurons can carry
phase-coded composition (resonant, Ih-bearing cells), and a constraint any production migration must honour
(use resonate-and-fire / resonant dynamics, not plain leaky integrate-and-fire).

**3. The resonate-and-fire mechanism is load-bearing.** The naive integrate-and-fire (fire at the first
`Re(z) > θ` crossing) retrieves at chance — it fires early, dominated by accumulation timing, not the
resultant phase. So phase-coded composition is *specific* to the resonant-phase readout; it is not a generic
property of any spiking neuron. This is an honest constraint, not interchangeability.

## Where this leaves the build

The spiking pipeline for the phasor substrate is now validated end-to-end, including the membrane step, with
the biological requirement characterized (high-Q resonate-and-fire neurons; the rule and readout each
faithful). The remaining option-1 work is the production-integration engineering (and the documented
two-attribute resonator cost on grounded codes), not open science. The "spiking-faithful, not fully
biological" caveat stands — the phasor-binding framework remains a biologically-grounded hypothesis.

## Verdict

**RESOLVES.** A genuine resonate-and-fire membrane preserves the learned phase readout (1.00), naive
integrate-and-fire fails, and phase coding requires a high-Q resonator (intrinsic resonant currents) — a
biology-translatable insight, and the last membrane-level fidelity gap is closed.
