# D3 spiking port, rung 2 — the TRANSITION learned THROUGH a spiking LIF hidden pool (surrogate grad); with rung 1 (re-discretization on spikes), the WHOLE recurrent step is now on-substrate

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_spiking_transition_derisk.py` (reuse-by-import of the group task + `sim.surrogate_grad`; numpy; NO `sim/` edit).
**Verdict:** GO (multi-seed) — the group-multiplication δ is learnable through a spiking nonlinearity.

## Context
The rate D3 GO found the mechanism (discrete-attractor). Rung 1 (`_d3_spiking_attractor_derisk.py`) put the RE-DISCRETIZATION on the Izhikevich bridge (S3 1.0, A5 0.933 via FS lateral inhibition). But the TRANSITION (δ: state×input→next-state scores) was still the rate-learned tanh-hidden weights, host-computed. The cheap-first design test proved the transition NEEDS a nonlinear hidden layer (pure-linear → step-delta 0.58). THIS rung realizes that hidden layer as a SPIKING LIF pool.

## The result — the DFA transition learns through a spiking LIF hidden pool
Rate-coded feedforward SNN: input `[emb[prev_state]; input_code]` → `W1` → a LIF HIDDEN pool (T=16 hard-reset LIF steps, rate = mean spikes) → `W2` → K next-state scores; trained THROUGH the spiking threshold by SURROGATE GRADIENT (`sim.surrogate_grad.atan_surrogate`) on the teacher-forced transition triples.

| S3, seeds 42/43/44 | step-delta |
|---|---|
| **SPIKING-hidden (LIF, surrogate)** | train 0.999 / same **1.000** / deeper 0.999 (every seed) |
| PURE-LINEAR control (identity hidden) | **0.582** (fails — δ not linearly separable) |

**⇒ the group-multiplication DFA transition is learned through a spiking LIF hidden pool (step-delta 1.0 ≫ pure-linear 0.58 — the nonlinearity is load-bearing, cleanly isolated).** A methodology self-catch: the first control used ReLU (`max(drive,0)`), which is ITSELF nonlinear (0.958, no separation) — corrected to a pure-linear identity control (0.58) so the isolation is valid.

## ⇒ the WHOLE recurrent step is now on-substrate
- **Rung 1: re-discretization** — spiking FS-WTA over K attractor pools (S3 1.0, A5 0.933). 
- **Rung 2: transition** — spiking LIF hidden pool learns δ (step-delta 1.0). 
Both halves of the discrete-attractor recurrent step (`s_t = re-discretize(δ(s_{t-1}, g_t))`) now run on spiking neurons. **The discrete-attractor recurrent multi-hop composition — the mechanism for the simulated recurrent sequence/language cortex — is fully realized on the project's spiking substrate.**

## Honest scope + the next rungs
- **Rungs 1 & 2 are validated SEPARATELY (each on spikes); wiring them into ONE on-bridge recurrent loop** (the transition's spiking scores drive the re-discretization pools, the winner feeds back as the next state's `emb`, on one bridge) is the INTEGRATION rung — a composition of the two validated pieces.
- **Per-step supervision:** the transition is trained on teacher-forced (state,input→next-state) triples. **REDUCE the supervision** (end-label/reward — does the discrete-attractor architecture make end-label credit tractable where the continuous RNN failed?) is the adversarial genuineness rung.
- **A5-scale** the spiking transition (K=60, cleaner codes) + the A5 FS-WTA tune.

## Landing
The mission-central "simulated recurrent sequence/language cortex" now has BOTH halves of its recurrent step realized on spikes (transition + re-discretization), composing to arbitrary depth (length-generalizing), grounded in the mechanism the rate arc found (discrete-attractor = CA3). The remaining work (one-loop integration, reduced supervision, A5-scale) is named + scoped. NO `sim/` edit anywhere in the D3 arc.

## Files
`research/runners/_d3_spiking_transition_derisk.py` (+ `_d3_spiking_attractor_derisk.py` rung 1, `_d3_group_composition_derisk.py` rate mechanism); findings `2026-07-09-D3-*.md`.
