---
type: finding
status: contributing
date: 2026-08-19
mechanism: independent-correctness-oracle
artifacts:
  - research/findings/raw/brian2_oracle/parity_metrics.json
  - tests/test_brian2_oracle.py
---

# An INDEPENDENT correctness oracle for the vanilla spiking core — Brian2 (zero shared code) reproduces our Izhikevich-2007 + AdEx point neurons, COBA synapse, and pair-STDP; spike timing is bit-exact, sub-threshold trajectories agree to ~1e-11 mV (float64 scheme identity) and ≤0.05 mV (float32 production)

**One-line verdict.** The project had NO independent check that its core spiking engine is numerically correct — a wrong-mechanism bug can still produce clean-looking numbers (an unseeded-substrate confound once voided a whole research arc here). Brian2 (Stimberg, Brette & Goodman, "Brian 2, an intuitive and efficient neural simulator", eLife 8:e47314, 2019; https://doi.org/10.7554/eLife.47314) shares ZERO code with our engine, so it is a far stronger cross-check than our own numpy path. Rebuilding a VANILLA subset in both simulators at the same params/dt and comparing: **no discrepancy** — the engine's Izhikevich-2007 update, AdEx update, COBA conductance/current, and soft-bound pair-STDP curve all match Brian2 within the tolerances below. This is a validation of the vanilla core, not of our custom mechanisms. <!--derived--> (the `10.7554/...` above is a DOI, not a measurement)

## What is covered (and what is deliberately NOT)

The oracle (`tests/test_brian2_oracle.py`) drives the engine's REAL pure-math kernels from `sim/kernels.py` on the numpy/CPU backend (`SIM_BACKEND=numpy` forced) in a minimal loop that mirrors the bridge's documented per-step order (exact `sim/bridge.py` line refs are in the harness), and builds the same neuron/synapse/rule in Brian2 with its own unit system, code generation, and event-driven synapse machinery. Nothing here re-expresses our biology; the kernels ARE the production dynamics/plasticity math.

- **(a) Izhikevich-2007 point neuron** — the production default neuron model (`IZH2007_RS_CORTICAL_PYRAMIDAL`: C=100 pF, k=0.7, vr=−60, vt=−40, vpeak=35, a=0.03, b=−2, c=−50, d=100), forward-Euler, constant-current drive.
- **(b) AdEx point neuron** — the second neuron model (C=281, g_L=30, E_L=−70.6, V_T=−50.4, Δ_T=2, a=4, τ_w=144, b=80.5, V_peak=−40, V_r=−70.6), forward-Euler with the engine's exp-argument clip to [−20, 5].
- **(c) COBA synapse** — exponential conductance decay (`decay=exp(−dt/τ)`) + driving-force current `I=g·(E−v)` under voltage clamp, for both AMPA (τ=5, E=0) and GABA (τ=10, E=−75).
- **(d) pair-based STDP** — the engine's soft (multiplicative) bounded, last-spike-time (nearest-neighbour) rule `fused_stdp_weight_update`, swept over Δt∈±{5,10,20,40} ms and w₀∈{0.2,1.0,1.8}.

**OUT of scope (no Brian2 equivalent):** our load-bearing CUSTOM mechanisms — multicompartment dendritic credit, BTSP plateaus, neuromodulator-gated plasticity, HTM/BDSP rules, the VSA composer. This oracle does not touch, validate, or close any of them, and it closes no capability.

## Achieved parity (real numbers; source `research/findings/raw/brian2_oracle/parity_metrics.json`, brian2 2.9.0 / numpy 1.26.4 / backend=numpy)

(full per-case table + provenance in `research/findings/raw/brian2_oracle/parity_metrics.json`.) Two comparisons per neuron model: **float64** isolates the integration SCHEME (engine kernels run in float64 vs Brian2 float64 → algorithmic identity); **float32** is the production dtype (vs Brian2 float64 → quantifies the precision gap). Spike-step indices are compared by exact integer equality.

| Subset | Case | Spikes eng/bri | Spike steps exact | max|Δv| float64 | max|Δv| float32 |
|---|---|---|---|---|---|
| Izhikevich-2007 | I=100 pA | 5 / 5 | yes (both) | 6.7e−11 mV | 4.5e−02 mV |
| Izhikevich-2007 | I=300 pA | 20 / 20 | yes (both) | 4.4e−13 mV | 3.1e−04 mV |
| Izhikevich-2007 | I=600 pA | 36 / 36 | yes (both) | 2.6e−13 mV | 3.5e−05 mV |
| AdEx | I=400 pA (subthr.) | 0 / 0 | yes (both) | 2.8e−14 mV | 8.5e−06 mV |
| AdEx | I=700 pA | 4 / 4 | yes (both) | 9.4e−13 mV | 9.5e−04 mV |
| AdEx | I=1000 pA | 14 / 14 | yes (both) | 6.8e−13 mV | 1.4e−04 mV |

| Subset | Case | max|Δ| (quantity) |
|---|---|---|
| COBA clamp | AMPA τ=5, E=0 | g 1.1e−16 nS · I 2.8e−14 pA |
| COBA clamp | GABA τ=10, E=−75 | g 4.4e−16 nS · I 1.4e−14 pA |
| Pair-STDP | worst over 24 (Δt,w₀) pairs | |Δw| 4.8e−08 |

Reading these: in **float64** every neuron trajectory matches to ~1e−11 mV or better across the FULL run (including spikes and resets) — the two integration schemes are algorithmically identical. In **production float32** the sub-threshold match is ≤0.05 mV (the ~0.045 mV worst case is a single-step transient at a fast spike upstroke, not a drift), and — the strongest result — **spike-step indices are bit-exact even at float32** in every case. The COBA conductance/current and the STDP weight-change curve match to machine / float32 epsilon. <!--derived--> (0.045 = rounded read of the float32 I=100 max|Δv| in parity_metrics.json)

## Mechanism differences reconciled (these are themselves the useful finding)

- **Integration scheme.** The engine uses single-step forward Euler at dt (`v += dt·f(v)`) for BOTH Izhikevich-2007 and AdEx. Brian2 was set to `method='euler'` (forward Euler) at the same `dt=1 ms` so the schemes coincide exactly; the COBA linear decay used Brian2 `method='exact'` (analytic exponential), which equals the engine's discrete `g·exp(−dt/τ)` at grid points. Choosing the wrong Brian2 method (e.g. 'exact' for the nonlinear neurons, or 'euler' for the conductance) would introduce a real, non-bug discrepancy — the method choice is part of faithfully reproducing the engine.
- **AdEx exp-arg clip.** The engine clips `(V−V_T)/Δ_T` to [−20, 5] before `exp` (overflow guard). This had to be replicated verbatim in the Brian2 equation (`clip((v-VT)/DT,-20,5)`); omitting it diverges near threshold where the exponential runs away.
- **Reset semantics.** Engine reset is `v←c; u+=d` (Izh) / `v←V_r; w+=b` (AdEx), applied when `v_new≥vpeak` AFTER the Euler step. Brian2 `threshold`/`reset` fire in the same order, and its `StateMonitor(when='end')` aligns to the harness's post-step record at **offset 0** (verified — no ±1 sample shift).
- **Refractory.** Production `refractory_period_steps=2` blocks spike DETECTION only (voltage keeps integrating; `bridge.py:7577,7595`). It is NON-BINDING at these firing regimes (min ISI ≈ 9 steps ≫ 2), so it was disabled to isolate the integrator; the traces/spikes are identical to production. This is a genuine semantic note: our refractory is a spike-detection mask, NOT a voltage clamp, unlike Brian2's default refractory.
- **Precision.** Production is float32; Brian2 is float64. The float64-vs-float64 comparison shows the residual is pure float32 quantization (≤~0.05 mV, spike timing unaffected), not an algorithmic difference.
- **STDP pairing.** The engine's rule is a soft-bounded, last-spike-time (nearest-neighbour) pair rule keyed on Δt=t_post−t_pre. An independent Brian2 event-driven **trace** synapse (apre/apost decaying at τ±) with the SAME soft-bound resets reproduces the exact curve for an isolated pair — two completely different codepaths (direct exp vs decaying traces) agreeing to 4.8e−08 confirms both the timing dependence and the multiplicative bound.

## Honest scope / caveats

- Validates the VANILLA point-neuron core (single neuron + isolated synapse + isolated pair rule), NOT network-scale integration, the sparse matvec propagation path, or any custom mechanism. A full closed-loop post-synaptic-potential match would additionally have to reconcile the 1-step synaptic-delivery ordering (the bridge applies the conductance increment for the NEXT step); the voltage-clamp PSC comparison used here avoids that ambiguity while still cross-checking the exact COBA kernel.
- **Environment.** brian2 2.9.0 (latest) references `np.ndarray.ptp`, removed in numpy 2.0, so it cannot import under the repo's pinned numpy 2.4.x. The oracle was RUN and PASSED (10/10) in a dedicated numpy 1.26.4 venv; in the repo's numpy-2.x venv the test SKIPS cleanly (module-level guard) rather than failing CI. See `requirements-dev.txt`.
- No discrepancy was found. Had the engine and Brian2 disagreed on something they should agree on, that would be a first-class potential-bug finding; this run found none — the vanilla core is corroborated by an independent simulator.

## How to run

```
python3.11 -m venv .venv-brian2 && . .venv-brian2/bin/activate
pip install "numpy==1.26.4" "brian2==2.9.0" pytest
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -m pytest tests/test_brian2_oracle.py -v
```
