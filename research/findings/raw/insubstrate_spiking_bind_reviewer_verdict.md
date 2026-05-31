# Adversarial review verdict: in-substrate spiking compositional bind/unbind

**Date:** 2026-05-31
**Reviewer mandate:** falsify the claim "spiking bind/unbind RESOLVES multi-seed (42,43,44) to
K=4 on real substrate concept codes; Hadamard computed by spiking coincidence neurons."
**Verdict: CLEAR** -- the claim survives scrutiny. All 7 exploit classes ruled out by code +
independent numpy re-derivation + a single-Izhikevich-neuron simulation of the operating point.

## Exploit-class results (each with cited evidence)

1. **Answer leakage -- CLEAN.** `est` (cleanup input) is driven exclusively by the spiking bind
   output `bound_*` rates (run_spiking lines 183-199); the original `codes[fi[k]]` re-enters only
   as the legitimate cleanup codebook `argmax(codes @ est)` (lines 200, 203). No bypass path.
2. **Is the nonlinearity spiking? -- CLEAN.** The only `role*concept` / `S*role` multiplies are
   inside `numpy_reference` (the labeled ceiling). `run_spiking` has no Hadamard multiply; the
   bind/unbind product is coincidence-bank `cp_firing_states` through the real synaptic-propagation
   (`effective_connections_matrix.T @ prev_fired`) + Izhikevich-threshold pipeline. The two numpy
   steps (superposition sum, ON/OFF opponency) are LINEAR and honestly disclosed.
3. **Control validity -- CLEAN.** `_wrong_role` excludes the true role; control reuses the same
   `bound`, changes only the query role. Elevation reproduced in a noiseless overlapping-code
   surrogate (0.187/0.105/0.071/0.052 at K=1-4, decreasing with K) -- a genuine algebra property,
   not rigged or trivial.
4. **Cleanup triviality -- CLEAN.** numpy control << 1.0; recovery degrades K4<K1; an ideal
   surrogate with the identical `_scale_to_current` normalization shows wrong-role control FAILS
   (0.05-0.19) while correct-role recovers (1.0) -- the binding genuinely carries identity.
5. **Seed independence -- CLEAN.** Three distinct caches; `||r42-r43||/||r42|| = 1.02`
   (uncorrelated); per-seed between-cos 0.699/0.656/0.667; roles re-drawn per seed.
6. **Operating point / 0.000 role-OFF -- CLEAN.** Single-Izhikevich sim (C=100,k=0.7,vr=-60,
   vt=-40,vpeak=35, bias=-1000, w=320): 0/1/2 sources -> 0.0000/0.013/0.060. Genuine supra-linear
   threshold AND; tonic bias fires nothing alone; role-OFF=0.000 is legitimate (sharper, if
   anything, than the surrogate estimate).
7. **Other "not spiking" risks -- CLEAN.** Bridge genuinely steps (`_run_one_simulation_step` x
   (RESET+RUN)); firing genuinely read; plasticity OFF (fixed-wiring computation); cupy backend,
   25600 neurons in the multiseed log; reviewed probe committed with no uncommitted divergence.

## Disclosed caveats (do NOT invalidate; finding states them)

- Inter-phase memory (superposition + opponency) is linear captured-rate arithmetic, not in-network
  spiking storage.
- K is firing-rate/readout-window bounded (~4 at window 150) -- a capacity, not a mechanism ceiling.
- Roles + concept drives are supplied, not parsed from input (no learned parser).

## Bookkeeping note (non-fatal)

The per-run numpy CONTROL column is a stochastic estimate of the cleanup-bias floor (varies
~0.1-0.5 with RNG/trial count); the spiking RECOVERY numbers are stable and match across the
standalone + multi-seed runs. The multi-seed table is canonical.
