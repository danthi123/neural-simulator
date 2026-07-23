# gap#5 replay-transition: R0 PROVES the numpy GO rode start=A (symmetric store); hetero-dep + encode-knobs NEGATIVE → the fix is phase-precession CAUSAL-STDP encoding (2026-07-23)

## The reframe, now PROVEN (candidate #3 research gate → R0)
The candidate-#3 research gate reframed the whole replay problem: the three prior negatives (intrinsic fatigue
silences-not-directs; E→E STD destroys the discrete chain; on-spikes gamma-WTA 1/3) and the numpy-vs-spikes gap
reduce to ONE root cause — **the stored between-assembly weights are near-symmetric, and the numpy "1.000 forward"
GO rode `start=assembly-A` + self-avoidance, NOT weight asymmetry.** A noise-ignited spiking replay cannot assume it
starts at A, so it goes forward only at chance.

**R0 diagnostic (start-randomization control on the numpy isolation, `_gap5_R0_start_randomization_diag.py`):**
```
seed 42: adj_fwd=143.3 adj_rev=142.0 asym=+1.26 | start=0 fwd=1.000 | start=RANDOM fwd=0.365  => RODE-START
seed 43: adj_fwd=136.5 adj_rev=137.5 asym=-1.01 | start=0 fwd=1.000 | start=RANDOM fwd=0.412  => RODE-START
```
Decisive: with the ignition point fixed at A the readout is 1.000 forward (the banked GO); with a RANDOM ignition
point (what a noise-driven spiking replay actually has) it collapses to ~0.33-0.41 = chance (n_mem=3). The weights
are symmetric (seed 42 asym +1.26; seed 43 even reverse-signed -1.01). ⇒ **reliable forward replay needs weight
ASYMMETRY built at ENCODING, not a cleverer readout over the existing symmetric store.**

## R1 cheap-first levers — BOTH NEGATIVE (bank the methods; capability stays OPEN)
Encode-only, seed 42, measuring adj_fwd/adj_rev on the extracted between-assembly W (`_gap5_R1_hetero_encode_sweep.py`):

**(a) heterosynaptic depression (`btsp_hetero_dep`, the research's cheapest sub-lever) — NEGATIVE, inverts it:**
```
hetero=0.00: within=173.7 adj_fwd=143.3 adj_rev=142.0 asym=+1.26 ratio=1.01x   (baseline: symmetric)
hetero=0.10: within=85.8  adj_fwd=63.7  adj_rev=80.3  asym=-16.56 ratio=0.79x   (REVERSE + within crushed)
hetero=0.50: within=29.6  adj_fwd=26.3  adj_rev=33.5  asym=-7.16  ratio=0.79x
```
Hetero-dep preferentially depresses the FORWARD link (drives asym NEGATIVE) and crushes the within-attractor
(174→30). Wrong lever.

**(b) encode knobs (within_refresh, chain_fwd) — NEGATIVE, no knob dials in asymmetry:**
```
wr=8 cf=24 (baseline): within=173.7 asym=+1.26 ratio=1.01x
wr=4 cf=24:            within=149.8 asym=-4.78 ratio=0.96x   (reverse)
wr=0 cf=24:            within=5.9   asym=+1.33 ratio=1.27x   (within COLLAPSES below reactivation floor)
wr=0 cf=48:            within=5.9   asym=+1.33 ratio=1.27x   (== cf=24 -> more chain events SATURATE)
```
The within-refresh is REQUIRED for the within-attractor (removing it collapses within 174→6, the RANK-2 blocker) but
its symmetric cross-links dilute any chain asymmetry to ~0; and doubling chain_fwd changes nothing (the chain encode
saturates). The code comment's "+2.66 chain asym" is not a usable forward bias — the raw chain weights are ~6 and
near-symmetric. (cf=96 arms still running; wr0_cf24==wr0_cf48 makes the saturation conclusion already decisive.)

## Through-line (sharpened) + the prescribed next mechanism
The BTSP encode is plateau-gated, NOT spike-timing-causal, so it produces near-symmetric between-assembly weights,
and no knob on it creates a forward bias. Per the research gate, the principled fix is **phase-precession CAUSAL-STDP
encoding**: drive the assemblies in a forward-swept temporal order (A leads B leads C, ~10-20 ms offsets = theta
phase precession) with an ASYMMETRIC (Bi-Poo) STDP rule on the ca3→ca3 recurrent — pre-before-post potentiates A→B,
post-before-pre depresses B→A → **adj_fwd ≫ adj_rev by construction** (the standard mechanism by which CA3 forms the
forward-biased recurrent weights that support forward replay; Skaggs-McNaughton phase precession + causal STDP,
Sato-Yamaguchi). The sim already has `fused_stdp_weight_update` (asymmetric) + `enable_stdp`; the current encode
DISABLES it in favour of BTSP. The build re-enables asymmetric STDP on the recurrent during the sequential chain
drive (which already presents A-before-B) — a runner-side encode change, NO `sim/` edit for the de-risk.

**GO gate (the load-bearing controls, per the research gate):** adj_fwd/adj_rev ≥ 2-3× AND within preserved (≥ ~27);
then on the spiking readout — forward ≫ scramble/no-encode floor, AND the two NEW controls: **ASYM-LESION**
(symmetrize the weights → forward collapses to chance = the asymmetry is load-bearing) and **START-INVARIANCE**
(noise-ignited forward holds regardless of ignition point, unlike the numpy start=0 artifact). 6-seed.

## Banked methods (THE LAW: a verdict on the METHOD, capability stays OPEN)
intrinsic-fatigue-alone (silences); E→E STD (destroys discrete chain); heterosynaptic-depression (inverts);
within_refresh/chain_fwd knobs (no asymmetry, saturate). The static CA3 completion this rides on is CLOSED
(intrinsic dendritic bistability, 2026-07-18) and unaffected. Next: the causal-STDP encode.
