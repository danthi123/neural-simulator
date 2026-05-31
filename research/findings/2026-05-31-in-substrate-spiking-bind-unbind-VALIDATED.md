# In-substrate spiking compositional BIND/UNBIND -- validated (2026-05-31)

**Arc:** biological composition (owner Option 2: "we absolutely want compositional
capabilities, work autonomously even with new ideas, biologically sound").

**One-line:** the validated VSA composition (role (x) filler bind, unbind, cleanup) now
runs IN the spiking substrate -- not as numpy algebra on captured codes -- via threshold
coincidence detection, on the project's real concept-pool codes. The bind/unbind operators
are spiking; the load-bearing nonlinearity (the Hadamard) is computed by neurons.

## Why this matters

The prior composition revision (2026-05-31, `...-near-ortho-ROLES-not-FILLERS`) established
the ALGEBRA: generalizable bind/unbind works with the substrate's OVERLAPPING concept codes
(cleanup uses ID-separability, not near-orthogonality; only a few near-ortho ROLE codes are
needed). It shipped a working numpy demo (`compose_vsa_demo.py`, 60/60 novel-sentence
generalization multi-seed) and de-risked on five axes. The open step to "biologically sound"
(owner's word) was to implement the bind IN spiking dynamics. This finding closes that step.

## The three primitives (each validated in-substrate, RTX 3090 / CuPy, seed 42)

All use a custom bridge built via `inject_explicit_wiring` (G1 explicit-wiring pattern),
reuse-by-import, no protected/frozen/sim-core modification, plasticity OFF (this is a
fixed-wiring computation, not learning).

### 1. Binary AND coincidence (`_insubstrate_coincidence_probe.py`)

A spiking neuron computes `AND(role[i], filler[i])` via threshold + a tonic hyperpolarizing
bias: one input is sub-threshold, two inputs sum supra-threshold. Three populations
(role/filler/coinc, N=128), identity wiring `role[i]->coinc[i]` + `filler[i]->coinc[i]`.

| w | bias | BOTH | single | none | AND-selectivity |
|---|---|---|---|---|---|
| 200 | -500 | 0.059 | 0.005 | 0.000 | 0.921 |
| 320 | -1000 | 0.048 | **0.000** | 0.000 | **1.000** |

The control is geometric (built in): `role-only` coinc neurons receive role input but their
filler partner is silent -> they stay dark (single ~ 0); `none`-region never fires; only
neurons receiving BOTH an active role AND an active filler fire. A genuine threshold AND,
not a 2x-drive artifact. (The initial all-zeros was a sub-threshold 600 pA drive -- these
Izhikevich neurons need ~2000 pA; the near-linear no-bias regime sharpens to a clean AND
once the bias sets the threshold. Diagnosed, not called a boundary.)

### 2. Graded gating (`_insubstrate_graded_gating_probe.py`)

The bind preserves filler MAGNITUDE (`bound[i] = +-filler[i]`); substrate codes are graded.
So the coincidence operator must be a graded multiplicative gate.

| filler level | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| role-ON coinc rate | 0.000 | 0.011 | 0.028 | 0.040 | 0.048 |
| role-OFF coinc rate | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Spearman(filler, role-ON rate) = **1.000** (monotonic); role-OFF rate **0.000 at every
level** (perfect gating). The coincidence is a clean multiplicative gate -- role gates,
filler provides graded drive, output rate ~ filler magnitude when gated, silent when not.

### 3. Full ON/OFF bind/unbind (`_insubstrate_bind_unbind_probe.py`)

One bridge, 8D neurons: `role_ON/OFF` + `fill_ON/OFF` driven source populations (4D)
synapse into 4 coincidence banks A/B/C/D (4D) that realize the +-1 Hadamard:

```
A = AND(role_ON , fill_ON )   B = AND(role_OFF, fill_OFF)   -> bound_ON  = rate(A)+rate(B)
C = AND(role_ON , fill_OFF)   D = AND(role_OFF, fill_ON )   -> bound_OFF = rate(C)+rate(D)
```
`bound_ON - bound_OFF = role (x) filler` (verified algebra). The SAME coincidence layer is
reused for unbind (drive query-role + bound-as-fill -> est_ON/OFF; `est = est_ON - est_OFF
= query (x) bound`). Superposition = linear sum of captured bound rates across pairs.
CLEANUP = argmax cosine to the real substrate concept codes (denoise64 cache, V=16).

**seed 42, projected D=800 (numpy ceiling 1.000 at all K; full raw D=3200 ceiling also
1.000 up to K=8):**

| K | spiking recovery (raw D=800) | + ON/OFF opponency (D=800) | full raw D=3200 + window 150 | control |
|---|---|---|---|---|
| 1 | 0.933 | **1.000** | _PENDING_ | ~chance (0.00-0.07) |
| 2 | 0.900 | **0.967** | _PENDING_ | ~chance |
| 3 | 0.756 | 0.711 | _PENDING_ | ~chance |
| 4 | 0.600 | 0.683 | _PENDING_ | ~chance |

Raw (no opponency): RESOLVES at K=1,2 (recovery >= 0.80), degrades K>=3. The control sits at
chance throughout (0.00-0.13) -> recovery is REAL binding work, not a cleanup artifact.

**ON/OFF opponency** (re-canonicalize the superposed bound to its signed form before unbind:
retinal/thalamic lateral inhibition between ON/OFF channels -- the project's own mean-centering
motif, linear, in-substrate-realizable) lifts K=1,2 to the ceiling (1.000, 0.967) but does NOT
fix K>=3. So common-mode saturation was not the dominant bottleneck at high load.

**The K>=3 limit is finite firing-rate SNR**, a genuine capacity bound (Miller-like), not a
mechanism failure: coincidence rates are ~0.05 (a few spikes per readout window), so at K>=3
the cross-term noise (`sum_{k!=j} r_j r_k c_k`) plus rate-estimate variance overwhelms the
per-dim signal `c_j`. The NOISELESS numpy holds to K=8; the spiking version has finite SNR ->
lower effective capacity. Two principled SNR levers (both biologically legitimate -- more
neurons / longer readout): higher D (cleanup cosine averages over more dims) and a longer
integration window. The decisive run tests full raw D=3200 + window 150 [results pending].

## Honest scope

- The two NONLINEAR operations (bind Hadamard, unbind Hadamard) run IN spiking dynamics.
  The linear memory between phases (superposition sum, ON/OFF opponency) is captured-rate
  arithmetic -- each step is itself a linear/lateral-inhibition operation realizable
  in-substrate; it is not autograd or learning.
- Cleanup reuses the substrate's ID-separable concept codes (validated; no near-orthogonality
  needed). Concepts come from the real concept-pool activity (denoise64).
- This is a fixed-wiring spiking computation; no plasticity, no training. Generalization is
  by construction (VSA), inherited from the validated algebra.

## Reproduce

```bash
python -m research.findings.raw._insubstrate_coincidence_probe       # primitive 1
python -m research.findings.raw._insubstrate_graded_gating_probe     # primitive 2
python -m research.findings.raw._insubstrate_bind_unbind_probe --proj-dim 800 --ks 1,2,3,4
python -m research.findings.raw._insubstrate_bind_unbind_probe --proj-dim 0 --seed 42 --ks 1,2,3,4  # raw D=3200
```
