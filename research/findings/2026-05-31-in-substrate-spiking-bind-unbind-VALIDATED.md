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

**Decisive seed 42, full raw D=3200 (no projection), readout window 150 steps:**

| K | numpy recovery | numpy control | **spiking recovery** | spiking control | recovery-vs-control gap |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.200 | **1.000** | 0.267 | +0.733 |
| 2 | 1.000 | 0.113 | **1.000** | 0.233 | +0.767 |
| 3 | 1.000 | 0.150 | **0.978** | 0.067 | +0.911 |
| 4 | 1.000 | 0.075 | **0.833** | 0.100 | +0.733 |

**RESOLVES to K=4.** Spiking recovery clears 0.80 at every K=1..4 (1.000, 1.000, 0.978,
0.833), and the recovery-vs-control gap is decisive everywhere (+0.73 to +0.91). The two
principled SNR levers got there: higher D (cleanup cosine averages over more dims) and a
longer readout window (more spikes per dim) -- both biologically legitimate (more neurons /
longer integration = the speed-accuracy tradeoff).

### The control floor is the overlapping-code cleanup-bias, NOT a spiking artifact

Scrutinizing the PASS: the wrong-query control is elevated at low K (spiking 0.27/0.23).
This is FAITHFUL to the algebra, not a spiking failure -- the noiseless numpy reference has
the SAME elevation (0.20/0.11 at K=1,2; chance 1/16=0.062), because the substrate codes are
highly overlapping (between-cos mean **0.699**, a large shared component). Unbinding with a
wrong role gives a sign-scrambled `(w (x) r) (x) c` that, with such overlapping codes, lands
on the target ~15-20% of the time in the algebra itself -- the documented cleanup-bias
(decreasing with K). So `control == 1/V` is unachievable with overlapping fillers; the
correct criterion is FAITHFULNESS (spiking control ~ numpy control, no EXTRA spiking failure)
plus a decisive recovery-vs-control gap -- both hold. The spiking faithfully reproduces BOTH
the algebra's perfect recovery AND its overlapping-code cleanup-bias.

### Path to K=4 (how the two earlier configs led here)

| config | K1 | K2 | K3 | K4 |
|---|---|---|---|---|
| D=800, window 60, raw | 0.933 | 0.900 | 0.756 | 0.600 |
| D=800, window 60, +opponency | 1.000 | 0.967 | 0.711 | 0.683 |
| **D=3200, window 150, +opponency** | **1.000** | **1.000** | **0.978** | **0.833** |

A CPU Poisson two-stage capacity model localized the K>=3 falloff to readout-window /
spike-count SNR (window 60 ~ 3 spikes/dim: K4=0.89, K6=0.78; window 150 ~ 7 spikes/dim:
K4=1.00 ideal). The GPU has extra noise (source-neuron stochasticity, threshold jitter) so
it needs a longer window than the ideal model, but the trend held: D=3200 + window 150 lifts
K3,4 over the bar. The capacity is firing-rate/window-bounded and extends with a longer
readout -- not a mechanism ceiling. [Multi-seed 43,44 confirmation in flight.]

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
