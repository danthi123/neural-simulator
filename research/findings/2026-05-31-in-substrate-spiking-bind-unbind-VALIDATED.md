# In-substrate spiking compositional BIND/UNBIND -- validated (2026-05-31)

**Arc:** biological composition (owner Option 2: "we absolutely want compositional
capabilities, work autonomously even with new ideas, biologically sound").

**One-line:** the validated VSA composition (role (x) filler bind, unbind, cleanup) now
runs IN the spiking substrate -- not as numpy algebra on captured codes -- via threshold
coincidence detection, on the project's real concept-pool codes. The bind/unbind operators
are spiking; the load-bearing nonlinearity (the Hadamard) is computed by neurons.

## VERDICT: RESOLVES, multi-seed (42,43,44) -- capacity K=4 to K=6, set by coincidence firing rate

| K | seed 42 | seed 43 | seed 44 | mean (bias=-1000, clean AND) |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | **1.000** |
| 2 | 1.000 | 1.000 | 1.000 | **1.000** |
| 3 | 0.978 | 0.911 | 0.978 | **0.956** |
| 4 | 0.833 | 0.833 | 0.917 | **0.861** |

Full raw D=3200 (no projection), readout window 150. Every seed clears the frozen 0.80 bar at
every K=1..4 at the clean-AND operating point. A HIGHER firing-rate operating point (bias=-500,
~2x coincidence rate = more dynamic range) extends this to **K=6 multi-seed** (mean 0.975/0.933/
0.856 at K=4/5/6, every seed >= 0.80 -- the Miller 7+-2 range; full table in the capacity ladder
below). numpy ceiling 1.000 all K. Control is faithful to the algebra's overlapping-code
cleanup-bias floor (not a spiking artifact -- see below); recovery-vs-control gap decisive
everywhere (+0.67 to +0.93). This is the owner's "biologically sound" composition, realized IN
spiking dynamics, validated across three seeds. Honest capacity correction (below): the readout
WINDOW does not extend capacity (that earlier expectation was falsified); the coincidence FIRING
RATE does, to K=6.

**Adversarial review: CLEAR.** A dedicated skeptical reviewer (instructed to falsify) ran all 7
exploit classes against the load-bearing probe and ruled out each, citing line numbers + an
independent numpy re-derivation + a single-Izhikevich-neuron simulation of the operating point:
(1) no answer leakage -- `est` is driven only by the spiking bind output `bound_*` rates, the
original concept re-enters only as the legitimate cleanup codebook; (2) the bind AND unbind
Hadamard are genuinely spiking (coincidence-bank `cp_firing_states` through the real synaptic-
propagation + Izhikevich-threshold pipeline) -- the only numpy steps are the two LINEAR ops
(superposition sum, ON/OFF opponency) the scope honestly discloses; (3) control valid (same
bound, only query role changes; `_wrong_role` excludes the true role); (4) recovery non-trivial
(numpy control << 1.0; K4<K1; role matters under identical normalization); (5) seeds are
genuinely independent substrate realizations (`||r42-r43||/||r42|| = 1.02`) with re-drawn roles;
(6) operating point a real supra-linear AND (0/1/2 sources -> 0.000/0.013/0.060; the -1000 pA
bias fires nothing alone); (7) bridge genuinely steps + reads firing, plasticity OFF. Bookkeeping
note from the review: the per-run numpy CONTROL column is a stochastic estimate of the
cleanup-bias floor (varies ~0.1-0.5 with RNG/trial count across runs); the spiking RECOVERY
numbers are stable and match exactly across the standalone and multi-seed runs. The multi-seed
table above is canonical.

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
it needs a longer window than the ideal model, but the trend held from window 60 to 150:
D=3200 + window 150 lifts K3,4 over the bar. Multi-seed (42,43,44) confirmed RESOLVES at all
K=1..4 (table at top); per-seed K=4 = 0.833 / 0.833 / 0.917.

### Capacity ladder: the window lever PLATEAUS by ~K=4 (honest correction)

A follow-up window-300 run (seed 42, K=4..8) tested whether a longer readout extends capacity
toward Miller 7. It does NOT:

| K | spiking recovery, window 150 | spiking recovery, window 300 |
|---|---|---|
| 4 | 0.833 | 0.850 |
| 5 | -- | 0.760 |
| 6 | -- | 0.600 |
| 7 | -- | 0.500 |
| 8 | -- | 0.438 |

Doubling the window 150 -> 300 barely moved K=4 (0.833 -> 0.850) and K=5,6,7 stay below the
bar. So the earlier "extends with a longer readout -> Miller 7" expectation (from the CPU Poisson
model) is FALSIFIED for the window lever: window 60 -> 150 helped a lot (K4 0.60 -> 0.83), but
150 -> 300 plateaus. The CPU model overestimated because it modeled ONLY Poisson spike-count
noise (which a longer window removes); the GPU has a WINDOW-INDEPENDENT bottleneck at K>=5 --
most likely the coincidence rate-resolution / dynamic range (rates live in [0, ~0.05], a coarse
graded scale, so at high K the many small bound components are under-resolved) and/or cross-term
interference, neither of which a longer readout fixes. Honest capacity at the validated operating
point (w=320, bias=-1000): ~K=4.

### The CAPACITY LEVER is firing rate, not window (seed 42 reaches K=6)

Testing the firing-rate lever directly: a less-negative tonic bias (-500 vs -1000) roughly
DOUBLES the coincidence firing rate (BOTH=0.097 vs 0.048 -> more dynamic range), at the cost
of some single-input leakage (single=0.040). seed 42, window 150:

| K | bias -1000 (window 300) | bias -500 (window 150) |
|---|---|---|
| 4 | 0.850 | 0.950 |
| 5 | 0.760 | 0.900 |
| 6 | 0.600 | 0.817 |

The higher firing rate extends capacity to **K=6** (all clear the 0.80 bar; the Miller 7+-2
range), where the longer window did not. This CONFIRMS the window-300 diagnosis: the K>=5
bottleneck was the coincidence rate-RESOLUTION (dynamic range), which more firing rate addresses
and a longer readout does not. The control did NOT degrade despite the leakage (it dropped to
~0.00-0.025; recovery-vs-control gap +0.82..+0.93) -- recovery stays clean.

**Multi-seed (42,43,44) at bias=-500 confirms K=6:**

| K | seed 42 | seed 43 | seed 44 | mean spiking recovery |
|---|---|---|---|---|
| 4 | 0.950 | 0.975 | 1.000 | **0.975** |
| 5 | 0.900 | 0.960 | 0.940 | **0.933** |
| 6 | 0.817 | 0.933 | 0.817 | **0.856** |

All three seeds clear 0.80 at every K=4,5,6; control near zero throughout. So the complete,
multi-seed-validated capacity story: **capacity is set by the coincidence FIRING RATE** -- K=4
at the clean-AND operating point (bias=-1000, single-input leakage 0.000), K=6 at the higher-rate
point (bias=-500, ~2x rate, leakage 0.040 that does NOT accumulate to hurt), all three seeds.
The window lever does not extend it; the firing-rate (dynamic-range) lever does, to K=6 (the
Miller 7+-2 range). This is a clean speed/precision tradeoff in the substrate: a higher
coincidence rate trades a touch of AND-selectivity for the dynamic range that holds more
superposed bindings.

## What this is and is not (honest scope)

IS: the two NONLINEAR composition operators (bind Hadamard, unbind Hadamard) computed by
spiking coincidence neurons, multi-seed, on the project's real overlapping concept codes,
recovering bound role-filler pairs to K=4 (a subject/verb/object + manner frame). Generalizes
by construction (VSA) -- any novel (role, filler) combination works with no training.

IS NOT: (a) a learned parser -- roles and concept drives are supplied; the system does not yet
infer role-filler structure from raw input (that is a downstream learning arc). (b) unlimited
capacity -- K is firing-rate/readout-window bounded (~4 at window 150; extends with a longer
readout, the biological speed-accuracy tradeoff). (c) fully end-to-end spiking storage -- the
linear memory between bind and unbind (superposition sum, ON/OFF opponency) is captured-rate
arithmetic, each step itself a linear/lateral-inhibition operation realizable in-substrate; it
is not autograd or learning.

## Next (future arcs, not this finding)

1. Capacity scaling: the WINDOW lever is exhausted by ~K=4 (above); the remaining lever is a
   higher firing-rate operating point (more dynamic range) -- under test. If that also plateaus,
   ~K=4 is the operating-point capacity and reaching Miller 7 would need a different code (e.g.
   higher per-dim rate resolution or a sparser bound representation). 2. Wire the bind layer to
   the existing concept POOLS (concepts already are pools) for a fully in-network path.
   3. Learned role-filler parsing (infer the bindings from input) -- the bridge from this
   fixed-wiring primitive to used-in-conversation composition.

## Honest scope

- The two NONLINEAR operations (bind Hadamard, unbind Hadamard) run IN spiking dynamics.
  The linear memory between phases (superposition sum, ON/OFF opponency) is captured-rate
  arithmetic -- each step is itself a linear/lateral-inhibition operation realizable
  in-substrate; it is not autograd or learning.
- Cleanup reuses the substrate's ID-separable concept codes (validated; no near-orthogonality
  needed). Concepts come from the real concept-pool activity (denoise64).
- This is a fixed-wiring spiking computation; no plasticity, no training. Generalization is
  by construction (VSA), inherited from the validated algebra.

## Downstream capability: a queryable relational FACT-MEMORY in the spiking substrate

The validated bind composes into the first conversational primitive: a small subject/verb/object
KNOWLEDGE BASE you can query relationally, all in spiking dynamics. A fact "dog chases cat" =
agent (x) dog + action (x) chase + patient (x) cat (3 spiking-coincidence bindings, K=3). Facts
are stored SEPARATELY (a numpy cheap-first confirmed this is the correct architecture: separate-fact
storage + cue-based retrieval = 1.000 multi-seed, while SUPERPOSING facts into one vector degrades
to 0.475 -- the multi-hop wall). A relational query "what does dog chase?" = spiking-unbind the
agent of each stored fact + cleanup to match the cue, then spiking-unbind the patient of the matched
fact.

Multi-seed (42,43,44), D=800, 2-fact KB, at the higher-rate operating point (bias=-500):
**single-fact 1.000, relational (find-by-agent -> read-patient) 1.000, control (no false match on an
absent cue) 1.000 -- all three seeds.** (At the clean-AND bias=-1000 point it is 2/3, seed 44 dipping
to 0.750; the higher firing rate -- the same lever that extends bind capacity to K=6 -- gives the
dynamic range for robust relational memory.) The KB SCALES with no degradation: a numpy cheap-first
holds relational query at 1.000 up to N=12 facts (vocab-limited at 16 distinct agents, all 3 seeds),
and the spiking version holds 1.000 at N=5 facts (seed 42, bias=-500) -- because separate-fact
storage means each fact is an INDEPENDENT K=3 bind, so adding facts adds no superposition
interference. Owner-facing demo:
`research/runners/compose_relational_memory_demo.py` stores "dog go north" + "cat come south" and
answers "what does dog have as object?" -> north, action -> go, with the absent-cue control returning
"(no fact found)". Honest scope: roles (agent/action/patient) are SUPPLIED, not parsed from raw input
(a learned role parser is the next arc); this is structured fact-memory with cue-based retrieval, not
open-ended relational reasoning over superposition (that degrades).

**Architectural finding -- separate storage is the universal structure mechanism (flat nesting
fails).** A cheap-first tested FLAT NESTED composition: encode "big dog goes north" as
agent (x) (modifier (x) big + noun (x) dog) + action (x) go + patient (x) north (a phrase as a
filler in one vector), then descend -- unbind agent to get the phrase, then unbind noun/modifier.
Result (seeds 42/43/44): outer single-level recovery 1.000, but DEPTH-2 descent collapses to chance
(noun 0.025-0.050, modifier 0.050-0.100; chance 0.062). Unbinding the outer role leaves the phrase
buried under full-magnitude cross-terms, so the depth-2 signal is ~1-of-5 terms and cleanup cannot
find it -- the same superposition/multi-hop wall. So the bind is a FLAT slot-filler (one level);
hierarchical structure (nesting, modification) must use SEPARATE storage + cue retrieval (the
validated relational-memory pattern: store "big dog" as its own {head: dog, modifier: big} fact,
reference dog as the agent, recover the modifier by cue), NOT flat nested binding. Separate-fact
storage is thus the universal architecture for structure in this substrate -- for multi-fact AND for
hierarchy -- because flat superposition/nesting hits the SNR wall. (Biologically sensible: distinct
structured items as distinct ensembles, bound by association, not all summed into one vector.)

## Reproduce

```bash
python -m research.findings.raw._insubstrate_coincidence_probe       # primitive 1
python -m research.findings.raw._insubstrate_graded_gating_probe     # primitive 2
python -m research.findings.raw._insubstrate_bind_unbind_probe --proj-dim 800 --ks 1,2,3,4
python -m research.findings.raw._insubstrate_bind_unbind_probe --proj-dim 0 --seed 42 --ks 1,2,3,4  # raw D=3200
python -m research.findings.raw._insubstrate_bind_unbind_probe --seed 42 --ks 4,5,6 --coinc-bias -500  # K=6 (firing-rate lever)
python -m research.findings.raw._vsa_relational_query_probe                          # cheap-first: relational fact-memory
python -m research.findings.raw._insubstrate_relational_memory_probe --coinc-bias -500  # spiking relational fact-memory
python -m research.runners.compose_spiking_bind_demo                                  # demo: bind subject/verb/object
python -m research.runners.compose_relational_memory_demo                             # demo: queryable SVO knowledge base
```
