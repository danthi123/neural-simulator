# gap#4 on-bridge spiking — SURPASS attempt REMOVES the compute confound → POWERED NO-GO: even the idealized ceiling doesn't learn; root cause = φ'-vanishing credit + a TONIC-PINNED FROZEN hidden representation at the sparse point-neuron operating point (2026-07-24)

## Why this supersedes the launch-bound framing
The banked gap#4 on-bridge NO-GO (`2026-07-24-gap4-onbridge-spiking-6seed-nothing-learns-LAUNCH-BOUND-compute-wall.md`,
936bce6e) said "nothing learns, launch-bound compute wall." Per THE LAW the surpass was TAKEN (not deferred): shrink the
task to a feasible scale so the compute confound is gone, then run the decisive diagnostic. **Result: the compute wall
was a confound — the real wall is the SPARSE POINT-NEURON OPERATING POINT, and it is a POWERED NO-GO (fails even under
the true gradient), with a precise substrate-read root cause and two named next levers.**

## What was built (runner-only, additive, NO `sim/` edit; construct-smoke passes)
`research/runners/_gap4_onbridge_spiking_selfpredict_derisk.py` (uncommitted additions, for review):
- **Vectorized `_read_ff_logical` / `_sync_transport`** — replaced a per-edge Python loop with a boolean-mask + fancy-index
  reconstruction, **~4× faster** for the transport_ceiling arm, byte-identical output. This alone dissolves most of the
  "launch-bound" cost for the ceiling arm → the compute-wall confound is removed.
- **`--bdsp-w-max`** (default 6.0 = parent) widens the BDSP FF clamp (kernel reads `cfg.bdsp_w_max` dynamically,
  bridge.py:8009); confirmed weights move freely past ±6 at ±30.
- `ff_weight_moved` + `train_acc` per-arm learning diagnostics.

## The task shrink (valid + depth-separating)
`n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16` → **k=5 classes** (from 9), n_train=720, chance 0.333.
Rate oracle 0.93–1.0, 1-layer 0.44 → **depth gap +0.556** (a real depth task). NB: `n_super=8` is too small — the oracle
collapses to chance (too few taught supers/class to generalize) → the task can't shrink below ~12 supers.

## The decisive diagnostic — the ceiling does NOT learn (read from the substrate)
Across depth 2 & 3, parent-strength drive, wide clamp (±30), per-layer credit normalization, pool_k up to 16, credit
windows up to 100 steps, up to 40 epochs — **the transport-ceiling (idealized weight-transport = the TRUE gradient)
oscillates at/below chance.** Root cause, measured directly:
1. **Vanishing credit in depth.** The descending credit is scaled by φ'(E)=E(1−E) each hop; the sparse operating point
   gives E≈0.04 → φ'≈0.04 ≪ 0.25. Injected apical per layer: **out 15.7 → H2 1.97 → H1 0.01 (~1600× attenuation)**; the
   bottom hidden layer's burst probability stayed exactly at baseline (`P=0.300`, dev=0) → **zero credit reaches it**.
2. **Normalization fixes the vanishing but does NOT unlock learning.** Rescaling each layer's credit to the output-error
   magnitude (≈ dendritic/homeostatic gain control) restored H1's credit — the net still oscillated at chance.
3. **Not finite-spike noise either.** At 100-step windows × pool 16 (~80 events/pool/window, good SNR), the **hidden
   linear-probe stayed EXACTLY at random-init decodability (0.344/0.333) at ep1/ep5/ep10** — the hidden representation is
   **FROZEN**: the forward hidden E is tonic-dominated and insensitive to the weight changes credit induces (weights drift
   ~2.5% but the representation doesn't move). No discriminative features form → the compressed spiking readout (all
   output E≈0.04) can't classify.

⇒ **The wall is the sparse-firing point-neuron operating point:** to keep layers non-silent it needs a tonic floor, but
that floor PINS the hidden firing rate, and the φ'-gated spiking BDSP credit — **even the true gradient** — cannot induce
structured hidden features against it. It fails under weight transport ⇒ **NOT merely a feedback-direction problem** —
it is credit efficacy + forward representability at this operating point. This confirms + sharpens the multiply-documented
gap#4 boundary (2026-07-14 credit-STRUCTURE 0/6; 2026-07-18 credit-DIRECTION; 2026-07-20 shared-readout).

## Verdict (per THE LAW — METHOD banked, CAPABILITY open, next levers named)
- **METHOD (make the current spiking BDSP pipeline learn via shrink + wide-clamp + more-epochs): POWERED NO-GO** — now
  with a precise root cause, not a compute confound. The learned-vs-fixed comparison **cannot run** (the upper bound is
  at chance, so there is no learnable regime to compare in).
- **The learned-vs-fixed deep-credit capability is a RATE GO already** (`56c90d67`); the SPIKING realization is blocked by
  the operating point, not the mechanism.
- **NEXT LEVERS (each a distinct arc; (a) is cheapest-first and being taken now):**
  - **(a) Rebalance tonic-vs-input drive** so the forward hidden is INPUT-driven, not tonic-pinned — the exact-invariant
    hidden linear-probe is the fingerprint to fix. Most spike-faithful; a drive-balance tuning, not a new mechanism.
  - **(b) Graded-state escape** — read/credit via the graded soma potential (analog/dendritic, point-neuron-limit-
    endorsed) or DECOLLE-style per-layer LOCAL readouts to bypass the multi-hop φ' vanishing. Less spike-faithful; a new
    mechanism class.

Scratchpad probes (learning curves, per-layer credit reads, linear-probes) under `scratchpad/`. This is a taken-surpass
outcome (compute confound removed, decisive diagnosis), NOT a deferral; lever (a) is the immediate continuation.
