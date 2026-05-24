# Direction E theta-gamma multiplexing — ALGEBRA VALIDATED (controls decisive)

**Date:** 2026-05-24
**Verdict:** ALGEBRA_PASS_CONTROLS_DECISIVE (substrate biologization pending)
**Frozen bar:** 0.80 multi-seed (NEVER tuned)
**Pillar:** ALGEBRA-only validation; mirrors the FHRR-algebra pattern from the biologization arc

## Summary

The Lisman-Idiart theta-gamma multiplexing mechanism (project
catalog N.16; brain encodes ordered sequences via ~7 gamma slots
within each ~125 ms theta cycle; sequence-position-i items fire at
gamma slot i) tested in numpy algebra. Multi-seed (42/43/44),
300 trials/load, vocab=16, N_DIM=256, biological-precision noise
sigma=0.05 (matches the resonate-and-fire biologization probe).

**Main probe:** PERFECT 1.000 at every tested load {2, 3, 5, 7}.
The N_GAMMA=7 catalog cap is the natural sequence-length ceiling
without multi-theta multiplexing.

**Noise stress:** algebra robust to noise sigma 0.05-5.0 (100x
biological); breaks above ~5-10 depending on code sparsity.

**Adversarial review** (three controls; mirrors the FHRR-
biologization adversarial reviews):
- (A) Permutation: 0.198 == 1/LOAD chance 0.200 (decoder
  genuinely uses slot phase, not pattern recognition alone)
- (B) No-slot-windowing: 0.193 == 1/LOAD chance 0.200 (slot
  windowing IS load-bearing; without it the decoder can't
  discriminate)
- (C) High-overlap vocab (0.32 measured concept-overlap; closer
  to substrate-realistic 0.45): 1.000 robust at all 3 seeds

VERDICT: CONTROLS_DECISIVE -- the theta-gamma positional binding
is genuinely slot-discriminating AND overlap-robust at the algebra
level. Pillar-eligible (algebra-only, mirrors how FHRR-algebra was
recorded before resonate-and-fire biologization).

## Background

Overnight 2026-05-23 -> 2026-05-24, the (c) generative-replay arc
fully characterised the sequence-storage gap on the v16
concept-pool substrate: REPLAY_DOESNT_REACTIVATE -- the substrate
stores SIMULTANEOUS engrams perfectly (multitag 91.7% multi-seed
at n=100/n=101) but NOT SEQUENTIAL slot-position structure
(diagnostic n=99).

Two catalog-grounded positional-binding mechanisms could close the
gap:
- **Direction A: ec_context** (D.01+D.02+D.11) -- SPATIAL code
  (200-neuron sparse per-position drive; engram captures word +
  position co-firing). Currently IN FLIGHT (~3 hr GPU).
- **Direction E: theta-gamma multiplexing** (N.16) -- TEMPORAL
  phase code (each gamma slot = a sequence position; items at
  position i fire at gamma slot i). Cheap-first algebra probe
  before substrate investment.

Per autonomous discipline (falsify-cheaply-first): the algebra
probe ran first (~40 s total wall across main + stress + review).
Algebra clears the frozen 0.80 bar decisively; substrate
biologization is justified.

## Mechanism (numpy reference)

```python
# Per concept: sparse activation pattern of N_DIM neurons,
# ~ACTIVE_FRAC active during one gamma window (~17 samples).
patterns[c, t, :] = sparse_mask(N_DIM, ACTIVE_FRAC)

# Encode: place pattern_c at gamma slot i within theta cycle.
ensemble[i*GAMMA_PERIOD:(i+1)*GAMMA_PERIOD, :] += patterns[c]

# Decode slot i: cosine-match the gamma window at slot i against
# every concept's pattern at slot 0; argmax wins.
window = noisy_ensemble[i*GAMMA_PERIOD:(i+1)*GAMMA_PERIOD, :]
predicted = argmax_c cosine(window.sum(0), patterns[c].sum(0))
```

The KEY property the controls validate: the decoder reads ONLY
slot i's window (not the full theta cycle), and only the concept
encoded at slot i is "in" that window. Without slot windowing, the
decoder sees all concepts in the sequence and cannot assign them
to specific slots.

## Code + reproducibility

- Main: `research/findings/raw/direction_E_theta_gamma_numpy_probe.py`
  -> JSON `direction_E_theta_gamma_numpy_probe.json`
- Stress: `direction_E_noise_stress.py`
  -> `direction_E_noise_stress.json`
- Adversarial review: `direction_E_adversarial_review.py`
  -> `direction_E_adversarial_review.json`

Commits: `1e14548` main + stress; `794f2f8` adversarial review.
Both remotes (origin + gitea).

Reuses no project module (genuinely net-new algebra probe;
mirrors the cheap-first FHRR probe pattern). No protected/frozen/
moat module modified. No autograd. NUMPY only.

## Honest scope

This is **algebra-only** validation. The numpy reference does NOT
have the additional noise sources a spiking substrate would have:
real spike-timing variability, inter-region transmission delays,
imperfect gamma window boundaries, refractory periods,
synaptic-strength variability. The algebra PASS justifies building
the spiking-substrate implementation; it is NOT itself a
substrate-validated capability claim.

This is the SAME standing the FHRR-algebra had before the resonate-
and-fire biologization in the FHRR-biologization arc. The
biologization there:
- (shortcut 1) function-first integrator neurons -> resonate-and-
  fire neurons (PASS)
- (shortcut 3) argmax clean-up -> attractor identification +
  separate familiarity gate (RESOLVED)
- (shortcut 2) oracle symbols -> mean-centered substrate activity
  (PASS via common-mode removal)

The theta-gamma biologization will follow the same pattern:
- Replace abstract "patterns[c]" with spiking populations
- Implement theta clock generator (substrate-internal)
- Verify gamma window boundaries hold under real neural noise
- Verify slot windowing reads what the algebra promises

If each of these biologization steps PASSes, the theta-gamma
positional binding becomes a substrate-validated capability and a
fully-biology-grounded complement to ec_context.

## Next steps (autonomous chain)

1. **Direction A FULL-SCALE** still in flight (~3 hr GPU). When it
   completes:
   - Smell test (`direction_A_smell_test.py`; 3 anti-cheat
     controls; ~10-15 min wall)
   - Dedicated fresh-agent adversarial review
   - Capability pillar n=103 IF PASS_CONTROLS_DECISIVE
   - If Direction A PASSes: theta-gamma becomes a COMPLEMENTARY
     biology-grounded mechanism on the same substrate (catalog has
     both)
   - If Direction A FAILs or COLLAPSES: theta-gamma is the
     principled fallback for substrate sequence storage

2. **Theta-gamma spiking substrate implementation** (design + write
   + GPU run) -- justified by this ALGEBRA validation.

3. **Synthesis** of the two mechanisms in one substrate (spatial
   ec_context + temporal theta-gamma) -- biology has both; they
   should compose.

## Honest ceiling (per project standing)

This is a numpy-algebra validation of a biology-grounded mechanism.
It is NOT a conversational capability, NOT fluent open-ended
language, NOT an LLM. The full project goal -- artificial life with
a proper brain analogue -- is incrementally served by validating
each biology-grounded primitive in its proper measurement regime,
propagating honestly, and iterating.
