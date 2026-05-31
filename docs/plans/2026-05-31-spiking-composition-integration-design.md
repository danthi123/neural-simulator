# Spiking compositional binding integration -- design sketch (2026-05-31)

**Context:** The 2026-05-31 composition revision (finding `...-near-ortho-ROLES-not-FILLERS`)
established, with a hardened cheap-first + a working demo (`research/runners/compose_vsa_demo.py`,
20/20 novel-sentence generalization), that generalizable compositional binding (role x filler)
works with the substrate's OVERLAPPING concept codes -- needing only a FEW near-ortho ROLE codes,
not near-ortho concepts; the bind is biologically realizable via ON/OFF coincidence (verified
exact, >=0 ops). The remaining step to "biologically sound" (owner's word) is to implement the
bind/unbind IN the spiking substrate, not as numpy algebra on captured codes.

## Goal

Spiking compositional binding: drive a few ROLE ensembles + concept pools (FILLERS), let the
substrate COMPUTE the bound composite + answer role queries, end-to-end, with the same
generalization the algebra shows. Pre-registered fixed bar (frozen): role-query recovery
>= 0.80 multi-seed at >= 3 bound roles, with a reproduce-the-failure control (no-binding) at
chance.

## Architecture (declarative BrainRegion framework; reuse-by-import)

1. FILLERS = the existing concept pools (their activity is the substrate concept code; already
   validated, between-cos ~0.7, ID-separable).
2. ON/OFF coding: represent each filler dimension by firing ABOVE vs BELOW a baseline. Cleanest
   substrate realization: a pair of populations (or use rectified deviation from the OU/baseline
   rate). This is the project's mean-centering (common-mode removal) made explicit as two
   rate channels.
3. ROLES = a few (3-4) distinct DISTRIBUTED ensembles (NOT disjoint sub-populations -- those
   FAIL per the finding's role-mode control). Each role = a fixed distributed +-1 pattern over
   the binding layer, realized as ON/OFF.
4. BIND = a coincidence-detector layer: one unit per (dimension, ON/OFF) that fires when BOTH
   the role channel AND the filler channel are co-active (AND/coincidence -- a standard
   dendritic/spiking operation; bound_ON = role_ON*filler_ON + role_OFF*filler_OFF, etc.).
   This EXACTLY computes role (x) filler (verified in numpy). Multiple bound pairs SUM in the
   binding layer (superposition).
5. UNBIND = drive the binding layer's coincidence units with a QUERY role -> the recovered
   filler appears as ON-OFF on the readout.
6. CLEANUP = the substrate's existing nearest-concept matching (the parallel-population matching
   / cosine-to-concept-pool readout already validated) selects the recovered concept.

## Cheap-first gate (before the full build)

Focused spiking probe on the cached n=98 substrate: add a small binding layer + 3 role ensembles;
bind 2-3 (role, concept) pairs; query each role; measure recovery vs a no-binding control.
RESOLVES (>= 0.80 + control at chance) -> build the full mechanism + multi-seed decisive + a
spiking compose demo. BOUNDARY -> the spiking coincidence bind loses the algebra's accuracy
(timing/noise) -> characterize honestly; the numpy-on-substrate-codes capability still stands.

## Discipline

Reuse-by-import (concept pools + matching readout byte-unchanged); the binding layer + role
ensembles are NEW declarative regions (no protected/frozen/moat/sim-core change). Cheap-first
before the full build. Frozen bar + reproduce-the-failure control. Honest scope: the numpy
algebra on substrate-derived codes is already validated; this is the in-spiking-dynamics
realization. Catalog grounding: ON/OFF push-pull coding (retina/thalamus), coincidence
detection (dendritic AND; cerebellar granule, CA1), distributed role codes (PFC mixed
selectivity). The owner can steer scope (full build vs the demo being sufficient for now).
