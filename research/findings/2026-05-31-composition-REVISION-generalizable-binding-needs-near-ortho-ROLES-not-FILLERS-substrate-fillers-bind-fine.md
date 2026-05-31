# Composition arc REVISION (positive): generalizable compositional bind/unbind works with the SUBSTRATE's OVERLAPPING fillers -- the near-orthogonality boundary is on a FEW ROLES (trivially feasible), NOT on the MANY FILLERS. This meaningfully reopens biological composition: the bound that I characterized this session (substrate concept activity can't be made near-ortho) blocks making MANY concepts near-ortho, but composition only needs a FEW near-ortho ROLE codes x MANY ID-separable (within>between) overlapping FILLER codes -- which the substrate HAS. Honest caveats: roles must be DISTRIBUTED near-ortho (disjoint sub-populations FAIL); abstract +-1 VSA isn't directly biological (neurons >=0 -> needs the >=0 gain-field realization); a cleanup bias inflates absolute accuracy.

**Date:** 2026-05-31
**Status:** Positive REVISION of the composition picture, from a hardened cheap-first numpy probe (anti-cheat + role-mode controls). Owner chose Option 2 (pursue composition). After check-existing-first found generative-replay + sequence-storage bounded, this probe tests the load-bearing question directly and overturns the naive "composition needs near-ortho concepts" reading.

## What was tested + result

VSA / gain-field compositional binding (role x filler; the only binding that GENERALIZES to novel combos -- bind ANY role x ANY filler on the fly): a 'sentence' = sum of K bound (role, filler) pairs; query a role -> unbind (Hadamard) -> cleanup to nearest filler. fillers = the substrate's REAL concept codes (denoise64 cache, centered, between-cos 0.70). Multi-seed 42/43/44, 60 trials.

| K | random near-ortho roles | overlapping roles (cos 0.67) | disjoint-subpop roles | broken-binding (anti-cheat) |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.133 | 0.406 |
| 2 | 1.000 | 0.947 | 0.111 | 0.314 |
| 4 | 1.000 | 0.807 | 0.103 | 0.188 |
| 8 | 1.000 | 0.619 | 0.129 | 0.115 |

chance = 1/16 = 0.062.

## The revision (sound) + the caveats (honest)

SOUND CORE FINDING: the FILLERS being overlapping (substrate concepts, between 0.70) does NOT block compositional bind/unbind -- with near-ortho roles, accuracy is 1.000 up to K=8. The unbinding works because cleanup uses ID-SEPARABILITY (within > between; the substrate has 16/16 identity), NOT near-ORTHOGONALITY (between -> 0). So the near-ortho boundary -- which is about making MANY concept-fillers near-ortho (impossible per this session) -- is NOT the relevant bar for composition. Composition needs a FEW near-ortho ROLE codes (agent/patient/action/manner -- 3-8 of them, trivially makeable distinct) x MANY ID-separable FILLERS (the substrate has). This is a MUCH smaller, FEASIBLE requirement.

CAVEATS (load-bearing, do not bury):
1. ROLES must be DISTRIBUTED near-ortho. Random distributed bipolar roles -> 1.000. But DISJOINT SUB-POPULATION roles (the naive "distinct dlpfc neuron block per role") -> ~CHANCE (0.10) -- block roles restrict the bind to 1/R of the dims, destroying SNR. So the simple "role = a distinct sub-population" implementation FAILS; roles must be distributed patterns over the whole population.
2. OVERLAPPING roles DEGRADE with load (0.81 at K=4 -> 0.62 at K=8). Roles need to be reasonably near-ortho; substrate-grounded (overlapping) roles only partially work. But only a FEW roles are needed, and few distributed patterns are easy to make near-ortho.
3. ABSTRACT +-1 VSA isn't directly biological (neurons fire >=0) -- BUT this is RESOLVED by ON/OFF rate coding + coincidence detection (all >=0): represent each value by a pair (ON = firing-above-baseline = max(x,0), OFF = below = max(-x,0)). The coincidence bind bound_ON = role_ON*filler_ON + role_OFF*filler_OFF, bound_OFF = role_ON*filler_OFF + role_OFF*filler_ON EXACTLY realizes the +-1 Hadamard product (role=+1 -> filler unchanged; role=-1 -> ON/OFF swap = -filler), using only >=0 multiply+add (coincidence detection, a standard dendritic/spiking operation). Readout = ON - OFF. So the BIND is biologically realizable by construction; ON/OFF coding = the project's mean-centering (baseline subtraction) doubled. The naive >=0 single-population gain-field DOES fail (cross-terms don't sign-cancel); ON/OFF restores the cancellation. The remaining open step is implementing ON/OFF concept coding + coincidence bind + cleanup in the actual SPIKING substrate (noise-robustness at 2x std is encouraging).
4. CLEANUP BIAS: the broken-binding anti-cheat is 0.41 at K=1 (above chance) -- substrate fillers have residual structure that cleanup partially recovers even from a wrong-role unbind; it washes out by K=8 (0.12). The binding adds REAL signal (gap 1.000 - 0.41 = 0.59 at K=1) but the absolute numbers are cleanup-bias-inflated.

## Why this matters (strategic)

Before this probe, the composition picture looked uniformly bounded (generative-replay, sequence-storage, near-ortho all bounded), implying composition needs months-scale richer training. This revises it: the near-ortho boundary blocks the WRONG thing (many near-ortho concepts) -- composition only needs a few near-ortho ROLES, and the substrate's overlapping fillers are FINE. So there is a plausible biological-composition path on the CURRENT substrate, contingent on (a) a >=0 gain-field binding that works, and (b) a few DISTRIBUTED (not sub-population) near-ortho role codes from a biological source. Both are open but far more tractable than richer training.

## Next (cheap-first, biologically-realizable)

Test the BIOLOGICAL realization: (1) >=0 multiplicative gain-field binding (role_gain >= 0 elementwise x filler) -- does it unbind/cleanup with substrate fillers + few distributed roles? (2) where do a few distributed near-ortho role codes come from biologically -- e.g. a few fixed random projections (cerebellar-granule / mushroom-body expansion), or a few engram-tagged distributed role patterns, NOT disjoint sub-populations. If the >=0 gain-field + distributed-role realization clears a frozen bar with the anti-cheat near chance, biological compositional generalization is a real capability on the current substrate.

## Discipline

Throwaway numpy probe; stdlib+numpy + cached substrate activity; no protected/frozen/moat module touched. The surprising first ESCAPE (1.000) was scrutinized HARDER than a FAIL: role-mode controls revealed disjoint-subpop roles fail + overlapping roles degrade; the broken-binding anti-cheat revealed a cleanup bias. The SOUND core (overlapping fillers bind fine; near-ortho is on roles-not-fillers) survives; the absolutes + the biological realization are honestly caveated. Frozen bar + reproduce-the-failure (broken binding) control built in. This is a genuine positive revision, not an overclaim.
