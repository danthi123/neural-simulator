# Step-3 cheap-first de-risk — do LIVE `cortex_it` RATE-derived grounded codes COMPOSE (not just recall)? **GO (mechanism, 4-object scope)**

**Date:** 2026-06-16
**Runner:** `research/runners/_step3_live_cortex_grounded_compose_probe.py`
**Raw:** `research/findings/raw/_step3_live_cortex_grounded_compose.json`
**Scoping (controller-verified):** `research/findings/2026-06-16-step3-compose-perceived-content-scoping.md`
**Precedent (verified):** `research/runners/_visual_grounded_composition_probe.py` (grounded V1-matrix phasor codes compose)
**Verdict:** **GO at the mechanism level** — 3 seeds (42/43/44), CPU `SIM_BACKEND=numpy` (512-neuron bridge,
~1 s/seed = the tiny-smoke carve-out). Exit 0.

---

## The question this closes

The (B) perception→memory milestone (`navigate_to_see_then_answer.py`, 6/6 seeds) maps its OWN honest boundary:
it can RECALL "I saw the apple" (stimulate the perceived ensemble's engram tag → read it back) but it **cannot bind
a perceived apple into a NEW role-filler fact** — the engram tag is an opaque pointer to an ensemble, not an
operand the composer's bind/unbind algebra can manipulate. That is the rate-vs-phasor cross-code wall: navigation
perception is a RATE code (`cortex_it` Izhikevich firing rate); the composer is a PHASOR code (Fourier Holographic
Reduced Representation — unit-magnitude complex, information in phase). Not commensurable.

The scoping doc's recommended dissolution is **shared grounded codes**: a fixed map sends the `cortex_it` RATE
percept onto a composer-ready phasor code, so a PERCEIVED object enters the EXISTING (validated) bundling algebra
and can be COMPOSED into a novel fact. The precedent `_visual_grounded_composition_probe.py` already showed grounded
phasor codes compose — but from a numpy V1 matrix, not the live spiking substrate. **This de-risk closes that one
gap:** source the grounded code from a LIVE `cortex_it` rate forward pass on a real `SimulationBridge`.

## What the probe does

1. Build the (B) probe's real bridge (`cortex_it` 256 neurons + `language_output`, Izhikevich, the same substrate
   the navigation perception writes into).
2. For each object: drive its percept into `cortex_it`, run `RATE_READ_STEPS=80`, accumulate `cp_firing_states`
   → a per-neuron **spiking firing-rate** vector (the grounded rate code — a real substrate response, not a numpy
   stand-in).
3. Project that rate vector through a FIXED random complex projection → a unit-magnitude phasor code. Fixed (not a
   free random code) = the phasor code is a deterministic function of the rate features → **grounded**.
4. Compose 2-role facts of two PERCEIVED objects: `fact = R_AGENT·code(a) + R_PATIENT·code(b)` (bind + bundle, the
   validated VSA primitive the production conversational system uses); unbind each role by `fact·conj(R)`; cleanup
   by nearest phasor in the codebook.
5. **The anti-cheat that separates COMPOSE from RECALL:** all ordered distinct-object pairs are split into
   MEMORIZED (a recall baseline stores their bound vectors) vs HELD-OUT (never composed before). A recall baseline
   (`_mem_recall`: nearest stored fact → return its remembered filler) is scored on the SAME held-out facts. The
   algebra composes held-out facts identically to memorized ones (shared codebook + roles); the recall baseline
   cannot recover a held-out fact's fillers because no stored fact matches it.

## Result (3 seeds)

| Seed | held-out compose **clean** | held-out compose **corrupt** | held-out **mem-floor** | memorized compose | memorized mem-floor |
|------|----------------------------|------------------------------|------------------------|-------------------|---------------------|
| 42   | 1.000                      | 1.000                        | 0.500                  | 1.000             | 1.000               |
| 43   | 1.000                      | 1.000                        | 0.500                  | 1.000             | 1.000               |
| 44   | 1.000                      | 1.000                        | 0.500                  | 1.000             | 1.000               |
| mean | **1.000**                  | **1.000**                    | **0.500**              | 1.000             | 1.000               |

Chance = 0.250 (1 of 4 objects).

**The discriminating evidence is the gap, not the absolute.** On HELD-OUT (never-composed) facts, the algebra
recovers BOTH fillers at 1.000 while the recall baseline manages only 0.500 → **compose generalizes to novel
(object, role) combinations; a lookup does not.** The memorized-facts column is the sanity check: the recall
baseline trivially gets memorized facts right (1.000), confirming the floor isn't broken — it specifically fails on
the held-out novel facts. The live-spiking-rate-derived grounded code carries enough object identity that the
percept enters the bundling algebra cleanly.

## HONEST SCOPE — why `1.000` is NOT the headline

- **4 objects (chance 0.25).** Cleanup over only 4 phasor codes is trivial — `1.000` clean is the expected ceiling,
  not a strong result. Likewise corrupt `1.000`: a noisy percept of one of only 4 objects still lands nearest its
  own clean code. The absolute compose accuracy is **inflated by the tiny vocabulary**; do not read it as a scaled
  validation.
- **What the result DOES establish:** the mechanism works — a live `cortex_it` spiking rate code, through a fixed
  projection, produces phasor codes separable enough to (a) clean up and (b) **generalize** (held-out 1.000 ≫
  mem-floor 0.500). This is the compose-vs-recall distinction holding on the live substrate, which is the load-
  bearing question the scoping doc posed.
- **The mem-floor is 0.500, not 0.250,** because a held-out fact `(a,b)` is often nearest a stored fact sharing `a`
  OR `b`, so the recall baseline gets one slot right by partial overlap. The floor being non-trivial makes the
  held-out ≫ floor gap a genuine separation, not an artifact of a degenerate baseline.

## NEXT (the obvious stress, GPU)

Scale the object vocabulary to 16–32 objects, where cleanup over the codebook is non-trivial, and test whether the
live `cortex_it` rate code (still 256 neurons) keeps the grounded phasor codes separable. If it holds, the shared-
grounded-codes route is de-risked at a realistic vocabulary; if it degrades, that localizes the limit to the rate
code's capacity at fixed neuron count (→ population code / more `cortex_it` neurons / a learned projection) — an
honest map either way. Run on GPU (`SIM_BACKEND=cupy`).

## BRAIN-BASED-ONLY accounting

The grounded code is a LIVE spiking rate read (`cp_firing_states` over the read window) on a real bridge — the
substrate's own response to the percept, not a host stand-in. The bind/unbind/bundle is the validated fixed VSA
primitive (the same one the production conversational system runs on the FHRR resonate-and-fire substrate). The
fixed projection is the shared-grounded-code map (the scoping doc's mechanism). Host code here is the de-risk
harness (driving the percept = the environment's sensory render; scoring). **No `sim/` edit.** This de-risks
COMPOSITION over PERCEIVED OBJECTS bound into facts; it is NOT the learn-the-whole-bind-including-multi-attribute-
bundling version (point-neuron bundling needs a fixed self-inverse / dendritic primitive per the 2026-06-16
capability map) — the dendritic rewrite stays the deferred owner call.
