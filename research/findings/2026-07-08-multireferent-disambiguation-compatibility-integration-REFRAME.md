# Multi-referent pronoun disambiguation — the 2026-06-17 NEGATIVE RESOLVED, and the specified mechanism REFRAMED (6-seed): the fix is the predicate-COMPATIBILITY signal + TEMPORAL INTEGRATION (a linear integrator = 1.000), which decisively defeats the recency/salience confound (0.000); the specified WTA BIASED-COMPETITION is NOT the necessary ingredient — a linear integrator of the RIGHT signal suffices and outperforms the LCA (which makes premature noise-driven commits). The missing ingredient was the SIGNAL, not the inhibition. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_multireferent_wta_derisk.py`. numpy (LCA + linear integrator). NO `sim/` edit.
**Verdict:** REFRAME-GO (6-seed) — resolves the documented NEGATIVE with a simpler mechanism than specified; honestly refutes the WTA-competition necessity.

## Why this ran
The open-domain research gate ranked "multi-referent WTA" as the one specified-but-unbuilt mechanism. The 2026-06-17 finding documented a NEGATIVE (which of several held referents a bare pronoun binds: recency + salience-boost both FAILED) and specified the fix as "winner-take-all biased-competition inhibition between referent attractors." This de-risk BUILDS the specified WTA (Usher-McClelland leaky competing accumulators) and tests it against the right controls — closing the boundary per "boundaries are undiscovered mechanisms."

## The result — 6-seed, K in (3,6,9), noise=0.9
```
                        K=3      K=6      K=9
WTA biased-competition  0.477    0.252    0.191     <- the SPECIFIED mechanism -- degrades with K, worse than linear
linear-integrator       1.000    1.000    0.999     <- compatibility signal + temporal integration (NO competition)
recency/salience readout 0.000   0.000    0.000     <- the 2026-06-17 confound (binds the recent distractor)
argmax(compat)+noise     0.441   0.267    0.191     <- single-shot, noise-fragile
no-bias (equal compat)   0.332   0.173    0.105     <- ~chance (no spurious winner)
```

## The reframe (honest)
- **The confound is real:** the recency/salience readout scores 0.000 — it binds the *recent/salient* distractor, not the predicate-compatible referent. This reproduces the 2026-06-17 failure and shows its cause: the WRONG SIGNAL.
- **The fix is the right SIGNAL + integration:** a linear temporal integrator of the predicate-COMPATIBILITY signal scores 1.000 at every K — it defeats the confound decisively. No competition is involved.
- **The specified WTA is NOT necessary:** the LCA biased-competition (the specified mechanism) scores 0.48→0.19 (degrading with K) — WORSE than the linear integrator. Its mutual inhibition + threshold commit make premature noise-driven commits and can suppress the correct accumulator. So the WTA competition is not the load-bearing ingredient; the RIGHT SIGNAL + integration is.
- **no-bias ~ chance** confirms no spurious winner (the integrator isn't fabricating a choice when compatibility is flat).

## Anti-cheats / controls
- recency-readout (the failing 2026-06-17 approach) = 0.000 → the confound is genuine (not a strawman).
- linear-integrator vs LCA = the fair isolation of the COMPETITION's value (both use the right signal + temporal averaging; only the LCA adds mutual inhibition) → the competition is shown non-necessary (and harmful here).
- no-bias → chance → no fabricated winner.

## Honest scope
This models the disambiguation as SELECTION-under-noise with a moderate (identifiable) compatibility bias — where a long linear integration trivially denoises (hence 1.000). It does NOT claim to solve the harder regimes: (a) genuinely AMBIGUOUS compatibility (the correct referent is not cleanly more compatible), or (b) the BINDING problem (computing the compatibility signal itself from the discourse). Those are harder follow-ons. The LCA's underperformance is partly parameter-sensitive (a differently-tuned attractor might match the integrator); the robust claim is that the WTA competition is NOT NECESSARY — a linear integrator of the right signal is sufficient.

## What this establishes
The 2026-06-17 multi-referent disambiguation NEGATIVE is resolved: the fix is to read the predicate-COMPATIBILITY signal and INTEGRATE it (a biological integrator neuron / persistent-activity accumulator), which defeats the recency/salience confound; the specified WTA biased-competition is not required. This corrects the specified mechanism (simpler than proposed) and is an honest first-class deliverable (a documented boundary → the actual, simpler mechanism). Follow-on: the ambiguous-compatibility + binding regimes; wire compatibility-integration disambiguation into the multi-turn console.

## Files
`research/runners/_realcorpus_multireferent_wta_derisk.py`; `tests/test_multireferent_disambiguation.py`. Prior: the 2026-06-17 multi-referent NEGATIVE; the open-domain research gate `2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md` (ranked #4).
