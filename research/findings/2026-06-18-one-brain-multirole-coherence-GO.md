# Roadmap phase 2, step 3c — phase coherence holds as a fact gains ROLES (2→3→4) on the persistent bridge: GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** (3 seeds × 2 D × 3 role-counts = 18/18).
The four prior de-risks proved a **2-role** fact (agent, action) chains register→register on ONE persistent bridge
(bind → bundle → unbind → cleanup → moat, ~5 ops, 1.000). The explicitly-named **top risk** was "phase coherence as
the chain lengthens." A realistic conversational fact has MORE roles — "the big warm dog goes" = agent(dog) +
action(go) + attribute(big) + attribute2(warm). Each extra role adds one bound vector to the bundled composite, so
every unbind sees the other R−1 binds as superposition **crosstalk**. This de-risk sweeps **R = 2, 3, 4** and measures
whether the full on-bridge chain still recovers EVERY role.

**Runner:** `research/runners/_phaseB_onebrain_multirole_coherence_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_multirole_coherence.json`

## Result — 3 seeds × {D=64, D=128}, per role-count

| roles R | on-bridge == truth | on-bridge == host (parity) | host == truth (FHRR baseline) | random-codebook anti-cheat (chance 0.050) |
|---|---|---|---|---|
| 2 (agent, action) | **1.000** | **1.000** | 1.000 | 0.017 |
| 3 (+ attribute) | **1.000** | **1.000** | 1.000 | 0.044 |
| 4 (+ attribute2) | **1.000** | **1.000** | 1.000 | 0.033 |

All on-bridge, register→register, no host round-trip: the R fillers are kicked, the binds settle, the bundle settles
the composite, then all R unbind synapses fire in parallel into R separate query registers, and R separate concept-score
blocks read their query register via `conj(codebook)` — the answer is the argmax of each block's membrane (re), exactly
the step-3a/3b on-bridge cleanup, R-wide.

## Reading

- **The substrate is faithful at every R** (on-bridge == host parity 1.000): the on-bridge FHRR (Fourier Holographic
  Reduced Representation) chain computes exactly what the validated numpy composer computes, through 4 bundled binds.
- **The named phase-coherence risk does NOT bite to R=4** at these D: a persistent bridge holds a **multi-attribute
  fact** and answers every one of its roles with no host round-trip — no phase-latch on the stored composite required
  up to four roles.
- **The anti-cheat collapses to chance** (random-codebook cleanup 0.02–0.04 vs chance 0.05): the recovery is the real
  matched filter, not an artifact.

This directly de-risks the production `OneBrainComposer`'s per-fact capacity (the §5 risk-A "phase coherence as the
chain lengthens" in `2026-06-18-production-one-brain-composer-scoping.md`): a single stored fact can carry up to four
role-binds and still be queried correctly on one brain.

## Honest scope + next

- This maps the **per-fact** role capacity (one composite, R bundled binds). It is complementary to, not a substitute
  for, the **multi-fact** store capacity (many facts living in plastic complex synapses on the persistent bridge — the
  scope's GAP A / STEP A1, the recommended next cheap-first de-risk). The two crosstalk sources are different: this is
  the within-composite bundle crosstalk; A1 is the across-fact tiling, which the scope keeps isolated by per-block
  tiling (facts not summed).
- R=4 is the composer's declared filler-role set (agent, action, attribute, attribute2); beyond that needs more roles
  (and at larger R the FHRR superposition SNR — host included — eventually limits, mitigated by larger D / split binds).
- The selection read is still a host argmax over the on-bridge per-block membranes — a legitimate body/output read
  (the scope's GAP C); the optional spiking winner-take-all is a later biologization.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_multirole_coherence_derisk \
    --seeds 42,43,44 --dims 64,128 --roles 2,3,4
```
