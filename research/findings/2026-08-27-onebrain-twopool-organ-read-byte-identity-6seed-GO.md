---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# One-brain two-pool ORGAN-READ byte-identity — 6-seed GO

**Verdict: GO (6/6 seeds).** The follow-on rung to the substrate-init merge de-risk (Vikunja #171). It closes the
gap that rung left open: the prior GO proved only that the 4 core cortical organs' per-neuron INIT arrays are
byte-identical when merged onto one `SimulationBridge` vs co-resident alone — it did NOT run the organ READ
pipelines nor the post-build topographic wiring those reads depend on. This runner does.

## What was verified (`_onebrain_twopool_organread_verify.py`, numpy, seeds 42/43/44/100/101/102)

The 4 core cortical organs — D2 surprise (1056) + E2 world-model (528) + E1 metacog (290) + D pragmatic (160),
N=2034 on ONE bridge — each run their REAL production read/judge pipeline (`SurpriseProductionOrgan.judge`,
`WorldModelProductionOrgan.expectation`, `MetacogProductionOrgan.judge`, `PragmaticProductionOrgan.interpret`) on
the merged substrate, compared against running the SAME shipped organ classes on today's production TWO pools
(`onebrain_merge_production.MergedSubstrate` #1 + `onebrain_merge_production2.MergedSubstrate2` #2). The runner
adds the post-build wiring the reads need (metacog's dense self-recurrent assembly loops + surprise's 3
block-diagonal topographic masks), applied in the same order the shipped builders use.

| Check | Result |
|---|---|
| Organ-read byte-identity, all 4 organs (per-neuron read state, merged vs two-pool) | **0.00e+00, 6/6 seeds** |
| Chat-answer preservation (rendered answer strings, all 4 organs) | **identical, 6/6** |
| gain-0 freeze holds pool-2 edges bit-frozen after the full train+read lifecycle | **6/6 (n≈26.3k–26.5k edges, d=0.00e+00)** |
| Legacy discriminator (per-region seams OFF → global param-het+wiring) diverges | **6/6 (byte-identity is not vacuous)** |

## The named cross-region hazard, handled (not walled)

Surprise trains Hebbian at `hebbian_max_weight=45` on the SHARED `cp_connections`, while pool-2 (metacog +
pragmatic) edges must stay frozen. Closed with the SAME primitive the parser-on-pool merge used: a permanent
per-synapse `cp_plasticity_rate_gain = 0.0` on every edge with both endpoints inside a metacog/pragmatic region.
The `gain0_freeze_check` verifies BOTH the structural gain array (0.0 over pool-2 edges, 1.0 elsewhere) AND the
outcome (pool-2 edge weights byte-identical before vs after every organ has trained/read on the shared bridge).
This protects two hazards, not one: surprise/world-model's shared Hebbian training never touches pool-2's weights,
AND metacog/pragmatic's own evidence-elicited reads (which fire on the shared bridge, unlike their standalone
Hebbian-off builds) never nudge their own edges into non-reproducible drift.

## Scope + what this is NOT

This is a DE-RISK, not the production flip. It proves the single-pool merge preserves every organ's read + answer
byte-identically — the safe-migration gate. The production flip to a single pool (retiring the two `MergedSubstrate`
classes) is the next rung. Per the owner-approved 2026-08-27 direction, this runner is also the working PROTOTYPE
for the general **merge-framework batched verify**: it already loops organs + applies the gain-0 freeze generically,
so the framework's parameterized verify generalizes it from 4 hardcoded organs to a declarative organ registry.

## Honesty note

Byte-identity-in-isolation is a MIGRATION-SAFETY proof (co-locating the organ didn't change its behavior), NOT the
one-brain GOAL — the goal is organs INTERACTING via cross-region synapses. The hand-declared block-diagonal masks +
assembly loops + gain-0 freezes are host scaffold; the faithful end state has regions that develop connectivity +
interact through learning. This rung earns the safe bulk migration; the integration phase then switches the gate to
functional (faculty still works + cross-region interaction emerges).

Runner: `research/runners/_onebrain_twopool_organread_verify.py` · artifact:
`research/findings/raw/_onebrain_twopool_organread_6seed.json` · branch `research/onebrain-twopool-organ-read`.
