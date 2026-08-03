---
type: finding
status: contributing
date: 2026-08-03
mechanism: laneC-plastic-source-memory
runner: research/runners/_laneC_plastic_source_memory_derisk.py
artifacts:
  - research/findings/raw/lanes/metacog/laneC_plastic_source_memory_6seed.json
  - research/findings/raw/lanes/metacog/laneC_plastic_source_memory_6seed.json.prov.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_plastic_source_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_plastic_source_6seed.json.prov.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_plastic_source_gpu_smoke_s42.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_plastic_source_gpu_smoke_s42.json.prov.json
---

# Lane C plastic source memory: learned source support and production wire-in GO

<!--derived-->
**One-line verdict.** A zero-initialized spiking Hebbian memory now learns whether a complete proposition was
externally experienced and can feed that support into the default-off self-schema honesty path without an expected
answer table, exact source fact, or primary-memory index at inference. The isolated six-seed gate is GO; in the
stressed production battery the combined path downgraded 46/46 familiar-but-wrong recalls, left zero wrong assertions,
preserved 475/475 hard abstentions, and added zero false accepts. The learned source signal itself detected 45/46 wrong
candidates and unnecessarily marked 5/133 correct recalls as unsupported, so this is a promoted burn-down rung rather
than final biological source monitoring.

## Role In The Whole Brain

Fluent recall and reliable recall are different. Before speech, the brain needs evidence that the proposition it is
about to express is connected to an actual experience rather than only to a strong cleanup winner. This source signal
should influence uncertainty in the self-model while remaining downstream of the existing hard unknown-fact boundary.

The previous exact-metadata floor solved the measured safety problem by consulting a Python fact. The independent RF
echo removed that direct lookup but still wrote a complete second memory by construction. This rung changes the
mechanism: experience changes source-supporting synaptic weights, and a later live proposition must reinstate source
population activity through those weights.

## Mechanism

- Four independent banks assign each `(query kind, cue, candidate)` proposition a sparse distributed assembly.
- Each proposition bank projects to an external-source population through a fully connected pathway whose weights are
  explicitly zeroed at construction.
- `observe_source_event(...)` co-activates the proposition and source populations while a named Hebbian learning gate
  is open. Primary-memory `store()` alone does not teach source evidence.
- Retrieval closes plasticity, drives only the live recalled proposition, and normalizes source spikes by a direct-drive
  liveness measurement.
- `plastic_source_consistency` can retain or lower the confidence entering the existing spiking self-schema relay. It
  cannot create content after the hard moat abstains.

The implementation is opt-in. Existing composer, trace, engineered-source, and self-schema behavior remains unchanged
unless the new mode is selected and source events are explicitly supplied.

## Isolated Six-Seed Gate

Artifact: `research/findings/raw/lanes/metacog/laneC_plastic_source_memory_6seed.json`.

| measure | result |
|---|---:|
| seeds | 6/6 GO |
| learned propositions accepted | 286/288 (99.3%) |
| wrong-answer propositions accepted | 0/288 |
| unknown propositions accepted | 0/72 |
| no-learning accepts | 0/72 |
| source-path lesion accepts | 0/72 |
| permuted teaching followed | 72/72 |
| original answer accepted after permuted teaching | 0/72 |
| learned weights changed | 6/6 |
| retrieval changed learned weights | 0/6 |
| minimum per-seed worst-case learned-vs-wrong margin | +0.019 | <!--derived-->

The CPU pool ran one seed per process over `42 43 44 100 101 102`. A separate RTX 3090 smoke established CuPy
execution of the same learning, wrong-candidate, and lesion path.

## Production Six-Seed Gate

Artifact: `research/findings/raw/lanes/metacog/laneC_self_schema_plastic_source_6seed.json`.

| measure | result |
|---|---:|
| verdict | GO |
| matched queries | 288 |
| correct matched recalls | 133 |
| familiar-but-wrong recalls | 46 |
| wrong recalls still asserted | 0 |
| learned-source mismatches on wrong recalls | 45/46 |
| learned-source false mismatches on correct recalls | 5/133 (3.8%) |
| correct recalls asserted | 70/133 |
| hard abstentions preserved | 475/475 |
| added false accepts | 0 |
| self-schema invocations on hard abstentions | 0 |

The one wrong candidate that the learned source circuit supported was still downgraded by the first-order confidence
signal. This matters: the result is a combined honesty path, not evidence that the source learner is a perfect truth
oracle. Conversely, five correct recalls were hedged because the learned source support was too weak. That is the
measured retention cost compared with the engineered echo.

The final seed-42 production run also passed on the RTX 3090 with `SIM_BACKEND=cupy`, confirming that the opt-in
integration executes on both supported backends.

## Causal Controls

- The source weights start exactly at zero and remain zero when learning is disabled.
- Source weights grow after explicit experience and remain unchanged during all retrievals.
- Disabling transmission through the learned pathway collapses accepted source support to zero across all seeds.
- Permuted experience teaches the permutation and rejects the original candidate.
- Unseen and wrong propositions do not inherit support from labels or fixed codes in the isolated battery.
- Primary `store()` without an experience event leaves the production mode unavailable and fail-closed.
- Default-off retrieval identity and the hard unknown-fact moat remain intact.

## Honest Boundary

This is not a complete biological source-monitoring system. The sparse proposition allocation is deterministic, the
host explicitly delivers the external-source teaching event, and the source populations live on a dedicated bridge.
Population normalization is host-read, and the production decision still passes through a host scalar and fixed
meta-to-self wiring before host thresholds choose assert, hedge, or soft abstention. The current circuit also learns
only external support for complete symbolic propositions; it does not yet distinguish heard, seen, self-generated,
inferred, or imagined sources in a lived multimodal episode.

## Next Mechanism

1. Move source populations onto the shared brain and connect their spikes directly to the dynamic ACC/aPFC and
   self-schema circuit.
2. Replace the explicit symbolic source event with sensory and corollary-discharge activity during a lived episode.
3. Learn several source classes and ambiguity, rather than one external-support bit.
4. Exercise source-based uncertainty inside the minimal grounded speech-action loop.
5. Keep the engineered RF echo and exact metadata floor only as comparison and safety scaffolds, not the preferred
   research mechanism.
