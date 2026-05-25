# Direction P-v3 KILLED early: discovered the proposed architectural fix is a STRICT SUBSET of the already-NEGATIVE 2026-05-22 ca1-variant + staged-recurrence work; substrate-consolidation dynamics-class arc is fully convergent across 5+ negatives; pivot to representation-class (per 2026-05-22 pre-registered next direction)

**Date:** 2026-05-25
**Status:** Direction P-v3 KILLED before multi-seed completion; partial seed-42 result captured (pre-A 0.375 untrained); rediscovery documented; arc-closure synthesis; representation-class pivot pre-registered

## What happened

Direction P-v3 was queued per the 2026-05-24 P-v2 finding doc as the
"CLS architectural fix" - wrap `build_biological_brain_regions` to
append 12 plastic `ca1 -> {noun,verb,adjective}_pool` `RegionPathway`
entries before bridge construction, then test if proper hippocampal-
only multitag encoding + SWR consolidation transfers to cortex.

P-v3 was scaffolded (`research/findings/raw/direction_P_v3_ca1_concept_pathways.py`,
commit 720523a), launched as background python (PID 15676), and
produced a partial seed-42 result before being killed:

```
seed 42 (P-v3 ca1->concept pathways)
  ADDED 12 extra ca1 -> concept_pool pathways (density 0.1, weight 2.0)
  fresh substrate built 0.15 min
  hippo region_filter: ['ca3', 'ca1', 'dg']
  encoding 8 pairs - encoded 0.37 min
  PRE-A (hippo active)     3/8 = 0.375  (below 0.50 bar)
  PRE-B (hippo silenced)   1/8 = 0.125
  SWR cycle (opening ca1->concept gate)  -- killed here
```

The partial 0.375 pre-A on FRESH (untrained) substrate is already
below the 0.50 bar, and the substrate has no Phase-1 trained weights
in the new pathways. The result would converge with P-v2's HIPPO_
ENCODING_INSUFFICIENT verdict.

## Why P-v3 was killed (the rediscovery)

A parallel literature pass during the P-v3 GPU wait found that the
exact "ca1 + 12 plastic concept-pool wires" architecture had ALREADY
been built and tested 3 days earlier:

**2026-05-22 ACh-staged recurrence variant finding** (`research/findings/2026-05-22-staged-recurrence-variant-NEGATIVE-verified-active-dynamics-gating-class-exhausted-converges-with-SPEAR.md`):

> "The ca1-variant substrate (concept pools + the 12 appended
> `ca1 -> concept-pool` consolidation wires) reused with its Phase-1
> checkpoint... After loading, recurrent excitatory connectivity was
> INSTALLED into each concept pool (30,335 edges across 12 pools...).
> Then: encode 4 compositional bindings, measure, run replay
> consolidation, measure at 20 and 60 cycles."

Result for that work was NEGATIVE with a verified-valid structural-
effect check (the installed recurrence genuinely transmits at 1.41x
whole-pool spread, so the negative is NOT an inert-mechanism artifact):

```
| phase                       | bound-adj pool rate | selective | permuted-ctrl |
|-----------------------------|--------------------:|----------:|--------------:|
| pre-recurrence              | 0.0024              | 2/4       | 1/4           |
| post-install pre-consolidat | 0.0023              | 1/4       | 2/4           |
| 20 replay cycles            | 0.0022              | 1/4       | 2/4           |
| 60 replay cycles            | 0.0020              | 1/4       | 1/4           |
```

The 2026-05-22 setup tested **ca1-variant + ACh-staged recurrence
amplification**. Direction P-v3 tests **ca1-variant alone** - a
strictly weaker configuration than the already-NEGATIVE setup. The
P-v3 result would be a known-NEGATIVE duplicate consuming 3-5 hr GPU
to confirm what is already documented.

The honest decision: KILL P-v3 immediately rather than complete it,
document the rediscovery, and pivot to the actually-frontier
direction.

## Why the P-v2 finding doc recommended P-v3 anyway (lesson)

The P-v2 finding doc (2026-05-24) listed P-v3 as the queued next
direction without referencing the 2026-05-22 ca1-variant work. This
appears to be a case of the autonomous chain losing thread continuity
across days - the 2026-05-22 finding closed the dynamics-class arc
and pre-registered "phase-coded vector-symbolic composition" as the
next direction, but the 2026-05-24 P-v2 chain re-derived the same
architectural fix as if it were novel.

**Discipline lesson**: before launching ANY new direction, grep the
prior findings dir for the proposed mechanism to confirm it is not
a duplicate of a closed arc. Specifically check for the architectural
substrate (regions + pathways added) and the mechanism class (gating
/ wiring / representation / scale) of the proposed fix. The
autonomous-runs principle says "use prior findings + commit log;
read them" - this is an explicit instance of that principle's
necessity.

## Cumulative convergent dynamics-class NEGATIVE findings (now 5+)

Per the 2026-05-22 staged-recurrence finding:

> "The compositional investigation has now tested, and exhausted, the
> entire class of 'fix the network dynamics' interventions:
>  - 8 architectures: gating, theta-multiplexing, disinhibition,
>    per-regime monitoring, cue-suppression, generative replay,
>    aggressive consolidation, pool-readout substitution
>  - difference-readout probe: the readout computation
>  - ca1-variant: the missing consolidation wire
>  - staged-recurrence variant: ACh-staged recurrent amplification"

Plus the 2026-05-24 work that re-derived but did not fully realize
duplication:

- Direction P (cortex-only multitag chat + SWR): TRIVIAL PASS (cortex
  bypass; the SWR mechanism was not load-bearing)
- Direction P-v2 (hippocampal-only engram + SWR): HIPPO_ENCODING_
  INSUFFICIENT (pre-A 0.167 below 0.50)
- Direction P-v3 (P-v2 + ca1 architectural pathways): duplicate of
  ca1-variant; killed before completion

Plus the 2026-05-24 (c) generative-replay decisive run:

- (c) loop n=99 pillar: 5.78% vs 6.25% chance at K={4,8,16}; the
  diagnostic probe confirmed REPLAY_DOESNT_REACTIVATE (post-replay
  cortex selectivity +0.006, essentially zero)

Convergent diagnosis (unchanged from 2026-05-22):

> "The compositional fix is not in the network dynamics. It is in
> the REPRESENTATION."

## Pre-registered next direction (2026-05-22, still binding)

> "The genuinely-missed thread, pre-registered by the SPEAR arc
> itself and never built: phase-coded vector-symbolic composition
> (Orchard 2023/2024 spiking-phasor / Fourier Holographic Reduced
> Representations). Instead of trying to make the cortical concept
> pools host a consolidated compositional attractor -- which the
> whole dynamics-gating class cannot do -- the shared theta-gamma
> rhythm CARRIES the composed representation as the phase of each
> spike within a cycle. Bind / unbind / superposition / cleanup
> become operations on a structured, decodable phase-coded object."

Status of representation-class work:

- FHRR algebra: PASS (Direction K numpy probe + multiple prior FHRR
  pillars n=84-94)
- FHRR + substrate-grounded phasors: PASS but substrate not load-
  bearing per reviewer BLOCK (Direction K no-teacher 1.000; smell
  test showed random phasors also PASS at N_DIM=3200)
- Biologized FHRR pipeline: 0.000 at substrate scale (Direction K
  reviewer fix #3 BOTH FAIL)
- Theta-gamma ALGEBRA: VALIDATED multi-seed (pillar n=103)
- Theta-gamma at substrate: BOUNDARY (Direction E Task 1: 0.250;
  Direction G HIPPO+theta-gamma: 0.333)

The representation-class is ALSO substrate-bounded at current scale.
Both dynamics-class (5+ negatives) and representation-class (3
convergent BOUNDARY) hit the SAME ceiling: substrate at 60-200
neuron pools is below the biological threshold for these mechanisms
to engage robustly.

## The actual frontier (autonomous next selection)

Two pre-registered options remain that have NOT been duplicated:

**Direction Q** (per AUTONOMOUS_STATE.md pre-registered): dlpfc_wm
scale-up 60 -> 1000 neurons + dense recurrent + dedicated PFC training
to attempt Wang 2002 attractor at proper scale. Tests the scale
hypothesis directly. Closes Direction I bound (PFC NMDA bistability
failed at 60-neuron substrate). Substantial; 1-2 weeks subagent-
driven build per pre-registered estimate.

**Direction 4 from 2026-05-24 post-c roadmap** (`docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`):
cross-bridge bio_brain_regions composition. Train multiple
bio_brain_regions bridges on different vocab categories (mirroring
G.20 sparse's 5-bridge pattern). Test cross-bridge mode-unification
on the union. ~per-bridge 30 min train; full ensemble ~3 hr;
cross-bridge probe ~10 min CPU.

**Direction 3 from same roadmap**: extend OPTION 3 / HIPPO-OPTION3 /
DLPFC-extension chain to 32, 64, 160 concepts on bio_brain_regions
(per-tier ~1.5-2 hr GPU).

Direction 4 is cheapest first; Direction 3 compounds on Direction 4;
Direction Q is the most substantial scale-up answer.

## What is preserved unconditionally

- All five substrate-readiness pillars (n=93/n=94/n=96/n=97/n=98)
  stand. Their validation is independent of the dynamics-class arc.
- The working deliverable Direction M (320-concept G.20 multi-bridge
  chat with multitag mechanism) stands.
- Direction R capacity envelope (50 assoc 80% top-1; 192 assoc 45%
  top-1 / 95% top-3) stands.
- Phase 1.3 SWR consolidation validated for DIRECT-BINDING tasks
  (not retracted; just clarified that it does NOT generalize to
  multi-slot sequence completion at this substrate scale).
- Parallel-matching mode-unification validated.
- No-confab moat 7/7 byte-identical.
- Frozen 0.80 bar unchanged.
- No protected/frozen/moat module modified.

## Discipline preserved

- P-v3 runner (`research/findings/raw/direction_P_v3_ca1_concept_pathways.py`)
  remains in the repo as documentation of the attempted direction +
  rediscovery audit trail; will not be re-run.
- Honest propagation: this finding doc + AUTONOMOUS_STATE update +
  both remote pushes document the rediscovery rather than silently
  pivoting.
- The KILL decision was made AT THE TIME OF discovery (mid-seed-42),
  not deferred to multi-seed completion - reviewer-style scrutiny
  applied promptly.
- The autonomous chain continues with the actually-frontier direction
  (Direction 4 cross-bridge bio_brain_regions composition, cheapest
  first per 2026-05-24 roadmap).

## Files

- Killed runner: `research/findings/raw/direction_P_v3_ca1_concept_pathways.py`
- Partial log: `research/findings/raw/direction_P_v3.log` (captured up
  to seed-42 SWR-cycle start)
- 2026-05-22 ca1-variant + staged-recurrence finding: `research/findings/2026-05-22-staged-recurrence-variant-NEGATIVE-verified-active-dynamics-gating-class-exhausted-converges-with-SPEAR.md`
- 2026-05-24 (c) loop diagnostic REPLAY_DOESNT_REACTIVATE: `research/findings/2026-05-24-c-loop-diagnostic-REPLAY_DOESNT_REACTIVATE-Phase-1-3-SWR-consolidation-validated-for-direct-binding-not-sequence-completion.md`
- 2026-05-24 (c) loop decisive NEGATIVE pillar n=99: `research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-loop-at-n-iterations-1-doesnt-produce-above-chance-completion-pivot-direction-identified.md`
- 2026-05-24 post-c direction roadmap: `docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`
- 2026-05-24 P-v2 finding (the source that re-derived P-v3): `research/findings/2026-05-24-DIRECTION-P-v2-HIPPO-ENCODING-INSUFFICIENT-substrate-consolidation-arc-CLOSED.md`
