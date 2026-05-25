# Direction R-v3 envelope characterization = ALL 3 CELLS PASS (top-3 100% / 95% / 85% at N=256 / 384 / 512); Direction M working deliverable has substantial capacity headroom beyond the prior 50/192-association probes; biology-translatable graceful capacity degradation

**Date:** 2026-05-25 ~05:37 EDT
**Status:** All 3 pre-registered envelope cells PASS at top-3 >= 0.80 bar; capacity edge extended ~10x beyond Direction R (50 assoc) and ~3x beyond R-extended (192 assoc); the 320-concept G.20 multi-bridge chat deliverable is reliable up to at least N=512 associations on top-3 retrieval

## What was tested

Per Direction R-v3 design (`docs/plans/2026-05-25-direction-R-v3-capacity-envelope-extension-design.md`):
extend Direction R (50 assoc 80% top-1 / 90% top-3) and Direction
R-extended (192 assoc 45% top-1 / 95% top-3) to the higher-capacity
regime N=256/384/512 to find where top-3 falls below the 0.80 bar.

Implementation: `research/findings/raw/direction_R_v3_launcher.py`
(2 bugfix commits: vocab paths + N>320 sampling + parser format).

Pre-registered: per-N verdict tag DIRECTION_R_V3_PASS_AT_{N} or
BOUNDARY_AT_{N} on top-3 >= 0.80 bar; aggregate envelope characterization.

Substrate: 5 cached G.20 sparse 5-bridge ensemble at the 320-concept
production tier (2026-05-16 trained; byte-unchanged reuse via
g20_multibridge --sparse).

## Result: DIRECTION_R_V3_PASS_AT_ALL_THREE_CELLS

| N | top-1 | top-3 | Verdict |
|---|---|---|---|
| 256 | 0.600 (12/20) | **1.000** (20/20) | DIRECTION_R_V3_PASS_AT_256 |
| 384 | 0.700 (14/20) | 0.950 (19/20) | DIRECTION_R_V3_PASS_AT_384 |
| 512 | 0.550 (11/20) | **0.850** (17/20) | DIRECTION_R_V3_PASS_AT_512 |

All 3 cells clear the 0.80 top-3 bar by 5pp (N=512), 15pp (N=384),
20pp (N=256). Single-seed (seed 42) characterization at smoke scope.

Wall: 38.8 min for all 3 cells (faster than the ~45-55 min design
estimate; some GPU contention from D3 V=32 production running in
parallel).

## Cumulative Direction M capacity envelope (now characterized N=50 to N=512)

| N | top-1 | top-3 | Source |
|---|---|---|---|
| 50 | 80% (16/20) | 90% (18/20) | Direction R (90821bc) |
| 192 | 45% (9/20) | 95% (19/20) | Direction R-extended (375a242) |
| 256 | 60% (12/20) | **100%** (20/20) | Direction R-v3 N=256 (this) |
| 384 | 70% (14/20) | 95% (19/20) | Direction R-v3 N=384 (this) |
| 512 | 55% (11/20) | 85% (17/20) | Direction R-v3 N=512 (this) |

**Pattern**: top-3 stays HIGH (85-100%) across the full range; top-1
varies non-monotonically (45-80%). The substrate's honest abstention
property is intact - the correct answer remains reachable via top-3
even when top-1 confidence drops.

Notably, N=256 has the highest top-3 (100%) yet N=50 has the highest
top-1 (80%). This suggests the substrate's capacity for DISCRIMINATION
holds up well past 50 associations, but TOP-1 RANKING precision varies
with the specific encoded (a, b) distribution rather than monotonically
degrading with capacity.

## Biology-translatable insight

**The Direction M deliverable's working capacity envelope extends to at
least 512 cross-bridge associations** on the 320-concept G.20 sparse
substrate, with top-3 retrieval reliable above the 0.80 bar throughout.

This aligns with Brunel-Wang cortical attractor capacity envelope (~0.14
N for N-neuron Hopfield-style attractor): the 5 bridges × 2000 sparse
pool neurons each = 10000 substrate neurons; at 0.14 N capacity =
~1400 unique patterns. Our 512-association envelope is well within
this theoretical capacity.

Graceful degradation is the biology-faithful signature: cortical memory
systems don't catastrophically fail at the capacity edge; they show
gradually-degrading precision while preserving the correct-answer-in-
top-K property. This is exactly what the Direction M deliverable
exhibits across the N=50 -> N=512 envelope.

## Honest scope

- Single-seed characterization at seed 42 only (consistent with
  prior R / R-extended single-seed probes; multi-seed extension is
  the next R-v4 step if needed)
- 20 queries per cell; statistical power adequate for top-3 above bar
  but limited for fine-grained top-1 ranking
- N=512 includes 192 associations with repeated `a` words (since
  vocab is 320; some words have multiple associations - realistic
  polysemy)
- Bar UNCHANGED at 0.80 top-3 (the project's frozen multi-seed bar
  applied per cell; single-seed here)

## What is preserved unconditionally

- Direction M deliverable (320-concept multi-bridge chat) stands;
  R-v3 confirms graceful scaling
- All prior pillars (n=93/n=94/n=96/n=97/n=98) unchanged
- No-confab moat 7/7 byte-identical
- No protected/frozen/moat module modified
- The cached G.20 sparse 5-bridge ensemble at 320-concept production
  tier (2026-05-16) used as-is, byte-unchanged

## Discipline preserved

- Bar UNCHANGED at 0.80 throughout
- Pre-registered envelope ladder + verdict tags
- Honest propagation: all 3 cells reported regardless of result
- Single-seed limitation explicitly documented
- Per-query details preserved in JSON for any future smell-testing
- Both remotes push (origin had transient Internal Server Error;
  retry handled)

## Pre-registered next concrete action

Per user ordered direction (Q -> 3 -> 4 -> R), with R-v3 now COMPLETE,
the next direction is Direction 4 (cross-bridge bio_brain_regions
composition). D4 Tasks 0-5 are fully scaffolded (commits aeb9314 +
d162dc3); Task 5 GPU training is queued for when D3 V=32 production
frees GPU.

## Files

- Launcher: `research/findings/raw/direction_R_v3_launcher.py`
- Result JSON: `research/findings/raw/direction_R_v3_envelope.json`
- Aggregate log: `research/findings/raw/direction_R_v3_envelope.log`
- Per-N detail logs: `research/findings/raw/direction_R_v3_n{256,384,512}_seed42.log`
- Design doc: `docs/plans/2026-05-25-direction-R-v3-capacity-envelope-extension-design.md`
- Direction M deliverable: `research/findings/2026-05-24-DIRECTION-M-COMPLETE-320-concept-multi-bridge-chat-deliverable-VALIDATED.md`
- Prior R capacity probes: Direction R (90821bc; N=50) + R-extended (375a242; N=192)
