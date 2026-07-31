---
type: plan
status: live
date: 2026-05-25
---

# Direction R-v3 design: Direction M capacity envelope extension at 256 / 384 / 512 associations

**Date:** 2026-05-25
**Status:** Brainstorm/design pass (pre-staged while D3 V=32 smoke trains; per user ordered direction Q -> 3 -> 4 -> R, this is the final direction; cheapest probe ~10-15 min CPU per cell)

## Goal

Extend the Direction R capacity envelope characterization of the
Direction M 320-concept G.20 multi-bridge chat deliverable. The
prior probes:
- Direction R (commit 90821bc): 50 associations -> top-1 80% / top-3 90%
- Direction R-extended (commit 375a242): 192 associations -> top-1 45% / top-3 95%

The pattern: top-3 stays at 90-95% even as top-1 degrades. The
substrate's honest abstention property is intact (doesn't confabulate
at high capacity; correct answer reachable via top-3).

**Open question**: at what N does top-3 fall below 80% (the project's
frozen multi-seed bar)? Localizing this gives the production-quality
capability envelope of the working deliverable.

## Biology-translatable framing

Real biology has graceful capacity degradation: as memory load
increases, retrieval precision drops gradually (not catastrophically).
The Direction M deliverable shows this exact pattern at 192 assoc.
Pushing to 256 / 384 / 512 measures the SLOPE of the degradation
curve, which is biology-translatable: cortical memory networks have
similar capacity envelopes (cf. Brunel-Wang mean-field analyses of
attractor network capacity ~0.14 N where N = neuron count).

## Approach

Reuse the Direction R-extended probe pattern byte-unchanged: 
`g20_multibridge.py --sparse` runner with `--scripted` flag. Generate
N "remember X is Y" commands followed by 20 queries on the cross-
bridge ensemble; measure top-1 / top-3 retrieval accuracy.

The probe is CPU-bound for inference (the bridges are pre-trained
and cached); GPU is used briefly during bridge load. ~10-15 min per
N value.

## Pre-registered test + bar

**Test**: for each N in {256, 384, 512}:
1. Load 5 G.20 sparse bridges (cached at production tier; bridges
   from 2026-05-16 commit)
2. Encode N "remember X is Y" associations sequentially via scripted
   commands
3. Query 20 randomly-selected associations
4. Measure top-1 and top-3 accuracy

**Bar UNCHANGED at 0.80** (project's frozen multi-seed bar):
- `DIRECTION_R_V3_PASS_AT_N`: top-3 >= 0.80 at the tested N
- `DIRECTION_R_V3_BOUNDARY_AT_N`: top-3 < 0.80 (capacity envelope
  edge identified)

The bar is per-N; the envelope characterization reports the
multi-N table.

## Cost estimate

- N=256: ~10-12 min CPU/GPU
- N=384: ~15-18 min
- N=512: ~20-25 min
- Total Direction R-v3 (3 cells): ~45-55 min wall

Cheap; can run immediately after D3 V=32 smoke frees GPU OR can
run in parallel if GPU has headroom. The bridges are ~10GB GPU
when all 5 loaded.

## Files

- Direction R-v3 runner (already exists implicitly): use
  `research/runners/g20_multibridge.py --sparse --scripted ...`
  with N "remember X is Y" + 20 queries
- Direction R-v3 launch script:
  `research/findings/raw/direction_R_v3_launcher.py` - generates
  scripted commands at each N + invokes g20_multibridge
- Direction R-v3 verdict aggregator:
  `research/findings/raw/direction_R_v3_aggregator.py` - reads
  per-N results + writes envelope characterization JSON

## Implementation note

The original Direction R / R-extended runners are not in the repo
as standalone .py files (only the .log + .json output); they were
ad-hoc scripted command sequences. Direction R-v3 should formalize
the pattern as a reusable launcher + aggregator so the envelope
characterization is reproducible.

## Continuation pointer

When D3 V=32 smoke completes AND D3 production run completes (if
launched) AND D4 GPU training completes (if launched), the GPU is
free for Direction R-v3. Cheapest cell first (N=256), then N=384,
then N=512. Report envelope shape in the findings doc.

Alternative: if D3 PARTIAL and Direction 4 is pivoted-to instead,
Direction R-v3 may be runnable in parallel with D4 training (D4
trains 5 bridges sequentially; R-v3 only loads 5 already-trained
bridges briefly).

## Discipline

- Bar UNCHANGED at 0.80 (top-3 only; top-1 is informational)
- No protected/frozen/moat modification
- No autograd
- CPU/GPU as needed; brief GPU for bridge loading + inference
- Honest propagation EVERY outcome both remotes
- The Direction R-v3 findings doc reports the multi-N table
  honestly: the envelope edge is itself the biology-translatable
  finding even if no single N PASSes
