---
title: "3-way neural superposition composes zero-shot at arity 3 (GO, 3-seed) — but on DISJOINT channels, so the bundling-capacity limit is NOT yet stressed"
date: 2026-08-09
type: finding
status: contributing
lane: composer
seeds: [42, 43, 44]
seed-waiver: 3-seed concrete scope (the retry after the first arity attempt failed on the agent StructuredOutput cap); zero-shot held-out recall is 1.00 on all 3 seeds (perfect, +0.96 above the chance floor), verify CONFIRMED. The GO is BOUNDED (disjoint-channel case) — the honest limit, not the magnitude, is the point; a 6-seed run adds little to a 3/3 ceiling result whose real caveat is scope (disjoint channels), not seed variance.
---

# Arity-3 composition is GO on disjoint channels; the real capacity limit (interfering terms) is untested

## Claim (GO with an honest scope caveat the verify surfaced)

<!--derived-->

Real facts have many attributes, not two. Extending the zero-shot composer to arity 3 (facts = (a,b,c), regenerate
= SUM of THREE per-primitive spiking-readout outputs): **zero-shot held-out recall = 1.00, 3/3 seeds** (K=4, N=64,
16 held out, chance 0.016), **equal to the arity-2 baseline (1.00, measured in the same run)**, floors at chance
(v2 0.042, flat 0.021; margin +0.96/+0.98 ≫ the +0.30 gate). Composition neural (3-way superposition, 3-way lesion
localises each block, 0 stored patterns, no leakage — verified). **GO on the capability (arity3_go=True 3/3).**

## The caveat (why this is a BOUNDED GO, not a capacity result)

<!--derived-->

The verify (CONFIRMED, not refuted) flagged the honest limit: **the three terms occupy DISJOINT channel index
ranges, so the SUM is really a CONCATENATION with zero inter-term crosstalk** (the exact-0.0 lesion sparing proves
it). So **arity-3 recall equals arity-2 BY CONSTRUCTION** — adding a third *non-interfering* attribute costs nothing,
but this does NOT stress the **bundling-capacity margin** (~1/√(#terms), Plate/Kanerva), which only bites when terms
SHARE channels and interfere. So the result shows: **multi-attribute composition works when attributes are separable
(disjoint), zero-shot** — a real, useful property — but the capacity limit (where superposition finally fails with
many interfering terms) is the untested next question, best probed by a larger-K / shared-channel arity sweep (the
composer-interference direction, `5ae5f8979`, applied to arity).

## Honest notes

<!--derived-->

- Verdict machine-status was GO 2/3 (s44 UNDEFINED) SOLELY from a spurious `sim_diff_empty=False` repo-hygiene flag
  (a pre-existing branch divergence from the dendritic-bind sim edit, NOT this arity work — `git diff HEAD -- sim/`
  empty; neither arity-3 commit touches sim/). The CAPABILITY is GO 3/3; substrate byte-identical all seeds.
- Small/concrete scope (K=4, 3 seeds, numpy) per the retry's tighter framing after the first arity attempt failed on
  the agent StructuredOutput cap. Standing composer idealizations (frozen random reservoir, host NLMS readout fit,
  host running-mean per primitive) carry over from the arity-2 GO unchanged.

Artifacts: `research/findings/raw/teacher_loop_arity3_composition_AGG.json` (+ s42/s43/s44). No `sim/` edit.

NEXT: a SHARED-channel arity sweep (interfering terms, larger #terms) to LOCATE the bundling-capacity limit — the
point where superposition needs binding even for same-attribute-type composition. NO-EXTERNAL-NEEDED: Plate/Kanerva
VSA capacity is the recorded grounding.
