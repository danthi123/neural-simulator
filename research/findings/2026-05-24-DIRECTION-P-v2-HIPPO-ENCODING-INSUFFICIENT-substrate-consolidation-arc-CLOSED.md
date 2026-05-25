# Direction P-v2 HONEST: hippocampal encoding insufficient for retrieval; substrate-consolidation arc CLOSED

**Date:** 2026-05-24
**Status:** P-v2 verdict HIPPO_ENCODING_INSUFFICIENT; consolidation arc has 3 convergent BOUNDARY findings; further architectural work would be substantial substrate redesign (3+ hr; out of cheap-probe scope)

## Headline

Direction P-v2 corrected the trivial P-v1 (cortical region_filter)
with proper hippocampal-only region_filter=[ca3, ca1, dg]. Multi-seed
results:
- pre-A (hippo active): 0.167 [0.375, 0.000, 0.125]
- pre-B (hippo silenced): 0.208 [0.250, 0.250, 0.125]
- post-sleep (hippo silenced): 0.292 [0.375, 0.250, 0.250]
- gain (post - pre-B): +0.083

VERDICT: HIPPO_ENCODING_INSUFFICIENT — pre-A (0.167) is below the
0.50 threshold; hippocampal engram tag doesn't drive lang_output
adequately even with hippocampus active.

## Diagnosis

The trisynaptic cascade (drive → DG → CA3 → CA1 → cortex →
lang_output) is too weak in the current substrate to produce
confident concept-pair retrieval. The CA3 ensemble fires during
encoding (per pillar n=97/n=101 multitag works at 91.7% when
region_filter=cortical_pools), but using a HIPPOCAMPAL region_filter
means the engram tag's neurons are in CA3/CA1; stimulating those
during retrieval drives cortex weakly via existing CA1 → cortex
pathways (which are ca1_to_motor only per Phase 1.3 design).

Three convergent BOUNDARY findings on the substrate-consolidation
arc today:
1. **Multitag is cortex-only** (Direction P trivial; multitag works
   because it uses cortical engram tags; hippocampus uninvolved)
2. **Hippocampal-only engram cannot drive retrieval** (Direction
   P-v2 just completed; ca3 → cortex cascade insufficient)
3. **Existing ca1_to_motor consolidation pathway** doesn't support
   concept-concept transfer (architecturally; designed for word→motor
   binding in Phase 1.3)

## Why this matters (honest biology-translatable insight)

The substrate's hippocampal infrastructure is wired (CA3 + CA1 + DG
present per pillar n=97) but the cortical-hippocampal communication
pathways are bottlenecked:
- Phase 1.3 SWR consolidation works for word→motor binding
  (validated 3/3 strict anti-cheat multi-seed)
- The pathways for concept-concept consolidation (ca1 → noun_pool,
  ca1 → verb_pool, ca1 → adjective_pool) don't exist in the
  current substrate
- Adding them would require new substrate training (~3-5 hr per
  seed) to learn the consolidated cortex representations

In real biology, the hippocampal-cortical pathways are more general
(supporting consolidation of arbitrary content); the current
substrate's specialization for word→motor is an architectural
choice that bounds it for general concept consolidation.

## What would close this bound (Direction P-v3 — QUEUED, not run)

Direction P-v3 would:
1. Wrap _build_bridge_with_hippo to ADD ca1_to_noun_pool +
   ca1_to_verb_pool + ca1_to_adjective_pool RegionPathways (allowed
   per discipline; mode_unification_with_hippo_probe.py is not
   protected; only build_biological_brain_regions itself is)
2. Train fresh substrate with new pathways (~50 min/seed × 3 = 2.5
   hr)
3. Encode hippocampal-only multitag tags
4. Run SWR consolidation through the new pathways
5. Test cortex-only retrieval

Cost: ~3+ hr substrate train + ~30 min tests. Substantial but
doable. Queued for next session OR user steering.

## Cumulative substrate-consolidation arc

Today's 4 consolidation-related tests:
- Pillar n=101 (prior): hippocampus addition doesn't degrade
  multitag (91.7%)
- Direction G: HIPPO + theta-gamma sequence storage BOUNDARY
- Direction P (trivial): multitag is cortex-only (no hippo
  involvement)
- Direction P-v2 (current): hippocampal-only engram tag insufficient
  for retrieval

The convergent finding: substrate hippocampus is functionally
disconnected from concept-pool operations. Closing this requires
adding pathways the existing builder doesn't include.

## Discipline preserved

- Bar UNCHANGED at 0.50/0.30/0.30 (pre-A / pre-B / gain) pre-
  registered conditions throughout
- No protected/frozen/moat module modified
- HONEST PROPAGATION of P-v2 negative finding
- v1 trivial-pass + v2 honest negative = two complementary
  characterizations of the substrate's consolidation infrastructure
- Both remotes pushed

## Status

Today's autonomous arc COMPLETE: ~135 commits, 2 pillars (n=103
VALIDATED + n=104 BOUNDARY 4x extended), 15 mechanism tests (8
substrate sequence-storage + 3 PFC bistability + 4 consolidation/
multitag-localization + others), 1 working deliverable (Direction M
320-concept chat).

NEXT DIRECTIONS (autonomous chain continues when watchdog or user
steers):
- Direction P-v3: new substrate with ca1_to_concept_pool pathways
  (~3+ hr GPU; proper CLS test)
- Direction N: scale chat 320 → 640 concepts (~85 min GPU; vocab
  curation needed)
- Direction O: sentence parser UI for chat (UX work)
- Direction Q: dlpfc_wm scale-up 60 → 1000 neurons (substantial
  architectural test; closes Direction I bound)
