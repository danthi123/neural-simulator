# Direction P HONEST: PASS is trivial — multitag is cortex-only; the test does NOT validate sleep consolidation

**Date:** 2026-05-24
**Status:** PASS-AT-FACE / TRIVIAL-AT-INSPECTION — pillar n=105 NOT claimed
**Discipline lesson:** scrutinize PASS harder than FAIL (caught BEFORE pillar claim)

## Headline

Direction P (multitag chat + Phase 1.3 SWR sleep consolidation; cached
HIPPO-OPTION3 substrate) returned multi-seed PASS at retention 1.00
(pre-sleep 8/8 FULL = 1.000; post-sleep with hippo silenced 8/8 FULL
= 1.000; 3 seeds [42,43,44]).

But the PASS is **trivial** when inspected:
- Multitag mechanism uses `region_filter = [concept_pools]` (cortical
  noun/verb/adjective pools), NOT hippocampal regions
- `commit_engram_tag()` places the tag's neurons in CORTEX
- `stimulate_tag()` drives those cortical neurons
- `lang_output_pattern_during_stim` reads cortical lang_output
- Silencing CA3 has **NO direct effect** on a cortical engram tag

The 1.00 retention demonstrates: **multitag retrieval is
hippocampus-independent on the HIPPO-OPTION3 substrate.** It does NOT
demonstrate: "associations transferred from hippocampus to cortex via
SWR consolidation."

## Discipline lesson (caught BEFORE pillar claim)

This is the same pattern as Direction K's reviewer-caught dim-overkill
(commit cf85b5f): a 1.000 PASS that passes "trivially" not because
the claimed biology mechanism is doing the work. Today's autonomous
discipline caught this MYSELF before recording pillar n=105 candidate
(unlike Direction K which required fresh-agent reviewer BLOCK to
catch). The standing protocol works.

## What the test actually demonstrates (honest finding)

**Multitag retrieval mechanism is CORTICAL-ONLY at this substrate
scale.** Specifically:
- Engram tags placed via `commit_engram_tag(region_filter=cortical_pools)`
  live entirely in cortical concept pools
- `stimulate_tag` and `lang_output` readouts operate on cortex
- The hippocampal substrate (CA3 + CA1) is present but UNINVOLVED in
  multitag encoding or retrieval
- Silencing CA3 (-2000 pA clamp) has no functional effect on multitag
- This is CONSISTENT with pillar n=101's finding "hippocampus addition
  doesn't degrade multitag at 91.7%"

The substrate's multitag conversational capability operates entirely
in cortex; hippocampus is wired but functionally orthogonal.

## What would PROPERLY test sleep consolidation

A true sleep-consolidation test would:
1. Encode associations EXCLUSIVELY in hippocampus (e.g., direct CA3
   patterns; engram tag with `region_filter=["ca3","ca1"]`)
2. Verify pre-sleep retrieval depends on hippocampus (silence
   hippocampus pre-sleep → retrieval fails)
3. Run SWR consolidation cycle (CLS mechanism transfers patterns to
   cortex via Schaffer collaterals + CA1 → cortex pathways)
4. Silence hippocampus post-sleep
5. Test if retrieval still works via cortex (proper CLS validation)

Direction P's recipe accidentally bypasses step 1 (uses cortical
engram tags) so the entire experiment is vacuous for the
sleep-consolidation claim. A future Direction P-v2 with proper
hippocampal-only encoding would be the real test.

## Cumulative status (corrected)

After today's complete arc (~125 commits, 2 pillars, 12 mechanism
attempts characterized including this Direction P honest correction),
the precise substrate capability landscape is:

**Validated deliverable conversational capabilities:**
- Multitag chat at 16 concepts on single substrate (91.7%; pillar
  n=100/n=101)
- 320-concept multi-bridge chat (verified Direction M; pre-trained
  G.20 ensemble; honest abstention works)
- Multitag retrieval is CORTICAL-ONLY (Direction P honest finding;
  hippocampus uninvolved)

**Bounded capabilities (pillar n=104 BOUNDARY extended 4x + this
correction):**
- Sequence-position retrieval: bounded across 7+ biology-grounded
  mechanisms
- PFC NMDA bistability: doesn't engage at 60-neuron substrate scale
  (Direction I closed)
- Sleep consolidation of multitag: not testable as currently designed
  (multitag is cortical; nothing for hippocampus to consolidate)

## Discipline preserved

- Bar UNCHANGED at 0.80 multi-seed throughout
- No protected/frozen/moat modification
- HONEST PROPAGATION of every outcome — including this trivial-pass
  catch
- No pillar n=105 claim made on Direction P (despite face-value
  1.00 PASS) because the mechanism doesn't validate the stated
  claim
- Reviewer-style scrutiny applied AT THE TIME OF RESULT, not deferred
- Both remotes pushed
