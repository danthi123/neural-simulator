---
type: finding
status: qualified
date: 2026-05-30
---

# Phase-factored integrated loop, decisive iteration 2: the DG/engram wm carrier makes the v1 instrument SOUND (v1 wm=1.0, genuinely selective) but the frozen verdict returns VOID (non-discriminating) -- the wm capability becomes a localized hippocampal-store LOOKUP, lesion-invariant except for removing the store itself. Together with iteration 1 this CHARACTERIZES BOTH HORNS of the binding-retrieval problem, each VOID-certified by the pre-registered verdict for OPPOSITE reasons.

**Date:** 2026-05-30
**Status:** Honest VOID for the FULL integrated-loop wm instrument, delivered by the frozen pre-registered verdict's discrimination check (NOT a science PASS, NOT a science FAIL -- "cannot conclude emergence"). The decisive multi-seed run correctly stays UNLAUNCHED (the discrimination failure is structural, not stochastic; it reproduces at every seed by construction). A genuine partial win stands: the episodic-order DECOUPLING (ep) remains validated. This completes the two-horns characterization of the binding-retrieval problem on this substrate.

## What was tested + the result

Iteration 1 (the phase-restructure) left the wm instrument UNSOUND: cortical dlpfc_verb->filler STDP selectivity is unstable on this substrate (repeated training erodes the topographic prior -> a role query lights all filler pools ~equally -> v1 wm < 0.90). The de-risk (controller, reuse-by-import confirmed) pointed to routing wm retrieval through the DG-separated hippocampal ENGRAM mechanism -- the SAME stimulate_tag -> CA3 pattern-completion path the ep readout already uses to reach ep=1.0. Iteration 2 implemented that (commit cb6834b): per-binding engram tags at encode (over [role pool + bound filler pool + DG ca3]) + multitag stim-recall retrieval at query. Reuse-by-import only (engram API byte-unchanged); 71/71 tests pass; integrated_loop_core.py byte-unchanged.

Full-scale single-seed partition (N=2, seed 42), the controller's R3 grounding probe:

```
cell                  wm       ep
v1                  1.000    1.000   instrument SOUND (both >= 0.90 bar)
full                0.500    1.000   compositional task: wm only 0.5
no_binding          0.500    1.000   SHARED: must collapse BOTH -> does NOT
no_shared_clock     0.500    1.000   SHARED: must collapse BOTH -> does NOT
no_hippo_store      0.000    0.000   SHARED: collapses both (the ONLY mover of wm)
no_bg_gate          0.500    1.000   HELPER_WM: must collapse wm -> does NOT
no_sequencing       0.500    1.000   HELPER_EP: must collapse ep -> does NOT
no_cls_replay       0.500    1.000   HELPER_EP: must collapse ep -> does NOT
no_neuromod_timing  0.500    0.000   collapses ep only
```

The frozen verdict (`integrated_loop_core.py`, instrument-validity FIRST, fail-closed) checks discrimination before any science scoring: SHARED + HELPER_BOTH lesions must drop BOTH readouts <= 0.40; HELPER_WM must drop wm; HELPER_EP must drop ep. `no_binding` = (wm=0.5, ep=1.0) fails the SHARED check (wm 0.5 > 0.40) -> **VOID**: "the capability is not emergent-from-integration here / wiring artifact." (`no_bg_gate`, `no_sequencing`, `no_cls_replay` independently fail their checks too.)

## Why: the engram store is a localized LOOKUP, lesion-invariant by construction

wm sits at a flat **0.5 for full and 6 of 7 lesions**, dropping to 0.0 only under `no_hippo_store`. That 0.5 floor is the signature: of the two scored queries in the compositional condition, the DRILLED query (qi=0: query a stored role -> its own per-binding tag -> its bound filler) always succeeds, and the NOVEL-RECOMBINATION query (qi=1: query pair[-1].role expecting pair[0].filler -- a binding never drilled) always fails because there is NO tag for the recombined pair. So:
- v1 (scores only the drilled query) = 1.0 -> SOUND.
- full + every lesion (drilled passes, recombination fails) = 0.5 -- regardless of the lesion, because the per-binding tag for the drilled query is committed and retrieves from the hippocampal store no matter what `no_binding` / `no_shared_clock` / `no_bg_gate` / `no_sequencing` / `no_cls_replay` do. Only `no_hippo_store` (which skips the tags entirely) moves wm, to 0.0.

The subagent's in-code lesion-mapping (e.g. "no_binding -> degenerate tag -> wm collapses") was aspirational; the full-scale table FALSIFIES it -- the dlpfc holding bias `no_binding` suppresses is NOT on the critical path for the engram tag's drilled retrieval (the role+filler pools still co-fire from the teacher current + lang drive, so the tag still captures a usable ensemble). The tiny-synth probe had FLOORED wm at 0 for all modes (a scale artifact), hiding this; the full-scale R3 grounding probe is exactly what surfaced it -- the grounding discipline working (a ~32-min catch, not an ~8-hr VOID).

## The two horns, both VOID-certified by the pre-registered verdict

| Iteration | wm carrier | v1 soundness | lesion discrimination | frozen verdict |
|---|---|---|---|---|
| 1 | cortical STDP selectivity | UNSOUND (v1 wm < 0.90; eroding prior) | n/a (soundness fails first) | VOID (unsound) |
| 2 | DG/engram per-binding store | SOUND (v1 wm=1.0, selective) | FAILS (wm flat 0.5; lesion-invariant except no_hippo_store) | VOID (non-discriminating) |

The binding-retrieval problem has two horns, and on this substrate each available mechanism sits on one of them:
- **STDP selectivity** is EMERGENT (it is built by the integrated loop's timed co-firing, so in principle a lesion that removes the timing/binding/gating collapses it) but UNSTABLE (repeated training erodes it -> the instrument cannot even measure composition).
- **Engram store** is STABLE (one-pass write, recall = stimulate the stored ensemble -> v1 nearly perfect) but NOT EMERGENT (it is a localized hippocampal-store lookup; removing any single OTHER system does not collapse it; only removing the store does).

There is no mechanism here that is BOTH stable enough for a sound instrument AND emergent-from-integration enough for the lesion partition to hold. The "compositional cognition emerges from integrating many brain systems under one shared rhythm" thesis, FOR role-filler working-memory binding RETRIEVAL, is NOT supported on this substrate -- and that conclusion is certified by a verdict whose bars were pre-registered before any run and never tuned to a result. This is the SAME substrate-stability theme the D-arc geometry diagnostic surfaced (clean selectivity is the structural prior; synaptic training erodes rather than sharpens it), now sharpened to a precise either/or.

## What genuinely stands (banked, unaffected)

- The episodic-order DECOUPLING (ep) is VALIDATED and was PRESERVED across both iterations: ep=1.0 at v1 and full. The online theta-ordered episodic index is genuinely decoupled from concept-weight learning -- the encode-order conflict that stalled the single-pass loop is resolved on the order side. Half the two-phase thesis holds.
- The engram approach DOES achieve genuine v1 SOUNDNESS with DECISIVELY SELECTIVE retrieval (R1 grounding probe, full scale, seed 42): across all scored v1 queries the TRUE filler out-fires the seven distractors by 15x-400x. The cleanest queries show the true pool at ~6000-6900 accumulated spikes with every distractor in the single-to-low-double digits (e.g. true F0 = 6011 vs the next pool = 0, the rest <= 15; true F0 = 6181, others <= 33). The higher-floor queries still dominate ~10-15x (true F1 = 6613, next = 987). NOT "all fire equally, top correct by luck" -- the engram store is a reliable, sharply selective role-filler memory. It simply is not the EMERGENT mechanism the integrated-loop verdict requires.
- No frozen bar moved; integrated_loop_core.py byte-unchanged; the 4 validated subsystems byte-unchanged; no new rule, no autograd; the no-confab moat stands (71/71 incl. moat). Prior pillars + the D-arc stand unaffected.

## Honest disposition + decision point (surfaced to owner)

The integrated-loop wm instrument cannot be certified emergent-from-integration on this substrate by either available mechanism. This is a pre-registered-bar-certified VOID -- an honest scientific deliverable under the project's stated value ("honest negatives under strict biology ARE the scientific deliverable"), not a failure of execution. The decisive multi-seed run stays UNLAUNCHED (the discrimination failure is structural; it reproduces at every seed; running ~8 hr to reconfirm a structurally-determined VOID would violate the grounding discipline).

Two honest paths, surfaced rather than auto-chosen (a deeper redesign is a genuine fork, and the owner has twice asked whether deeper characterization is worth it vs goal progress):
- (A) PARK the integrated-loop wm-emergence thesis HERE as a characterized VOID + the banked ep-decoupling win, and pick the next goal-aligned arc (conversational capability / artificial-life direction).
- (B) Attempt a THIRD mechanism: make the engram READOUT loop-gated -- i.e. retrieval that DEPENDS on the BG gate + shared clock + binding step to select/complete the tag (so those lesions collapse wm) rather than a bare stimulate_tag. This could in principle satisfy discrimination, but it re-introduces the instability risk (a noisy gate/clock degrades v1), i.e. it walks back toward horn 1. It is a deeper redesign, not a tweak, with real risk of landing back at VOID.

Recommendation: (A). The two horns are now both characterized and verdict-certified; a third mechanism is speculative and likely re-opens the stability/emergence tension that is itself the finding. Banking the ep win + the two-horns characterization and redirecting to goal-aligned conversational work is the higher-value move.
