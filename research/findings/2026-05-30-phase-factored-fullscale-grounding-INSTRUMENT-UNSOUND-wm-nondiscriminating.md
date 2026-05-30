# Phase-factored full-scale grounding probe: INSTRUMENT UNSOUND -- wm (concept-query) readout is near-chance and non-discriminating at full scale; ep collapses for the genuine composition; the decisive run is NOT ready (would VOID). Diagnosis before any multi-hour run.

**Date:** 2026-05-30
**Status:** NOT a science result. A single full-scale grounding cell (N=2, seed 42; 32.2 min) surfaced an instrument-soundness failure BEFORE the ~8 hr decisive run. The frozen verdict would VOID (v1 wm = 0.5 < the 0.90 soundness bar). This is the grounding/timing probe doing its job. Diagnosis is the next step (iterate the integration fidelity, per the design's no-hand-back discipline).

## What the probe showed

Full-scale single cell, N=2, seed 42 (research/findings/raw/phase_factored_fullscale_timing_probe.txt):
```
v1   {wm: 0.5, ep: 1.0}
full {wm: 0.5, ep: 0.0}
lesions: no_binding (0.5,1.0) no_shared_clock (0.5,1.0) no_hippo_store (0.5,0.0)
         no_bg_gate (0.5,1.0) no_sequencing (0.5,1.0) no_cls_replay (0.0,1.0)
         no_neuromod_timing (0.5,0.0)
```
Wall: 32.2 min/cell -> the full decisive run (3 loads x 3 seeds, N=8 heaviest) would be ~6-10 hr.

## Two distinct problems (honest diagnosis)

**Problem 1 (BLOCKER) -- wm readout is non-discriminating.** wm sits at 0.5 (chance for the 2-filler query) for v1, full, and 5 of 7 lesions. Critically, the lesion that SHOULD collapse wm -- no_bg_gate (the working-memory helper) -- does NOT (wm stays 0.5). A working wm readout would show no_bg_gate wm <= 0.40. So wm is not measuring real role->filler selectivity; it is near-chance noise. v1 wm = 0.5 < 0.90 fails instrument soundness -> the verdict VOIDs. The one anomaly (no_cls_replay wm = 0.0) is the only sub-chance value and does not fit the wm helper-partition either. This is a wm-side wiring/scale issue at full scale, NOT the composition science (which cannot be measured while the instrument is unsound).

(The Task-4 adversarial review correctly predicted the tiny-synth wm=0.0 was a gate-abstention scale artifact and that full-scale wm depends on "whether the binding produces correct role-selective filler above 650." The full-scale probe answers: it does NOT yet -- wm is at chance even on the trivial v1 bind. So the prediction's conditional resolved negative, which is exactly the instrument-soundness question the probe was meant to settle.)

**Problem 2 (substrate caveat materialized) -- ep collapses for the genuine task.** ep = 1.0 for v1 (trivial no-gap bind) but 0.0 for full (genuine gapped composition). The order index survives the trivial bind but NOT the gap+consolidation of the real task. This is precisely the residual-coupling the cheap probe could not settle (its toy had near-orthogonal reps with tiny common-mode; the D-arc measured real reps have LARGE common-mode 0.18-0.68, so consolidation moves them substantially and the stored-content index nearest-match fails). The consolidation-updates-index insurance, as wired, is not sufficient at full scale -- OR the gap itself (delay between encode and readout) breaks it. This is the scientifically interesting failure mode, but it is confounded by Problem 1 (cannot measure composition while wm is unsound).

## Why this is a good catch, not a failure

The grounding/timing probe spent 32 min to reveal the instrument is unsound, instead of ~8 hr producing a VOID. This is the cheap-first / grounding discipline working exactly as designed. The decisive multi-seed run is correctly NOT launched on an unsound instrument.

## Next step (diagnosis, no hand-back)

Per the design's iterate-following-biology discipline:
1. Diagnose Problem 1 (wm non-discriminating). The wm readout is a structural copy of the parked validated runner's, but the parked runner was a different loop; the role->filler binding + query + gate scale must be checked AT THIS full scale for THIS two-phase controller. Targeted single-cell probes (32 min each) varying one factor (e.g. teacher/binding drive, readout window, gate threshold calibration for these pools, query timing).
2. Diagnose Problem 2 (ep survival) only after wm is sound -- likely the consolidation-updates-index path needs to genuinely re-bind the index post-drift, or the EP readout must be taken with index-update rather than relying on the stored old content.
3. Each fix is an integration-fidelity correction (reuse-by-import preserved; no frozen bar moved; no protected module touched). A clean v1-sound instrument is the prerequisite for any science verdict.

## Discipline

No bar moved. No protected/frozen/moat module touched (the controller + verdict are unchanged; this is a probe finding). No science claim -- the instrument is unsound, so the composition question is unmeasured. Honest: the decisive run is not ready; the wm readout needs fixing first; the ep substrate-caveat is real and awaits a sound instrument to measure.

## Root-cause diagnosis (code inspection, post-probe)

The wm readout (phase_factored_loop_gate.py:400-443) drives a ROLE code onto language_input and ranks the cortical FILLER pools (noun_pool_F*) by firing, gated at DEFAULT_THRESHOLD=650 -- i.e. it is a role->filler BINDING query ("what filler is bound to this role?"), inherited structurally from the parked runner. For it to score above chance it needs a learned role->filler association that reactivates the correct filler when the role is queried. Two candidate root causes (a distinguishing probe is the next step):

- **Hypothesis A -- the role->filler binding is not built/maintained.** The single-pass parked loop drilled role->filler via teacher-forced training epochs. The phase-factored split replaced that with Phase 1 (one-shot engram tag of the co-firing ensemble) + Phase 2 (offline consolidation). A one-shot engram tag is an index, not a drilled synaptic role->filler pathway; if neither phase strengthens lang(role)->filler_pool for the specific binding, the role query retrieves a filler only at chance -> wm=0.5. The working-memory query may need the PFC working-memory maintenance arm (dlpfc) to hold the role->filler binding, which is listed in the design but may not be wired into THIS readout path.

- **Hypothesis B -- Phase 2's consolidation mechanism builds the wrong selectivity.** The parked iteration-4 built queryable selectivity with the validated train_word_to_pool (shuffled co-firing + topographic prior). Phase 2 here uses run_concept_replay_phase (SWR replay of engram tags) -- a DIFFERENT mechanism that consolidates hippocampus->cortex but may not strengthen the lang(role)->filler_pool readout pathway the wm query reads. If so, the offline phase builds the wrong thing for this readout.

Distinguishing probe (next): measure wm right after Phase 1 (before Phase 2), and with Phase 2 swapped to train_word_to_pool vs run_concept_replay_phase. If wm is sound after Phase-1-only or with train_word_to_pool, the issue is Phase 2's mechanism (B); if wm is chance even then, the role->filler binding/maintenance is missing (A).

Either way this is an integration-fidelity correction (reuse-preserved; no frozen bar moved; no protected module touched), exactly the design's iterate-following-biology step. The ep substrate-caveat (Problem 2) is downstream of a sound wm instrument.
