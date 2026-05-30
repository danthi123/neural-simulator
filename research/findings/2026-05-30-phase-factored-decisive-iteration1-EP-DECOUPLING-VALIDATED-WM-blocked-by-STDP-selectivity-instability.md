# Phase-factored integrated loop, decisive iteration 1: the two-phase ENCODE-ORDER DECOUPLING is VALIDATED (ep sound, preserved through the selectivity-move); the WM instrument is blocked by a SUBSTRATE-LEVEL representation-stability problem (STDP selectivity erodes the topographic prior) -- NOT a phase-placement bug. An honest, deep NEGATIVE with a precisely-characterized root cause.

**Date:** 2026-05-30
**Status:** Honest non-success for the FULL integrated-loop instrument, with a GENUINE PARTIAL SUCCESS (episodic-order decoupling validated) and a precisely-localized blocker. NOT a science PASS (the decisive multi-seed run stays unlaunched: v1 wm < 0.90). NOT a pillar. The phase-factored thesis is HALF-validated.

## What was tested + the result

The full-scale grounding probe (2026-05-30) found the integrated-loop instrument unsound (v1 wm=0.5). Diagnosis localized it to a phase/design mismatch: concept selectivity was being trained in Phase 1 IN-ORDER (winner-take-most) rather than Phase 2 SHUFFLED as designed. The corrected fix (commit 06b13c1) restructured the controller exactly as the design prescribes:
- Phase 1 (online, in-order): freeze the selectivity plasticity gates -> writes the ORDER INDEX only (engram + theta-gamma slot order).
- Phase 2 (offline, SHUFFLED): the validated v16 shuffled teacher co-firing + STDP builds concept selectivity, in a cross-mode-identical deterministic shuffle (rng-faithfulness preserved); the SWR consolidation (ca1->concept index update insurance) is kept.

Full-scale v1 re-probe (seed 42, N=2): **V1_EP = 1.000, V1_WM = 0.000.**

## VALIDATED: the two-phase encode-order DECOUPLING (the design's core claim)

Freezing selectivity in Phase 1 (so the in-order pass writes only the order index) did NOT break ep -- it stayed 1.000. This genuinely validates the load-bearing half of the two-phase thesis: **the online theta-ordered episodic index is decoupled from concept-weight learning.** The encode-order conflict that stalled the single-pass loop (selectivity needs shuffle, order needs no-shuffle, contradictory in one pass) IS resolved on the order side -- order is written online, untouched by the offline selectivity training. This is a real result: the phase factorization works for what it was designed to do on the ep side.

## NEGATIVE: the wm instrument is blocked by STDP-selectivity INSTABILITY (a substrate problem, not a phase problem)

Moving selectivity to Phase-2-shuffled did NOT make wm role-selective (0.0). Checkpoint probes (full-scale, on the real controller path) characterized the root cause precisely:

- The **topographic prior ALONE** (no training) already gives CLEAN 2/2 role->filler selectivity (the prior's lang(role)->bound-filler x6 boost works). Phase-1-frozen index-write holds 2/2 selectivity through ~epoch 9.
- But **repeated STDP selectivity-training over the 14 epochs ERODES the prior's clean margin** via cumulative non-selective potentiation: the UNBOUND filler pools creep up epoch-over-epoch and overtake the bound filler by ~epoch 6 (the bound filler's absolute count also grows, but the unbound grow faster).
- This happens in EVERY variant tested: full role+filler co-fire, role-code-only co-fire, with explicit off-target suppression, and with SWR disabled. So it is NOT the dlpfc broad bias, NOT the filler-code confound, NOT the SWR consolidation, and NOT the phase placement or presentation order. It is intrinsic to running selectivity-STDP at the prescribed dose on this substrate.

**The deep tension (the binding-retrieval problem, localized):** the only STABLE selectivity source on this substrate is the topographic PRIOR -- but the prior is LESION-INVARIANT, so it cannot satisfy the lesion-collapse requirement of the verdict (a lesion must collapse the capability it is responsible for; a prior that no lesion touches cannot). The only LESION-ABLATABLE selectivity source is STDP -- but STDP selectivity is UNSTABLE (erodes the prior over training). So there is no mechanism here that is BOTH stable enough to give a sound wm instrument AND lesion-ablatable enough to satisfy the emergence-from-integration verdict. This is the precise, substrate-level form of the binding-retrieval problem the parked loop also struggled with (its wm readout carried extensive unresolved "which pools steal a weak bound pool's activity" diagnostics).

## Connection to the D-arc geometry finding (same substrate theme)

This is the SAME representation-stability problem the Direction-arc geometry diagnostic surfaced: dedicated-pool concept geometry is clean (near-orthogonal) from the topographic prior, but DEGRADES under accumulated training / common-mode noise (the V=320 envelope bend). There, more training eroded cross-bridge orthogonality; here, more STDP erodes role->filler selectivity. Two arcs, one substrate-level finding: **on this substrate, the clean selectivity is the structural prior, and synaptic training (STDP) erodes rather than sharpens it.** That is a real, biology-translatable result about this architecture's learning dynamics.

## Honest disposition + next biology-identified direction (no hand-back)

- The two-phase ENCODE-ORDER DECOUPLING is VALIDATED (ep sound, preserved). The phase factorization works for order.
- The full integrated-loop instrument is NOT soundly buildable as currently wired, because wm role-addressable selectivity has no stable-AND-ablatable source on this substrate. This is an honest NEGATIVE for the FULL loop's instrument soundness, with a fully-characterized root cause. The decisive multi-seed run stays unlaunched.
- Next biology-identified direction: a selectivity mechanism that is BOTH stable and lesion-ablatable. The candidate the D-arc itself pointed to: hippocampal DG pattern-separation as the stable-but-ablatable selectivity source (DG orthogonalizes representations and IS a lesionable subsystem), replacing STDP-on-cortex as the selectivity carrier. OR a consolidation rule that STABILIZES selectivity (homeostatic / anti-Hebbian normalization preventing the unbound-pool creep) rather than the plain Hebbian STDP that erodes it. Either is a deeper redesign of the selectivity carrier, the genuine next step -- NOT a tweak.

## Discipline

No frozen bar moved (integrated_loop_core byte-unchanged). No protected/frozen/moat module touched (byte-empty diff; moat 7/7). No autograd. The fix is the faithful design-aligned restructure; ep soundness genuinely achieved; wm soundness an honest NEGATIVE with a characterized substrate-level cause. NOT overclaimed: the two-phase thesis is half-validated (order decoupling works; selectivity carrier is the open problem). Prior validated results (pillars, the no-confab moat, the D-arc) stand unaffected. Commit 06b13c1.
