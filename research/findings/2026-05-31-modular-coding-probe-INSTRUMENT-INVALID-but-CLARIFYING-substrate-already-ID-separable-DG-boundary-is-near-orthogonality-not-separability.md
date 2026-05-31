# Modular-redundant-coding cheap probe = CANNOT-CONCLUDE (instrument-invalid), but the scrutiny that retracted its false auto-PASS produced an IMPORTANT clarification: the raw substrate concept activity is ALREADY 16/16 identifiable (within-concept cosine 0.896 > between-concept 0.768; perfect nearest-neighbour identity with NO DG / projection / coding at all). So (1) the probe's id~1.0 was trivially inherited from the raw activity, measuring nothing about the code; (2) the DG "FUNDAMENTAL BOUNDARY" framing conflated basic SEPARABILITY/identifiability (the activity already has it) with NEAR-ORTHOGONALITY for clean VSA binding (between ~0, a much higher bar) -- the boundary is real for the near-orthogonality bar only; (3) clean rate-based k-WTA reaches between 0.45 (better than the spiking DG's 0.66 but far from near-orthogonal), so neither spiking nor clean coding cheaply yields VSA-near-orthogonal symbols. Converges on accepting the oracle for VSA binding + advancing the validated P4 retrieval stack.

**Date:** 2026-05-31
**Status:** Honest result of the cheap-first modular-redundant-coding probe (the night-synthesis P3(a) "grid/conjunctive code" candidate for the DG boundary). The auto-verdict said RESOLVES; controller scrutiny (scrutinize-a-PASS-harder-than-a-FAIL + bug-discovery-first) retracted it to CANNOT-CONCLUDE and uncovered a clarification that refines my own earlier DG-boundary finding. No spiking grid-module build justified.

## What the probe did + the false auto-PASS

Cheap numpy probe (research/findings/raw/_modular_redundant_coding_probe.py): M independent random-projection k-WTA modules over the SAME cached substrate concept activity the DG drove (denoise64_seed{42,43,44}.npz, store=first-32-obs mean, query=last-32-obs mean, between-concept cosine ~0.77-0.82). Two decoders (concat-cosine + per-module majority-vote = the grid-cell redundant decode). M=1 was meant to be the single-DG control reproducing the boundary failure; M>1 was meant to test whether modular redundancy escapes it.

The auto-verdict printed RESOLVES. CONTROLLER SCRUTINY caught that it was INVALID:

## The two scrutiny catches (why CANNOT-CONCLUDE)

1. THE M=1 CONTROL PASSES (within 0.642 / between 0.454 / id 0.979) at EVERY projection density tested (dense gaussian 1.0, sparse 0.05, very-sparse 0.01) and every M. So the reproduce-the-failure control does NOT reproduce the spiking DG failure (which was within ~0.59 / between ~0.66, unseparated, on a firing-rate metric). A clean DETERMINISTIC rate-based random-projection k-WTA separates+stabilises the activity regardless of M -> the probe cannot isolate any modular benefit (everything passes). Instrument-invalid. (The auto-verdict's bug: it never checked whether M=1 ALSO threads; fixed in the probe -> now prints CANNOT-CONCLUDE.)

2. THE id METRIC IS SATURATED BY THE RAW ACTIVITY. Direct check of the raw store/query mean vectors (no projection at all): within-concept cosine 0.896, between-concept 0.768, and nearest-neighbour identity 16/16 = 1.000. The substrate activity is ALREADY perfectly 16-way identifiable. So id~1.0 under any code is trivially inherited, not a contribution of modular/clean coding.

## The clarification (refines my own DG-boundary finding)

The DG FUNDAMENTAL BOUNDARY finding (2026-05-31-DG-...-FUNDAMENTAL-BOUNDARY) concluded "the substrate's overlapping activity CANNOT be transformed into a separated-AND-stable compositional symbol ... by ANY single competitive-sparse-coding stage." Two corrections from this probe:

- BASIC SEPARABILITY/IDENTIFIABILITY is ALREADY PRESENT in the raw activity (within 0.896 > between 0.768; id 16/16). No DG needed for concept identification. So "cannot be separated" is too strong -- the activity is separable enough for identity/retrieval (which is exactly why the multitag/engram retrieval stack WORKS).
- The genuine bar the DG was failing is NEAR-ORTHOGONALITY (between -> ~0) required for clean VSA/FHRR binding without cross-talk -- a MUCH higher bar than identifiability. The spiking DG reached between ~0.66; a clean rate-based k-WTA reaches ~0.45; neither reaches near-orthogonal (~0). So the honest boundary is: VSA-near-orthogonal separable+stable symbols are NOT cheaply achievable from the substrate activity (spiking loses reliability getting there; clean coding plateaus at ~0.45). The "fundamental to any single competitive-sparse-coding stage" claim is too strong (a clean k-WTA threads the identifiability+reliability bars); the correct claim is the NEAR-ORTHOGONALITY bar is unmet.

- Secondary: the spiking DG's specific failure (within collapses to 0.2-0.3 at separated sparsity) is a SPIKING-DYNAMICS artifact (stochastic near-threshold spike-based k-WTA flips winners between a concept's two halves) -- a DETERMINISTIC top-k on the same inputs is stable. So the DG's instability is implementation-specific, not intrinsic to competitive sparse coding.

## Disposition (converges on P4; no new biological build justified)

- Modular/grid redundant coding: UNTESTED (the probe can't test escape when the control passes + the metric is saturated). NOT pursued further -- a faithful test would need the spiking substrate where M=1 actually fails, and the clarification above shows the real bar (near-orthogonality) is the issue, not module count.
- This CONVERGES on the night-synthesis P3(c): the substrate is ID-separable (sufficient for the validated multitag/engram/G.20 retrieval stack -- the working artifact), but VSA-near-orthogonal binding needs the external (oracle) near-orthogonal code, which is an engineering component. Accept it; advance the validated P4 stack (directional multi-hop shipped this session).
- Honest scientific value: a clean retraction of a false auto-PASS + a correction to my own "fundamental boundary" overclaim (the boundary is the near-orthogonality bar, not separability; and the spiking instability is implementation-specific). Negative/clarifying results under strict scrutiny ARE the deliverable.

## Banners applied
- 2026-05-31-DG-biologization-FUNDAMENTAL-BOUNDARY-...md: "fundamental to any single competitive-sparse-coding stage" refined (the activity is already ID-separable; the unmet bar is near-orthogonality; spiking instability is implementation-specific).
- 2026-05-31-DG-boundary-alternative-codes-survey-...md: the specified modular cheap probe came back instrument-invalid (control passes + id saturated); modular escape untestable cheaply.

## Discipline

Throwaway CPU probe only; no protected/frozen/moat/sim/runner module touched; stdlib+numpy + the cached activity. The frozen instrument-validity-first gate was APPLIED (the probe's auto-RESOLVES was retracted because the M=1 control passed -> CANNOT-CONCLUDE), not the science bar moved. The raw-activity confound (already 16/16 separable) was checked directly before drawing any conclusion -- bug-discovery-first cut both ways (I scrutinised my OWN numpy pass, not just the spiking DG's fail). No spiking build started on the false PASS.
