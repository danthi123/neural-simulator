---
type: plan
status: live
date: 2026-05-31
---

# Preparation: Direction (A) richer representation learning at scale -- 2026-05-31

**Owner decision:** pursue (A) -- learn richer, less-overlapping concept representations to push word
recognition past the ~28-word / 54%-concept-wins wall -- "but spend time in preparation ensuring we make
the most of all the time spent on compute."

This doc is the preparation: it ensures the expensive run (~100 GPU-hr if needed) is launched only on a
de-risked premise, with the right objective, efficient compute, and pre-registered bars. **Status: IN
PROGRESS** (external survey DONE; internal asset map running; cheap-first gates next).

## The problem, precisely (from tonight's de-risk)
- 28-word pool-label recognition 0.571; 16-word 0.812. Cheating-audit finding.
- The wall is NOT a readout artifact: even among concept pools only (motors excluded), concept words win
  just 13/24 = 0.54, and down-weighting motors makes it WORSE. So the substrate's LEARNED concept
  representations become too overlapping as vocabulary grows -- an upstream representation-learning problem.
  Finding 2026-05-31-frontend-wall-not-cheap-motor-rebalance-needs-redesign.md.

## External survey reframe (the compute-protecting finding)
Full survey synthesized 2026-05-31 (background agent). The decisive reframe:
- The project's already-bounded separability mechanisms (hippocampal DG k-WTA ~0.66; Foldiak learned
  decorrelation 0.30 but over-sparsified/dead-codes; fixed random projection 0.45 floor) ALL attack a
  DIFFERENT problem: post-hoc READOUT-TRANSFORM of the substrate's already-fixed overlapping activity toward
  VSA near-orthogonality (cos->0, a HIGH bar). NONE changes the UPSTREAM learned representation.
- The 54%-wins limit is upstream representation learning -> the bounded mechanisms do NOT subsume it.
- Ruled out: predictive coding (~100x costlier than BPTT); SPA (= the G.20 Kanerva oracle, assigns not
  learns -- subsumed); sparse coding (= bounded Foldiak).
- TWO genuinely-untried, biologically-grounded, NON-100hr levers target the actual cause:

  1. **Expansion recoding + Hebbian refinement** (Lindsay 2017 / Rigotti-Fusi 2013 mixed selectivity):
     random high-dim expansion (mixed selectivity) THEN a LOCAL multiplicative Hebbian rule that
     expands/decorrelates the per-concept reps (paper: decoding 70.5% -> 83.2%). Cheap, local, no gradient.
     Genuinely different from fixed random projection (no learning) and Foldiak (sparsifies outputs).
  2. **e-prop** (Bellec-Maass three-factor local rule approximating online BPTT): constant-memory, strongly
     biological (three-factor = the project's existing paradigm). On small-vocab keyword spotting (Google
     Speech Commands, directly analogous) ~91% from scratch in real-time; within a few % of BPTT. The
     project's Phase-2 falsification used full BPTT, NOT e-prop -> genuinely untried.

## The compute-protecting GATE (pre-registered)
**The ~100hr BPTT is NOT yet earned.** Run the two cheap-first falsification gates FIRST:

- Gate 1 (expansion+Hebbian, CPU, on cached 28-concept activity): random expansion to ~4-8K dims +
  multiplicative Hebbian refinement -> nearest-neighbor concept classification. PASS if it beats the
  fixed-expansion baseline AND lifts concept-separability notably above the 0.54 wall. ~minutes.
- Gate 2 (e-prop SRNN, ~1 GPU-hr, toy 28-word classification): PASS if it beats 0.54 concept-wins.
- If EITHER passes -> a cheaper-than-100hr path to richer reps exists -> pursue it, 100hr avoided/deferred.
- If BOTH fail -> the 100hr representation-learning commitment is EARNED (and we design it efficiently).

Input data for the gates: research/findings/raw/_28concept_activity_seed42.npz (multi-sample per-neuron
concept activity, captured 2026-05-31).

## Compute-efficiency plan (for the run that does get launched) -- TO FINALIZE with the internal asset map
- Live separability instrumentation + KILL-SWITCH: measure concept-separability DURING training; abort a run
  that is not improving (do not burn hours on a failing run).
- Checkpoint/resume (sim/lineage.py + save_checkpoint/load_checkpoint) so a crash never loses hours.
- GPU throughput: the project documented ~25% GPU utilization historically (perf roadmap); maximize before
  scaling. Cloud H100 option exists (scripts/deploy_to_cloud.sh, docs/plans/2026-05-05-cloud-h100-
  deployment.md) for the big run.
- Pre-registered FROZEN bars + anti-cheat controls (permuted-label / untrained controls) so the result is
  trustworthy, not an uninterpretable burned run.
- Reuse-by-import; no protected/frozen/moat-module modification; honest propagation of every outcome.

## Internal asset map (DONE) -- sharpens the plan substantially
Key findings (full report in the session; sources cited there):
- **BPTT is ALREADY decisively bounded for THIS goal.** Phase 2.3a (134K params) next-char pretraining
  gave 22% W->A (< 28% random init); Phase 2.3b (50M params) made it WORSE (inter-word cosine 0.72->0.85)
  -- "scale fixes it" FALSIFIED. The char-level objective produces MORE overlapping features for
  phonetically-similar words at larger scale. Q4 concept-level-objective probe = VOID at cheap scale. So a
  naive BPTT scale-up is the WRONG big bet.
- **The existing contrastive runner is NEGATIVE** (current-injection contrastive made the motor-N bias
  worse, 12.5% W->A). Not reusable as an objective.
- **The decisive lesson (b):** the near-orthogonality floor (~0.48 between-concept) is set by the
  substrate's intrinsic per-pair overlap (~0.75), FLAT across N=4..16, and is NOT moved by any coding stage
  that operates on the SAME activity (DG / Foldiak / random all bounded). "A richer-representation run that
  still operates on the same substrate activity will hit the same floor. The only path to near-orthogonal
  codes is concepts whose activity is less-overlapping BY CONSTRUCTION during acquisition (different
  training distribution or a much larger model)." -> So the survey's expansion+Hebbian lever (operating on
  the same activity) is LIKELY in the bounded class; the cheap-first must check this, not assume it.
- **The compute-protecting fact:** the substrate's 16-concept ACTIVITY is already 100% nearest-neighbor
  identifiable (within 0.896 > between 0.768) even though pool-argmax recognition is only 81%. So the
  "front-end wall" may be a LOSSY-READOUT artifact (pool-argmax collapses each 200-neuron pool to a scalar),
  not a representation limit. **GATE 1 tests exactly this** -- if the 28-concept codes are highly
  decodable, the fix is a better readout (cheap, NO 100hr).
- **If representation learning IS needed, BPTT is the wrong tool.** The internal map's evidenced
  alternatives for spent compute: (1) scale the validated G.20 sparse-distributed architecture (160/320 ->
  640+, D8 infra scaffolded); (2) the VSA gain-field role-binding with a few DISTRIBUTED near-ortho ROLE
  codes (the only composition path with a positive result, 1.000 at K=4; needs near-ortho ROLES not
  FILLERS). These build on validated positives and do NOT repeat bounded work.
- Reusable byte-unchanged: sim/bptt_snn.py (numpy ref), scripts/deploy_to_cloud.sh, BridgeLineage
  (checkpoint/resume), concept_pool_demo --save/load-bridge, the denoise64 caches. Perf: GPU util ~30-50%
  (memory-bandwidth-bound sparse SNN); fp16_synapse_state + reset/stim trims ~2.7x before cloud; H100 ~6-8x.

## ⛔ VERDICT RETRACTED 2026-06-01 — GATE 1 = DONE, VALIDATED: the 28-word wall is a REAL representation limit (NOT a cheap readout fix)
Finding 2026-05-31-GATE1-frontend-wall-is-a-real-representation-limit-at-28-words.md ⛔ (RETRACTED, verdict only, 2026-06-01 — the "representation limit" conclusion was confounded by an UNDERTRAINED bridge: the _v17 28-word bridge saw ~50 events/word vs the 16-word control's 200, and a matched 150-event 28-word bridge gives clean recognition 0.893, not 0.64. The pipeline + the 16-word control below stand; only the cross-vocab representation-limit conclusion dies. See `research/findings/2026-06-01-GATE2-overturns-GATE1-28word-wall-is-undertraining-not-representation-limit.md`). With a validated
pipeline (16-word positive control reproduces NN 0.91 > pool-argmax 0.80 at k=1, both 1.000 at k=4 = the
internal map's "lossy readout / 100% identifiable"), the 28-word fair head-to-head shows the full-code
decoder is WORSE than pool-argmax at every averaging level (k=4: 0.527 vs 0.402), plateauing at ~0.53-0.64
(NOT 1.000 like 16 words). It is OVERLAP not noise (averaging doesn't fix it). So the lossy-readout escape
that works at 16 words does NOT extend to 28 -> a genuine representation-capacity transition. Cheap readout
fix is OUT; representation learning at ACQUISITION is genuinely needed. (Several intermediate runs were
caught as capture-faithfulness bugs and corrected -- the discipline.) -> The 100hr is warranted IF an
acquisition-level lever is needed, but must target G.20-scaling or VSA-roles or acquisition-level e-prop /
expansion+Hebbian -- NOT the bounded BPTT, NOT post-hoc transforms.

## Original gate order (Gate 1 now resolved -> representation limit)
1. GATE 1 (CPU, decisive, DONE = REPRESENTATION LIMIT): is the 28-word wall a lossy READOUT or a representation limit?
   (a) pool-argmax (the wall ~0.57) vs (b) nearest-centroid on mean-centered full code vs (c) learned linear
   decoder. If a proper decoder clears ~0.80 -> READOUT artifact -> cheap fix, NO 100hr. NOTE: first run was
   a CAUGHT BUG (capture state-drift -> pool-argmax 0.234 != the probe's 0.571; no mean-centering; p>>n
   logreg overfit to chance) -- fixed (thorough between-capture reset + mean-centering + PCA), re-running.
2. GATE 1b (only if Gate 1 = representation limit): does a LEARNED transform on the same activity
   (expansion+Hebbian, the survey lever) beat the fixed-projection floor? The internal map predicts bounded;
   the data decides.
3. GATE 2 (only if 1 + 1b say representation limit AND a learned-on-same-activity method helps): the real
   acquisition-level question -> the owner-strategic fork, with BPTT DE-PRIORITIZED (bounded) in favor of
   G.20-scaling or VSA-role-binding per the internal map.

The 100hr is earned ONLY if Gates 1 + 1b both say "representation limit, not fixable on the same activity."
