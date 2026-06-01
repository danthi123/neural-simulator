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

## Open inputs (pending)
- Internal asset map (running agent): the existing CONTRASTIVE runner (text_train_contrastive.py) + its
  result, the 0.46 ceiling audit, BPTT infra reusability, the single most important prior-failure lesson.
  CRITICAL: if the contrastive runner already beats 0.54, a cheaper path may already exist.
- Phase 2.3a NEGATIVE lesson (next-char features did not transfer): the new objective must target SEPARABLE
  CONCEPT codes directly, not a surrogate task (next-char) whose features don't separate concepts.
