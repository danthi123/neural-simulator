# 🎉 D cue-recall RESOLVED — a LEARNED sparse recurrent heteroassociative memory — 2026-06-05

The D boundary the owner asked to resolve — **cue-direction associative recall** (drive concept `a` alone → recall its
associate `b`), the DIRECT learned mechanism the dense v16 could only do at ~27.5% (≈ chance) — is **RESOLVED**:
multi-seed clean completion, anti-cheat-clean, genuinely learned. Runner: `research/runners/_D_sparse_heteroassoc.py`.

## The resolution
A **sparse recurrent heteroassociative memory** (Marr 1971 / Treves-Rolls CA3 autoassociator; McClelland CLS): a
shared pool with SPARSE concept patterns (K-of-N), a **PLASTIC excitatory recurrent** (the heteroassociative weights),
and FS inhibition. Encode = co-activate `a`'s + `b`'s sparse patterns → the recurrent `a↔b` GROWS by Hebbian co-fire
(zero-init → learned; the SELECTIVITY emerges from co-firing — NOT set). Recall = drive `a`'s pattern alone → the
recurrent completes `b`'s pattern, selectively.

## Result (GPU, seeds 42/43/44)
- **Clean cue→associate completion: post-encode AND post-SWR, all seeds** (the run() pipeline, 2 pairs: 2/2 each seed).
- **Bidirectional 4/4** on seed 42 (the direct test: c0↔c1, c2↔c3 — driving either completes the other; associate
  cosine 0.2–0.4, all non-associates ≈ 0 → selective).
- **Anti-cheat PASSED (the smell-test):** with a PERMUTED encoding (0→3, 1→2), driving c0 completes **c3** (the
  encoded associate, not c1) and driving c1 completes **c2**. The completion FOLLOWS the encoding ⇒ it is genuinely
  LEARNED, not a fixed/structural artifact.

## The three engineering layers solved (each was a real blocker, isolated in turn)
1. **Learning strength.** The bridge's STDP without reward forms eligibility but never applies it (weight stays 0) —
   so use direct Hebbian co-fire. But the default Hebbian caps at `hebbian_max_weight=1.0` and floors every edge at
   `hebbian_min_weight=0.05` (broad background). Fix: cap 45, floor 0, faster rate → the co-fired `a→b` reaches
   functional strength; non-co-fired stay 0 (clean selectivity).
2. **Propagation.** Even a strong recurrent didn't fire the target: a SPARSE pattern gives fan-in ≈ K·density ≈ 72,
   8× below the lang→pool's ~600, so the per-target synaptic input sits just above rheobase → marginal firing
   (target 0.15 vs cue 6.9). Fix: compensate the sparse fan-in with a high per-synapse weight (45) — the total drive
   then matches the lang→pool. This is the strong-sparse CA3 recurrent (biologically the CA3 collaterals ARE strong
   and sparse), not a cheat: the SELECTIVITY is learned; the strength is a synaptic parameter. (FS inhibition is NOT
   the suppressor — verified: target firing is unchanged at fs_inh ∈ {1.2, 0.3, 0}.)
3. **Read-out.** Measure the RECURRENT output: drive the cue, accumulate pool firing, EXCLUDE the cue's own
   directly-driven neurons, cosine to each pattern. Excluding the cue removes the pattern-overlap confound (else
   overlapping concepts rank by shared directly-driven neurons, not by the learned association).

## Correction to the earlier finding (important honesty)
`2026-06-05-D-swr-consolidation-dense-code-NEGATIVE.md` attributed the dense-v16 failure to the "heteroassociative
capacity wall." **That analysis was WRONG.** A numpy Hopfield heteroassociative resolves 4/4 for BOTH sparse AND dense
codes at this scale (4 pairs / N=2000) — capacity was never the binding constraint here. The v16 failure was that the
cross-pool **did not learn/propagate** (the open caveat that finding itself flagged: "confirm the cross-pathway weight
actually grows"). The real blockers were the three above (learning cap, propagation strength, read-out), now solved on
the sparse substrate. The sparse substrate still matters (sparse codes keep the completion CLEAN/selective and scale
the capacity), but the dense-v16 null was a learning/propagation failure, not a capacity wall.

## What this means for cheat D
Cheat D's residual was "the association weights are SET (outer-product), not Hebbian-LEARNED." This resolution
demonstrates the **LEARNED** direct heteroassociative cue→associate recall on the substrate, multi-seed,
anti-cheat-clean — the biology-faithful mechanism (CA3 recurrent autoassociation + Hebbian learning) the SET
outer-product was standing in for. The direct cue-only mechanism (the 27.5% boundary) is lifted to clean completion.

## Artifacts
`research/runners/_D_sparse_heteroassoc.py` (build + co-replay encode + recurrent-output completion + permuted
anti-cheat). NO `sim/` edits (reuses the brain-region framework + Hebbian + `generate_sparse_patterns`). Honest note:
the high recurrent weight (45) is the strong-sparse-CA3 regime compensating the sparse fan-in; the next rung is
wiring this into the conversational agent's dlPFC association memory (replacing the set-from-Python `c2d` edges with
this learned recurrent) — the original cheat-D integration target.
