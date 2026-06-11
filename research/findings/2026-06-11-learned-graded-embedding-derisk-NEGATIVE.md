# Learned graded-similarity embedding — cheap-first de-risk: NEGATIVE (the brain-based Hebbian learner produces NO graded structure; the host ceiling proves the structure IS there)

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_derisk_probe.py` (built by the design subagent; controller-adopted + run after the subagent orphan-yielded). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090). **Raw:** `research/findings/raw/_lge_gpu_seed42.json`. **Scope:** single-seed (42) — the result is a MECHANISTIC collapse, decisive at single seed; the host ceiling + the collapse are seed-robust signatures.

> **Verdict: NEGATIVE_no_structure.** The recommended brain-based mechanism for the dual/CLS architecture's one unbuilt piece — a Hebbian co-occurrence learner (`LearnedAssocGraph`) + a diffusion graded read-out — **does not produce graded semantic structure at all.** The learned codes collapse to near-uniform; generalization is at chance and *worse than random codes*. The decisive control: a host distributional-semantics method (positive pointwise mutual information + singular value decomposition, "PPMI+SVD") on the **exact same co-occurrence data** recovers the structure **perfectly** (generalization 1.000). So the corpus genuinely contains the structure and the architecture routes it fine — **the brain-based learning mechanism is the specific failure** (brain-vs-host gap +0.76). This is the classic "biological learning is weaker than backprop-on-big-data" wall, in a sharp collapse form — and it was caught in **4.5 GPU-minutes, before any months-scale build.**

## Why this ran
The dual / complementary-learning-systems (CLS) architecture is fully de-risked on the substrate but only with SYNTHETIC graded codes; the one unbuilt piece is a LEARNED graded-similarity cortex embedding (related concepts → similar codes). The design opening move (`docs/plans/2026-06-11-learned-graded-embedding-design.md`) recommended Option A — a Hebbian / distributional co-occurrence embedding, with the core learner already built (`research/runners/learned_assoc_graph.py`). The owner chose to cheap-first de-risk the LEARNING MECHANISM before committing the build. This is that falsification.

## Method
A toy-but-real co-occurrence corpus (48 concepts = 8 hubs + 40 members, 280 facts) with a KNOWN ground-truth graded similarity S_true, including **53/80 second-order pairs** (concepts that never co-occur directly but share neighbours — the genuine cat~dog case). The brain-based Hebbian learner (`LearnedAssocGraph.store_fact`, a real 2,300-neuron / 2.76M-synapse spiking bridge, 20 store-cycles) learns the co-occurrence graph; a diffusion read-out turns it into codes. Gates: G1 structure recovery (Pearson vs S_true), G2 generalization (held-out-neighbour inference), G3 cortex-channel round-trip, plus a PERMUTED-CO-OCCURRENCE control and a labelled host PPMI+SVD ceiling.

## Results (seed 42, GPU)

| Gate | Result |
|---|---|
| **G1 structure recovery** | **FALSE** — codes collapsed to near-uniform (within-cos 0.955 ≈ between-cos 0.956, margin −0.001); Pearson(S_learned, S_true) = **−0.024** (≈ 0). Second-order cat~dog: shared-neighbour cos +0.956 vs between-cluster +0.956, **margin +0.000, not recovered.** |
| **G2 generalization** | **FALSE** — graded acc **0.237** (chance 0.250); orthogonal 0.119; permuted-property 0.244. |
| beats-random baseline | **FALSE** — learned 0.237 < random-Gaussian **0.312** (the learned codes are *worse than random*). |
| permuted-co-occurrence collapses (anti-cheat) | TRUE — but trivially (the real codes are *also* at chance; the control is uninformative when the intact capability is at chance). |
| **G3 cortex-channel round-trip** | TRUE (Pearson +1.000, binding identity 1.000) — the architecture routes fine; it is the *learned codes* that lack structure. |
| **HOST CEILING (PPMI+SVD, labelled, NOT the deliverable)** | **gen 1.000, Pearson(S, S_true) +0.932, graded TRUE** — the structure is fully extractable from the same data. **brain-vs-host gap = +0.76.** |

**Overall: NEGATIVE_no_structure.** Total elapsed 269.8s (~4.5 min on GPU — ~10× faster than the CPU run's ~45 min/seed; the GPU correction, owner-flagged, validated).

## Diagnosis (preliminary) + next step
The corpus contains the structure (host PPMI+SVD: gen 1.000) and the architecture routes it (cortex-channel +1.000) — so the failure is localized to the **brain-based learning + read-out**. The codes collapsing to ~uniform (≈0.956 all-pairs) points at one of two causes, which the follow-up diagnosis separates:
1. **The diffusion read-out over-smooths** — diffusion (alpha 0.5, 2 steps) over a dense connected graph (nnz 2.4M, mean weight ~1) drives toward the uniform stationary distribution. If the learned recurrent weight matrix W *does* carry the graded structure, a better read-out (or the missing PPMI-style marginal normalization) would recover it. **This is the fixable case.**
2. **The Hebbian learn doesn't capture graded W** — if W itself is degenerate (saturated/uniform), the learning rule is the failure and a different brain-based rule (predictive / contrastive-Hebbian) is needed.

The notable structural hint vs the host: PPMI+SVD's power is the **PPMI normalization** (divide co-occurrence by the marginals — removes the high-frequency-concept dominance that otherwise collapses everything together). The brain-based read-out has no such normalization — a strong candidate for the collapse, and a brain-based divisive-normalization analogue is the first fix to try.

## Honest framing
- This is a genuine NEGATIVE for Option A *as implemented* — not a "weak generalization" but a total collapse. The cheap-first falsification did exactly its job: it killed the recommended build mechanism in minutes, before the months-scale commitment.
- The host PPMI+SVD ceiling is the TARGET (it proves the ceiling exists and is reachable from the data); the job is a *brain-based* learner that reaches it. The next step is the diagnosis above, then the brain-based fix (likely a divisive-normalization read-out, or a different learning rule).
- Single-seed; the collapse + the host ceiling are mechanistic/seed-robust, so multi-seed is a confirmation, not the deciding test.

**No banking** — reported exactly as found; the diagnosis localizes the fix.
