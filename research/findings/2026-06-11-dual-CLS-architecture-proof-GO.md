# Dual / CLS architecture proof — GO (the round-trip PRESERVES graded similarity)

**Status:** GO. Cheap-first (CPU/numpy, multi-seed 42/43/44, reuse-by-import, NO substrate build,
NO GPU) ARCHITECTURE-PROOF de-risk that gates the dual / complementary-learning-systems (CLS) build.
**Date:** 2026-06-11. **Probe:** `research/runners/dual_cls_architecture_proof_probe.py`.
**Raw:** `research/findings/raw/_dual_cls_proof_multiseed.json` (+ `_dual_cls_proof_20260611_084735.json`,
the seed-42 validation run). **Spec implemented:** `docs/plans/2026-06-11-dual-CLS-architecture-design.md` §4.

## Headline

**All six gates GREEN on all three seeds → the dual architecture is VIABLE.** The single
load-bearing number — whether the encode→decorrelate→bind→retrieve→decode ROUND-TRIP preserves the
graded similarity that generalization needs (the inverse of the binding problem) — is **Pearson(S, S')
= +0.891 / +0.898 / +0.843 (mean +0.877)** at the binding-viable + identity-working operating point,
against a permuted-S baseline of ≈ 0 (−0.06 to +0.01). **Decorrelation does NOT destroy similarity
when the CA1→cortex decode link is learned.** The risk the design flagged as "the deepest technical
unknown" (§5.2 / risk ii) does NOT materialize on a synthetic graded codebook.

| Probe | Gate | seed 42 | seed 43 | seed 44 | verdict |
|---|---|---|---|---|---|
| **A** generalization | A1 graded ≥ 0.7 | 1.000 (4.0×) | 1.000 (4.0×) | 1.000 (4.0×) | **PASS** |
| **A** decisive contrast | A2 orthogonal collapses | 0.119 | 0.256 | 0.237 | **PASS** (≈ chance 0.25) |
| **A** headline anti-cheat | A3 permuted-S collapses | 0.219 | 0.163 | 0.212 | **PASS** (≤ chance) |
| **B** binding | B parity ≈ 1.000 | PASS | PASS | PASS | **PASS** (reused verbatim) |
| **C** round-trip identity | C1 ≥ 0.9 viable point | 1.000 | 1.000 | 1.000 | **PASS** |
| **C** round-trip similarity | **C2 Pearson ≥ 0.7** | **+0.891** | **+0.898** | **+0.843** | **PASS** |

⇒ **GO. Recommend scoping the learned graded-similarity embedding build (the one genuinely-new piece).**

## The synthetic graded codebook (design)

Per §3.4 / §4.1: a controlled **"category factor + concept residual"** generator (in
`build_graded_codebook`). N = 8 clusters × 5 concepts = 40 concepts in dim = 256. Each concept code =
`(1 − residual_frac)·category_factor[cluster] + residual_frac·per_concept_residual`, residual_frac =
0.55, then mean-removed + unit-normalized (native convention, asserted). Concepts in the same cluster
share a category direction → systematically CLOSER; different clusters have independent category
factors → LOW between-cluster cosine. This is graded and **semantic-by-construction** (cluster =
"category"), the controlled stand-in for the not-yet-built learned embedding.

**Measured graded structure (the unit check):** within-cluster cosine **0.395 / 0.402 / 0.398**,
between-cluster cosine **−0.001 / −0.007 / +0.002**, margin **0.396 / 0.410 / 0.396** (within ≫ between,
`is_graded=True` all seeds). The **orthogonal control** codes (`generate_sparse_patterns`, K=100/N=2000)
have between-cos ≈ 0.0008 (equidistant by construction) — the decisive A2 contrast codebook.

## Probe A — GENERALIZATION (held-out-neighbour property inference)

**Task (§4.1).** Properties are assigned so CLUSTER predicts property (semantic inheritance: canids
share property P; `assign_properties`, n_props = 4 → chance 0.25). In each cluster HOLD OUT one
concept (never entered in the property table), train on the rest, then infer the held-out concept's
property from its **k=3 nearest TRAINED neighbours** via a similarity-weighted vote (`similarity_vote_infer`
— the cortex "read whatever code arrives" stand-in, NOT the exact-inverse algebra). Averaged over 20
random held-out splits per seed (so the contrast is statistically tight, not coarse).

- **A1 (graded PASSES): 1.000 every seed (4.0× chance).** Graded similarity lets a held-out concept
  inherit its cluster-mates' property — the "cat~dog" generalization, demonstrated.
- **A2 (the DECISIVE CONTRAST — orthogonal FAILS): 0.119 / 0.256 / 0.237 ≈ chance (0.25).** The
  IDENTICAL test on the project's orthogonal sparse codes collapses to chance — **proving the
  generalization is SIMILARITY-driven, not a generic retrieval trick.** Graded generalizes; orthogonal
  cannot (they are equidistant — the documented Option-A limitation, reproduced as the contrast).
- **A3 (HEADLINE anti-cheat — permuted-similarity collapses): 0.219 / 0.163 / 0.212 ≤ chance.**
  Shuffling the property labels (decoupling property from cluster/code structure) collapses
  generalization to chance — so the win is NOT code overlap unrelated to meaning. This is the mandatory
  headline control (the analogue of the whitening de-risk's reproducibility headline), and it is clean.

## Probe B — BINDING preserved (positive control, reused verbatim)

`run_binding_poscontrol` imports `cortex_sparse_attractor_poscontrol_probe.run_seed` unchanged. On the
decorrelated sparse codes (between-cos ≈ 0.0008, unit-check PASS): Hopfield/argmax **parity 1.000 at
flip p ≤ 0.3**, completion 1.000 at keep ≥ 0.15, the attractor **collapses on the correlated denoise64
codes** (hopfield ≈ 0.06 ≤ 2× chance while argmax = 1.000), and the **noise-cue no-hallucination
anti-cheat is decisive** (max concept freq 0.09 ≤ 3× chance). gate_a ∧ gate_b ∧ noise-cue OK = PASS,
all three seeds. The validated binding side composes in the dual setting with zero regression.

## Probe C — THE ROUND-TRIP (the load-bearing new number)

**Pipeline (§4.3).** graded cortex code → **ENCODE** (DG-style fixed random projection + top-k WTA
sparsifier = the DG PV-basket feedforward-inhibition analogue; `make_dg_encoder`) → **BIND/RETRIEVE**
(Hopfield attractor over the decorrelated expansion, noised cue p=0.1, settle — Probe B's validated
path; `hopfield_retrieve_all`) → **DECODE** the settled state back toward cortex via a ridge linear map
learned from the codebook (the CA1→cortex consolidation-pathway analogue; `fit_decoder`). Then
**Pearson(S_orig, S')** between the off-diagonals of the original graded cosine matrix and the
round-tripped one. **Sweep** expansion sparsity/capacity; **permuted-S baseline** (decoder fit on
row-shuffled codes → Pearson must be ≈ 0).

**The encode decorrelates (precondition for binding):** every sweep point drives the expansion
between-cos to 0.009–0.025 (< 0.15) — the DG random-projection + top-k strongly decorrelates even at
K/N = 0.20. So **binding is viable at every operating point**; what varies is the attractor's IDENTITY
recovery (capacity-limited) and, tracking it, the Pearson.

**Sweep (seed 42 shown; 43/44 identical shape — full sweep in the raw JSON):**

| pool | K | K/N | expansion cos | bind viable | identity | **Pearson(S,S')** | permuted baseline |
|---|---|---|---|---|---|---|---|
| 2000 | 400 | 0.20 | 0.022 | yes | 0.600 | +0.750 | −0.053 |
| 2000 | 200 | 0.10 | 0.019 | yes | 0.750 | +0.739 | −0.056 |
| 2000 | 100 | 0.05 | 0.015 | yes | 0.625 | +0.692 | −0.060 |
| 4000 | 100 | 0.025 | 0.012 | yes | 0.825 | +0.795 | −0.037 |
| **8000** | **100** | **0.0125** | **0.009** | **yes** | **1.000** | **+0.891** | **−0.062** |

- **C1 (binding round-trips): identity 1.000** at a binding-viable point on every seed (pool=8000/K=100;
  pool=4000/K=100 also reaches 1.000 on seed 44). The right concept comes back.
- **C2 (LOAD-BEARING — similarity survives): Pearson +0.891 / +0.898 / +0.843 (mean +0.877)** at that
  binding-viable + identity-working point, vs permuted baseline ≈ 0. **The decorrelation is REVERSIBLE
  in the similarity dimension when the decode link is learned** — a stored episodic fact round-trips
  through the hippocampal side WITHOUT losing its graded similarity.

**Honest nuance (capacity, not a wall).** Pearson and identity both IMPROVE as the expansion gets
sparser (larger pool at K=100): more capacity → both a cleaner attractor and a better-conditioned
linear decode. So the operating point that satisfies C2 is the SAME sparse regime the validated binding
uses (K=100, large pool). There is no tension between "sparse enough to bind" and "preserves similarity"
— they co-optimize. (At the denser K=400 end, Pearson is still +0.75 but identity dips to 0.60 — the
attractor, not the similarity, is the limiter there.) The round-trip-destroys-similarity risk does not
appear at any point in the sweep: the **lowest** Pearson observed anywhere is +0.69 (seed 42, K=100/pool=2000),
still far above the ≈ 0 permuted baseline.

## Decision logic (stated explicitly)

Per the spec's decision logic: **GO** requires A1 ∧ A2 ∧ A3 ∧ B ∧ C1 ∧ C2 — all satisfied on all three
seeds. The round-trip Pearson is high (mean +0.877 ≥ 0.7) at an operating point where binding also works
(identity 1.000, expansion cos 0.009). **The fast bidirectional codec form of the link survives** — the
fallback (encode-fast / consolidate-slow) is NOT needed (though it remains available and is more
biology-faithful if the learned-embedding substrate later proves noisier than this synthetic codebook).

## The decisive next step

**Scope the learned graded-similarity embedding build — the one genuinely-new piece (§3.4, risk i).**
The architecture is proven viable on ideal graded codes, so the expensive substrate arc is now
justified (it was gated on exactly this cheap proof). The build target: a LEARNED semantic embedding in
which related concepts cluster (cat~dog close, cat~bicycle far), grounded in co-occurrence /
shared-context / shared-attribute statistics, realized on the spiking substrate — the §3.4 verdict's
central new sub-problem (denoise64 is correlated-but-not-semantic; V1 is graded-but-perceptual-and-
discarded; concept-pool codes are orthogonal-by-design). The encode (DG)/retrieve (CA1) plumbing this
proof exercised is ~80% built and individually validated (§2); the new work is the graded-cortex code
itself + reading it for inference + wiring the graded-cortex ↔ decorrelated-expansion pair (never
connected this way — risk v).

**Honest scope caveats carried forward (do not let the GO over-claim):**
1. **This is an ARCHITECTURE proof on a SYNTHETIC graded codebook, not a capability on the real
   substrate.** The graded structure was constructed (category factor + residual); the learned-embedding
   arc must still PRODUCE such structure on spiking neurons, which is the deep, unproven piece (risk i,
   A-5). A clean architecture GO de-risks the *shape*, not the *substrate*.
2. **"Generalization" here = similarity-based property inheritance** (a real, measurable, CLS-grounded
   capability), NOT open-ended analogy / schema reasoning (risk iii). The honest deliverable is "the dual
   architecture supports similarity-based generalization that the flat composer cannot," exactly as
   measured by A1-vs-A2.
3. **The decode link was trained on the codebook the system knows** (the consolidation analogue). The
   on-substrate confirmation (§4.4) — wiring the real DG separation + real CA1 link on a small
   `SimulationBridge` and re-running A/B/C — is the next gate BEFORE any GPU build, and the DG may behave
   differently on graded structured input than on the random sparse input it was P1-validated on (risk v).
   That integration risk is now front-loaded as the immediate follow-on, not discovered at build time.

## Anti-cheats (all clean)

- **Orthogonal-codes contrast (A2):** generalization collapses to chance (0.12–0.26) on the project's
  decorrelated codes — proving it is similarity-driven.
- **Permuted-similarity (A3, headline):** generalization collapses to ≤ chance when property↔cluster is
  decoupled — proving it is meaning-driven, not code-overlap.
- **Round-trip permuted-S baseline (C2):** Pearson ≈ 0 (−0.06 to +0.01) when the decoder is fit on
  row-shuffled codes — proving the +0.877 true Pearson is a real similarity-preservation signal, not a
  pipeline artifact.
- **Binding noise-cue (B):** pure-noise input does not hallucinate a concept (max freq ≤ 3× chance).
- Native binary, mean-removed code conventions asserted (the positive control's unit check); multi-seed
  42/43/44.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners.dual_cls_architecture_proof_probe \
    --seeds 42,43,44 --out research/findings/raw/_dual_cls_proof_multiseed.json
# ~52 s, CPU only. Defaults: 8 clusters x 5 concepts, dim 256, 4 properties,
# round-trip sweep pool in {2000,4000,8000} x K in {400,200,100}, flip 0.1, ridge 1e-2.
```

## Sources / cross-references

- Spec: `docs/plans/2026-06-11-dual-CLS-architecture-design.md` (§4 de-risk, §5 risks).
- The falsification that forced the pivot: `research/findings/2026-06-11-option-B-whitening-derisk-NEGATIVE.md`.
- Reused harnesses: `research/runners/cortex_sparse_attractor_poscontrol_probe.py` (Probe B + the
  noise-cue anti-cheat + native conventions, 1.000 binding parity);
  `research/runners/concept_pool_sparse_distributed.generate_sparse_patterns` (the orthogonal control +
  expansion codes).
- CLS theory: McClelland-McNaughton-O'Reilly 1995; Kumaran-Hassabis-McClelland 2016; Teyler-Rudy 2007
  (hippocampal indexing). Catalog D.12 (DG separation), D.13 (CA3 completion), N.14 (systems consolidation).
