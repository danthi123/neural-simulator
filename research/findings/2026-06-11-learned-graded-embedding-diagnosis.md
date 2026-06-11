# Learned graded-embedding collapse — DIAGNOSIS: it is the **Hebbian LEARN**, not the read-out (the recurrent saturates to a near-uniform blob that doesn't even track the co-occurrence counts; PPMI/divisive-normalization can't recover structure that isn't in W)

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_diagnose.py` (NEW). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090). **Raw:** `research/findings/raw/_lge_diagnose_seed42.json`. **Scope:** single-seed (42), GPU, foreground — a MECHANISTIC localization (saturation + the host ceiling are seed-robust signatures); learn ran ONCE (~2.3 min) and the pipeline was dissected at each stage off that one learned W.

> **Verdict: LEARN_FAILURE — the collapse is the spiking-Hebbian LEARN, NOT the read-out.** The de-risk doc's preliminary prime suspect (a missing PPMI / divisive-normalization in the read-out) is **falsified**: applying PPMI **and** brain-based divisive-normalization **directly to the brain-learned W** — the suspected fix — still recovers **nothing** (best Pearson(sim, S_true) **+0.039**, generalization at chance), because the Hebbian recurrent W is a **saturated, near-uniform blob that does not even track the co-occurrence counts** (Pearson(W, raw_counts) = **+0.062**; W off-diag mean 0.864, std 0.235, CV 0.27; recurrent mean 0.998, max 44.8). There is no graded structure in W for any read-out to recover. The host PPMI+SVD on the **raw counts** still hits **+0.932 / gen 1.000** — the structure is fully in the corpus, the spiking learn destroyed it. **The fix is a different brain-based learning rule (a normalized / competitive / predictive Hebbian that does not saturate), NOT a better read-out.**

## Method (what was dissected)
Learn ONCE on the de-risk's exact corpus (48 concepts = 8 hubs + 40 members, 280 facts, 53 second-order cat~dog pairs) with the brain-based spiking-Hebbian learner (`LearnedAssocGraph`, 2,300-neuron / 2.76M-synapse Izhikevich bridge, 20 store-cycles, 139 s on GPU). Extract the learned recurrent W [Nc×Nc] (mean a→b recurrent weight between each concept-pair's sparse patterns — the SAME extraction the de-risk's `learn_assoc_matrix` does). Then compute Pearson(stage-similarity, S_true) **and** the generalization gate at **each** pipeline stage:

1. **STAGE W** — the learned weights, BEFORE any read-out: member↔member submatrix rows of W → cosine.
2. **STAGE diffusion** — the de-risk's current read-out (diffusion α=0.5, 2 steps) → reproduce the collapse.
3. **STAGE PPMI/divnorm** — the **prime-suspect fix**, applied **to the LEARNED W** (not the raw counts): (a) brain-based divisive-normalization only (Carandini–Heeger, M/√(rowsum·colsum)), (b) PPMI+truncated-SVD on W, (c) divnorm+SVD on W.

Plus the labelled host PPMI+SVD ceiling (on the **raw counts**, NOT the deliverable) and a Pearson(W, raw_counts) faithfulness check.

## Per-stage results (seed 42, GPU)

| Stage | Pearson(sim, S_true) | permuted-S baseline | within / between cos | graded? | generalization (chance 0.250) |
|---|---|---|---|---|---|
| **STAGE W** (learned weights, no read-out) | **−0.026** | −0.007 | 0.540 / 0.550 | False | 0.250 (1.00×) |
| **STAGE diffusion** (current read-out) | −0.024 | −0.002 | 0.955 / 0.956 | False | 0.237 (0.95×) |
| STAGE divnorm-only (on W, brain-based) | −0.047 | −0.014 | 0.221 / 0.238 | False | 0.319 (1.27×) |
| **STAGE PPMI+SVD (on LEARNED W)** | **+0.039** | −0.020 | 0.500 / 0.486 | False | 0.237 (0.95×) |
| STAGE divnorm+SVD (on LEARNED W) | −0.040 | −0.005 | 0.973 / 0.973 | False | 0.269 (1.07×) |
| random-Gaussian baseline | — | — | — | — | 0.312 |
| **HOST CEILING (PPMI+SVD on RAW counts)** | **+0.932** | — | — | **True** | **1.000** |

**Anti-cheat (the deliverable fix runs on the brain-learned W, not the counts):** Pearson(W, raw_counts) = **+0.062** (≪ 0.999 → the PPMI/divnorm fix operates on the LEARNED W, distinct from the raw counts — it is NOT silently re-deriving the host ceiling). Pearson(sim_PPMI(W), sim_PPMI(counts)) = +0.039 (the W-fix is nowhere near the host-on-counts result). Every "recovered" claim's permuted-S baseline is ≈0, as required — but no stage actually recovers, so this is moot.

## Localization (the decisive logic)
- **The READ-OUT is exonerated.** STAGE W — the raw learned weights, *before any read-out at all* — is already at Pearson −0.026 / margin −0.010 / not graded. The diffusion read-out then over-smooths (within/between both → 0.955) but it is destroying structure **that was never there**. You cannot blame a read-out for collapsing a signal its input never carried.
- **PPMI / divisive-normalization on W does NOT rescue it** (best +0.039). This is the single most important negative: the de-risk doc hypothesized the missing marginal normalization (PPMI's power on the host counts) was the fix. Applied to the *learned W* it does nothing — because **W has no marginal-removable graded structure to expose.** PPMI works on the host counts precisely because the counts encode the hub-mediated second-order ties; W has lost them.
- **Root cause — W does not track the co-occurrence counts and is saturated.** Pearson(W, raw_counts) = **+0.062** (essentially uncorrelated). W off-diag mean **0.864** with std only 0.235 (CV 0.27) — a near-uniform blob. The recurrent itself: mean **0.998**, max **44.8**, nnz 2.40M of 2.46M possible pool↔pool edges → the plastic excitatory recurrent has **filled in and saturated**: after 20 store-cycles of co-firing dense overlapping sparse patterns (K=100 in N=2000, with NO normalization, NO competition, NO LTD/decay on the recurrent), nearly every pool→pool weight has grown to a similar value. The graded co-occurrence signal (which pairs co-fire *more*) is swamped by the uniform potentiation floor.
- **The structure is genuinely in the data.** Host PPMI+SVD on the raw counts: Pearson +0.932, gen 1.000, graded True. The corpus is fine; the architecture routes fine (the de-risk's cortex-channel was +1.000); **the spiking Hebbian learn is the specific, isolated failure.**

**Corroborating mechanism (from a tiny-scale CPU smoke, 3×3 concepts, only 2 store-cycles):** there W tracked the counts at Pearson **+0.724** (faithful learned count). At the deployed 20-cycle full scale it drops to **+0.062**. → the de-correlation of W from the counts is **training-driven saturation**: few cycles = faithful graded W; many cycles on an un-normalized recurrent = uniform blob. (Smoke was wiring-validation only and was deleted; the direction is the point.)

## The honest answer to the three diagnostic questions
1. **Per-stage Pearson(·, S_true):** STAGE W **−0.026** (ungraded before any read-out) → STAGE diffusion **−0.024** (the documented collapse) → STAGE PPMI/divnorm-on-W **best +0.039** (the suspected fix fails) ‖ host ceiling **+0.932**.
2. **Is the collapse the READ-OUT or the LEARN?** **The LEARN.** W is degenerate/near-uniform and uncorrelated with the counts *before* any read-out; no read-out (diffusion, PPMI, divisive-norm, or +SVD) recovers structure that isn't in W.
3. **Does a brain-based divisive-normalization read-out on the SPIKING-LEARNED W recover the structure toward the host ceiling?** **No** (+0.039 vs +0.932; generalization at chance). Divisive-normalization is the right *idea* but it must act **inside the learning** (so the recurrent never saturates), not as a post-hoc read-out on an already-saturated W.

## Explicit next step
**Do NOT re-run the de-risk with a different read-out — that path is closed (the read-out is not the bug).** Pivot to the **learning rule**, the cheapest-first being a normalization/competition that prevents the recurrent from saturating, so the *learned W itself* tracks the graded co-occurrence (then the existing read-out, or PPMI on a faithful W, will work):

1. **(cheapest, retest first) De-saturate the Hebbian recurrent.** Add a brain-based homeostatic / normalizing term to the recurrent so it does not fill to a uniform floor: synaptic scaling / weight normalization (Turrigiano), an LTD/decay arm (so non-co-firing pairs lose weight, restoring contrast), shorter store-cycles (the smoke's +0.724 at 2 cycles vs +0.062 at 20 is direct evidence less potentiation preserves the graded signal), or sparser/less-overlapping patterns (lower K, larger N) so co-fire overlap is informative rather than saturating. **Cheap re-test:** sweep store-cycles ∈ {2,5,10} and a recurrent decay/scaling factor, re-measure Pearson(W, raw_counts) and Pearson(sim_W, S_true) with THIS diagnose runner. If a setting restores W↔counts ≫ +0.06 and sim_W↔S_true > 0, then **PPMI/divnorm on that faithful W should reach the ceiling** — and the read-out question reopens favorably.
2. **(if de-saturation alone is insufficient) A competitive / predictive Hebbian rule** — Oja's rule (built-in normalization, no saturation), BCM (sliding threshold → selectivity), or contrastive/predictive Hebbian (the distributional-semantics-grounded rule: predict-the-context, the spiking analogue of word2vec's objective that the host ceiling effectively uses). This is the deeper rewrite if (1) plateaus.

The diagnosis has done its job: it killed the wrong fix (a better read-out) in 2.3 GPU-minutes and pointed the build at the actual bottleneck (a non-saturating Hebbian learn). **No banking** — reported exactly as found.
