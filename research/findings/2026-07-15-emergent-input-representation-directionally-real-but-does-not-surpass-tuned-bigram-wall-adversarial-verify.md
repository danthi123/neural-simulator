# The emergent input-representation gate: directionally REAL (input representation is the lever + reservoir context is real) but it does NOT surpass the TUNED first-order bigram wall at tractable scale — an adversarial-verify + P1/P2 controls REFUTED the "surpasses the bigram wall" overclaim

**Date:** 2026-07-15 · **Status:** HONEST NEGATIVE on the headline (the deliverable), with a genuine directional sub-result. 6-seed WikiText (V=200, n_pool=200, m_embed=48, n_sent=1500), numpy CPU, NO `sim/` edit. Caught by a read-only adversarial reviewer + the P1 (memoryless-embedding) and P2 (tuned add-k bigram) controls it demanded.

## The chain that led here (the a-1 deep-research convergence)

The reservoir-LM read-out-depth arc closed at tractable scale (bigram-level, per the bigram-starvation confound). An a-1 deep-research pass surfaced that the frontier had already CONVERGED (6-seed, first-principles, `2026-07-11-R3-REFRAME` + `2026-07-11-content-addressable-...`) on the **INPUT REPRESENTATION** — not the memory horizon, not recurrent credit, not node-perturbation (a committed adversarial-verify had retired that). Two controls this session confirmed the direction:
- **Memory-horizon sweep = NULL:** `adapt_win_hi` 300→3000 (10×) and `beta` 1→3 changed the 2-stage CE by ≤0.03 (all ~3.57–3.60, never beats the bigram). The reservoir-LM's ceiling is NOT the memory horizon → input-bound.
- So I tested the MISSION-preferred (emergent/unsupervised) version of the input-representation lever: does a co-occurrence-PPMI-SVD structured `W_in` (developed from the corpus, on the emergence bar) beat the random/one-hot `W_in`?

## The result (6-seed WikiText V=200; a 2-stage read-out on a FIXED rate reservoir)

| baseline | held-out next-token CE (nats) |
|---|---|
| **TUNED add-k first-order bigram (k=0.1) — the TRUE wall** | **3.183** |
| struct = reservoir + emergent PPMI `W_in` | 3.227 |
| embed-only = memoryless PPMI embedding (no reservoir) | 3.341 |
| add-1 bigram (the ARTIFACTUAL wall) | 3.373 |
| random-`W_in` reservoir | 3.454 |
| perm-embed (alignment broken; rank/scale-matched) | 3.522 |

**Two honest reads:**
1. **GENUINE, directional (within the reservoir-LM family):** the emergent PPMI input representation is load-bearing — struct beats random-`W_in` by **−0.227** and the rank/scale-matched **permuted-embedding anti-cheat by −0.295 (6/6 seeds)** → the win is real **token-specific distributional structure**, not a spectral/scale artifact. And the **reservoir context is real**: struct (3.227) < embed-only (3.341) on **6/6** → the recurrence adds value beyond the memoryless first-order embedding. This confirms the R3 direction (input representation is the lever) at tractable scale.
2. **HONEST NEGATIVE (the overclaim REFUTED):** the reservoir-LM does **NOT surpass the natural-language bigram WALL**. The add-1 bigram (3.373) is a weak/undersampled smoother; the **TUNED add-k first-order bigram (3.183) BEATS the full reservoir-LM (3.227) on 6/6 seeds**. No supra-first-order (beyond-bigram) capability is demonstrated. "Surpasses the bigram wall" is false against the correct baseline.

## Why this was nearly an overclaim — the adversarial-verify caught it

A read-only adversarial reviewer (dispatched before any GO commit) code-reviewed the runner + the 6-seed numbers and flagged the exact confounds:
- **BIGRAM-LEAK:** `E` is a rank-48 SVD of the ADJACENT-co-occurrence PPMI = a low-rank smoother OF the bigram (Levy-Goldberg). "Beats the bigram" against **add-1** only proves better smoothing, not higher-order structure. It demanded **P1** (memoryless-embedding control) and **P2** (a tuned add-k / Kneser-Ney bigram — is the add-1 "wall" an undersampling artifact?).
- **Anti-cheat metric:** the original `perm ≥ rand` gate cracked on seed 100 (a small spectral residual); the robust metric is **struct − perm** (rank/scale-matched, robustly −0.29). Fixed in the runner.
- P1 came back struct < embed-only (reservoir context IS real — a point FOR the reservoir), but **P2 came back decisive against the headline**: the tuned bigram (3.183) beats struct (3.227). The add-1 3.373 was indeed an artifact.

The reviewer's verdict — **OVERCLAIMED** — was correct. The genuinely-supported claim is the narrow one (read #1); "surpasses the wall" (read #2 negative) is not established and is in fact refuted by the tuned baseline.

## What this means (the honest re-localization)

- **The reservoir-LM next-token PERPLEXITY lens is genuinely exhausted at tractable scale** — and now CONFIRMED against the correct (tuned) baseline, not just the add-1 artifact. Even the emergent input representation + real reservoir context does not beat a tuned first-order bigram at 1500 WikiText sentences. This RE-CONFIRMS the owner's ceiling finding (`feedback_run_ceiling_early_and_keep_gpu_busy`: no model — reservoir or transformer — beats a well-tuned bigram at these scales; the natural-language higher-order signal is thin → data/scale-bound).
- **The R3 +4-nat "learn W_in beats fixed" win is a LARGE-scale (n_pool=300, ~2M-token) result.** At tractable scale the input-representation effect exists directionally but sits BELOW the tuned-bigram wall. ⇒ **the pinned SMALL-scale spiking W_in de-risk would face the SAME wall** — a spiking-tractable operating point will not show a "beats the wall" result; the R3 property needs large data (expensive on spikes). This adversarial pass SAVED a likely-doomed small-scale spiking build (an honest-negative-that-prevents-wasted-compute, the point of the cheap gate).
- **The mission (LLM-like conversation) does not hinge on beating the tuned-bigram perplexity.** The working conversational capability is the EMERGE structure-from-experience path (categories/inheritance/grammar/grounded conversation, discovered + spoken on spikes), which does not depend on this perplexity metric. The perplexity lens was a proxy that is bigram-bound at tractable scale for ALL models.

## The honest frontier (re-localized)

1. **DROP the reservoir-LM next-token perplexity lens at tractable scale** as a route to "beat the wall" — it is a confirmed data/scale wall for all models (this + the owner's ceiling). Pursuing it requires LARGE data (the R3 regime), which is the only place the input-representation win exceeds the tuned bigram — an expensive, scale-gated arc, not a cheap next step.
2. **The mission-central path is the EMERGE structure-from-experience conversational capability** (working: discover categories → reason → speak on spikes) — extend it toward richer/open conversation, on the emergence bar.
3. **The parallel MEMORY frontier: CA3 self-forming RECALL** (the sparse-synchronous full-scale Kopsick model; formation is 6-seed GO this cycle, functional completion is the pinned scale-bounded next).

## Artifacts
- Runner: `research/runners/_reslm_controlled_lag_eprop_derisk.py` — `language_input_repr_gate` (+ `--input-repr-gate`, `--m-embed`) with the perm-embed anti-cheat + the P1 embed-only bigram-leak probe; the gate metric is struct−perm (not perm−rand).
- P2 tuned-bigram control: `research/findings/raw/_reslm_P2_tuned_bigram.log`. Gate logs: `raw/_reslm_inputrepr_{6seed,bigramleak_6seed}.log`; horizon-null control: `raw/_reslm_horizon_sweep.log`.
- Method note: the cheap-first gate + adversarial-verify + P1/P2 controls worked exactly as designed — they converted a tempting "surpasses the wall" GO into an honest, precisely-scoped negative BEFORE any commit or any expensive spiking build.
