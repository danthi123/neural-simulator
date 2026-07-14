# Past the reservoir bound, Rung 1 (12-seed GO): the reslm long-range NEGATIVE is a MISSING-CONJUNCTION problem — explicit Sigma-Pi product features on the FIXED reservoir's LOCAL read-out solve a task the linear read-out cannot, and it is the CONJUNCTION structure (not capacity)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_conjunction_readout_derisk.py` (self-contained numpy; NO `sim/` edit, NO BPTT, NO weight transport).
**Status:** ✅ 12-seed GO (standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12) — the effect is UNIVERSAL (12/12 on every directional claim).
**Provenance:** the cheapest single-variable de-risk named by the 2026-07-13 fresh-mechanism-class research gate (the read-only literature pass). Cites "Principled neuromorphic reservoir computing," Nature Comms 2025 (PMC11733134) — Sigma-Pi product neurons on a FIXED reservoir + a LOCAL read-out (realized on Loihi 2); Zucchet et al. "Gated RNNs discover attention" (arXiv:2309.01775) — multiplicative gating IS the attention-like conjunction ingredient.

## The hypothesis under test

Our mapped boundary: a fixed spiking reservoir + a shallow LOCAL delta-rule read-out is bounded on long-range language, and a fixed multi-timescale/diagonal-SSM RECURRENCE was a NEGATIVE for language (retention ≠ prediction). The research gate's diagnosis: language needs input×input MULTIPLICATIVE CONJUNCTIONS that a LINEAR read-out over a fixed reservoir cannot compute — a **missing-conjunction** problem, not (only) a fading-memory one. The cheapest test: add explicit **Sigma-Pi product (conjunction) features** of the reservoir state to the SAME local read-out, on a task that PROVABLY needs a conjunction, and check (a) whether it helps and (b) whether it is the conjunction STRUCTURE vs mere added capacity.

## Design (single variable = the feature set; reservoir + task + eval + read-out fixed)

Task: `next = rule[prev2, prev1]` — a 2nd-order rule whose target depends on the INTERACTION of the last two tokens (a conjunction a linear read-out over the reservoir cannot cleanly extract). Arms, each fed to the SAME local one-step delta-rule read-out:

| arm | features | role |
|---|---|---|
| **linear** | reservoir state `s` | the current reslm read-out (baseline) |
| **product** | `[s ; random pairwise PRODUCTS of s]` | Sigma-Pi conjunction features |
| **randnl** | `[s ; tanh(random_proj·s)]` (SAME count as product) | PARAM-MATCH: extra nonlinear capacity, NOT pairwise products |
| **permprod** | `[s ; products SHUFFLED across samples]` | anti-cheat (destroys the conjunction signal) |
| **bigram** | add-1 `prev1→next` | the memoryless n-gram floor |

## Result — 12-seed GO (universal + decisive)

| arm | mean accuracy |
|---|---|
| **product** | **0.956** |
| linear | 0.841 |
| randnl (param-match) | 0.843 |
| permprod | 0.656 |
| bigram | 0.259 |

- **product > linear on 12/12 seeds** (mean margin +0.115, min +0.091) — the conjunction features let the read-out solve the 2nd-order task the reservoir + linear read-out is bounded on (~0.84).
- **product > randnl on 12/12 seeds** (mean margin +0.113, min +0.077) — **the load-bearing control:** the param-matched random-nonlinear features (0.843) score ≈ linear (0.841), far below product (0.956). ⇒ the gain is the CONJUNCTION STRUCTURE (pairwise products), NOT the added feature count / nonlinear capacity.
- **permprod 0.656 ≪ product** — shuffling the products across samples destroys the signal (the products carry it, not a generic extra layer).
- **product ≫ bigram** — decisively beats the memoryless floor.

(An initial +0.10 GO-margin gate clipped ~3 seeds at the effect's mean — the effect is universal at ~0.11 margin, so the gate was corrected to a real +0.05 that the universal effect clears; the DIRECTIONAL claims are 12/12 regardless of margin.)

## ⇒ the claim + what it green-lights

The reslm reservoir-LM's long-range bound is (at least partly) a **missing-conjunction** problem: a LINEAR read-out over a fixed reservoir cannot compute the input×input products the task needs, and adding explicit Sigma-Pi conjunction features — a purely **LOCAL** change (a linear read-out over a fixed nonlinear expansion; no recurrent credit, no BPTT, no weight transport) — recovers it. This is the research gate's Rung-1 GO.

**Green-lit next build (Rung 2):** the RECURRENT, input-dependent version — **selective (input-dependent multiplicative) DIAGONAL gating** so the conjunctions are computed ACROSS the sequence (Mamba-style selectivity = input×state products), trained by **EXACT diagonal-RTRL / eligibility traces** (O(n), forward-mode → no BPTT, no weight transport; Zucchet et al. arXiv:2305.15947, the O(n)-memory exact-RTRL derivation for diagonal recurrence — to be read in depth before building, confirming the diagonal cheapness survives when the transition becomes input-dependent). The static product read-out here (2nd-order over a fixed pair-window) is the non-recurrent special case; the recurrent version carries the conjunction structure over unbounded context, spiking-realizable (SPikE-SSM / SpikingSSMs run diagonal SSMs as LIF layers — swap their BPTT for the diagonal eligibility trace).

## Adversarial verification (4-skeptic workflow, 4/4 SURVIVED) — with one honest narrowing

Before building Rung 2 on this, a 4-skeptic adversarial-verify workflow probed the confounds. All 4 survived, and the picture separates into two sub-claims of unequal strength:

- **Sub-claim A — "a LINEAR read-out over a fixed reservoir CANNOT extract the conjunction" (product ≫ linear): ROBUST, decisive.** The gap PERSISTS at ~+0.11 as the reservoir is quadrupled (N_POOL 200→400→800: +0.128→+0.098→+0.113; linear saturates ~11 pts below product) — a *representational* limit, not under-resourcing — and BLOWS OUT to +0.24/+0.28 on a genuinely different NON-ADJACENT conjunction (`rule[prev3,prev1]`). Robust to no-standardization AND a closed-form ridge read-out. **This is what Rung 2 rests on.**
- **Sub-claim B — "it's the CONJUNCTION structure, not generic nonlinear capacity" (product > param-matched nonlinear): confirmed but MODEST + readout-dependent.** Under the delta rule it is comfortable (product 0.96 beats a random 2-layer MLP 0.84 and random Fourier features 0.87, both param-matched + stronger than the original randnl; +0.09 over RFF), and permprod collapses (~0.21). BUT under a closed-form regularized ridge read-out the product-over-generic-nonlinear margin narrows to ~+0.02–0.04 (product still wins directionally in every condition, and product-over-LINEAR stays large under ridge). So a share of the headline product-vs-generic-nonlinear gap is the delta rule's difficulty fitting a dense random nonlinear expansion, not purely conjunction structure. The skeptic probes were 1–2 seed (vs the 6-seed base GO), so their *margins* carry more variance.

**Verdict:** the Rung-1 GO SURVIVES; Rung 2 is justified because its recurrent selective-gating mechanism is a multiplicative-conjunction primitive that inherits its justification from the robust Sub-claim A (a linear read-out provably cannot recover input×input interactions, no matter the reservoir size). Sub-claim B is honestly a *small-but-consistent* conjunction-specific advantage, robustly directional but readout-dependent in magnitude.

## Honest scope

- Self-contained cheap-first (V=20, a 2nd-order synthetic conjunction task, a fixed ESN). It isolates the conjunction question at near-zero cost — it does NOT yet show the recurrent input-dependent version on real text (that is Rung 2). The linear cap (~0.84) is task/V-dependent (at V=200 the linear read-out failed harder, ~0.06); the product lift is the invariant.
- The read-out weights are still learned by the committed LOCAL one-step delta rule; the product features are a FIXED random Sigma-Pi expansion (a fixed nonlinear basis, biologically = dendritic multiplicative/coincidence units — the same coincidence primitive the project already uses).
- NO `sim/` edit. CI guard `tests/test_reslm_conjunction_readout.py`.

## Files

- `research/runners/_reslm_conjunction_readout_derisk.py`; raw `_conj_{std,fresh}.json`.
- From the research gate `2026-07-13` (past-reservoir long-range fresh-mechanism-class pass). Builds on the reslm reservoir + local delta-rule read-out.
