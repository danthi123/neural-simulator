# SCALE mechanism (the key insight) — the reservoir's aggregate CE plateau vs the bigram on real text HIDES a clear structure: the reservoir DECISIVELY beats the bigram at MID context depth (2–5 tokens) but LOSES at DEEP context (6+) — the running-cumulative fading-memory limit. The next lever is the WM buffer, not a bigger reservoir

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_context_depth_derisk.py` (reuse-by-import: the Rung-1 machinery + the real-corpus loader; TinyStories V=200, 2-seed 42/100; NO `sim/` edit, NO BPTT).
**Verdict:** **The scale plateau is not an architecture ceiling — it is a fading-memory limit with a known fix.** The aggregate next-token CE has the reservoir hovering AT the bigram on real text (the data sweep). Breaking held-out CE down by CONTEXT DEPTH (tokens seen before the prediction) reveals WHY: the reservoir **beats the bigram decisively at mid-depth (2–5 tokens)** — it captures higher-order structure a bigram is blind to — but **loses at deep context (6+)** because its running-cumulative feature washes the early tokens out. The aggregate is the average of the two. ⇒ the lever for deep context is the Rung-2 **WM buffer** (hold distal tokens the fading reservoir forgets) + recency-aware features, NOT a bigger reservoir.

## Result — reservoir − bigram CE margin by context depth (2-seed 42/100; +margin = reservoir beats bigram)
| context depth (tokens seen) | n_pool=300 margin | n_pool=600 margin |
|---|---|---|
| 1 | +0.143 | −0.680 (bigger reservoir OVERFITS short context) |
| 2 | **+0.307** | −0.004 |
| 3 | **+0.308** | **+0.311** |
| 4–5 | **+0.151** | **+0.215** |
| 6–9 | −0.018 | +0.054 |
| 10+ | −0.024 | +0.045 |

- **The reservoir's real contribution is at MID depth (2–5).** A bigram uses ONLY the previous token; the reservoir's fading memory over the prefix captures a few tokens of higher-order context there, beating the bigram by +0.15 to +0.31 nats — exactly where higher-order structure lives.
- **It LOSES at deep context (6+) with the running-cumulative feature** (n_pool=300): the mean-over-prefix state washes out the early tokens, so a diffuse average is less predictive than the bigram's sharp previous token. This is the SAME fading-memory limit the ladder documented (Rung 1 ~depth-3; Rung 2 built the WM latch precisely because the reservoir forgets distal referents).
- **Bigger reservoir (600) trades short for deep:** worse at depth-1 (−0.68, short-context over-parameterization) but positive at deep (6+: +0.05). More units = more memory capacity, but it over-parameterizes the easy short-context predictions.

## ⇒ the reframed scale conclusion + the next mechanism
The aggregate-CE plateau (`-reservoir-size-vs-data-levers`) is NOT "the reservoir+read-out is merely bigram-level on real text." It is a **superposition**: a genuine mid-depth WIN (the reservoir's higher-order-context contribution, which no bigram has) plus a deep-depth LOSS (the running-cumulative fading-memory washout). So the path to a robust edge is NOT raw scale (bigger reservoir overfits short context; more data helps the bigram) — it is the **already-built Rung-2 WM buffer** applied on real text: a non-fading latch holds distal tokens the reservoir forgets, which should convert the deep-context LOSS into a WIN (the bigram is structurally incapable of distal context). Recency-aware features (per-window alongside running-cumulative) should also help short/mid. This is a sharp, actionable next mechanism, discovered by the boundary-surpassing workflow (break the plateau down → find where the reservoir wins → identify the exact limit + its known fix).

## OPEN (the decisive next experiment)
Add the Rung-2 WM-buffer / recency features to the real-corpus read-out and re-run the context-depth analysis: does the deep-context (6+) margin go POSITIVE (the reservoir+buffer beating the bigram where the bigram is blind)? If yes, the emergent generator has a robust, mechanistic advantage over the bigram on real text at the depths that matter for discourse — the real scale signal. (The co-scale aggregate grid is running in parallel to complete the aggregate picture.)

## Files
`_emerge_reservoir_lm_context_depth_derisk.py`; raw `research/findings/raw/_ctxdepth/np{300,600}_s{42,100}.json`. Follows `2026-07-11-SCALE-reservoir-size-vs-data-levers-on-real-text.md`; ties to Rung 1 (fading memory ~depth-3) + Rung 2 (the WM latch).
