# The SSM-extract LANGUAGE escalation at toy scale is a NULL DISCRIMINATOR (a-1-confirmed thin-deep-signal confound), not a mechanism verdict — the memory-horizon GO stands; the decisive test must run in the validated-signal regime (TinyStories 23.7M / V=2000, where the +1.9-nat growing-with-depth target is established)

**Date:** 2026-07-13
**Runner:** `research/runners/_ssm_reservoir_lm_derisk.py` (numpy-CPU; reuse-by-import `Vocab`/`fit_bigram`/`bigram_ce` + `load_sentences`; self-contained ridge read-out + temperature-calibrated CE; NO `sim/` edit).
**Status:** HONEST NULL — the toy-scale next-token-CE task cannot evaluate the multi-timescale mechanism's language value (no reservoir of any structure beats a memoryless bigram at deep context). The memory-horizon 6-seed GO (`2026-07-13-SSM-fixed-structured-multitimescale-reservoir-SURPASSES-fading-memory-ceiling-6seed-GO.md`) is unaffected. Follows the ceiling-first + a-1 discipline; the decisive test is named.

## What ran (seed 42, TinyStories, n_sent=800, V=1000, block=64, reset-per-block so within-block position = context depth)
Four FIXED reservoirs + a shallow local read-out (ridge → temperature-calibrated CE), CE bucketed by context depth, vs a **by-depth bigram control**:
| arm | shallow CE (1–4) | mid CE (5–16) | deep CE (17+) | deep state ‖x‖ | deep next-token acc |
|---|---|---|---|---|---|
| random ESN (`tanh(W@x)`, mixing) | **5.773** | 6.217 | 6.363 | 4.24 | 0.137 |
| multitimescale diagonal (linear, `A·x+u`) | 5.768 | **6.130** | 7.633 | 10.75 (drifts) | 0.084 |
| mtbounded diagonal (`tanh(A·x+u)`) | 5.980 | 6.887 | 8.172 | 4.71 | 0.070 |
| hetero-leaky-ESN (mixing + heterogeneous leak) | 5.937 | 6.338 | 7.424 | 0.70 | 0.082 |
| **bigram (by-depth control)** | 6.137 | 6.081 | **6.158** | — | — |

## The load-bearing read (systematic + a-1): this is a NULL DISCRIMINATOR, not a mechanism boundary
- **No reservoir beats the by-depth bigram at DEEP context** (best reservoir 6.363 > bigram-deep 6.158). The bigram's CE is ~flat across depth (6.14/6.08/6.16) — it is memoryless, so "difficulty" is constant. At **deep positions the exploitable predictive signal is bigram-level**: there is essentially nothing a fixed reservoir can add.
- The plain ESN DOES beat the bigram at **shallow** (5.773 vs 6.137, +0.36 nats) — recent-context memory is real and exploitable — but that advantage vanishes by deep. This is the documented **"reservoir wins mid, loses deep = fading-memory limit"** (`2026-07-11-SCALE-reservoir-wins-mid-depth-loses-deep-*`).
- **a-1 (`--corpus finding`) confirms the confound is already in our record:** `2026-07-11-CEILING-...long-range-signal-is-THIN-at-this-scale-*` — at 1.7–5M words / small vocab, **even a well-trained full transformer is WORSE than a bigram at deep context (−0.06→−0.38, monotonically worse)**; long-range signal is thin here for ANY model. My n_sent=800/V=1000 result is exactly that regime.
- ⇒ the toy next-token-CE task **cannot confirm or refute** whether the multi-timescale structure helps language. A negative here would be scale-confounded, not a substrate verdict.

## The mechanism MAP the arms still give (informative, not a verdict)
- **Artifact ruled out** (systematic-debugging): the linear diagonal's deep failure looked like unbounded drift (deep ‖x‖ 10.75) — but **bounding it (mtbounded, tanh) fixes the drift (‖x‖ 4.71) yet makes language WORSE at every depth.** So drift is not the cause.
- The real distinction: a **pure diagonal reservoir (bounded or not) lacks nonlinear cross-channel MIXING** — it can *hold* a distant cue (the memory-horizon GO = pure retention) but cannot *compute* the local n-gram conjunctions that dominate next-token prediction, which the ESN's `tanh(W@x)` recurrence provides. The mixing ESN beats the bigram at shallow; the diagonal does not.
- The hetero-leaky-ESN (mixing + heterogeneous leak) is starved here (deep ‖x‖ 0.70 — the leak+input scaling suppress the mixing); its parameterization needs the validated-scale regime to tune meaningfully, not toy scale.

## ⇒ The decisive next test (iterate, don't stop): the VALIDATED-SIGNAL regime
Our record pins the regime where deep-context signal is real and the target is quantified: **TinyStories 23.7M words / V=2000** (and WikiText-103), where a transformer AND a full-backprop LSTM both capture **+0.5→+1.9 nats growing-with-depth** long-range (`2026-07-11-CEILING-*`). The mission-relevant question — *does STRUCTURING the fixed recurrence (multi-timescale) capture more of that growing-with-depth long-range than a random reservoir, with only a local read-out (no BPTT, per the R3 reframe)?* — is only meaningful there. The memory-horizon GO shows the multi-timescale diagonal is a candidate **non-fading distal store** (the exact role the "reservoir loses deep" finding named the WM-buffer for). NEXT: run the 3-reservoir comparison (random / multitimescale / hetero-leaky-ESN) at the validated scale, by-depth vs bigram AND vs the +1.9 ceiling — first reproducing the reservoir's mid-depth win (validates the discriminator), then reading whether multi-timescale extends the win to deep context.

Runner: `_ssm_reservoir_lm_derisk.py` (scale-up = larger `--n-sent` + a batched-block reservoir for tractability). NO `sim/` edit.
