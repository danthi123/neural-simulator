# SCALE (the decisive test) — the emergent generator beats the bigram on REAL TEXT (TinyStories), not just the controlled template grammar: the dynamics-earned next-token generation is not a toy-corpus artifact

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_realcorpus_derisk.py` (reuse-by-import: the exact Rung-1 machinery — `Vocab`/`ReservoirStates`/`train_readout`/`eval_ce`/`fit_bigram`/`fit_trigram` — over a REAL corpus; NO `sim/` edit, NO BPTT; CPU numpy).
**Verdict:** **3/3 GO — the scale frontier's foundational question is answered YES (modestly): the emergent generator beats the bigram on REAL natural text on all three seeds.** Rung 1 was validated on the bounded EMERGE-62 template grammar; the honest open question was whether the same fixed-spiking-reservoir + one-step-local-delta generator holds on a REAL corpus, where the bigram is a MUCH stronger baseline. On TinyStories (V=200), the reservoir's held-out next-token **cross-entropy beats the bigram on all 3 seeds** (mean advantage 0.18 nats) and the permuted-corpus control collapses on all 3 — so it captures REAL higher-order structure. Honest nuance: the advantage is on CE/perplexity (the primary LM metric), not top-1 accuracy (the bigram is slightly higher there).

## Result — TinyStories, V=200 (top-200 words; 1400 train / 300 held-out sentences per seed)
| Seed | reservoir CE (acc) | bigram CE (acc) | trigram CE | permuted-corpus CE | GO |
|---|---|---|---|---|---|
| 42 (dev) | **3.264** (0.320) | 3.416 (0.365) | 3.962 | 3.922 | **yes** |
| 100 (blind) | **3.152** (0.327) | 3.393 (0.367) | 3.946 | 3.852 | **yes** |
| 43 (blind) | **3.307** (0.309) | 3.453 (0.356) | 4.003 | 3.915 | **yes** |

(chance = log V ≈ 5.30.) **The reservoir beats the bigram on cross-entropy on every seed** (advantage 0.15–0.24 nats, mean 0.18), and the **permuted-corpus control** (scramble each training sentence's word order + a fresh read-out) rises to **≥ the bigram every seed** (3.85–3.92) — the reservoir's advantage is destroyed when the real word-order structure is removed, so it is capturing genuine higher-order context, not an artifact. The trigram (~3.95–4.00) is WORSE than the bigram at this data scale (it overfits), so the reservoir's fading-memory context beats what a fixed higher-order n-gram can do without overfitting.

## Honest reading
- **CE vs accuracy.** The reservoir wins on CE (better-calibrated full next-token distributions → lower perplexity) but the bigram wins on top-1 accuracy (0.365 vs 0.320). On real text the single most-likely next word is often a very high-frequency function word the bigram nails; the reservoir spreads probability better across the plausible continuations (lower CE). CE/perplexity is the standard LM objective, so the reservoir's advantage is on the metric that matters for generation — but the accuracy gap is reported honestly, not hidden.
- **Modest, not dramatic.** A ~0.15-nat CE edge on a 300-neuron reservoir over a strong bigram is a modest advantage — but it is a REAL one (permuted-corpus collapses), on REAL text, from a fixed reservoir + a local one-step read-out with NO backprop-through-time. It establishes that the emergent-generation path is not confined to the controlled template grammar.
- **What it does NOT claim.** This is a next-token-LM CE result at V=200 on TinyStories — NOT fluent open-domain generation, NOT the full ladder (generalization/order/WM) on real text, NOT a transformer replacement. It is the FOUNDATION (dynamics-earned next-token prediction) shown to hold on natural text; scaling the reservoir + the ladder mechanisms to real fluent conversation remains the big open build.

## ⚠ SCALE-SCOPE CAVEAT (added same-day — read this GO at its regime)
This 3-seed GO is at V=200, **1400 train sentences**, n_pool=300 — a regime where the **bigram is data-starved** (many bigrams unseen under add-1 smoothing). The follow-up data sweep (`2026-07-11-SCALE-reservoir-size-vs-data-levers-on-real-text.md`) shows that at **5000 sentences the bigram improves more than the reservoir and OVERTAKES it** (margin +0.152 → −0.032 at n_pool=300), and that bigger reservoirs OVERFIT at fixed 1400-sentence data. So this GO is a genuine **foothold** — the emergent generator's dynamics-earned prediction is real (permuted-corpus collapses at every data scale) — but it is NOT a robust, scale-monotone win: the edge is regime-dependent (small-data + moderate-reservoir), and a robust advantage would require co-scaling reservoir size with data (the decisive open test). Do not over-read the +0.18 nats.

## ⇒ significance
The single most load-bearing open question of the emergent-generation arc — does the reservoir generator's dynamics-earned next-token prediction survive OFF the controlled template grammar, on REAL text? — is answered YES (modestly): on TinyStories it beats the bigram on cross-entropy with the real-word-order control collapsing. The path is not a toy-corpus artifact. NEXT: larger reservoir + more data (does the CE margin grow with scale?); the ladder's generalization/order/WM mechanisms on real text; harder corpora (WikiText); and the substantial build toward fluent generation.

## Files
`_emerge_reservoir_lm_realcorpus_derisk.py` (`--corpus`/`--vocab`); raw `research/findings/raw/_rc/ts_s{42,100,43}.json`; reuses the Rung-1 machinery; corpus `data/corpus/tinystories.txt`.
```
python -m research.runners._emerge_reservoir_lm_realcorpus_derisk --seeds 42 --corpus data/corpus/tinystories.txt --vocab 200
```
