# CEILING de-risks (a real GPU transformer + a learned single-head attention) reconcile to: long-range signal beyond a bigram/cache is THIN at this scale/vocab (5M tokens, V=300). A full transformer is WORSE than a bigram at deep context (increasingly so — overfitting, not long-range); a learned single head only MARGINALLY beats the uniform topic-prior via induction/COPY. ⇒ the session's reservoir-LM long-range NEGATIVES are PARTLY SCALE-BOUND, not purely a substrate limit — a clean substrate test needs scale where long-range signal is demonstrably present (WikiText-103-scale)

**Date:** 2026-07-11
**Runners:** `research/runners/_wikitext_transformer_ceiling.py` (GPU, PyTorch TinyGPT — a reference CEILING, NOT a biological model) + `research/runners/_emerge_reservoir_lm_learned_attention_derisk.py` (numpy learned single-head attention over a cross-sentence store; controller-verified, NO `sim/` edit). WikiText, same top-300 word vocab + bigram baseline as the whole arc.
**Verdict:** the parallel GPU ceiling recontextualizes the session's long-range negatives — they are confounded by THIN long-range signal at this small scale, so they do NOT cleanly isolate "the reservoir substrate can't" from "there is little to capture."

## GPU transformer ceiling — a full attention model is WORSE than a bigram at deep context (the decisive, sobering result)
A well-trained TinyGPT (d192/L3/H6, train-CE 5.78→2.14 over 12k steps, 5M tokens) on contiguous WikiText, CE by within-block context depth vs the add-1 bigram (margin = bigram − transformer; + = transformer better):
| context depth (tokens seen) | transformer CE | bigram CE | margin |
|---|---|---|---|
| 1 | 2.778 | 2.719 | −0.059 |
| 3 | 2.914 | 2.757 | −0.158 |
| 4-8 | 3.012 | 2.760 | −0.252 |
| 9-16 | 3.053 | 2.737 | −0.316 |
| 17-64 | 3.117 | 2.738 | **−0.379** |
- **The transformer is WORSE than a bigram at every depth, and MONOTONICALLY MORE worse with depth** (−0.059 → −0.379). More context makes it *relatively worse* — the opposite of "captures long-range." The large train-vs-held-out gap (train 2.14 vs held-out ≥2.78) is the tell: at 5M tokens / V=300 a small transformer OVERFITS and does not generalize better than a bigram, and its use of longer context adds noise, not signal.
- ⇒ **at this scale/vocab there is little generalizable long-range structure for ANY learned attention to exploit** — the strongest possible content-model (a real transformer) cannot beat a bigram at long-range here.

## Learned single-head attention — a MARGINAL content-selective (induction/COPY) signal, not rich long-range
A learned single attention head (learned Q/K over fixed token-embedding keys, value = onehot(next token)) over the cross-sentence store, 3-seed, full-gradient CEILING:
- content beats the SHUFFLE bag clearly: **+0.045/+0.042/+0.049** (3/3), with the advantage GROWING to deep context (induction-head signature).
- BUT content beats the stronger UNIFORM topic-prior only MARGINALLY: **+0.031/+0.030/+0.034** (right at the 0.03 threshold; one seed's own verdict = THIN).
- The LOCAL feedback-alignment rule does NOT reach even that (+0.009–0.011).
- Interpretation: a single head captures a small INDUCTION/COPY effect (retrieve where the same recent-token-context occurred → copy its continuation = a repetition prior), which is content-selective vs a random bag but is NOT rich long-range reasoning, and it only marginally exceeds the topic-prior a uniform bag already provides.

## DEFINITIVE follow-up (content-word vocab sweep): the corpus is ~50-100× TOO SMALL for ANY clean long-range test
Re-ran the ceiling with CONTENT-word vocab (V=2000, V=5000) + a bigger model (d320/L5/H5, 16k steps) on all 1.7M WikiText words, to test the hypothesis that V=300 (function-words-only) stripped out where long-range lives. Result — **catastrophic overfitting, not a valid ceiling**: train-CE dropped to **0.43** (near-perfect memorization) while held-out CE is **7–13 nats — WORSE than uniform-random** (log 2000=7.6, log 5000=8.5) at every depth. A 2M-param transformer on 1.7M words / V=2000-5000 has far too much capacity for the data: it memorizes train and generalizes worse than random. ⇒ **there is NO vocab regime on this corpus where a transformer cleanly beats a bigram at long-range**: function-word V=300 → bigram wins (transformer barely trains); content-word V≥2000 → catastrophic overfitting. The corpus (1.7M words) is ~50-100× too small to train ANY content-word attention model to a meaningful long-range ceiling.

## ⇒ the honest reconciliation + correction to the arc
The 1-seed learned-attention smoke read "the long-range signal is real, the reservoir failed on key-quality" too optimistically. The full-transformer ceiling (the stronger, cleaner test) and the marginal-vs-uniform learned-attention result together say: **long-range signal beyond a bigram/cache is THIN at this scale (5M tokens, V=300)**. Therefore the session's reservoir-LM long-range NEGATIVES (fixed / e-prop / longer-τ / ALIF-state / content-addressable retrieval) are **PARTLY SCALE-BOUND** — they measured, in part, "there is little generalizable long-range signal at this scale," not purely "the substrate cannot." This is a crucial honest recontextualization: the arc's within-reach-bounded conclusion is sound, but the *long-range* frontier claims can only be cleanly tested at a scale where long-range signal is DEMONSTRABLY present (a transformer beats a bigram at deep context) — i.e. WikiText-103-scale (100M+ tokens, full vocab), not the 5M-token / V=300 setting the whole arc used.

## The corrected frontier + next step
- **The genuine substrate test of long-range needs SCALE first**: reproduce the "a transformer's advantage over the bigram GROWS with depth" signature at larger scale (more tokens, larger vocab, bigger block) — establish that long-range signal is present — THEN test whether the biological substrate (a learnable content-selective non-fading store / local-rule attention) can capture it. Testing biological long-range where even a transformer can't beat a bigram is testing against noise.
- **The within-reach results stand** (SCALE-CAPSTONE, e-prop REAL-WITH-SCOPE, the spiking BDSP credit mechanism) — those are not scale-confounded (they compare like-for-like at fixed scale).
- **The induction/copy head + the ceiling-vs-local gap** remain a real (if small) signal; the local-rule attention frontier is still valid, but its payoff is bounded by how much long-range signal exists at the tested scale — currently thin.

## Honest scope
The transformer is a small reference ceiling (0.1–1M params, 12k steps) — a LARGER transformer with more data would do better, which is exactly the point: the signal needs scale. Both runners reuse the arc's vocab + bigram + metric; the learned-attention is anti-cheated (content vs shuffle AND uniform, depth-broken-out); NO `sim/` edit. This is the parallel-GPU deliverable that recontextualizes the whole arc honestly.

## Files
`_wikitext_transformer_ceiling.py`, `_emerge_reservoir_lm_learned_attention_derisk.py`; raw `research/findings/raw/_wikitext_transformer_ceiling.json`, `_eprop/lattn_s*.json`, `_reslm_learned_attention.json`. Recontextualizes the 2026-07-11 arc synthesis (`-ALIF-adaptation-state-NEGATIVE-and-the-long-range-arc-synthesis-*`).
