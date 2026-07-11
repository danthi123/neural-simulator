# SCALE — the fading-memory deep-context loss is FIXED: a reservoir + recent-token read-out ROBUSTLY beats the bigram at EVERY context depth on real text (aggregate +0.47 nats, +0.49–0.51 at deep context) — the earlier "plateau at the bigram" was a missing-feature artifact, not an architecture ceiling

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_ngram_hybrid_derisk.py` (reuse-by-import: the Rung-1 reservoir + read-out + the real-corpus loader + the context-depth buckets; TinyStories V=200, 2-seed 42/100, 2500 train sentences; NO `sim/` edit, NO BPTT).
**Verdict:** **The emergent generator has a ROBUST, mechanistic advantage over the bigram on real text — once the read-out reads the recent token alongside the reservoir state.** The context-depth analysis showed the reservoir-only read-out beats the bigram at mid-depth but LOSES at deep context (its running-cumulative feature washes the recent tokens out). The standard echo-state fix — give the read-out the recent token identity (what the fading state loses) ALONGSIDE the reservoir's higher-order context — makes it beat the bigram at EVERY depth, most at deep context. The read-out is still a shallow linear-softmax trained by the one-step local delta rule (NO BPTT); it simply reads `[reservoir_state ⊕ onehot(recent tokens)]`.

## Result — reservoir + recent-K tokens vs bigram, CE margin by context depth (2-seed, n_pool=300; + = generator beats bigram)
| read-out feature | aggregate margin | d1 | d2 | d3 | d4–5 | **d6–9** | **d10+** |
|---|---|---|---|---|---|---|---|
| **K=0** reservoir only | +0.121 | +0.14 | +0.31 | +0.31 | +0.15 | **−0.02** | **−0.02** |
| **K=1** reservoir + prev token | **+0.467** | +0.17 | +0.43 | +0.60 | +0.51 | **+0.51** | **+0.49** |
| K=2 reservoir + prev 2 tokens | +0.390 | +0.16 | +0.35 | +0.53 | +0.40 | +0.42 | +0.44 |

- **K=1 (reservoir + previous token) beats the bigram at EVERY depth** — and the deep-context margin FLIPS from −0.02 (reservoir-only, the fading-memory loss) to **+0.49–0.51**. Aggregate margin jumps +0.121 → **+0.467**.
- **The reservoir's genuine contribution is cleanly isolated at deep context.** A bigram uses ONLY the previous token; the K=1 read-out has the previous token TOO, so its +0.49–0.51 nats at deep context is *purely the reservoir's higher-order memory* adding value beyond the bigram — exactly where the bigram is structurally blind (>1 token back).
- **K=2 (+prev 2 tokens) is slightly worse than K=1** (+0.390) — the extra one-hot adds parameters/mild overfit; one recent token is the sweet spot at this scale.

## Why this is legitimate (not "just adding the bigram's feature")
Reading the input alongside the reservoir state is the STANDARD echo-state read-out (Jaeger 2001: `W_out·[state; input]`), and biologically a cortical read-out can access both the sustained recurrent state and the recent afferent input. The read-out is unchanged in KIND — a shallow linear-softmax, one-step local delta, NO backprop-through-time. The point is not that the hybrid beats the bigram (it contains the bigram's feature, so of course it should) — it is that **the reservoir's higher-order context adds ~0.5 nats BEYOND the bigram at deep positions**, which is the reservoir's real, isolated contribution to next-token prediction on natural text.

## ⇒ significance — the scale story, corrected
The earlier "reservoir hovers AT the bigram / plateaus" concern (`-reservoir-size-vs-data-levers`, `-mid-depth-wins-loses-deep`) was NOT an architecture ceiling — it was the reservoir-only read-out discarding the sharp recent-token signal it washes out. Restoring it (a trivial, principled feature change) gives a **robust +0.47-nat advantage over the bigram at ALL depths**, growing to ~+0.5 at deep context. So on real text the emergent generator (fixed spiking reservoir + a one-step-local-delta read-out over `[state ⊕ recent token]`) is meaningfully, mechanistically better than the bigram, precisely where higher-order/discourse context matters. This is the boundary-surpassing workflow end-to-end: aggregate plateau → context-depth diagnosis → fading-memory limit → the recent-token fix → a robust win.

## ⚠ RIGOR CALIBRATION (same-day — the bigram is a WEAK bar; measured against a LEARNED K-gram)
The +0.47 "beats bigram" is MOSTLY recovering 2–3-gram context, not deep reservoir memory. Isolating the reservoir's contribution by comparing `[reservoir ⊕ recent-K tokens]` against `[recent-K tokens only]` (a LEARNED K-gram, a much stronger baseline than the add-1 bigram), 2-seed on TinyStories:
| reservoir's contribution beyond a learned K-gram (nats, +=reservoir adds) | d1 | d2 | d3 | d4–5 | d6–9 | d10+ |
|---|---|---|---|---|---|---|
| vs learned 1-gram (K=1) | −0.21 | +0.01 | +0.19 | +0.20 | +0.19 | +0.17 |
| vs learned 3-gram (K=2) | −0.27 | −0.20 | −0.02 | +0.01 | +0.06 | +0.08 |
| vs learned 4-gram (K=3) | −0.29 | −0.24 | −0.14 | −0.04 | +0.02 | +0.03 |

- A **learned 3-gram already beats the add-1 bigram by +0.31** — so most of the reservoir+token "win over the bigram" is just better-estimated short-range n-gram context.
- The reservoir's **genuine contribution BEYOND a learned 3-gram is SMALL** (+0.02–0.08 nats at deep context only) and it **HURTS at short context** (−0.2 to −0.3, over-parameterizing the easy predictions).
- **Honest conclusion:** on TinyStories (simple, short-range-dominated text) the emergent reservoir generator is ≈ a learned 3-gram + a modest genuine long-range residual. Its higher-order memory is real but mostly redundant with a few recent tokens on this corpus. The residual SHOULD grow on text with real long-range dependencies (WikiText) — the pinned next test. This corrects the headline: the robust win is over the BIGRAM; over a strong learned n-gram the edge is a small deep-context residual.

## OPEN / next
- Confirm the K=1 hybrid's robustness as DATA scales (does the +0.47 hold/grow at 8000+ sentences and larger vocab — where the reservoir-only edge had vanished but this hybrid should not, since it strictly dominates the bigram).
- Compare against the TRIGRAM (a fairer higher-order baseline than the bigram) — does the reservoir still add value beyond a well-estimated trigram at deep context? (The trigram overfits at small data; at more data it is the real bar.)
- The Rung-2 WM buffer is the deeper distal mechanism (the recent-token feature is a K-token register; the buffer is a non-fading latch) — relevant when the dependency is beyond the recent K.

## ⚠⚠ WIKITEXT RESULT — the "residual should grow on harder text" hypothesis is FALSIFIED (see the SCALE CAPSTONE finding, 2026-07-11)
The pinned test — does the reservoir's genuine contribution beyond a learned n-gram GROW on harder real text (WikiText) — came back the OPPOSITE way. On WikiText (V=300, 3000 sentences, 2-seed 42/100) the reservoir's contribution beyond a learned K-gram is NEGATIVE or ~zero at EVERY context depth (K=1 deep −0.04…−0.01; K=2/K=3 all-depth negative, down to −0.79 at short context). The reservoir is genuinely active (res+tok CE 3.093 ≠ tok_only 2.967; res+tok still beats the add-1 bigram +0.044) — its contribution is simply net-negative on harder text. So the TinyStories deep-context residual (+0.02–0.08) does NOT generalize; it was largely n-gram-recovery on simple, repetitive text. **The honest capstone conclusion:** the fixed 300-neuron reservoir + linear read-out is n-gram-competitive-at-best on real text and adds no usable long-range structure on harder text (cause = capacity ceiling and/or overfitting-at-this-data-budget; not separated). Full analysis + the next-mechanism gate: `2026-07-11-SCALE-CAPSTONE-reservoir-is-ngram-competitive-not-transformer-competitive-on-real-text.md`.

## Files
`_emerge_reservoir_lm_ngram_hybrid_derisk.py` (`--k-recent`); raw `research/findings/raw/_hybrid/np300_s{42,100}.json`. Follows the context-depth finding (`-reservoir-wins-mid-depth-loses-deep-*`).
