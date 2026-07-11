# The GENUINE long-range test (CROSS-SENTENCE content-addressable retrieval over a persistent store) is a clean 3-seed NEGATIVE: content-addressing over the reservoir substrate (fixed OR e-prop-learned keys) is NO BETTER THAN a random/uniform cross-sentence token BAG — the retrieval's whole benefit is a topic/cache prior (shuffle-invariant), not content-addressing. ⇒ the genuine long-range content-addressable win is UNACHIEVED on the reservoir substrate; it needs DEEPER REPRESENTATIONS (deep credit), the deep frontier

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_crosssentence_retrieval_derisk.py` (built by a controller-verified build subagent; contiguous-document loader → passages of ~10 in-order sentences, passage-level train/eval split, a persistent cross-sentence store of PRIOR sentences' (reservoir-state key, next-token value), retrieval over it interpolated with the base; reuse-by-import; NO `sim/` edit — verified byte-empty). Numpy, WikiText, 3-seed.
**Verdict:** closes the rung-2 long-range investigation with an honest NEGATIVE — the mechanism the whole arc hypothesized (content-addressable retrieval to reach long-range) does NOT beat a cache/bag prior even cross-sentence, on the reservoir substrate.

## Why this is THE genuine test (and why the prior "win" was a confound)
The committed interp "long-range CE-win" was CORRECTED to a within-sentence-token CACHE (shuffle-invariant), because retrieval was intra-sentence over ≤16-token sentences (a shuffled retrieval = the same sentence-bag). The genuine content-addressing test the shuffle control can actually adjudicate is CROSS-SENTENCE: a persistent store over PRIOR sentences of a contiguous passage, where a RANDOM cross-sentence bag should NOT help (too diffuse) but a content-addressed retrieval of the RELEVANT prior context SHOULD. The load-bearing signal is **content − shuffle at cross-sentence positions** (NOT content − base, which any bag inflates).

## Result — 3-seed cross-sentence CE (learned keys, passages=200, n_pool=200)
| seed | base | content | shuffle | uniform (bag) | content−SHUFFLE (real signal) | best arm |
|---|---|---|---|---|---|---|
| 42 | 4.585 | 3.637 | 3.647 | **3.520** | +0.010 | uniform |
| 43 | 3.257 | 3.187 | 3.193 | **3.161** | +0.006 | uniform |
| 44 | 3.953 | 3.469 | 3.465 | **3.387** | −0.004 | uniform |
- **content − shuffle ≈ 0** (mean +0.004; negative on seed 44; d10-99 by-depth +0.010/+0.005/−0.004) — content-addressing is statistically indistinguishable from a random-key cross-sentence bag. No content signal, and it does not grow with depth.
- **The UNIFORM bag is the BEST arm on ALL 3 seeds** (uniform < content < shuffle) — the retrieval's entire benefit is the *marginal distribution of prior-sentence next-tokens* (a topic/recency CACHE prior, Merity-2016 neural cache); actually attending by content-similarity slightly HURTS vs just averaging.
- **content − base is large but confoundable** (+0.95/+0.07/+0.48) — exactly the number the correction warns about: base is weak cross-sentence, so ANY bag lifts it; the uniform arm proves that lift is not content-addressing.
- The arms genuinely differ (content ≠ shuffle ≠ uniform) and sharper attention (β=8) did not rescue content — a real measurement, not a harness bug.

## ⇒ the honest close of the rung-2 long-range arc + the deep frontier
The reservoir-LM arc hypothesized that a non-fading, content-addressable store reaches the long-range structure the fading reservoir cannot. Across every honest test this session that hypothesis FAILS on the reservoir substrate:
- fixed reservoir-state keys: `content ≈ shuffle` (bad keys);
- LEARNED (e-prop) keys: `content ≪ shuffle` on the APPEND metric (content-addressing is real there) but the retrieval does NOT beat base;
- INTERPOLATION: beats base at deep but SHUFFLE-INVARIANT (a within-sentence cache, corrected);
- CROSS-SENTENCE (this, the genuine test): `content ≈ shuffle`, the uniform bag is best — NO content signal.
**In every case the retrieval's benefit is a cache/topic BAG prior (a simple, known LM trick), NOT content-addressable long-range retrieval.** The binding limit is the KEY QUALITY: the reservoir keys (even e-prop-within-reach-learned) do not discriminate cross-sentence linguistic context finely enough for content-addressing to beat a bag. ⇒ genuine long-range content-addressable retrieval needs MUCH better keys = **learned representations encoding genuine distal structure, which requires reaching distal dependencies during learning = biological DEEP CREDIT** (the dendritic / deep-credit substrate — the owner's standing priority and the arc's independently-re-derived fundamental lever). A retrieval bolt-on over reservoir keys does not get there; the deep-credit substrate is the genuine remaining frontier.

## Honest scope
3-seed cheap-scale (200 passages, n200, learned keys). The negative is robust + mechanistically coherent (uniform-best across all seeds, content−shuffle≈0, matches the corrected interp finding + the mechanism). A much larger scale COULD in principle surface a small content signal (the number to watch = content−shuffle growing positive at d10-99, which it does not), but the direction is clean and the burden is now on a better-key (deep-credit) substrate, not a bigger retrieval. numpy rate-level; NO `sim/` edit.

## Files
`_emerge_reservoir_lm_crosssentence_retrieval_derisk.py`; raw `research/findings/raw/_eprop/xsent_s*.json`, `_reslm_crosssentence_smoke.json`. Closes the arc: `2026-07-11-learned-keys-plus-interpolation-*` (the corrected confound) + `-content-addressable-retrieval-needs-LEARNED-keys-*` (the convergence on deep credit) + `-the-long-range-wall-is-the-fading-STATE-*`.
