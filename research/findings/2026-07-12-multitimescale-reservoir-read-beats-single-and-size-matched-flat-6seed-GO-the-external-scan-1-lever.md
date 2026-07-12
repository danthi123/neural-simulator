# Multi-timescale reservoir read (the #1 external-scan lever) beats single-timescale AND a size-matched flat reservoir AND the bigram — 6-seed GO, the gain is TIMESCALE DIVERSITY not more units

**Date:** 2026-07-12
**Status:** ✅ 6-seed GO — a genuinely-new, cheap, reuse-by-import, emergent lever (surfaced by the deep-credit external-scan Workflow, ranked #1) works on our own reservoir. Runner `research/runners/_reslm_multitimescale_derisk.py`; NO `sim/` edit, NO shared-runner edit.
**Frontier:** the emergent-generation SCALE frontier. Ueda 2025 + our probes: the fixed reservoir is n-gram-bounded at small scale; the DeepESN literature says multi-timescale FORWARD state reaches more with the same units. This tests the ONE relative question tractable at our scale.

## Mechanism (reuse-by-import, cheap)
The DeepESN multi-timescale lever (external scan #1; Gallicchio-Micheli / ICANN-2025 time-scales-in-DeepESN): read the SAME on-bridge spiking reservoir (EMERGE-82 OnBridgeLSM) at MULTIPLE timescales and concat → the read-out. Slow = `running_cumulative` (whole-prefix mean spike-rate); fast = `per_window` (recency, the token's `_T_STEP` window). MULTI = concat(slow, fast) per token → 2·n_pool features. No new reservoir, no learning-rule change, two deterministic reads of the same washed trajectory.

## Result — 6-seed (42/43/44/100/101/102), TinyStories V=200, n_pool=300
| arm | held-out next-token CE (mean) | vs MULTI |
|---|---|---|
| single-timescale (running_cumulative, n_pool feats) | 3.391 | +0.119 |
| **MULTI-timescale (concat slow+fast, 2·n_pool feats)** | **3.272** | — |
| size-matched FLAT (single-timescale, **2·n_pool** units) | 3.630 | +0.358 |
| bigram | 3.675 | +0.403 |

**MULTI beats single on 6/6 · beats the size-matched flat on 6/6 · beats the bigram on 6/6.** The load-bearing anti-cheat: the size-matched flat reservoir (2·n_pool units, SAME feature dim as MULTI, single timescale) is WORSE than even the single-timescale n_pool read (3.630 vs 3.391) — more units alone HURT at fixed data (data-starved, consistent with the R3 SCALE finding) — so MULTI's win is **timescale DIVERSITY, not the extra features / more units.** A clean relative improvement to the emergent generator's above-bigram margin.

## Honest scope + escalation
- A RELATIVE CE improvement (−0.119 nats over single) at our tractable scale — real + 6-seed + anti-cheat-confirmed, but modest and (per Ueda) the reservoir path is scale-gated overall (bounded ~60% BLiMP ceiling at 16-65k units/100M words; this lever helps reach that ceiling with fewer units, it does not exceed it).
- Two timescales only (slow+fast). More timescales / a true multi-leak STACK (N reservoirs, staggered leaks — the full DeepESN) is the next rung; the leak-shuffle control + a real-scale (TinyStories-23.7M / WikiText-103) run are the fuller anti-cheat/ceiling escalation the external-scan doc specified.
- Emergence-bar: this is a FORWARD-STATE representation lever (multi-timescale reading of a fixed reservoir), reuse-by-import, no hand-built capability — a legitimate cheap step on the reservoir substrate; it is NOT the path PAST the reservoir's bounded ceiling (that is learned recurrence / deep credit = the standing dendritic frontier).

## Files
`research/runners/_reslm_multitimescale_derisk.py`; `raw/_multits_s{42,43,44,100,101,102}.json`. Surfaced by `2026-07-12-deep-credit-on-spikes-external-scan-verdict-...md` (ranked lever #1). Builds on the R3↔generation convergence + Ueda 2025 (arXiv:2503.01724).
