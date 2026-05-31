# P4 (validated conversational stack): multi-hop (2-hop transitive) reasoning WORKS under clean conditions -- 8/8 on the 160-concept G.20 multitag stack via iterated single-hop chaining. Honest scope: this is the EASIEST case (all-distinct words, no hub competition at hop-2); the decisive HUB-REUSE + multi-seed scrutiny is in flight. The mechanism is two chained ~100%-reliable single-hop lookups linked by a shared tag-name middle term, not learned inference.

> ## RESOLVED by the hub-reuse decisive test (2026-05-31): DEGRADES-WITH-FANIN
> The hub-reuse + multi-seed scrutiny (finding 2026-05-31-P4-multihop-hub-reuse-DECISIVE-DEGRADES-WITH-
> FANIN-...md) is in: the clean 8/8 below did NOT generalize. Multi-seed full-2hop = 0.833 at hub fan-in 2
> but COLLAPSES to 0.000 at fan-in 8 (chance 0.094). The clean 8/8 was the fan-in-1 easiest case (each
> middle term in exactly one chain -> no hub competition). Bottleneck LOCATED: hop-1 is fine (flat 0.83);
> the entire loss is at hop-2 -- querying a crowded hub returns its many incoming nouns and buries the one
> outgoing edge, because multitag retrieval is undirected/aggregate-ranked. Real association graphs are
> hub-heavy, so the realistic regime is the failing one. A principled cheap fix (directional tag-name
> filtering at hop-2) is queued. Read this doc through that resolution.

**Date:** 2026-05-31
**Status:** Clean-condition PASS for 2-hop transitive reasoning on the validated G.20 multitag conversational stack (the first P4 step after pivoting from the DG-biologization boundary). Genuine result (beats the prior corrected-NEGATIVE 0.25 via a DIFFERENT mechanism), but its final verdict depends on the hub-reuse + multi-seed scrutiny now running. The honest mechanistic read is recorded.

## What was tested + result

Pivoting from the DG-biologization fundamental boundary to advance the WORKING conversational capability (the G.20 multitag stack: 160 concepts, single-hop retrieval 90%, cross-bridge encode, hierarchy, yes/no, tokenization), the first step attacked its known frontier: MULTI-HOP reasoning. Does chaining the validated 90% single-hop multitag retrievals give reliable 2-hop transitive inference (A->B encoded, B->C encoded, query A -> B -> query B -> C, reaching C without A->C ever encoded)?

Probe `research/findings/raw/_multihop_reasoning_test.py` (throwaway): loaded the 5 sparse 160-concept bridges (74s), encoded 8 chains A->B->C with ALL-DISTINCT words (apple->big->hot, river->wet->cold, dog->fast->small, cat->soft->warm, tree->tall->green, sun->yellow->white, moon->cool->blue, fire->red->dry), A=noun (cross-bridge to adj B), B,C=adj (intra-bridge).

| metric | result |
|---|---|
| single-hop confirmation (apple -> ?) | big @ rate 869 (decisive) |
| HOP-1 (A -> B top-1) | 8/8 = 1.000 |
| HOP-2 marginal (true B -> C top-3) | 8/8 = 1.000 |
| FULL 2-HOP transitive (A -2hop-> C top-3) | 8/8 = 1.000 |
| anti-cheat (A->C NOT directly retrievable) | 8/8 = 1.000 |
| chance (top-3 of 32 adj) | 0.094 |

VERDICT (pre-registered): MULTI-HOP WORKS under these conditions (2-hop 1.000 >= 0.50, >> 0.25 prior, >> 0.094 chance; anti-cheat 1.000). The 2-hop genuinely traverses A->B->C (e.g. query "big" -> [apple, hot, new] surfaces apple via the apple_big tag and hot via the big_hot tag).

## Honest scope + scrutiny (the load-bearing caveats)

1. EASIEST case: the chains used ALL-DISTINCT words, so each middle term B appears in only ONE chain -> hop-2 has ZERO competition. A realistic association graph has HUB concepts ("big" = many things are big); querying a hub returns many competing associates, and the 2-hop must still surface the correct C. UNTESTED here. THE DECISIVE SCRUTINY (hub-reuse + multi-seed) is running now (subagent a3d6187f2cb233796, _multihop_hubreuse_test.py, varying hub fan-in 2/4/8 across seeds 42/43/44). The clean 8/8's final status (ROBUST vs DEGRADES-WITH-FANIN vs NEGATIVE) depends on it.
2. Seed 42 only, single run, n=8.
3. Does NOT contradict the prior corrected-NEGATIVE (chain transitive 1/4): that used a DIFFERENT mechanism (compose_concept_chain_test, cross-pool STDP weight growth on a 16-concept v16 bridge). This is engram-tag stim-recall over multitag aggregation -- a different, cleaner mechanism.
4. HONEST mechanistic read: the 2-hop here is two chained ~100%-reliable single-hop lookups linked by a shared middle term that appears in BOTH the A_B tag and the B_C tag. Stacking two near-lossless lookups composes losslessly. This is clean associative chaining, NOT learned inference. Whether it is a useful "reasoning" capability depends on robustness to hub crowding (the in-flight test).

## Disposition

A genuine clean-condition multi-hop result on the validated stack -- the first positive P4 step. The decisive hub-reuse + multi-seed test determines whether it is a ROBUST conversational reasoning capability or a clean-case-only artifact. Either is honest. reuse-by-import; no protected module touched; nothing overclaimed (the mechanistic caveat is foregrounded).
