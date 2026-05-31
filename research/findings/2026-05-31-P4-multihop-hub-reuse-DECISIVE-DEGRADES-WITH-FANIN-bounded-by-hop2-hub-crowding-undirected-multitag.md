# P4 multi-hop hub-reuse DECISIVE = DEGRADES-WITH-FANIN. 2-hop transitive chaining on the validated 160/32-concept G.20 multitag stack is REAL at low hub fan-in (multi-seed mean full-2hop 0.833 at fan-in 2, >> chance 0.094) but COLLAPSES to 0.000 at fan-in 8. The clean 8/8 prior result was the fan-in-1 easiest case. Bottleneck mechanistically LOCATED: hop-1 (find the hub) is flat/fine at 0.83 across all fan-in; the entire loss is at hop-2 -- querying a crowded hub returns its many INCOMING nouns and BURIES the one OUTGOING edge, because multitag retrieval is UNDIRECTED/aggregate-ranked. A principled cheap fix (directional tag-name filtering at hop-2) is identified for follow-up.

**Date:** 2026-05-31
**Status:** Controller verdict (DEGRADES-WITH-FANIN) on the decisive multi-seed hub-reuse scrutiny of the P4 multi-hop reasoning capability. Honest partial/bounded result on the validated conversational stack. The clean-condition 8/8 finding (2026-05-31-P4-multihop-...PASS...md) is hereby corrected: it did NOT generalize to realistic hub reuse. Subagent a3d6187f2cb233796 ran the decisive test (seeds 42/43/44, 32-concept sparse tier, ~19 min wall); this doc is the controller's scrutinized verdict.

## Decisive measurement (3 seeds; hub fan-in 2/4/8; G.20 sparse multitag)

Hub graph: one hub per fan-in level, pairwise-disjoint nouns/Cs so the ONLY thing varying between levels is hub crowding. fan-in 2 hub `fast` ({ball,key}->fast; fast->small); fan-in 4 hub `hot` ({bird,flower,leaf,fruit}->hot; hot->dry); fan-in 8 hub `big` ({apple,river,dog,cat,tree,fish,mouse,frog}->big; big->red). A=noun (cross-bridge to adj HUB); HUB+C=adj (intra-bridge). Encode/recall via the shipped query_concept/encode_pair (imported, g20_multibridge.py byte-unmodified).

| fan-in | hop-1 (A->H top-1, mean) | hop-2 (C in crowded-hub top-3, mean) | full 2-hop (mean) |
|---|---|---|---|
| 2 | 0.83 | 1.00 | **0.833** |
| 4 | 0.83 | 0.33 | **0.333** |
| 8 | 0.83 | 0.00 | **0.000** |

Chance (specific C in top-3 of 32 adj): 0.094. Anti-cheat (A->C not retrievable before HUB->C edge): 13/14, 14/14, 14/14 across seeds -- confirmed both by construction and empirically.

PRE-REGISTERED VERDICT (frozen before the run): full-2hop 0.833 at fan-in 2 (>= 0.50, rules out NEGATIVE; >> chance, real) but 0.000 at fan-in 8 (< 0.50, rules out ROBUST) => **DEGRADES-WITH-FANIN**. The capability is real but bounded by hub crowding.

## Controller scrutiny (the verdict survives; the bottleneck is located, not generic)

1. The DEGRADES verdict is MECHANICALLY CORRECT on the recorded numbers (fan-in 2 >= 0.50 not NEGATIVE; fan-in 8 < 0.50 not ROBUST).
2. The bottleneck is LOCATED, not a generic failure: hop-1 is FLAT at mean 0.83 across all three fan-in levels -- finding the hub from the noun is never the problem. The collapse is ENTIRELY at hop-2. The full-2hop and hop-2-marginal columns track each other exactly -> the loss is the hop-2 hub query, not chain-propagation noise or drift.
3. The mechanism is FUNDAMENTAL, not a tuning bug: multitag retrieval is UNDIRECTED / aggregate-ranked -- querying a concept returns ALL tags containing it, ranked by aggregate rate. A hub with 8 incoming noun-edges + 1 outgoing C-edge -> the 8 incoming nouns dominate the ranking and the outgoing C is buried below top-3. The per-trial logs are literal: querying `big` (fan-in 8) returns [tree, river, apple] / [dog, cat, river] (the incoming nouns); `red` (the C) never reaches top-3. This is an inherent property of the representation, not a fixable parameter.
4. The fan-in 4 seed-dependence is informative (transition zone): seed 43 gets 4/4 (the HUB->C edge survives the crowd of 4) while 42/44 get 0/4 -- so ~4 incoming edges is the coin-flip threshold; by 8 it is unanimously buried.
5. NOT too pessimistic: the clean 8/8 prior result depended entirely on each middle term having fan-in 1 (exactly one incoming association). Real association graphs have HUB concepts ("big", "hot" -- many things are big). So the realistic regime is the crowded one, where it fails. The honest capability statement is: multi-hop chaining works only when middle terms are low-fan-in, which is the exception not the rule.

## Honest disposition + the located cheap follow-up

This is an honest BOUNDED capability on the validated stack: 2-hop transitive retrieval is real but collapses under realistic hub reuse, because undirected multitag retrieval cannot distinguish a hub's INCOMING edges from its one OUTGOING edge at hop-2. It does NOT contradict the clean 8/8 -- it CONTEXTUALIZES it as the fan-in-1 easiest case.

The bottleneck is located precisely enough to suggest a PRINCIPLED CHEAP FIX worth testing next: the multitag tags are NAME-ORDERED ("remember a is b" -> tag "a_b", cue-first). So at hop-2, when querying the hub for its OUTGOING edge, filter to tags where the hub is the FIRST token (hub_*) and ignore the incoming X_hub tags. If the encode order is reliable, this directional filter should surface C even at fan-in 8 -- rescuing multi-hop. This is a throwaway probe (reuses the tag store + recall logic, adds a name-direction filter; no protected-module edit). FOLLOW-UP DECISION: if directional filtering rescues fan-in 8 multi-seed (>= 0.50) -> multi-hop becomes a robust retrieval-reasoning capability; if it does not -> the limit is deeper (encode-order not reliable / the substrate aggregates regardless) and multi-hop is honestly bounded to low-fan-in chains. Either is an honest characterization.

Per the night-synthesis P4 frame: this is instrumental capability characterization of the working stack (which is itself the oracle-shortcut-in-another-form, not a substrate biologization); the biologization BOUNDARY (DG separation-vs-reliability) stays the banked biology-translatable deliverable.

## Discipline

Throwaway probe only (research/findings/raw/_multihop_hubreuse_test.py); g20_multibridge.py byte-unmodified; nothing of the shipped stack changed; reuse-by-import. Pre-registered three-state bar set before the run, not tuned. The PARTIAL was scrutinized for the bottleneck location + whether it is too pessimistic (it is not -- realistic graphs are hub-heavy). The clean 8/8 prior finding is honestly corrected, not quietly superseded. Subagent ran synchronously to completion and reported real measured numbers (no hand-back of an unfinished run).
