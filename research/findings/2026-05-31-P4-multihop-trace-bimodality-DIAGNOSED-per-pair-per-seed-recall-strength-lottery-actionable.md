# P4 multi-hop "trace" bimodality DIAGNOSED: it is per-pair x per-seed RECALL-STRENGTH variance (an engram-binding lottery), NOT a directional-filter flaw and NOT seed-global. Stim'ing the encoded tag and reading the target adjective's recall rank/32 shows big->red is rank 2 (seed 42) / rank 8 buried (seed 43) / rank 1 (seed 44) -- exactly mirroring the multi-hop 8/8 / 0/8 / 6/8. Other pairs are weak on OTHER seeds (hot->dry rank 4 on seed 44; cold->wet rank 8 on seed 44; seed 43's hot->dry is the STRONGEST at rank 1). So each pair's engram binding strength is a per-seed structural lottery; where the target falls below trace's top-3, multi-hop fails. Actionable in principle (reinforce/strengthen weak bindings); the directional filter is sound.

**Date:** 2026-05-31
**Status:** Diagnostic of the RESCUED-but-BIMODAL directional multi-hop "trace" capability (finding 2026-05-31-P4-multihop-directional-fix-...). Localizes the residual limiter precisely. Throwaway probe (research/findings/raw/_perseed_binding_diagnostic.py); g20_multibridge.py reused byte-unchanged via import.

## Measurement (stim each encoded tag, read TARGET adjective's recall rate + rank among 32 adj)

All HUB->C pairs are intra-bridgeC_adj, so only bridgeC_adj loaded per seed (cheap). For each pair: encode_pair -> recall_rates(tag) -> target's rate + rank.

| pair | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| big->red | 645 / rank 2 | **235 / rank 8** | 889 / rank 1 |
| hot->dry | 537 / rank 1 | 998 / rank 1 | **359 / rank 4** |
| fast->small | 578 / rank 2 | 821 / rank 1 | 579 / rank 2 |
| cold->wet | 548 / rank 1 | 521 / rank 2 | **208 / rank 8** |

(rank 1 = target is the top recalled concept; trace uses top-3, so rank > 3 = the target is missed.)

## Diagnosis (decisive)

- The multi-hop "trace" bimodality is RECALL-STRENGTH: where the encoded tag, when stimulated, recalls the target adjective at rank > 3, trace's hop-2 misses it. big->red rank 8 on seed 43 EXACTLY explains the multi-hop 0/8 on seed 43; rank 1-2 on seeds 42/44 explain the 8/8 and 6/8.
- It is PER-PAIR x PER-SEED, NOT seed-global: seed 43 is the WORST for big->red (rank 8) but the BEST for hot->dry (rank 1, rate 998). Seed 44 is best for big->red (rank 1) but worst for cold->wet (rank 8). So no seed is globally bad; each (pair, seed) is an independent engram-binding lottery.
- It is NOT the directional filter: the filter correctly isolates the big_red tag on every seed (verified in the directional finding); the tag simply recalls 'red' weakly on seed 43's bridge.
- It is NOT a chaining problem: the weakness is at single-tag recall (hop-2), confirmed by the direct stim measurement here.

## Why the lottery (mechanism)

The sparse engram encode (encode_pair_engram_sparse) drives the two concepts' sparse patterns and tags the co-firing neurons in one pass. How strongly the target's neurons get captured depends on the per-seed sparse-pattern overlap of the two concepts + the stochastic co-firing during the single encode pass. For some (pair, seed) combinations the target's neurons are weakly captured -> weak recall. This is the same per-seed structural variance that gates the whole stack, here localized to single-pass engram capture strength.

## Actionable fix (specified) + the cheap test in flight

The fix is to STRENGTHEN weak bindings so the target recalls at rank <= 3 on every seed. Two routes: (a) the compose-engram balanced-teacher pattern (drive BOTH concepts strongly during encode -- but the SPARSE encode path has no exposed teacher_pA knob, so this is a deeper change to encode_pair_engram_sparse, not rushed); (b) REINFORCEMENT -- re-encode the weak pair multiple times (no code change) and check whether the target's recall rank improves. The reinforcement test (_perseed_binding_reinforce.py) is running now on the two weak cases (seed 43 big->red, seed 44 cold->wet). If reinforcement raises the target to rank <= 3 -> a simple actionable fix (encode weak pairs more); if not -> the per-seed sparse-pattern structure caps the binding and the deeper balanced-teacher encode (or accepting the lottery) is the honest disposition.

## Disposition

The "trace" multi-hop capability is shipped and SOUND (directional filter correct); its bimodality is now precisely diagnosed as per-pair-per-seed engram recall-strength (a binding lottery), with a specified actionable direction (reinforce / balanced-teacher encode). This completes the multi-hop characterization: clean PASS -> hub-crowding DEGRADES -> directional RESCUE -> bimodality DIAGNOSED -> actionable. Honest frame: this is instrumental P4 usability of the working retrieval stack; the biological-symbol-grounding BOUNDARY (near-orthogonality unmet) remains the banked biology-translatable deliverable.

## Discipline

Throwaway probe only; g20_multibridge.py byte-unchanged (reused via import); GPU/CuPy; reuse-by-import; no protected/frozen/moat edit. The diagnosis was pre-registered (the probe's READ line stated the encoding-strength vs structural disambiguation before the run) and the result decisively localized recall-strength (rank correlates with the multi-hop per-seed result). The deeper balanced-teacher fix is NOT rushed; the cheap reinforcement test runs first.
