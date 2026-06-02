# Flat-distinct RESOLVES: robust cross-bridge biological composition on structured facts (the honest fix) -- 2026-06-02

## Context
The hierarchical-320 shortcut (bind each concept with its bridge-role to dodge the duplicate-pattern problem
without retraining) was RETRACTED: on STRUCTURED facts (noun/verb/adjective) it hit the nesting wall --
full-3-slot QA 0.000/0.950/1.000 at seeds 42/43/44, catastrophic at seed 42. The honest path: DISTINCT FLAT
codes via distinct-seed retraining (single-level composition, no 2nd binding level).

## Result -- distinct-flat codes, the SAME structured test
Retrained bridgeB (verbs) @ seed 43 and bridgeC (adj) @ seed 44 (bridgeA nouns @ seed 42 existing) -> 192
DISTINCT FLAT codes (between-concept cos mean 0.108, max 0.604 -- distinct, no duplicates). STRUCTURED SVO
composition (agent=noun / action=verb / patient=adj), full-3-slot QA, composition seeds 42/43/44:

| seed | hierarchical (retracted) | **flat-distinct** |
|------|-------------------------:|------------------:|
| 42 | 0.000 | **1.000** |
| 43 | 0.950 | **1.000** |
| 44 | 1.000 | **1.000** |
| mean | 0.650 | **1.000** |

VERDICT: RESOLVES. Distinct flat codes compose ROBUSTLY on the realistic structured-fact distribution at ALL
seeds, INCLUDING seed 42 where the hierarchical shortcut scored 0.000. Removing the extra binding level
removed the nesting wall.

## Why this is trustworthy (scrutinized -- I overclaimed once, so the PASS is checked hard)
- The test is on STRUCTURED facts (noun/verb/adj), the SAME distribution that EXPOSED the hierarchical
  failure -- not random fillers. So the 1.000 is on the right distribution.
- Multi-seed (42/43/44), all 1.000.
- Codes genuinely distinct (max-cos 0.604, no duplicates) -- distinct-seed retraining gave distinct patterns.
- This is a SINGLE-level bind (composition-role x flat-code), like the within-bridge 64 that is also robust;
  the only thing that ever failed was the EXTRA nesting level of the shortcut.

## Conclusion
Robust biological composition over CROSS-BRIDGE structured facts is achievable at 192 concepts (3 banks),
multi-seed perfect, via distinct-seed retraining. This covers the realistic SVO structure (noun agent / verb
action / adjective patient). The path to full 320 (5 banks) is the SAME mechanism: retrain bridges D + E with
distinct seeds -> 320 distinct flat codes -> robust structured composition (in flight). The brain-analogue
mechanism does robust structured relational reasoning at cross-bridge scale -- on solid footing, the honest
way, after the shortcut was caught and corrected.

## The arc (honest)
overclaim (hierarchical 320 'RESOLVES' on random fillers) -> demo caught it (0/6 structured) -> retraction
(nesting wall, structured 0.0/0.95/1.0) -> honest fix (distinct flat codes) -> RESOLVES (structured
1.0/1.0/1.0 multi-seed). The discipline recovered a real capability from a false claim.
