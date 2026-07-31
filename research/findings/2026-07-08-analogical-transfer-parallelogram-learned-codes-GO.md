---
type: finding
status: qualified
date: 2026-07-08
mechanism: analogy
---

# OPEN-WORLD INFERENCE #3 — analogical transfer A:B :: C:? via the parallelogram on the brain's LEARNED co-occurrence codes (GO, 6-seed): the parallelogram (d = b − a + c) recovers the correct analogy target 1.000, BEATS a "just C's nearest neighbour" baseline (0.000 — so it is genuine analogy, NOT retrieval), and collapses to chance under PERMUTED codes (0.083). The over-claim trap the research gate warned of is avoided by foregrounding the baseline + permuted controls. Validated on CLEAN FACTORED emergent codes (the regime the gate predicted works); entangled/bundled real codes need clean mining first (honest boundary). NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_analogical_transfer_derisk.py`. Reuse-by-import (`learn_stream_codes`). numpy. NO `sim/` edit.
**Research gate:** `2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md` (ranked #3; flagged sound-but-over-claim-prone).
**Verdict:** GO (6-seed) — analogical transfer on clean factored learned codes, with the over-claim controls foregrounded.

## The mechanism
Each concept word co-occurs with BOTH its category context AND its attribute context (two independent emergent factors), so its learned co-occurrence code is ~ (category component + attribute component). The analogy A:B :: C:? — where A=(c1,a1), B=(c1,a2), C=(c2,a1) — transfers the a1→a2 attribute shift across categories: `d = code(B) − code(A) + code(C)` cancels the category and applies the attribute shift, and the nearest member to d is the target (c2,a2). This is the VSA/word2vec analogy op ("king − man + woman = queen") on the brain's own learned codes.

## The result — 6-seed (3 categories × 4 attributes × 3 members, 60 analogy quads/seed)
```
parallelogram (d = b - a + c -> nearest member)      = 1.000 every seed   -- recovers the correct (c2,a2) target
baseline (just C's nearest neighbour, no analogy)    = 0.000 every seed   -- C's neighbour is in C's OWN cell, NOT the target
PERMUTED codes (shuffle code<->word)                 = 0.083 (~= chance)  -- collapses; the learned structure is load-bearing
chance                                               = 0.083
```

## Anti-cheats (the over-claim trap foregrounded, per the gate's warning)
- **Baseline (C-neighbour = 0.000)** — the KEY control the 2026-05-14 over-claim lacked. C's nearest neighbour is another member of C's OWN (c2,a1) cell, NOT the analogy target (c2,a2). The parallelogram SHIFTS to the correct different-attribute cell → it is genuine analogical transfer, not retrieval of C's neighbour. A 1.000-vs-0.000 gap.
- **Permuted codes (0.083 = chance)** — shuffling which code belongs to which word collapses the parallelogram to chance → the analogy rides the learned code structure, not coincidence.
- GO gate: the parallelogram must beat BOTH controls by >0.30 AND exceed 0.70, every seed (met: 1.000 vs 0.000 and 0.083).

## Honest scope (the gate's prediction confirmed)
The research gate flagged analogy as "sound-but-bounded — a NEGATIVE on raw BUNDLED codes; the clean-mining path fixes it." This de-risk validates the CLEAN-FACTORED regime directly: the 2-factor co-occurrence stream gives each word one category + one attribute context, so its code is cleanly (category + attribute) additive → the parallelogram works perfectly. The honest boundary (consistent with the gate): on ENTANGLED/bundled real codes (where category and attribute are not cleanly separable), the parallelogram would degrade — those need clean relational mining (the regimeB path) first. This result establishes the mechanism works when the codes carry clean factored structure; the real-corpus entangled case is the follow-on (as with the spreading-activation real-code step).

## What this establishes
The second open-world-inference mechanism (after spreading-activation completion): the brain can answer A:B :: C:? analogies by the parallelogram on its own learned co-occurrence codes, genuinely (beating the retrieval baseline) and collapsibly (permuted → chance). Combined with #1 (spreading-activation completion), the brain now has two validated cheap spiking-substrate mechanisms for inference BEYOND its stored inventory. Follow-on: real-corpus analogies on clean-mined (regimeB) relational codes; wire analogy into the console.

## Files
`research/runners/_realcorpus_analogical_transfer_derisk.py`; `tests/test_analogical_transfer.py`. Reuses `_emergent_vocab_breadth_scale_derisk.learn_stream_codes`. Prior: the research gate; the #1 mechanism `2026-07-08-spreading-activation-completion-open-world-inference-GO.md`.
