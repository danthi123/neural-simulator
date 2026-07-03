# EMERGE-75 — A→W VOCAB SCALING via the G.20 multi-bridge route (BOUNDARY)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge75_aw_vocab_scaling_derisk.py`
**Test:** `tests/test_emerge75_aw_vocab_scaling.py`
**Raw:** `research/findings/raw/_emerge75_aw_vocab_scaling.json`
**Verdict:** BOUNDARY (6-seed) — the mechanism works (all-word ground-truth accuracy **1.000**, isolated overflow A→W rate **1.000**, overflow genuinely spiking: BRIDGE-C pool→language_output LESION collapses the decode to **0.146** [engine-lesion 0.000]), but **3 full-render surfaces regress vs the token spell** at this training budget (the GO bar requires 0). The gate-first MOAT is intact (0 spell + 0 producer invocations on abstains). An honest scale/data residual, not a mechanism wall or moat breach.

## What was attempted

Generalize the spiking A→W read-out to arbitrary content vocabulary via the G.20 multi-bridge route: a
`UnifiedNeuralSpell75` dispatching each word to the bridge holding its pool, adding a **third bridge (BRIDGE-C)** for the
overflow words (13 object nouns + the new function words `to`/`on`/`is`), so the EMERGE-72 broadened constructions render
EVERY word on spikes. Same `concept_speak_demo` A→W recipe (build + topographic bias + orthogonal codes +
`train_word_to_pool`), trained inline + cached; the content (BRIDGE-A, EMERGE-67) and function (BRIDGE-F, EMERGE-68)
engines reused verbatim.

## The exact gap (fill from the raw json)

The content words spell 100% on BRIDGE-A and the 5 original function words 100% on BRIDGE-F (EMERGE-67/68 GO), and the
OVERFLOW words on BRIDGE-C decode 100% in ISOLATION (overflow A→W rate **1.000**, overflow slot accuracy **1.000**,
lesion-collapse **0.146** = genuinely spiking). The residual is at the **full-render** level:
- all-word ground-truth accuracy: **1.000** (every word decodes to a correct surface)
- BRIDGE-C raw A→W rate: **1.000** (bar 0.90) · overflow slot accuracy: **1.000**
- **render regression vs the token spell: 3 mismatches** across the 6-seed run — the GO bar is 0, so this is a BOUNDARY.

The residual is therefore NOT an isolated-decode failure (the overflow words each decode correctly) but a **full-render /
training-budget** effect: BRIDGE-C co-trains **3 high-frequency closed-class prepositions** (`to`/`on`/`is`) with 13
content nouns on one 16-pool bridge, and (as EMERGE-68 named for its function words) the closed-class words' orthogonal-band
A→W codes are the harder read (they co-occur with everything — the EMERGE-62 Goldilocks signature), so in the full
sequence their read occasionally regresses relative to the exact token surface.

## Next step (a scale/data lever + bounded pool-assignment change, NOT a new mechanism)

1. Train BRIDGE-C at the **fully-validated scale** (`n_per_pool=500`, more events, ~17 min/seed) to reach the documented
   per-pool discrimination.
2. OR **split** the function/content overflow across TWO bridges (the G.20 route): a dedicated closed-class BRIDGE for
   `to`/`on`/`is` (as EMERGE-68's BRIDGE-F did for the original function words) + a content BRIDGE for the object nouns.

## What is NOT compromised

- The MOAT is **not** breached (0 spell/producer invocations on abstains, by construction) — if it were, that would be
  BLOCKING, and we would NOT weaken it.
- The EMERGE-72 CONSTRUCTION render itself is unchanged (only the SPELL is the residual); the token-spell default path is
  byte-identical.
- Reuse-by-import; NO `sim/` edit.

## Honest scope

The overflow vocab is a bounded 16 words on one extra bridge; the EMERGE-73 adjective constructions are out of scope
(their mining is EMERGE-73's boundary; their adjective A→W is a further follow-on). NOT open prose (R4).
