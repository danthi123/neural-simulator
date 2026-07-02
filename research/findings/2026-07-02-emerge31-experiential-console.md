# EMERGE-31 / the CAPSTONE console — LEARN CATEGORIES FROM EXPERIENCE, then CONVERSE + INFER. The owner OBSERVES members co-occurring with contexts (no labels); the brain DISCOVERS the grouping (EMERGE-30); a property taught via ONE member is INFERRED for a DIFFERENT member the brain only ever OBSERVED in the same contexts — the full observe → learn → infer → converse loop on one spiking brain. NO `sim/` edit. (6/6-seed robust.)

**2026-07-02 (autonomous).** Runner `research/runners/_emerge31_experiential_console.py` (`--demo` / `--script` / interactive); CI guard `tests/test_emerge31_experiential_console.py` (4 tests). Reuse-by-import (`_emerge14` + `_emerge12`), composes EMERGE-30 (emergent grouping) + EMERGE-29 (console); NO `sim/` edit; CPU numpy-backend.

## What it does — the master-directive loop in one artifact
```
you> a robin lives-with a nest        brain> ok -- I've seen a robin with a nest.     (OBSERVE, no category named)
you> a sparrow lives-with a nest       ...
you> a robin lives-with a treetop      ...     (robin & sparrow share nest + treetop -> emergent group)
you> a sparrow lives-with a treetop    ...
you> a robin can fly                   brain> ok -- a robin can fly.                  (TEACH via ONE member)
you> can a robin fly?                  brain> Yes, a robin can fly.
you> can a sparrow fly?                brain> Yes, a sparrow can fly.     (INFERRED -- sparrow never told; only OBSERVED
                                                                          in the same contexts as robin)
you> can a sparrow swim?               brain> I don't know whether a sparrow can swim.   (honest -- not in the fish group)
you> can a shark fly?                  brain> I don't know what a shark is.              (moat -- never observed)
```
6/6 seeds (42/43/44/100/101/102): sparrow→fly inferred, pike→swim inferred, sparrow→swim abstains, shark→moat — all correct.

## Mechanism (emergent, no inference engine, no transformer)
"X lives-with Y" learns X-content → Y-context (on-bridge Hebbian co-occurrence, the committed `sim/` three-term kernel). Members sharing contexts learn to activate the same context cells → the emergent category. "a X can P" is taught by presenting X, priming its learned contexts, and binding P to (X + its contexts) → the property attaches to the shared context. "can a Y P?" reads P directly (if Y is the taught member) or via the shared context (Y → emergent context → P) → a co-observed member inherits. A member never observed drives no context → the moat abstains. The regex front end ("a X lives-with a Y" / "a X can P" / "can a X P?") is the legitimate world/keyboard interface; all cognition is on the spiking bridge.

## Significance — the whole arc in one loop
This is the culminating demonstration of the emergent toward-language+semantics arc (EMERGE-15..31): a brain that LEARNS ITS STRUCTURE FROM EXPERIENCE (categories discovered from observed co-occurrence, EMERGE-30), that you TALK TO and TEACH (EMERGE-25/29), and that INFERS beyond what it was told (inheritance, EMERGE-26), with an honest no-confab moat — all emergent, unsupervised, on one spiking substrate, no transformer, NO `sim/` edit. It is the observe → learn → infer → converse loop the master directive asks for, in a single interactive artifact.

## Honest scope + next
- One shared context token per observation is the clean cheap-first; overlapping/varied contexts (each member a random subset of a feature pool) + an HTM Spatial-Pooler forming a NEW shared column block are the robustness follow-ons (as for EMERGE-30).
- The console demonstrates single-property inheritance from experience; folding in multi-level taxonomy (EMERGE-27), transitivity (EMERGE-28), grammatical production (EMERGE-23), and growth (EMERGE-24) into this one experiential loop is the next unification. Cancellation on emergent codes is a follow-on.

## Artifacts
`research/runners/_emerge31_experiential_console.py`, `tests/test_emerge31_experiential_console.py`. Prior: `2026-07-02-emerge30-emergent-superordinate-GO.md`, `2026-07-02-emerge29-inference-console.md`.
