# EMERGE-29 / toward-language+semantics — the CONVERSATIONAL INFERENCE CONSOLE: teach the emergent spiking brain an is-a taxonomy + class properties, then ask questions whose answers were NEVER told — it INFERS them by inheritance, with an honest no-confab moat. Unifies EMERGE-25 (talk-to-and-teach console) + EMERGE-26/27 (emergent Collins-Quillian inheritance). NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge29_inference_console.py` (`--demo` / `--script` / interactive); CI guard `tests/test_emerge29_inference_console.py` (4 tests). Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend.

## What it does (the north-star: a brain you talk to that INFERS)
The owner teaches a taxonomy and properties in plain sentences, then asks inference questions the brain was never told the answers to:
```
you> a robin is a bird          brain> ok -- a robin is a bird.
you> a bird is an animal         brain> ok -- a bird is an animal.
you> a bird can fly              brain> ok -- a bird can fly.
you> an animal can breathe       brain> ok -- an animal can breathe.
you> can a robin fly?            brain> Yes, a robin can fly.               (inherited 1 level up -- never told)
you> can a robin breathe?        brain> Yes, a robin can breathe.          (inherited 2 levels up)
you> can a robin swim?           brain> I don't know whether a robin can swim.   (honest -- not inherited)
you> can a zzz fly?              brain> I don't know what a zzz is.          (moat -- unknown concept)
```

## Mechanism (emergent, no inference engine, no transformer)
Each concept has a CONTENT block; "x is a y" gives x the shared code of y (x inherits y's content block, transitively up the chain), so a member's code overlaps its whole taxonomy. "a y can P" potentiates y's content block → P via the committed `sim/` three-term kernel. Asking "can a x P?" presents x's content + all ancestor blocks (walk the is-a chain); the ancestor that owns P primes P through the learned pathway → x inherits it though x's own code was never bound to P. A concept with no taxonomy drives nothing → the moat abstains ("I don't know what a zzz is"). A known concept whose queried property is not inherited abstains honestly ("I don't know whether…"), never a false "no". The natural-language front end (regex for "a X is a Y" / "a Y can P" / "can a X P?") is the legitimate world/keyboard interface; all cognition is on the spiking bridge.

## Significance
This makes the whole inference arc (EMERGE-26/27 inheritance) TALKABLE-TO: the owner converses with the emergent spiking brain, builds its knowledge, and asks questions it reasons out by inheritance up a multi-level is-a hierarchy — with an honest moat that abstains rather than confabulate. It composes the EMERGE-25 teach-live console with the EMERGE-26/27 inheritance mechanism; reuse-by-import, NO `sim/` edit.

## Honest scope + next
- Uses the inheritance inference (EMERGE-26/27). Transitive-ordering queries (EMERGE-28 "is B greater than D?") and grammatical grounded production (EMERGE-23/25) are separate modes not yet folded into this one console — a straightforward next unification.
- The taxonomy is TAUGHT (told is-a links); it does not yet EMERGE from experience/co-occurrence — the deferred R-c residual (the next deep-research gate: the is-a structure must arise from statistics/perception via the PPMI stream cortex + replay).

## Artifacts
`research/runners/_emerge29_inference_console.py`, `tests/test_emerge29_inference_console.py`. Prior: `2026-07-02-emerge27-multilevel-taxonomy-GO.md`, `2026-07-02-emerge26-emergent-inheritance-GO.md`, `2026-07-02-emerge25` (grounded growing console).
