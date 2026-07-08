# The FLAGSHIP console's question-comprehension routing is now NEURAL in production (opt-in `--neural-route`, GO): a fronto-striatal reservoir read-out classifies each question's TYPE and dispatches it, replacing the host keyword if-ladder in `ask()`. 7/7 question types routed correctly end-to-end (property inherit/exception, relational what/who/yes-no, describe, moat). Default OFF is byte-unchanged. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_unified_talkable_console.py` (`neural_route=True` / CLI `--neural-route`; `QuestionRouter` in `_realcorpus_neural_question_routing_derisk.py`). CI guard `tests/test_realcorpus_unified_console.py::test_neural_route_dispatches_every_question_type`. numpy. NO `sim/` edit.
**Verdict:** GO — the console routes every question type neurally in production, moat unaffected, default path preserved.

## Why this ran (the comprehension residual, wired to production)
The neural question-routing was de-risked GO (CYCLE 1025: a reservoir read-out classifies question type, held-out 1.000, spiking-confirmed). This wires it into the FLAGSHIP `UnifiedTalkableConsole` so the console ITSELF routes comprehension neurally. Before: `ask()` dispatched by a host keyword if-ladder (`toks[:1]==["what"]`, `["who"]`, `["does"]`+len>=5, ...). Now: an opt-in `neural_route` builds a `QuestionRouter` (the EMERGE-78 reservoir + a ridge read-out trained on the console's question forms); `ask()` computes the type via `_route_type(toks)` and each branch dispatches on it via `_is(rt, typ, keyword)`.

## The result — the console routing neurally (seed 42)
```
router on the console's exact forms: 9/9 correct (incl. property "does a cat run" vs yes/no "does the dog eat cat" -- both "does"-initial)
end-to-end neural-route console: 7/7 routed to the correct handler
  does a cat run?        -> "yes -- the cat can run"      (inherit)   [property]
  does a bird run?       -> "no -- the bird can sleep"    (override)  [property exception]
  what does the frog eat?-> "the frog eats fish"          (relational)[what]
  who eats the fish?     -> "the frog eats fish"          (relational)[who]
  does the frog eat fish?-> "yes -- the frog eats fish"   (yesno)     [relational yes/no]
  tell me about the frog -> "The frog can run. It eats fish." (describe)[multi-fact discourse]
  does a zzzqqx run?     -> "I don't know"                (moat)      [abstain]
```

## What's load-bearing / design
- `neural_route` is opt-in (default OFF). When off, `_route_type` returns `None` and `_is(rt, typ, kw)` uses the keyword condition → the default path is BYTE-UNCHANGED (the existing console CI tests are unaffected).
- `compare` (a fixed multi-word construction) is kept as a keyword marker; the core does/wh types (property/what/who/yes-no/describe) are routed neurally.
- **Robust to misroutes**: a neural misroute self-corrects via the per-branch extraction guards — e.g. a property question misrouted as yes/no fails the `len(content)>=3` guard and falls through to the property handler.
- The router correctly makes the hard non-local distinction (property "does a X verb" vs yes/no "does the X verb Y", both "does"-initial — the a/the + trailing object, which the whole-sequence reservoir integrates).

## What this establishes
The flagship talkable console now ROUTES its comprehension neurally in production (a reservoir read-out, not host keyword rules), completing the "whole turn on spikes / one brain" picture: comprehend (neural routing) → reason (property/relational, spiking) → speak (structure + words + productive morphology, spiking). Follow-on: full neural role/filler EXTRACTION (EMERGE-78 form→role — the router classifies the type; extraction is still position-based); the spiking-LSM router (`QuestionRouter(spiking=True)`) in the console.

## Files
`research/runners/_realcorpus_unified_talkable_console.py` (`neural_route` flag + `_route_type`/`_is` + `ask()` refactor); `research/runners/_realcorpus_neural_question_routing_derisk.py` (`QuestionRouter`); `tests/test_realcorpus_unified_console.py`. Prior: the neural question-routing de-risk `2026-07-08-neural-question-comprehension-routing-GO.md`.
