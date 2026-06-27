# Tier-2 console integration — BATCH 2 (transitive · common-ground · tense) wired into the FIRST-CHAT CONSOLE

**Date:** 2026-06-27
**Status:** **DONE.** The three remaining GO'd Tier-2 capabilities are now usable in the first-chat console, in ONE
additive pass (the chat was not edited three separate times). Tier 2 is now **5/5 wired** (2.1-A analogy + 2.2
chain-of-thought landed in the prior pass, commit `6efbebaf`; this pass adds **2.3 transitive · 2.4 common-ground ·
2.5 tense**). Reuse-by-import; **NO `sim/` edit, NO composer edit** (the three composers are subclasses imported
verbatim). The no-confab MOAT holds on every route; the DEFAULT `--rubric` regression is **byte-identical (10/10,
moat 0, MIXED, PASS)** before and after.

## What shipped (each route's genuine-vs-curated status, flagged honestly)

### 2.3 — transitive inference (`is A bigger than B?` / `A > B`) — REGIME-A (curated axis), genuine reasoning over it
- **Mechanism (genuine):** the route COMPARES two items' learned 1-D ordinal-map POSITIONS — the order is read from
  the learned GEOMETRY, so it generalizes to **never-trained non-adjacent pairs** (tiny vs huge), exactly the
  validated `_transitive_ordinal_map_derisk` mechanism (Betasort-asymmetric update; finding
  `2026-06-27-tier2.3-transitive-ordinal-map-GO.md`). The de-risk's `learn_positions` is locked to its 7-item
  `ABCDEFG` ladder, so the console **replicates the same validated update body** (one short function) — exactly as
  the console replicates `build_communicable_brain`'s body.
- **HONEST SCOPE = REGIME A (like the analogy KB):** the axis is a **GIVEN curated size ladder** (`tiny < small < big
  < huge < giant`), NOT corpus-learned. Confirmed empirically: the 1,454-concept brain's vocab carries **no size
  scale** (0 of the ladder words are in vocab; no `small`/`big`/`large`/`tiny`/… at all). The corpus has no clean
  total order — exactly as anticipated — so the ladder is given, identical to how the analogy KB carries its own
  curated `king/queen/prince` items independent of the brain's corpus codes. An item **not on the axis ABSTAINS**
  ("I only compare things on a scale I've been given") — the moat; never a fabricated order.
- Live: `is tiny bigger than huge?` → "No -- it's the other way around: huge is bigger than tiny" (held-out
  non-adjacent); `is huge bigger than small?` → "Yes" (held-out); `giant > tiny?` → "Yes" (operator form);
  `is tiny smaller than big?` → "Yes" (smaller-direction); `is dog bigger than cat?` → moat abstain.

### 2.4 — common-ground / audience design (`what does X Y?` on a user-stated fact) — **GENUINE** (live discourse)
- **Genuine:** operates on the console's **live per-session discourse ledger**, not a given listener model. When the
  user STATES a fact this session, it is recorded as **SHARED** (mutually known) via `CommonGroundComposer.store_cg`;
  the brain's own pre-loaded facts are **PRIVATE** (only the brain knows them). At response time, a `what does X Y?`
  query whose fact the user already stated is **ACKNOWLEDGED** ("As you mentioned, … — you told me that, so I won't
  belabour it") instead of re-told (the competent audience-design move); the brain's PRIVATE facts fall through to
  the normal certain-lead discuss (volunteered). The SHARED/PRIVATE tag is bound/read through the validated RF tag
  mechanism (finding `2026-06-27-tier2.4-common-ground-GO.md`); `should_volunteer` confirms the tag before the ack.
- This is exactly the "LEARNED common-ground ledger (updated at each accepted contribution, Clark grounding)" the
  de-risk named as the natural follow-on to its host-SET demo — here it is wired to the real conversation.
- The ledger starts EMPTY, so the rubric/demo (which make no statements) are byte-unchanged.
- Live: after `the boy went to the park`, `what does boy go?` → "As you mentioned, the boy went to the park …";
  after `dog will chase cat`, `what does dog chase?` → "As you mentioned, the dog will chase the cat …" (composing
  2.4 audience-design with 2.5 tense). `what does girl go?` (NOT stated, a PRIVATE/unknown fact) does NOT
  acknowledge — it falls through to the normal path.

### 2.5 — tense/aspect (a user statement echoed back tensed) — **GENUINE** (input tense → output tense)
- **Genuine:** the user's **input tense is detected from the surface verb form** (PAST via the irregular table
  `_PAST`/regular `-ed`; FUTURE via `will V`; PRESENT otherwise — a legitimate lexical front-end, like the existing
  `_surface_morphology` and the de-risk's own inflection table) and DRIVES the echo via `TenseAspectComposer`
  (`store_tensed` + `render_tensed`; finding `2026-06-27-tier2.5-tense-aspect-GO.md`). The object binds to the
  verb's frame role (GOAL for motion verbs → "went **to** the park"; else patient → "ate the apple").
- Live: `the boy WENT to the park` → "the boy went to the park" (PAST); `dog WILL chase cat` → "the dog will chase
  the cat" (FUTURE); `the girl EATS the apple` → "the girl eats the apple" (PRESENT).

**2.4 + 2.5 share one statement route** (`_statement_response`): a declarative SVO records the fact (tensed, as
SHARED) AND echoes it back tensed in a single turn — the genuine wiring of both, over the live discourse.

## The no-confab MOAT — 0 leaks across the full live transcript

A 16-turn live transcript on the real 1,454-concept brain (all 3 new routes — genuine results + honest abstains —
plus OLD Tier-0 phatic, Tier-0.4 unknown-word, Tier-2.1-A analogy, Tier-2.2 chain-of-thought) → **0 moat leaks**
(`audit_moat` per turn). The new-route records carry empty `emitted_propositions`, so the moat audit is clean by
construction; the moat is enforced INSIDE each op (transitive abstains off-axis; common-ground only acks a fact the
user literally stated; tense only echoes a fully-vocab SVO the user actually said). The transitive route abstained
on `dog/cat` and `apple/ocean` (off the curated axis), and chain-of-thought honestly dead-ended on `cat`.

## DEFAULT regression UNCHANGED

`SIM_BACKEND=numpy python -m research.runners.first_chat_console --rubric` — **byte-identical** before and after the
edits: `RUBRIC SCORE: 10/10 (moat leaks: 0)`, `MIXED`, `VERDICT: PASS`. `--demo` → 0 leaks (CLEAN).

## Route-collision safety

The new regexes do not misfire on the existing probes. The transitive prose/operator regexes require `is X
bigger/smaller than Y` / `X > Y` (unambiguous). The statement route is checked LAST (after every question route),
rejects anything with a `?`/`:` or a leading interrogative/route-trigger, and requires exactly 3 content words with
a KNOWN verb in the middle — verified against all 19 existing probe forms (what-is-X, is-X-like-Y, which-X,
what-does-X-Y, A:B::C, analogy-prose, starting-from-X, what-comes-after-X, greeting, tell-me-more/stop,
unknown-word): 0 misfires.

## Tests / CI

- Imported composer CI guards: `test_common_ground_composer.py` + `test_tense_aspect_composer.py` +
  `test_transitive_ordinal_map.py` → **27 passed, 2 GPU-skipped**.
- Prior-pass analogy guard `test_factored_relation_analogy.py` → 7 passed, 1 GPU-skip (no regression from editing the
  same file).
- `py_compile` clean. The console edit is **+306 lines additive** (1 import-comment line extended; everything else new).

## Files

- `research/runners/first_chat_console.py` — +3 route regexes + helpers; `__init__` lazy-builds the ordinal map
  (curated ladder) + the common-ground composer (brain facts pre-loaded PRIVATE) + the tense composer + the session
  ledger (all guarded → graceful abstain on any failure); `_transitive_response`, `_statement_svo` /
  `_statement_response`, `_common_ground_ack`; three `respond()` branches (transitive early, common-ground inside the
  `what does X Y` route, statement last). Imports `CommonGroundComposer`, `TenseAspectComposer` + `inflect`,
  `past_tense`/`_PAST`; replicates the de-risk's Betasort update body for the ladder.

## Honest residuals / deferred

- **2.3 is regime A** (a given curated axis) — the same honest boundary as analogy. A corpus-LEARNED ordinal axis (the
  fully-on-bridge self-organizing embedding, the de-risk's §3.1 reuse target) is the deferred follow-on; this brain's
  corpus carries no usable scale.
- **2.4 is a one-bit listener model** (shared vs private) over the live discourse — NOT full ToM / false-belief (the
  recursive-agent-modelling wall). The ledger is per-session (not persisted across console restarts).
- **2.5 is a 3-valued tense tag** — NOT full event-semantics (aspect, reference time, event chaining; Tier 3).
- The composers are standalone (their own kb on the brain's codes), so the main composer's recall + moat are
  byte-untouched. The brain-based claim for the tag bind/render is the validated RF substrate (the de-risk's
  GPU-parity tests); the console reuses them on CPU (the numpy oracle path).
