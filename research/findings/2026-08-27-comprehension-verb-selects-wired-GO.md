---
type: finding
status: live
date: 2026-08-27
mechanism: comprehension-verb-selects-cue-lexicon
---

# Comprehension cue-lexicon wire-in — the corpus-learned VERB_SELECTS patient-slot cue is wired behind a default-OFF flag, EXACTLY mirroring the already-wired ANIMACY cue

**2026-08-27. Verdict: WIRED-GO (a real bug was found by adversarial verification and fixed before this
landed — see the dedicated section below).** Closes the residual both sibling findings named: the ANIMACY
half of the comprehension organs' shared "VOCAB CEILING" scaffold was wired 2026-08-27
(`research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md`, `BRAIN_LEARNED_ANIMACY_CUE`,
default OFF); the VERB_SELECTS half was de-risked the same day
(`research/findings/2026-08-27-comprehension-verb-selects-corpus-learned-GO.md`, 6-seed GO numpy + spiking) but
explicitly left unwired ("no production wiring was attempted this session ... symmetric to how the ANIMACY
cue's own wire-in followed its de-risk in a separate session"). This finding is that wire-in — the arc-closing
rung toward production-default for the comprehension organ's cue lexicon.

## What was wired, and how it mirrors the ANIMACY cue

`comprehension_production_organ.py` gained a single new choke point, `_verb_selects_of(v)`, that is the exact
verb-side analogue of the already-wired `_animacy_of(n)`:

```python
def _verb_selects_of(v: str):
    sel = VERB_SELECTS.get(v)
    if sel is not None:
        return sel
    if not learned_verb_selects_enabled():
        return None
    lex = _get_learned_verbselects_lexicon()
    lex.set_lesion(learned_verb_selects_lesioned())
    cls = lex.classify(v)
    if cls is None:
        return None
    return {"agent": "animate", "patient": "animate" if cls == "animate_patient" else "inanimate"}
```

The hand `VERB_SELECTS` table (`_phaseB_multicue_competition_spiking_derisk.py:64`) is checked first, always
— byte-identical whether or not the new flag is set. Only when the hand table misses AND
`BRAIN_LEARNED_VERB_SELECTS=1` does it fall through to
`_comprehension_learned_verbselects_cue_derisk.LearnedVerbSelectsLexicon` (the de-risk's own spiking-realized
lexicon, reused by import — not reinvented). The learned lexicon's "animate_patient"/"inanimate_patient"
classification is re-expressed in the SAME per-slot `{"agent": ..., "patient": ...}` shape the hand table
uses; the AGENT slot is "animate" for every hand-typed verb too (the table never varies it — see the de-risk's
own docstring), so filling it in as "animate" here loses no information the hand table itself ever encoded.

Every `v in VERB_SELECTS` / `VERB_SELECTS.get(v)` membership test in the organ was converted to call this
single choke point: `_lemma_verb` (base-form recovery), `competent()` (the fully-covered / fully-OOV
competence test), and `repair_target()`'s OOV branch.

**The one piece with no noun-side analogue: propagating the learned verb into the spiking read.**
`cue_evidence`'s own `verbfit` computation (`_phaseB_multicue_competition_spiking_derisk.py:120`,
`_verbfit_vote`) reads `VERB_SELECTS.get(verb)` as a **module-global** lookup — unlike `ANIMACY`, it has no
`permute_map`-style indirection to redirect through. `_evs_for_organ` closes this the same way the ANIMACY
cue closes the noun side (a canonical-proxy substitution), one level up: when the flag is on and the surface
verb misses the hand table but the learned lexicon covers it, the verb string passed into `cue_evidence` is
replaced with a canonical proxy verb of the SAME learned patient-selectional class — `"watch"` for
animate-patient, `"eat"` for inanimate-patient — so the existing, untouched, already-validated
`SpikingRoleCompetition` (D4's `mean_auc_semantic=1.0`, `mean_auc_lesion=0.5` circuit) reads the correct
verbfit vote for it, without editing `_phaseB_multicue_competition_spiking_derisk.py`:

```python
v_eff = v
if learned_verb_selects_enabled() and v not in VERB_SELECTS:
    sel = _verb_selects_of(v)
    if sel is not None:
        v_eff = "watch" if sel["patient"] == "animate" else "eat"
```

This is byte-identical to the unmodified call when the flag is off or the verb is already hand-covered.

**An inherited, honest quirk carried over unmodified, not introduced here.** Because every hand-typed verb's
AGENT slot is "animate" (uninformative), `_verbfit_vote` is structurally 0 (no vote) for any verb whose
PATIENT slot is ALSO "animate" — `sel["agent"] == sel["patient"]` makes `fits_agent`/`fits_patient` agree for
every noun, so neither condition in `_verbfit_vote` fires. This is not a defect of the proxy trick: it is how
the hand table's own two "patient=animate" verbs (`chase`, `watch`) already behave (the intended MOAT
design — a symmetric verb genuinely carries no verbfit-cue signal). Choosing `"watch"` as the animate-patient
proxy therefore reproduces the hand table's own existing behavior for that class exactly, rather than
introducing a new asymmetry; only the inanimate-patient class (proxy `"eat"`) contributes a real
discriminating verbfit vote, exactly as it already does for the hand-covered `eat`/`push`/`carry`/`bite`/
`kick`/`grab`.

## A real regression was caught by verify-go, and fixed before this GO — not glossed over

Adversarial skeptics (`verify-go`) dispatched against the first draft of this wiring found a genuine bug in
`_lemma_verb`, distinct from anything the animacy cue's own wiring exhibits: with `BRAIN_LEARNED_VERB_SELECTS`
ON, an **existing hand-covered verb's inflected surface form** could resolve through the learned lexicon
before the correct hand-table base was ever tried — because the learned lexicon's vocabulary is corpus-wide
(any frequent content word), not verb-specific, so an inflected token like `"pushed"` or `"bites"` can itself
be a real corpus word with its own (unrelated) learned score. The first-draft `_lemma_verb` checked
`_verb_selects_of` (hand table THEN learned lexicon) on the raw form, then on each suffix-stripped candidate
IN ORDER, returning on the first hit — so a learned-lexicon hit on an early candidate could pre-empt the
correct hand-table candidate later in the list.

**Measured, not hypothesized:** flag ON, pre-fix, 3 of the 8 hand-table verbs' common inflections
mis-lemmatized: `"pushed"` → `"pushed"` (unlemmatized; should be `"push"`), `"bites"` → `"bit"` (should be
`"bite"`), `"kicked"` → `"kicked"` (unlemmatized; should be `"kick"`). This is a real regression on the
wired-ON path — it does not affect the default-OFF production floor (`_verb_selects_of` is a hand-table-only
fast path when the flag is off, so `_lemma_verb`'s behavior for these verbs was untouched with the flag
unset), but it would have surfaced the moment `BRAIN_LEARNED_VERB_SELECTS` was ever flipped on, silently
changing the comprehension read for EXISTING, already-covered sentences — exactly the kind of default-flip
land-mine this repo's discipline exists to catch before it ships.

**Fix:** `_lemma_verb` now tries the RAW form and every suffix-stripped candidate against the hand table
**only** first (`cand in VERB_SELECTS`, no learned-lexicon consultation in this pass); the learned-lexicon
fallback (the same candidate order, via `_verb_selects_of`) runs **only if no hand-table candidate matched at
all**, so it can never pre-empt a correct hand-table lemma. Re-verified after the fix:

* **Systematic 24-form sweep** (all 8 hand verbs × 3 inflections each: `-s`/`-ing`/`-ed`): `_lemma_verb`
  matches `off == on == <correct base>` for **24/24 forms** post-fix. Pre-fix, an independent adversarial
  skeptic (dispatched separately, before this fix landed) found **12/24 forms mis-lemmatized**, and — more
  seriously — **3/24 (`chased`, `watched`, `watching`) resolved to a genuinely DIFFERENT selectional class**
  than the hand table's true class (the learned lexicon is a binary animate/inanimate-patient classifier with
  no way to represent `chase`/`watch`'s symmetric "patient=animate" class, so a stray learned-lexicon hit on
  a candidate like `"chase"` itself, if it had scored, could have silently swapped the verb's selectional
  category) — a materially different single-process margin was measured on `"the wolf watched the owl"`
  pre-fix (0.0493 flag-off vs 0.1396 flag-on, ~3x). Both failure modes are now closed.
* Fresh-process, single-call A/B (the rigorous byte-identical methodology, not a same-process sequential-call
  comparison — same-process repeated `judge()` calls on one organ exhibit this project's own documented
  chaotic inter-turn spiking jitter regardless of the flag, which is NOT this bug and was checked separately
  by calling `judge()` four times with the flag OFF and unchanged, confirming the jitter is orthogonal) on
  `"the fox pushes the rock"`, `"the wolf bites the apple"`, `"the cat kicked the ball"`: **zero-line diff**,
  flag OFF vs flag ON, each a fresh process with exactly one `judge()` call.
* A new CI regression guard, `test_flag_on_does_not_break_existing_inflected_hand_verbs`
  (`tests/test_comprehension_learned_verbselects_cue.py`), pins all of the above; the full 9-test file and
  every other suite listed below were re-run post-fix and still pass.

The rest of this finding's verification (byte-identical-off, load-bearing-on, lesion-reverts, moat) was
re-run against the FIXED code, not the pre-fix draft — every number below reflects the fixed state.

## New flags (mirror `BRAIN_LEARNED_ANIMACY_CUE` / `BRAIN_LEARNED_ANIMACY_LESION` exactly)

* `BRAIN_LEARNED_VERB_SELECTS` (default OFF): extends VERB_SELECTS competence to the learned open-vocab cue.
* `BRAIN_LEARNED_VERB_SELECTS_LESION` (default OFF, only meaningful when the cue flag is on): zeros the
  F_anim/F_inanim coupling (`LearnedVerbSelectsLexicon.set_lesion`) — every open-vocab `classify()` call
  abstains, reverting coverage to the flag-OFF scope.

No `webapp/server.py` plumbing was needed — like the ANIMACY flag, this is a pure `os.environ` read inside
the organ, reached on every request via the organ's existing, already-`wired` call path (per
`docs/TERMS.md`'s "wired" row: reachable from `/api/brain-chat` on some request).

## Verification (6-seed methodology N/A here — the organ-level wiring check is deterministic, not seeded;
the underlying cue's own 6-seed GO is the cited de-risk)

### 1. Flag-OFF byte-identical (verified in the data, not inferred from reading the code)

**Organ-level, isolated from conversational state.** `organ.judge(...)`/`organ.competent(...)`/
`organ.repair_target(...)` on the same 4 sentences `_gateB_repair_production_verify.py` uses ("the book
carries the cup", "the wolf watches the owl", "the wug blickets the glorp", "the wolf bites the apple"),
captured to JSON (full float precision, `sort_keys`) on the pre-edit code (`git stash` on
`comprehension_production_organ.py` only) and on the post-edit code with `BRAIN_LEARNED_VERB_SELECTS` unset.
`diff` on the two captures reports **zero lines of difference, exit code 0** — an exact match, not an
eyeballed one.

**Full production-turn regression** (`research/runners/_gateB_repair_production_verify.py`, through
`webapp.server.brain_chat`, numpy backend): all 6 existing checks pass with the new flag unset —

```
[PASS] role_agent_targeted
[PASS] role_animate_generic_targeted
[PASS] oov_token_named
[PASS] no_false_repair_on_comprehensible
[PASS] lesion_collapses_to_bare_abstain
[PASS] flagoff_bare_abstain_no_key
ALL_OK=True
```

### 2. Load-bearing verification (vary the cue, then lesion it)

`research/runners/_comprehension_learned_verbselects_wire_verify.py`, output
`research/findings/raw/_comprehension_learned_verbselects_wire_verify.json`. `"the dog clean the cup"` —
"clean" is a real corpus verb NOT in the hand VERB_SELECTS table (verified: `"clean" not in VERB_SELECTS`,
also asserted at import time in the de-risk module); "dog"/"cup" are hand-table-covered nouns, isolating the
test to the VERB cue alone.

| condition | `competent()` | `judge()` |
|---|---|---|
| flag OFF (default) | `False` | `None` (out of scope, unchanged) |
| flag ON | `True` | `{margin: 0.3375, threshold: 0.24861111111111112, comprehended: True, ...}` |
| flag ON + `BRAIN_LEARNED_VERB_SELECTS_LESION=1` | `False` | `None` |
| flag OFF again | `False` | `None` |

The flag-ON row shows the extended coverage is LOAD-BEARING (the organ now judges a sentence it previously
passed through unchanged). The lesioned row is an exact dict match to both flag-OFF rows
(`lesioned_reverts_to_flag_off_exact_match: true` in the artifact) — the diff this flag introduces VANISHES
under the lesion, not merely shrinks.

### 3. Moat check (0-confab on genuinely unknown verbs)

Same artifact, `moat_check_oov`. `"the wug blickets the glorp"` (verb and both nouns off the learned graph,
flag ON): `classify("blickets")` = `None` (abstain — "blickets" is in no vocab the learned lexicon indexes, so
no current drives either F_anim/F_inanim pool and they tie at exactly 0). `judge()` returns
`comprehended: False` (margin=0.026388888888888892, well below the 0.24861111111111112 threshold) and
`repair_target()` correctly names all three tokens as OOV (`loadbearing: "host_lexical"`, `oov_tokens: ["wug",
"glorp", "blickets"]`). The learned cue never invents a selectional class for a verb it has no evidence for.

### 4. No-regression on the comprehension organ's existing behavior

* `tests/test_comprehension_learned_animacy_cue.py` (the sibling cue's own CI guard): 8/8 still pass —
  unaffected by this change.
* `tests/test_comprehension_learned_verbselects_cue.py` (new CI guard, mirrors the animacy test file
  structure exactly, PLUS the regression guard below): 9/9 pass (held-out coverage, off-graph abstain, lesion
  collapse, flag-off byte-identity, flag-on load-bearing, lesion-reverts, moat, inflected-hand-verb
  regression guard).
* `tests/test_gap3_spiking_feature_compat.py` (the F_anim/F_inanim mechanism both cues reuse): 7/7 pass.
* `tests/test_multireferent_biased_competition.py`: 5/5 pass.
* `tests/test_all_capabilities_on.py`: unaffected (16 pre-existing GPU-only skips under the numpy backend,
  unrelated to this change — confirmed via `-rs`: "the attributed / frame / neural-render bridges are
  GPU-validated Hebbian parsers").
* `.venv/bin/python tools/check_docs.py`: W1=0, W2=0, both document-structure rules pass.

### 5. Regression guard (the lemma-order bug, see above) — the systematic 24-form sweep

`tests/test_comprehension_learned_verbselects_cue.py::test_flag_on_does_not_break_existing_inflected_hand_verbs`
pins the 3 originally-caught cases (`pushed`/`bites`/`kicked`) plus a 5-verb sample of the wider sweep. The
full 24-form matrix (all 8 hand verbs × `-s`/`-ing`/`-ed`) was independently re-verified by hand at the
terminal post-fix: **24/24 lemmatize identically flag-off vs flag-on**, matching each verb's correct base.

## Status against the load-bearing terms (`docs/TERMS.md`)

Per the "wired" / "on-by-default" / "scaffold-retired" / "integrated" ladder: this is **wired** (reachable
from `/api/brain-chat` on some request, exactly as its sibling ANIMACY cue already is) but the flag defaults
OFF, so it is **NOT on-by-default**, and the hand VERB_SELECTS table is not removed (an EXTENSION, not a
replacement), so it is **NOT scaffold-retired**. The correct status is the partial one: wired (default-off),
not integrated — identical status to the ANIMACY cue's own wire-in. **Per the task scope, the default was
deliberately kept OFF**: flipping BOTH `BRAIN_LEARNED_ANIMACY_CUE` and `BRAIN_LEARNED_VERB_SELECTS` to
default-ON together is a SEPARATE joint flip-soak, the next rung (mirroring how other faculties in this
project have gone through a dedicated multi-seed flip-soak before a default flip).

## Files

* `research/runners/comprehension_production_organ.py` — `learned_verb_selects_enabled`,
  `learned_verb_selects_lesioned`, `_verb_selects_of` (new, mirrors `_animacy_of`); `_evs_for_organ` extended
  with the verb canonical-proxy substitution; `_lemma_verb`/`competent`/`repair_target`'s OOV branch converted
  to the single `_verb_selects_of` choke point. No other file needed edits (no `webapp/server.py` plumbing,
  same as the ANIMACY cue).
* `tests/test_comprehension_learned_verbselects_cue.py` — new CI guard, 8/8 passing.
* `research/runners/_comprehension_learned_verbselects_wire_verify.py` — new: produces the byte-identity /
  load-bearing / moat artifact below.
* `research/findings/raw/_comprehension_learned_verbselects_wire_verify.json` — the organ-level load-bearing +
  moat-check artifact (with `.prov.json` sidecar, auto-attached by `research/runners/__init__.py`).
* `research/findings/raw/_gateB_repair_production_verify.json` — re-run post-edit (flag unset), all 6 checks
  still `ALL_OK=true`.

## Residuals (declared, ride existing burn-down items)

* Both `BRAIN_LEARNED_ANIMACY_CUE` and `BRAIN_LEARNED_VERB_SELECTS` default OFF — a joint flip-soak (6+ seeds,
  the project's standing bar for a default-flip generalization claim) is the deliberate next rung, out of
  scope here.
* The verb canonical-proxy trick ("watch"/"eat") means `SpikingRoleCompetition` never sees the ACTUAL
  open-vocab VERB's own spiking representation during the role-competition read — only its patient-selectional
  category, via a stand-in verb. This mirrors the identical, already-declared residual on the noun side (the
  "dog"/"ball" proxy) — a fuller closure would drive a dedicated cue population per open-vocab word/verb rather
  than borrowing a hand-table proxy's population.
* The animate-patient proxy class ("watch") structurally contributes no verbfit vote (see above) — this is
  inherited from the hand table's own pre-existing behavior for that class, not a new gap, but it means the
  learned cue's load-bearing demonstration in this finding necessarily used an inanimate-patient held-out verb
  ("clean"); a held-out animate-patient verb (e.g. "help", "feed") extends COMPETENCE identically but its
  `judge()` margin is governed by the SAME animacy-cue-only dynamics the hand `chase`/`watch` entries already
  exhibit (potentially ambiguous on a two-animate-noun sentence — by design, not a defect of this wire-in).
* The corpus (`data/corpus/tinystories.txt`) is a gitignored, regenerable cache, exactly as the sibling
  finding notes; not re-pinned here since this session made no numpy-label-propagation changes (only reused
  the already-committed de-risk artifacts and the deployment-mode lexicon, which does not use the held-out
  split).
