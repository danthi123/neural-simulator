---
type: finding
status: measured
date: 2026-09-01
mechanism: np-entailment-moat-gate-participial-pronoun-coverage-widening
lane: open-ended-honesty
seeds: [42]
seed-waiver: A deterministic parsing-level verify (spiking role assignment trained once at a fixed seed,
  then pure string extraction + a lexicon lookup + entailment CLASSIFICATION over fixed sentences) — the
  evidence is catch/leak booleans, false-positive booleans, and byte-equality against the actual pre-edit
  module loaded from `git show HEAD:...`, none of which a seed sweep would move. Same waiver shape as the
  copula-coverage widening this extends (`research/findings/2026-09-01-np-entailment-copula-coverage-
  widening.md`) and the parent gate's own wiring verify.
instrument: research/runners/_np_entailment_participial_pronoun_coverage_verify.py — calls `webapp.
  np_entailment_moat_gate.gate_sentence` directly (PARSING-level: builds the same tiny 126+82-neuron
  BridgeParser/NPHeadBinder pair the live gate itself builds; no 15k-LTM brain) over a 6-case fabrication
  battery (3 participial, 3 pronoun-referent) + a 15-case true-sentence battery, flag on vs off, a
  byte-identical-off check against the actual pre-edit file content loaded from git HEAD, and a
  regression re-check of the copula-coverage widening's own saved battery (the two widenings share the
  `_CATEGORY_WORDS` lexicon, which this arc also extended).
runner: research/runners/_np_entailment_participial_pronoun_coverage_verify.py
external: NO-EXTERNAL-NEEDED — this widens an already-built, already-measured live gate
  (`webapp/np_entailment_moat_gate.py`, its own 2026-09-01 copula-coverage GO widening) to close the two
  remaining construction classes that same day's copula-coverage finding named as still-uncaught
  (not new-mechanism research).
artifacts:
  - research/findings/raw/_np_entailment_participial_pronoun_coverage_verify.json (GO: 6/6 fabrication
    catch, 0/15 false positives, byte-identical-off on all 21 cases, 0/13 regressions on the
    copula-coverage widening's own saved battery)
  - research/findings/raw/_np_entailment_copula_coverage_verify.json (re-run after this edit: unchanged
    GO, no regression — the `_CATEGORY_WORDS` widening this arc did is additive-only)
  - research/findings/raw/_np_entailment_moat_gate_wiring_verify.json (parent gate's own wiring verify,
    re-run after this edit: unchanged GO, no regression)
---

# Widening the NP-entailment moat gate to participial and pronoun-referent constructions — the two remaining named-uncaught classes, closed behind a new flag (default OFF)

Artifact: `research/findings/raw/_np_entailment_participial_pronoun_coverage_verify.json` (runner verdict: GO).

**One line.** `research/findings/2026-09-01-np-entailment-copula-coverage-widening.md` named PARTICIPIAL
("bordering France", "founded in 1892") and PRONOUN-REFERENT ("It's often associated with ...")
constructions as still-uncaught after its own copula-coverage widening — this arc closes both behind a
new flag, `BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE` (default OFF), plus optionally
generalizes the shared `_CATEGORY_WORDS` lexicon (sport → +nationality/profession/religion) per that
finding's own "natural extension" note, verified with zero regression to the copula-coverage widening it
shares that lexicon with.

## The gap, as named by the prior finding

That finding's "Honest limits" section: *"Participial and pronoun-referent constructions are still
untouched. The soak named these too ('bordering ...', 'It's often associated with ...'); this widening
is copula-only... Both remain a named residual."* The moat-safety soak's own "Five concrete before/after
examples" gave the real shape directly: example #2 (`city_of_knoxville_tennessee`) kept "bordering
Virginia to the north and North Carolina to the west" (a false claim) through both the parent-only and
+NP-entailment arms; example #4 (`college_for_interdisciplinary_studies`) had its subject as the pronoun
"It", unresolved to any antecedent, carrying a fabrication ("Columbia University") that survived even the
safest arm.

Both classes fail for the SAME structural reason the copula case did: `split_clauses`'s blanket
comma-split severs a participial phrase from its subject ("City, bordering Virginia, ..." → "City" /
"bordering Virginia" / ... — the second segment has no subject, so even a participle-aware per-clause
check would find an empty subject span and bail), and a pronoun subject never text-matches the retrieved
topic string (the pre-existing per-clause path's item (b) always treats "It ..." as off-topic and no-ops,
since "it" ≠ "castleford_f_c"). Both need their own whole-sentence extraction, the identical fix shape
the copula-coverage widening already established — not a new mechanism, an application of the same one.

## What was built (additive, flag-gated, monotonic)

`webapp/np_entailment_moat_gate.py`, behind `BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE`
(default OFF, checked by the new `participial_pronoun_coverage_enabled()`):

- **`_participial_wide_extract(sent)`** — takes the sentence's leading comma-segment as the subject (same
  convention as `_copula_wide_extract`), then scans the LATER comma-segments for the first one whose
  opening word is a recognized relational participle (`_PARTICIPIAL_ACTION_MAP`: border(ing), found(ed),
  built/build(ing), discovered/discover(ing), designed/design(ing), created/create(ing),
  constructed/construct(ing), established/establish(ing), located/locate(ing) — a small, explicit,
  extensible lexicon, the SAME posture as `_CATEGORY_WORDS`). Canonicalizes it to a relation-key action
  string and takes the rest of that segment (minus one leading preposition, e.g. "in"/"by") as the
  object. Backs off on negation or an unrecognized first word — honest coverage limits, not guesses.
- **`_participial_relation_conflict(canonical_action, object_text, facts, topic_norm_loose)`** — fires
  ONLY when the SAME topic has a store fact for that SAME canonical relation whose patient, loosely
  normalized, has NO overlap (neither substring-contains the other) with the extracted object — e.g.
  predicate "founded in 1892" vs a stored `founded=1919` fact conflicts; the object "1919" against the
  same fact does not. A topic with no fact for that relation at all never trips this — the identical
  "when unsure, don't touch it" posture `_copula_category_conflict` already takes.
- **`_pronoun_wide_extract(sent)`** — fires ONLY when the sentence's very first word is a third-person
  pronoun (it/he/she/they) immediately followed by a copula, contracted ("It's", "They're") or explicit
  ("It is"), with the same negation/present-participle/passive backoff guards `_copula_wide_extract`
  already uses. Because `gate_sentence` runs once per already-known topic (the single entity the whole
  reply is about — `post_filter`'s per-sentence loop passes the same `topic` to every sentence), a
  leading pronoun subject is treated as standing for that topic directly, no text-match needed. The
  extracted predicate is checked with the SAME, UNCHANGED `_copula_category_conflict` the copula-coverage
  widening already built — no new conflict logic for this shape.
- **`_CATEGORY_WORDS` widened** (the optional generalization the prior finding named as the natural next
  step): now also covers NATIONALITY (20 words), PROFESSION (13 words), and RELIGION (9 words) alongside
  the original SPORT family (16 words) — purely additive (no existing word removed or reassigned), used
  by BOTH the original copula path and the new pronoun-referent path.

`gate_sentence` runs both new checks as a second early-return block, AFTER the copula-coverage block and
BEFORE the per-clause loop (same reason as the copula fix: the per-clause loop never sees a
comma-severed subject or resolves a pronoun); it can only ADDITIONALLY drop a sentence the rest of the
gate already kept, the same monotonic contract every check in this module holds.

## Verify (parsing-level, RAM-light — no 15k-LTM brain)

`research/runners/_np_entailment_participial_pronoun_coverage_verify.py`, run locally
(`SIM_BACKEND=numpy`, the same tiny 126+82-neuron BridgeParser/NPHeadBinder pair the live gate itself
builds; peak local RSS was the existing tiny parsing nets, not the LTM brain — no pool/GPU dispatch
needed, `free -m` checked before running: ~28GB available).

**Fabrication battery (6 cases, must be caught flag-ON / must leak flag-OFF):** 3 participial
(present-participle "bordering" vs a `borders` fact; past-participle "founded in <year>" vs a `founded`
fact, with the leading-preposition strip exercised; "discovering" vs a `discovered` fact) + 3
pronoun-referent (two sport-family conflicts via "It's"/"They're", one nationality-family conflict via
"It's an American ..." — the last one exercising the same-day lexicon widening). **Result: 6/6 leaked
flag-OFF, 6/6 caught flag-ON — new-catch rate 1.0.**

**True battery (15 cases, false-positive check, must be UNCHANGED flag-ON vs flag-OFF):** the same 3
participial shapes with the CORRECT value substituted; a participial relation the store has NO fact for
at all (must never trip — no opinion to conflict with); a negated, an unrecognized-verb, and a comma-less
participial (three conservative backoffs); the same 2 pronoun sport shapes with the CORRECT sport; a
no-category-word pronoun predicate; negated/present-participle/passive pronoun predicates (three
conservative backoffs mirroring the copula path's own guards); a pronoun that is NOT the sentence's first
word (out of scope by construction); plus the parent gate's own saved `offtopic_agent_untouched` /
`grounded_kept` safety cases. **Result: 0/15 false positives — every case identical flag-on vs flag-off.**

**Byte-identical-off, measured against the actual pre-edit file (not the diff):** all 21 cases run
through the CURRENT module (new flag off) and the ORIGINAL module (loaded via `git show
HEAD:webapp/np_entailment_moat_gate.py` — this branch was cut from `origin/main` after the copula-coverage
widening landed, so HEAD here IS that widened-but-not-yet-this-arc's-edits version) produce IDENTICAL
`gate_sentence` output on every case. **`all_byte_identical_off: true`.**

**Regression check on the copula-coverage widening's own battery (shared `_CATEGORY_WORDS` edit):** all
13 cases from `_np_entailment_copula_coverage_verify.py`'s own FABRICATION_CASES + TRUE_CASES, re-run
against the CURRENT module with the copula flag on the SAME as that verify's own methodology (this new
flag off), compared to that finding's own saved verdict artifact. **0/13 mismatches — the category-lexicon
widening introduced zero behavior change on the battery it was built against.** Additionally, both
`research/runners/_np_entailment_copula_coverage_verify.py` and
`research/runners/_np_entailment_moat_gate_wiring_verify.py` (the parent gate's own wiring verify) were
independently re-run after this edit — both unchanged GO.

## Honest limits (named, not hidden)

- **Small, explicit, extensible lexicons, not general parsers.** `_PARTICIPIAL_ACTION_MAP` covers 9
  relation families (border/found/build/discover/design/create/construct/establish/locate); an
  unrecognized participle (e.g. "sitting near the river", "nestled in the mountains" — real prose from
  the soak's own transcripts) is left untouched by construction, an honest coverage limit measured
  directly in the true battery (`participial_unrecognized_verb`), not a guess.
- **Comma-appositive shape required for participial.** A participial phrase NOT set off by a comma from
  its subject ("Castleford FC bordering Wakefield is a rugby club", no comma at all) is conservatively
  skipped (`participial_no_comma` true-battery case) — the whole-sentence extraction requires ≥2
  comma-segments, matching the same convention `_copula_wide_extract` already uses.
- **Pronoun resolution is "sentence-initial only, always = the known topic."** This is a real, declared
  simplification: it will not catch (nor false-positive on) a pronoun appearing mid-sentence, or one whose
  true antecedent is a DIFFERENT entity than the topic (e.g. "The stadium, built in 1990, seats 40,000. It
  cost $2M." where "It" could plausibly mean the stadium, not the topic team) — `pronoun_not_sentence_initial`
  in the true battery measures the conservative (never-fires) side of this; the antecedent-ambiguity side
  is not exercised because it cannot false-positive under this design (a mid-sentence pronoun is never
  even attempted), only under-catch, the same "when unsure, don't touch it" posture as the rest of the
  gate.
- **Conflict-only, never coverage-driven.** Exactly like the copula widening, both new checks fire ONLY on
  a positive conflict with a recognized store fact/category word — a topic with no matching relation fact,
  or a predicate with no recognized category word, is never touched (`no_matching_relation_fact`,
  `pronoun_no_category_word`). This means the EXACT real-traffic soak examples (Knoxville's specific
  border claim against facts this store may not hold as a `borders` relation at all; "Columbia University"
  against a `location` fact with no lexical overlap at all) are not guaranteed caught by this widening —
  the battery demonstrates the CONSTRUCTION CLASS is now checkable when the store DOES hold a
  same-relation fact, not that every possible real-traffic instance of these shapes is caught. A larger
  real-traffic re-run of the 2026-09-01 soak with this flag added (the same natural next rung the parent
  finding named) would sharpen this past the hand-built battery.
- **Category lexicon still finite.** Nationality/profession/religion add 42 words across 3 new families;
  this is illustrative generalization, not an exhaustive typology (no age, no political-affiliation, no
  further country-name family, etc. — left for a future widening if the soak's next real-traffic
  measurement motivates it).
- **This flag is NOT wired to any default.** `BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE`
  is reachable from `/api/brain-chat` on some request (the parent gate already is, via `post_filter`'s
  per-sentence `gate_sentence` call), so it is *wired* per `docs/TERMS.md`, but it is default-OFF (opt-in,
  same as its sibling flags) — not on-by-default, not integrated/production-default. No production default
  was flipped by this change (owner-UX-gated per the task).

## Bottom line

Both constructions the copula-coverage finding named as still-uncaught — participial phrases and
pronoun-referent sentences — are now checkable behind a new default-OFF flag, additive and monotonic,
with a measured zero false-positive rate on a 15-case true battery, a measured 1.0 catch rate on a 6-case
fabrication battery spanning both construction classes, a measured byte-identical-off against the actual
pre-edit module, and a measured zero-regression on the copula-coverage widening's own saved battery (the
shared category-lexicon edit). The category-word conflict mechanism remains conflict-only by design
(never fires merely because a relation/category is unrecognized or unlisted), so real-traffic instances
outside the recognized lexicons or without a same-relation store fact remain an honest, named residual —
a larger real-traffic soak re-run with this flag added is the natural next rung, appropriately deferred
(RAM-light parsing-level verify only, per this task's scope) rather than run here.
