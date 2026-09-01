---
type: finding
status: measured
date: 2026-09-01
mechanism: np-entailment-moat-gate-copula-widening
lane: open-ended-honesty
seeds: [42]
seed-waiver: A deterministic parsing-level verify (spiking role assignment trained once at a fixed seed,
  then pure string extraction + a lexicon lookup + entailment CLASSIFICATION over fixed sentences) — the
  evidence is catch/leak booleans, false-positive booleans, and byte-equality against the actual pre-edit
  module loaded from `git show HEAD:...`, none of which a seed sweep would move. Same waiver shape as the
  parent gate's own wiring verify (`research/findings/2026-09-01-np-entailment-moat-gate-wired-into-
  live-open-ended-postfilter.md`), which this widens.
instrument: research/runners/_np_entailment_copula_coverage_verify.py — calls `webapp.
  np_entailment_moat_gate.gate_sentence` directly (PARSING-level: builds the same tiny 126+82-neuron
  BridgeParser/NPHeadBinder pair the live gate itself builds; no 15k-LTM brain) over a 5-case fabrication
  battery + an 8-case true-copula battery, flag on vs off, plus a byte-identical-off check against the
  actual pre-edit file content loaded from git HEAD.
runner: research/runners/_np_entailment_copula_coverage_verify.py
external: NO-EXTERNAL-NEEDED — this widens an already-built, already-measured live gate
  (`webapp/np_entailment_moat_gate.py`, its own 2026-09-01 GO wiring verify) to close a coverage gap that
  same day's real-traffic moat-safety soak MEASURED (not new-mechanism research).
artifacts:
  - research/findings/raw/_np_entailment_copula_coverage_verify.json (GO: 5/5 fabrication catch, 0/8
    false positives, byte-identical-off on all 13 cases)
  - research/findings/raw/_np_entailment_moat_gate_wiring_verify.json (parent gate's existing wiring
    verify, re-run after this edit: unchanged GO, no regression)
---

# Widening the NP-entailment moat gate to copula constructions — the castleford_f_c miss, closed behind a new flag (default OFF)

Artifact: `research/findings/raw/_np_entailment_copula_coverage_verify.json` (runner verdict: GO).

**One line.** The 2026-09-01 moat-safety soak measured that `webapp.np_entailment_moat_gate.gate_sentence`
never touches copula ("is a ...") sentences by design (scope item (d)), and real Qwen prose is dominated
by copula — so the gate changed 0/12 known-topic replies on real traffic, including a concrete miss:
`castleford_f_c` called a "professional football club" when the store's only sport fact is
`rugby_leauge`. A new flag, `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` (default OFF), adds a
narrow, additive-only widening that catches this specific class — a copula predicate naming a category
(today: a sport) that conflicts with a different, recognized category in a store fact for the same
topic — without reopening the false-reject risk scope (d) was written to avoid.

## The gap, reproduced

The soak's own transcript (`research/findings/raw/_open_ended_bundle_moat_soak_full.json`, arm A, known,
topic=`castleford_f_c`): Qwen writes "Castleford FC, commonly known as Castleford F.C., is a professional
**football** club based in Castleford, West Yorkshire, England." against a store holding only
`(castleford_f_c, country, united_kingom)` and `(castleford_f_c, sport, rugby_leauge)`. This sentence
survives BOTH the parent-only arm and the +NP-entailment arm untouched.

Two independent reasons the pre-widening gate cannot catch this, both confirmed directly against the real
data during this build (not assumed from reading the diff):
1. **Scope item (d)** excludes copula outright — the parsed action normalizes to "is", so the per-clause
   loop `continue`s before ever calling `classify_claim`.
2. **`split_clauses`'s blanket comma-split** (used by that same per-clause loop) severs the subject from
   the copula clause whenever an appositive sits between them: "Castleford FC, commonly known as
   Castleford F.C., is a professional football club..." splits into `"Castleford FC"`,
   `"commonly known as Castleford F.C."`, `"is a professional football club based in Castleford"`, ... —
   the third clause starts at "is" with NO subject, so even a copula-aware per-clause check would find an
   empty subject span and bail. **Both had to be fixed for the concrete case to close**, not just (1).

## What was built (additive, flag-gated, monotonic)

`webapp/np_entailment_moat_gate.py`, behind `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` (default
OFF, checked by the new `copula_coverage_enabled()`):

- **`_copula_wide_extract(sent)`** — a whole-sentence (not comma-pre-split) copula extraction: finds the
  first `is`/`are`/`was`/`were`, takes the subject as the sentence's LEADING comma-segment before it
  (tolerates the appositive that severs the per-clause path), and the predicate as the text after it up to
  the next comma/connective. Backs off (returns `None`, no-op) on a negated, present-participle
  ("is playing football" — progressive aspect, not identity), or passive ("was built by...", already
  `segment_clause`'s own passive pass) predicate — host lexical segmentation only, same category of
  preprocessing `segment_clause` already does, never a role decision.
- **`_normalize_loose(s)`** — separator-invariant subject/topic comparison used ONLY by this new path
  (the pre-existing per-clause path's `_normalize` is untouched): collapses underscores/hyphens/
  periods/whitespace and drops a leading article, so the Wikidata-slug topic `castleford_f_c` matches the
  human-readable subject text extraction pulls from real prose, `castleford fc` — still an EXACT-identity
  comparison after the collapse, not a fuzzy one.
- **`_copula_category_conflict(predicate_text, facts, topic_norm_loose)`** — a small, explicit,
  extensible lexicon (`_CATEGORY_WORDS`, today: 16 common sport names/synonyms — the concrete measured
  miss's own domain) fires ONLY when the predicate names a recognized category word AND a store fact for
  the SAME topic names a DIFFERENT recognized word in the SAME family (substring-tolerant, so the store's
  real `rugby_leauge` typo still hits "rugby"). An unrecognized predicate, or no matching-family fact,
  never trips this — the same "when unsure, don't touch it" posture scope (a)-(d) already take.

`gate_sentence` runs this as an early-return check BEFORE the pre-existing per-clause loop (needed,
because that loop never sees the severed subject); it can only ADDITIONALLY drop a sentence the rest of
the gate already kept, the same monotonic contract the module's original scope already declares.

## Verify (parsing-level, RAM-light — no 15k-LTM brain)

`research/runners/_np_entailment_copula_coverage_verify.py`, run locally (`SIM_BACKEND=numpy`, the same
tiny 126+82-neuron BridgeParser/NPHeadBinder pair the live gate itself builds; peak local RSS was the
existing tiny parsing nets, not the LTM brain — no pool/GPU dispatch was needed for this layer).

**Fabrication battery (5 cases, must be caught flag-ON / must leak flag-OFF):** the exact real-traffic
castleford sentence (appositive-comma shape), the same conflict with no appositive, a "soccer" synonym
variant, a second sport pair (chicago_bulls: basketball vs claimed baseball), and a second underscore-slug
topic (leeds_rhinos: rugby_leauge vs claimed cricket). **Result: 5/5 leaked flag-OFF, 5/5 caught
flag-ON — new-catch rate 1.0.**

**True-copula battery (8 cases, false-positive check, must be UNCHANGED flag-ON vs flag-OFF):** the same
castleford sentence with the CORRECT sport (rugby) substituted, the parent gate's own saved
`copula_untouched` case (Canada, no category word), the Eiffel Tower landmark case, a negated predicate
("is not a football club"), a present-participle predicate ("is playing football"), the passive
Eiffel-Tower-built-by-Gustave-Eiffel case, and the parent gate's own `offtopic_agent_untouched` /
`grounded_kept` safety cases. **Result: 0/8 false positives — every case identical flag-on vs flag-off.**
Also checked directly (ad hoc, not part of the battery file): the parent gate's own 3 real saved
known-topic Qwen replies (canada/france/morocco) through `post_filter` with BOTH flags on — byte-identical
to flag-off, no new false rejects on that real saved data either.

**Byte-identical-off, measured against the actual pre-edit file (not the diff):** all 13 cases run
through the CURRENT module (flag off) and the ORIGINAL module (loaded via `git show HEAD:webapp/
np_entailment_moat_gate.py` into an isolated namespace) produce IDENTICAL `gate_sentence` output on every
case. **`all_byte_identical_off: true`.**

**No regression to the parent gate:** `research/runners/_np_entailment_moat_gate_wiring_verify.py`
re-run after this edit — unchanged GO (all preconditions still hold, real-data regression still
byte-identical).

## Honest limits (named, not hidden)

- **Sport-only lexicon.** `_CATEGORY_WORDS` covers 16 sport names/synonyms — the concrete measured miss's
  own domain, not a general type-conflict detector. Extending to other mutually-exclusive category
  families (nationality, profession, religion, ...) is the natural next lever, deliberately not attempted
  here (scope discipline: build to the measured miss, not a speculative generalization).
- **Participial and pronoun-referent constructions are still untouched.** The soak named these too
  ("bordering ...", "It's often associated with ..."); this widening is copula-only, per the task's
  explicit "at least the copula construction" framing. Both remain a named residual.
- **A copula predicate with NO recognized category word is still fully out of scope** — the exact
  `copula_untouched` false-reject risk scope (d) was written to avoid is still avoided by construction,
  not just by luck: `_copula_category_conflict` returns `False` (silent no-op) whenever it finds no
  recognized-word conflict, which is the common case for elaborative descriptive prose.
- **Small batteries (5 + 8 hand-built cases), not a corpus sweep.** Precise for what it tests, but not a
  claim about the full space of real Qwen copula sentences beyond the 3 saved known-topic replies checked
  ad hoc above. A larger real-traffic re-run of the 2026-09-01 soak with this flag added (the natural next
  rung) would sharpen the catch-rate number past this hand-built battery.
- **This flag is NOT wired to any default.** `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` is reachable
  from `/api/brain-chat` on some request (the parent gate already is), so it is *wired* per
  `docs/TERMS.md`, but it is default-OFF (opt-in, same as its parent flag) — not on-by-default, not
  integrated/production-default. No production default was flipped by this change (owner-UX-gated per the
  task).

## Bottom line

The task's concrete ask — "castleford is a football club" vs the store's rugby_league fact caught as a
contradiction — is closed, GO, behind a new default-OFF flag, additive and monotonic, with a measured
zero false-positive rate on the true-copula battery and a measured byte-identical-off against the actual
pre-edit module. The full 6-seed / broader-corpus sweep and the participial/pronoun constructions are not
attempted here (see limits); a full-brain integration smoke that this flag actually fires through the real
open-ended path with the 15k-LTM brain is the natural next rung, appropriately routed to the pool per this
task's RAM-safety scope, not run locally.
