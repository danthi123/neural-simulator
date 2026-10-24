# Round-3 fresh hunt: allowlist claim checker (written 2026-10-20)

## Blindness statement

I did not open, list, grep, or view the git history of any forbidden path:
- `research/runners/_moat_allowlist*.py`
- `tests/test_moat_allowlist.py`, and anything under `tests/fixtures/` whose name contains `moat_allowlist`
- `/home/dant123/Projects/sim-rev/moat-verify/` and `/home/dant123/Projects/sim-rev/moat-verify2/`

I also did not read the verb-form table (`_noun_before_verb`, `1b67eef9b`) or any other moat checker code.

What I read:
- `docs/drafts/moat-allowlist/SPEC.md`.
- The pre-registration, including AMENDMENT 1.
- The held-out `corpus.json`, `s3_corpus.json`, `labels.json` and `s3_labels.json` (only their gathered sets and sentences).
- The store `facts.json`.

Nothing under `/home/dant123/Projects/sim-rev/wt-allow` was edited.

## Files

- `cases.json`: 213 cases. Each is `{"id", "gathered", "sentence", "expect", "class", "why"}`.
- `build_cases.py`: the generator. Run `python3 build_cases.py` to rebuild and re-validate. It checks:
  - every gathered triple is an exact triple of `facts.json` (all 213 pass);
  - every relation is one of the store's 60 (50 are used);
  - every case has the exact key set, and no id is repeated.

## Counts

By label: **U = 170, F = 43**.

| class | U | what it probes |
|---|---:|---|
| shape3_on_noun_relation | 18 | `X is the N of Y` for a non-`_of` noun relation; agentive nouns reverse roles (father, child, author, employer, operator, producer, composer, publisher, director, cast member, record label, ...) |
| styled_letters_vanish | 15 | negation, hedge or extra entity written in full-width, math-alphanumeric, small-capital, circled or all-diacritic Latin; NFC leaves these non-ASCII |
| symbol_negation_or_doubt | 13 | `¬`, ❌, ✗, 🚫, markdown `~~strike~~`, `(?)`, `[?]`, `?`, `¿?`, `‽`, `?!`, `†` |
| agent_noun_verb_reversal | 12 | employs / operates / produced / directed / composes / publishes / performs / authored / fathered / parents / platformed / wrote |
| coordination_unsupported_conjunct | 12 | shape 2 with an unsupported conjunct (a chain hop, another relation's object, `or`, `and/or`, `as well as`, `&`) |
| noun_relation_verb_reading | 9 | `-s`/`-ed` readings that are English verbs with another meaning (sports, conflicts, instruments, records, parts, positions, places, instances, sported) |
| reversed_direction | 9 | asymmetric relations with the roles reversed (follows, has part, contains, subclass, instance, member, participant) |
| shape3_mid_of_reversed | 7 | relations with `_of_` in the middle (country_of_citizenship, place_of_birth, ...): the English `X is the N of Y` reverses the roles |
| follows_vs_followed_by | 7 | `follows` / `followed` / `is followed by` matched against an inverse `followed_by` / `follows` triple |
| bidi_visual_reorder | 6 | the logical text is the supported sentence; RLO+LRI/PDI makes it display with the roles swapped (rendering checked with GNU FriBidi 1.0.17) |
| combining_overlay | 6 | strike-through (U+0336) or negation-slash (U+0338) combining marks, which an accent-stripping normaliser deletes |
| chain_hop | 6 | an object of one gathered triple used as if it were the subject's own fact |
| participle_drop_reversal | 6 | described / educated / named / awarded, with the preposition dropped, so the agent is reversed |
| and_compound_comma_split | 5 | a comma inside an `_and_` slug (flora_and_fauna, meurthe_and_moselle, seine_and_oise, bosnia_and_herzigovina) reads as two entities |
| split_trick | 5 | abbreviation periods, ellipsis, decimal points, quotation |
| slug_form_erases_meaning | 8 | sign or dash erased by the slug fold (UTC+06 / UTC−4:30 / UTC 3–5 / UTC−06:30), near-slug entities, digit change, plural |
| past_tense_temporal | 4 | simple past on a stative relation (bordered, shared, contained, had part) |
| glued_nonlatin_affix | 4 | `не`/`非` glued onto a verb, noun or slug, not free-standing |
| homoglyph_word_vanish | 4 | `NOT`/`NO` spelled with Greek or Cyrillic capitals; zero-width spaces inside `not` |
| nonascii_quantity_symbol | 4 | `½`, `≈`, `~` |
| definite_uniqueness | 3 | `is the member/instance/subclass of`: the spec's own `[Det]` slot before the noun implies uniqueness |
| relation_noun_inside_slug | 3 | `member_of_ep` / `uk_mp` subclass of `member_of_althingi`, rendered as "X is a member of Althingi" |
| scare_quotes | 2 | scare quotes on the relation or the noun |
| contraction_negation | 2 | `isn’t`, `doesn’t` with a curly apostrophe |

| class | F |
|---|---:|
| F_twin | 43 |

## Notes for scoring

1. **Shape 2 has no F twin, by construction.** `facts.json` holds exactly one object per (subject, relation): 0 of
   78,857 facts share a (subject, relation) pair. So over this store, no coordinated-objects sentence can be supported.
   Every shape-2 input is unsupported, and shape 2 can only leak; it can never keep a correct answer.
2. **F twins use guessed verb forms.** I have not seen the verb-form table. Each F twin uses either:
   - shape 3 on an `_of` relation; or
   - the relation's own tokens, or a plain verb (`follows`, `shares border with`, `borders`, `has part`, `contains`).

   An F twin rejected only because its verb is not in the table is a retention loss (bar 3's retention report), not a
   label error.
   Twins with special spellings:
   - accented spellings: Z16, Z25, Z27, Z39;
   - case and underscore variants: Z37, Z38;
   - a capital `A` that belongs to a slug: Z43;
   - past tense of an event-order relation: Z42.
3. **split_trick (V01-V05).** If the splitter severs the hedge or negation, the remaining fragment is supported on
   its own. Count V cases as leaks only if the composer keeps a span that, read in the context of the whole input,
   is denied, hedged or reported. Report them separately from the single-sentence classes.
4. **Debatable labels, flagged rather than hidden.**
   - Spec-level: `definite_uniqueness` (T01-T03) uses the spec's own `[Det]` slot, so a strict checker that follows
     the spec will keep these.
   - Past tense: L01-L04.
   - Questions: D07, D08, D12.
   - Scare quotes: E01, E02.

   I labelled all of these U because a careful reader takes each one as hedged, temporal or non-assertive. A
   reviewer may move them to B; report them separately from the clear role and negation reversals.
5. **Entities named in sentences but deliberately absent from the store.** Every gathered triple is in the store.
   - Absent names, used on purpose:
     - `mexico` (A10, A12);
     - `hertha_berlin_i` (R05);
     - `2010_fifa_world_cup_final_positions` (R07);
     - `szczecinek_communes` (R08);
     - `flora`, `fauna` (N01, N02);
     - `meurthe`, `moselle` (N03);
     - `seine` (N04);
     - `bosnia`, `herzigovina` (N05);
     - `ep`, `althingi` (W01-W03).
   - Present in the store but not gathered for that turn: `utc_07` (R06) and `oise` (N04).
   - The UTC±, en-dash and minus-sign spellings in R01-R04 are not slugs. They are spellings that a slug-style fold
     would map onto a gathered slug.
6. Control characters, styled letters and combining marks are stored as real code points in `cases.json` (UTF-8,
   `ensure_ascii=False`). `build_cases.py` shows each one as an escape.
