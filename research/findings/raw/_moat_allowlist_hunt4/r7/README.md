# Round-7 blind hunt: allowlist claim checker (written 2026-10-20)

## Blindness statement

I did not open, list, grep, cat or view the git history of any of these:
- `research/runners/_moat_allowlist*.py` (including `_moat_allowlist_names_build.py` and `_moat_allowlist_caps_build.py`),
  `research/runners/_moat_claim_entailment_derisk.py`;
- `tests/test_moat_allowlist*.py`, `tests/test_moat_hunt_runner.py`, or any `tests/fixtures/` file whose name contains
  `moat_allowlist`;
- `research/findings/raw/_moat_allowlist_caps/`, `research/findings/raw/_moat_allowlist_names/`,
  `research/findings/raw/_moat_allowlist_hunt4/run_hunt.py`, any `results*.json` or `r3_rerun*.json`, or
  `/home/dant123/Projects/sim-data/wikidata5m/`;
- anything under `/home/dant123/Projects/sim-rev/` other than the permitted read-only paths and this directory.

I listed no directory (this one was made with `mkdir -p`). I ran no `git` command of any kind, and no checker, test
or measurement. Nothing under `/home/dant123/Projects/sim-rev/wt-allow5/` was edited.

**What I read** (all under `/home/dant123/Projects/sim-rev/wt-allow5/`, read-only):
- `docs/drafts/moat-allowlist/SPEC.md`, in full.
- `research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md`, in full, with AMENDMENTS 1-5.
- `research/runners/_knowledge_core_curate.py`, through two commands only: `grep -n "def sanitize" -A 40`, which
  printed lines 52-92 (`sanitize`, all of `pick_clean_alias`, the `def _alias_map(...)` line and its docstring), and
  `grep -n "^_TOK_RE"`, which printed line 45.
- `research/findings/raw/_moat_allowlist_hunt3/`: `README.md` (its first 80 lines), `adjudication.json` (all of it),
  `cases.json` (the key list of the first row, and every sentence, loaded by my validator for the repeat check).
- `research/findings/raw/_moat_allowlist_hunt4/r4/` and `r5/`: `adjudication.json` (`final_counts`, the
  `relabel_to_B` id set) and `cases.json` (id, class and sentence of every expect-U row not relabelled B, plus every
  sentence for the repeat check). I did not read the r4 or r5 `README.md`.
- `research/findings/raw/_moat_allowlist_hunt4/r6/`: `README.md` (all of it); `adjudication.json` (its key list,
  `_doc`, `final_counts`, `bar3_count_note`, `label_crosstab`, `split_rule`, `blindness_note`, `provenance`,
  `backend`, `device`, the first two `relabel_to_B`, `reasons` and `judge_labels` entries, and both judges' labels and reasons for
  every expect-U row that the two judges did not both read as F); `cases.json` (those rows' sentence and gathered
  set, every row whose sentence names Solomon, Marshall, Thun or Biel, and every sentence for the repeat check).
- `research/findings/raw/_moat_slug_norm_heldout/`: byte sizes (`wc -c`) of all six files; `corpus.json` (top-level
  keys and the first one, `_doc`); `labels.json` (top-level keys); `s3_corpus.json` (top-level keys, `_doc`, the
  first `raw_files` entry); `s3_labels.json` (top-level keys and the first five labels). I did not read `s2_*`.
- The store `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json`, through my own scripts
  (kept in my session scratchpad, outside the repository).

**Seen by accident, disclosed.**
1. The session context held the project `CLAUDE.md`, the memory index and a git-status snapshot of
   `/home/dant123/Projects/sim`, all supplied by the system. The memory index names the earlier moat decision (the
   B2h slug-aware verifier held default OFF). None of it describes the allowlist matcher, its tables or its results.
2. The round-6 README names two results files in `_moat_slug_norm_heldout/` (`results_strata_allowlist_smoke.json`,
   `stub_population_allowlist_smoke.json`), and the prereg names `measure_strata.py`, `composer_records.py` and
   `heldout_numbers.py`. I opened none of them.
3. The spec and prereg name the names table, the caps table, their builders and the verb-form table commit. I opened
   none of them.
4. The round-6 adjudication shows which round-6 cases the judges read as faithful. That is a label record, not a
   checker verdict; Amendment 5 already states which round-6 cases build 4 kept.

## Files

- `cases.json`: 270 cases, each `{"id", "gathered", "sentence", "expect", "class", "why"}`; 1 adds `"arguable": true`.
  Characters are stored as real code points (UTF-8, `ensure_ascii=False`), including U+00A0, U+2007, U+2009, U+202F
  and line feeds.
- `build_cases.py`: run `python3 build_cases.py`. It writes `cases.json` from its own list and checks:
  - every gathered triple is an exact triple of `facts.json` (78,857 facts, 60 relations; 18 relations used);
  - 1-4 triples per case, the exact key set, ids prefixed `R7-` and unique, sentences unique;
  - no sentence is identical to any of the 758 round-3 to round-6 sentences;
  - at least 170 U and 40 F;
  - 26 display-space self-checks: mapping the four display spaces to U+0020, collapsing runs and lower-casing the
    named span gives a gathered slug with underscores as spaces, and the span occurs in the sentence.

## Counts

By label: **U = 216, F = 54**. Arguable: 1 (labelled U).

| class | U | F | what it probes |
|---|---:|---:|---|
| copula_fusion_display_space | 22 | 0 | **the leak probe.** Round 6's "Solomon is located in time zone ..." (`solomon_is`, `marshall_is`, `thun_be`), with the last space of the name written as U+00A0, U+2007, U+2009, U+202F or a run of them. Build 4 counts those as a space inside a name; the case asks whether build 5's clause-word rule, which splits the displayed name "on spaces", splits on them too. Also new relations for the class (`described_by_source`, `instance_of`, `member_of`, a copula-less "member of", "located in" for the time zone), all lower case, and two sentences |
| relation_token_reuse_hedge | 14 | 0 | hedges and negations built from words already in the relation labels ("in part", "part of the time", "part time", "at no point", "in some instances", "in name only") |
| modality_aspect | 15 | 0 | modals and aspect that change truth (may, might, could, used to, is about to, is set to, has stopped, has yet to, is rumoured to) |
| comparative_superlative | 10 | 0 | rankings and comparisons the triple does not state (most famous, longest border, larger member than) |
| apposition_relative_smuggle | 12 | 0 | a relative clause or appositive smuggling a second, false claim beside a supported one |
| ellipsis_gapping | 9 | 0 | a gapped or elided second conjunct asserting an ungathered triple ("and June Jul", "and Marshall Is too") |
| counterfactual_conditional | 8 | 0 | counterfactuals that presuppose the fact is false ("would have been ... but he never joined") |
| report_attribution | 8 | 0 | the fact reported as rumour, myth, disputed or wrongly claimed |
| quantity_numeral | 12 | 0 | counts, lengths and frequencies in words and digits (two Robbie Williams, 9 km, 12 times a year) |
| genitive_of_scope | 9 | 0 | an "of" phrase that moves the claim to another entity (the board of, the capital of, the son of) |
| symmetric_reciprocal | 8 | 0 | "each other" / "one another" on asymmetric relations, adding the reversed triple |
| time_index_change | 10 | 0 | dates and change-of-state verbs (since 1975, left, quit, joined in 2004, no longer) |
| cross_triple_swap | 17 | 0 | a realistic mix-up of two gathered facts, several with overlapping slugs (`eugene_or` / `greater_eugene`, a county for its city) |
| shape2_cross_relation_conjunct | 8 | 0 | shape 2 whose added conjunct is the same subject's object under another relation (birthplace, class, instrument as a membership) |
| near_negative_adverb | 8 | 0 | nearly, almost, hardly, scarcely, all but |
| exclusive_focus | 7 | 0 | only, alone, sole, exclusively |
| generic_quantified_class_subject | 4 | 0 | each / any / all over a class, contradicted by a gathered instance (Kushiro City in Korea standard time; Freiberg in Mittelsachsen) |
| lexical_negation_antonym | 12 | 0 | negation by content verbs (lacks, fails to, excluded, absent) and antonyms that reverse order (precedes, comes before) |
| name_truncation_other_entity | 8 | 0 | dropping part of a name leaves a different real entity (Bern, Jersey, Guinea, Papua, the Asian Games series, the Commonwealth of Poland) |
| name_extension_other_entity | 8 | 0 | adding words makes a different real entity (UN Security Council, Lesser Poland, June Carter, a tribute band) |
| numeral_inside_name_change | 4 | 0 | a changed numeral inside a name (Jackson Six, XXXI Olympic Games, Fourth Asian Games, third chromosome) |
| shape2_duplicate_variant | 3 | 0 | round 6's duplicated object with a display space or the verbatim slug in the second copy, and "another" |
| F_twin | 0 | 54 | supported twins (see note 3) |

## Arguable case (labelled U)

- **R7-A20** "Thun" + U+00A0 + "be located in Canton de Berne." The subjunctive or imperative reading of "be" may be
  taken as garble rather than a claim about a town called Thun (round 6's R6-CF07 was the ordinary-space version).

## Notes for scoring

1. **Where a leak can come from.** Under the spec every token outside an entity span, the relation and a/an/the
   rejects, so most classes here are expected rejects that add adjudicated U cases for bar 3's power. The class built
   to be kept is `copula_fusion_display_space`: every token is a gathered name, a relation word or the period, the
   name equals the gathered slug once the display spaces count as spaces, and the sentence reads as round 6's U/U
   case R6-CF01. R6-CF03 ("Solomon Is located ...", capital I) was read as faithful by both round-6 judges, so every
   case here writes the fused word in lower case.
2. **Display spaces.** U+00A0, U+2007, U+2009 and U+202F render like an ordinary space. Judges should read them as
   spaces; the anonymised copy may show them as escapes. `\n` in R7-A18 and R7-F52 is a line feed between sentences.
3. **F twins.** Each restates gathered triples with no extra content. Retention probes: display spaces inside names
   with the copula after the full name (F04, F05, F08, F45, F46), the verbatim slug (F07), no final period (F10, F13,
   F19, F20), "the" before an object (F17, F24, F41), three and two distinct facts in one reply (F43, F44, F47, F52,
   the last with a line feed), and the store's own misspellings and wrong values (F29 New Jeresy, F40 Kushiro City in
   Korea standard time). A twin rejected only because its wording is not in the verb-form table is an answer loss,
   not a label error.
4. **Relation wording was guessed.** I have not seen the verb-form table. Wordings used: "is located in time zone",
   "located in time zone", "is described by source", "described by source", "is a member of", "is an instance of",
   "is part of", "is a participant of", "is a sibling of", "has part", "follows", "is followed by", "shares border
   with", "contains", "is twinned with", "is named after", "is educated at", "is located in". A U case rejected only
   for its wording is a correct reject.
5. **Store values.** The store puts Kushiro City in Korea standard time and the class Grosse Kreisstadt in Bayern;
   the generic cases (R01-R04) rest on those gathered contradictions, not on world truth. Several slugs are the
   store's own aliases (`pole_land` Poland, `may_sickness` May, `new_jeresy` New Jersey, `old_zealand` Zeeland,
   `wacko_jacko` Michael Jackson). In every U case the unsupported meaning sits in words or name boundaries the
   sentence adds, not in the alias itself.
