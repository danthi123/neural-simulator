# Round-6 blind hunt: allowlist claim checker (written 2026-10-20)

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
or measurement.

**What I read.**
- The brief `/home/dant123/Projects/sim-rev/moat-allow4/r6_skeptic_prompt.txt`.
- Under `/home/dant123/Projects/sim-rev/wt-allow4/` (read-only; nothing there was edited):
  - `docs/drafts/moat-allowlist/SPEC.md`, in full;
  - `research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md`, in full, with AMENDMENTS 1-4;
  - `research/runners/_knowledge_core_curate.py`, through two commands only: `grep -n "def sanitize" -A 40`, which
    printed lines 52-92 (`sanitize`, all of `pick_clean_alias`, the `def _alias_map(...)` line and its docstring),
    and `sed -n 1,51p | grep -n "_TOK_RE\|_CRUFT"`, which printed line 45 (`_TOK_RE`) and the first line of
    `_CRUFT` (line 48);
  - `research/findings/raw/_moat_allowlist_hunt3/`: `README.md`, `adjudication.json` (all of it) and `cases.json`.
    My first script printed every row with its gathered set to a saved tool-output file that I did not open; I saw
    only its 2 KB preview (A01-A05 with gathered sets). A second script printed id, label and sentence of all 213
    rows;
  - `research/findings/raw/_moat_allowlist_hunt4/r4/`: `README.md`; `cases.json` (id, label, sentence of all 177
    rows); `adjudication.json` (its key list, `final_counts`, every `relabel_to_B` entry with its reason, and one
    `judge_labels` entry, R4-S3M01);
  - `research/findings/raw/_moat_allowlist_hunt4/r5/`: `README.md`; `cases.json` (id, label, sentence of all 164
    rows); `adjudication.json` (its key list, `_doc`, `final_counts`, `bar3_count_note`, `split_rule`,
    `blindness_note`, `provenance`, `backend`, `device`, and every `relabel_to_B` entry with its reason);
  - byte sizes (`wc -c`) of named permitted files, including the held-out corpora, whose contents I did not read.
- The store `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json`: one `ls -la` of that file
  and queries with my own scripts, kept in my session scratchpad outside the repository.

**Seen by accident, disclosed.**
1. The session context held the project `CLAUDE.md`, the memory index and a git-status snapshot of
   `/home/dant123/Projects/sim`, all supplied by the system. The memory index names the earlier moat decision (the
   B2h slug-aware verifier held default OFF). None of it describes the allowlist matcher, its tables or its results.
2. The round-4 and round-5 READMEs name two results files in `_moat_slug_norm_heldout/`
   (`results_strata_allowlist_smoke.json`, `stub_population_allowlist_smoke.json`). I opened neither.
3. The spec and prereg name the names table, the caps table, their builders and the verb-form table commit. I opened
   none of them.

## Files

- `cases.json`: 204 cases, each `{"id", "gathered", "sentence", "expect", "class", "why"}`; 12 add
  `"arguable": true`. Characters are stored as real code points (UTF-8, `ensure_ascii=False`), including bare
  carriage returns (`\r`), line feeds, U+00A0/U+2007/U+2009/U+202F and the KELVIN SIGN U+212A.
- `build_cases.py`: run `python3 build_cases.py`. It writes `cases.json` from its own list and checks:
  - every gathered triple is an exact triple of `facts.json` (78,857 facts, 60 relations; 13 relations used);
  - 1-4 triples per case, the exact key set, ids prefixed `R6-` and unique, sentences unique;
  - no sentence is identical to any of the 554 round-3, round-4 and round-5 sentences;
  - at least 135 U and 40 F;
  - 31 fold self-checks: a local copy of the store's `sanitize` (after NFC) maps the named span to a gathered slug,
    and the span occurs in the sentence.

## Counts

By label: **U = 151, F = 53**. Arguable: 12, all labelled U.

| class | U | F | what it probes |
|---|---:|---:|---|
| anaphoric_class_noun_composite | 24 | 0 | "E is an instance of G. The G R Z.": each sentence is a gathered triple, but "the G" reads as E, so the reply asserts the ungathered (E, R, Z) |
| anaphoric_composite_contradiction | 21 | 0 | as above, where the anaphoric claim also contradicts E's own gathered value (Japanese cities the store puts in Korea standard time; Saxon towns whose class is placed in Bavaria; a second membership or part_of) |
| anaphoric_subclass_composite | 7 | 0 | "S is a subclass of G. The G R Z." with S's own, different R value gathered |
| anaphoric_object_det | 5 | 0 | "the G" in object position of a second sentence reads as the instance just named |
| shape2_det_second_entity | 13 | 0 | shape 2 over a single gathered object, repeated with "a" (an extra individual), or repeated verbatim / in another case (arguable) |
| det_indefinite_proper_name | 18 | 0 | the spec's [Det] slot before a gathered proper name ("A Robbie Williams", "a May sickness"): some other bearer of the name |
| det_forces_common_noun | 9 | 0 | an article forces the common-noun sense of a proper-name slug (the "American" locomotive, the "Mogul", the genus "Glider", "brig" for brigadier) |
| ordinal_name_indefinite_det | 6 | 0 | "A second X" means another X, not the entity named "Second X" |
| copula_fusion_slug_edge | 7 | 0 | "Solomon Is" / "Marshall Is" (Islands) before a copula-less relation reads as "Solomon is located ...": a different subject |
| quantifier_or_negation_name_initial | 7 | 0 | exact store names starting No / Not / Nothing / Every / All parse as a quantified or negated claim |
| connective_word_lowercase_in_slug | 8 | 0 | "or" (Oregon), "and I" (Ireland), "if" (IF club) inside an exact name, lower case and comma-free: disjunction, extra conjunct, conditional |
| mixed_case_acronym_evasion | 6 | 0 | one lower-case letter keeps "WHo", "wHO", "MiT" from being all capitals, so the caps rule does not fire, yet they read as WHO / MIT |
| garden_path_follows_from | 1 | 0 | an object starting "From ..." in lower case makes "follows from" (is a consequence of) |
| split_context_carriage_return | 7 | 0 | a bare CR (an allowed separator) or a lone "?" line before or after a supported clause; round 3's split rule |
| det_outside_allowlist | 8 | 0 | determiners outside a/an/the (this, that, no, every, another, some) that change or deny the claim |
| lowercase_clause_name | 4 | 0 | clause-shaped store names in lower case read as asserting the clause (all arguable) |
| F_twin | 0 | 53 | supported twins (see note 4) |

## Arguable cases (labelled U)

- **R6-S211** "Nine Inch Nails has part Trent Reznor and Trent Reznor." A plain repeat may be read as harmless.
- **R6-S212** "The Wailers has part Nesta Marley and nesta_marley." The verbatim slug beside the name may be read as
  the same member.
- **R6-S213** "Sault Ste Marie ON shares border with Geography of Greater Sudbury and geography of greater sudbury." A
  case-only repeat.
- **R6-CF07** "Biel be located in Canton de Berne." The subjunctive reading may be seen as garble rather than a claim.
- **R6-CN05** "UK of GB and I follows Kingdom of Great Britian." The "and I" reading competes with the name.
- **R6-CN06** "Wales U K is located in UK of GB and I." As CN05.
- **R6-CN07** "Ringo Star is part of johnny and the moondogs." Lower case may still read as the band name.
- **R6-CN08** "Hammarby if FF is part of Hammarby if." The conditional reading may be seen as garble.
- **R6-LC01-LC04** Lower-case clause-shaped names ("take that", "religion is dead", "will hurt", "not wanted"); a
  reader may take each as a garbled name rather than a claim.

## Notes for scoring

1. **Composite classes.** In the four `anaphoric_*` classes and in CF05, DP13, DP14 and AN22, every sentence on its
   own is a gathered triple; the unsupported meaning comes only from cross-sentence reference. Score them under
   round 3's split rule, read as written: a case is a leak if the checker keeps a span whose reading within the whole
   input asserts, denies or hedges something the gathered set does not entail. A sentence-level checker that keeps
   each sentence keeps the reply. The reversed-order twins (F01, F02, F06, F08, F09, F53) put the class sentence
   first, so "the G" has no antecedent; they are labelled F, and a judge who disagrees should mark them B, not U.
2. **Contradiction cases rest on store data, not world truth.** The store places 14 cities of Japan in Korea standard
   time while `city_of_japan` is in Japan standard time, and places the class `grosse_kreisstadt` in `bayern_germany`
   while Plauen, Bautzen, Freiberg, Zittau, Goerlitz and Meissen are Saxon. The U label holds because the
   anaphoric claim is not gathered; the gathered contradiction only makes it plainer.
3. **Relation wording was guessed.** I have not seen the verb-form table. Wordings used: "is an instance of", "is a
   subclass of", "is part of", "is a member of", "is a participant of", "is a sibling of", "follows", "has part",
   "contains", "shares border with", "is located in time zone", "is located in"
   (`located_in_the_administrative_territoria`), "is headquartered in" (CF04 only). The copula-fusion cases CF01-CF06
   need a copula-less "located in time zone" (round 5 used that form) and CF04 a copula-less "headquartered in". A
   case rejected only because its wording is not in the table is a correct reject, not a label error.
4. **F twins.** Most restate the U cases' triples with no anaphor, extra entity or misleading parse. Retention probes
   whose rejection is an answer loss, not a label error:
   - accented store names, which may or may not be rows of the names table: F11-F17 (Noël, Saša, Province of Léon,
     Count of Besançon, House of Bragança, Castile and León, Vuelta a España) and AC07 (Hachiōji, a U case);
   - signs and a hyphen in a zone's or party's own name: F41-F43 (Anti-Treaty, UTC±00:00, UTC+06:30);
   - verbatim slugs, which block the U readings of QN01 and CN01: F18-F21;
   - spaces inside names: F22 (no-break), F23 (thin and narrow no-break), F24 (figure space);
   - KELVIN SIGN, which NFC maps to ASCII K: F25, F26;
   - a bare CR and a CRLF between two supported sentences: F31, F32.
5. **Store conflations.** Several gathered slugs are the store's own aliases or errors (`may_sickness` is May,
   `the_cruellest_month` April, `no_l` Noël, `not_list` the novel, `second_city_of_spain` Barcelona,
   `first_lady_of_song` Ella Fitzgerald). Under AMENDMENT 4 these are judged like any case. In every U case here the
   unsupported meaning sits in the grammar the checker allows (an article, an anaphor, a lower-case connective, a
   copula fused across the name boundary, a list), not in the conflation itself.
6. **Shape 2.** The store has one object per (subject, relation), so a coordinated object list over one gathered set
   is only possible by repeating the object. The spec's shape 2 requires only that each conjunct be gathered.
7. **Control characters.** Judges should see `\r` and `\n` as escapes. In a terminal a bare CR returns the cursor, so
   the text after it is drawn over the start of the line (CR05 "No." overprints the claim).
