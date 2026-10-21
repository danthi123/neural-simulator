# Round-5 blind hunt: allowlist claim checker (written 2026-10-20)

## Blindness statement

I did not open, list, grep, cat or view the git history of any of these:
- `research/runners/_moat_allowlist*.py`, `research/runners/_moat_claim_entailment_derisk.py`,
  `research/runners/_moat_allowlist_caps_build.py`;
- `tests/test_moat_allowlist*.py`, `tests/test_moat_hunt_runner.py`, or any `tests/fixtures/` file whose name
  contains `moat_allowlist`;
- `research/findings/raw/_moat_allowlist_caps/`, or any `results*.json` anywhere;
- anything under `/home/dant123/Projects/sim-rev/` beyond the permitted read-only paths and this output directory.

I listed no directory. I ran no `git` command of any kind, and no checker, test or measurement.

**What I read.** All paths are under `/home/dant123/Projects/sim-rev/wt-allow3/`, read-only; I edited nothing there.
- `docs/drafts/moat-allowlist/SPEC.md`, in full.
- `research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md`, in full, with AMENDMENTS 1, 2 and 3.
- `research/runners/_knowledge_core_curate.py`, through these three commands only:
  - `grep -n "def sanitize"`, which returned line 52;
  - `sed -n 52,90p`, which printed `sanitize`, all of `pick_clean_alias` (lines 60-87) and the `def _alias_map(...)` line (line 90);
  - `grep -n "_TOK_RE *="`, which returned line 45.
- `research/findings/raw/_moat_allowlist_hunt3/`: `cases.json` (all 213 rows), `README.md` and `adjudication.json`.
  - My first script printed every row with its gathered set to a saved tool-output file, which I did not open.
  - I saw only that file's 2 KB preview (A01-A04 with their gathered sets).
  - A second script printed id, label, class and sentence for every row.
- `research/findings/raw/_moat_allowlist_hunt4/r4/`: `cases.json` (id, label, class and sentence of all 177 rows) and
  `README.md`.
- `research/findings/raw/_moat_allowlist_hunt4/r4/adjudication.json`: its key list, its first 6,000 characters, and
  every judge-label pair that was not U/U (reasons cut at 150 characters).
- The store `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json`:
  - one `ls -la` of that single file;
  - queries with my own scripts, kept in my session scratchpad outside the repository.

The held-out corpora `_moat_slug_norm_heldout/{corpus,labels,s2_*,s3_*}.json` were permitted, but I did not read them.

**Seen by accident, disclosed.**
1. The round-4 `README.md`, which I was permitted to read, names two results files in `_moat_slug_norm_heldout/`
   (`results_strata_allowlist_smoke.json`, `stub_population_allowlist_smoke.json`). I opened neither.
2. The session context held the project `CLAUDE.md`, the memory index and a git-status snapshot of
   `/home/dant123/Projects/sim`, all supplied by the system.
   - The memory index names the moat work (B2h slug-aware verifier held OFF).
   - The git snapshot shows recent commit subjects about idle-replay.
   - None of them describes the matcher, its tables or its results.
3. The spec and prereg name the verb-form table's commit and the caps-table path. I opened neither.

## Files

- `cases.json`: 164 cases. Each is `{"id", "gathered", "sentence", "expect", "class", "why"}`, and 3 add
  `"arguable": true`.
  - All characters are stored as real code points (UTF-8, `ensure_ascii=False`).
- `build_cases.py`: run `python3 build_cases.py`. It writes `cases.json` from its own case list, then checks:
  - every gathered triple is an exact store triple (166 of 166). The store still has 78,857 facts and 60 relations;
    10 relations are used;
  - the exact key set, with only `arguable` optional;
  - 1-4 triples per case;
  - ids and sentences are unique;
  - no sentence is identical to any of the 390 round-3 or round-4 sentences;
  - at least 110 U and 35 F;
  - **140 fold self-checks over 139 cases.** Each names the displayed span the attack relies on. A local copy of the
    store's `sanitize` (lowercase; runs of `[^a-z0-9]` become `_`; strip `_`; cut at 40; rstrip `_`) must map that span
    to a slug of the gathered set, and the span must occur in the sentence.

## Counts

By label: **U = 117, F = 47**. Arguable: 3, all labelled U.

| class | U | F | what it probes |
|---|---:|---:|---|
| numeral_extension_foldable_letter | 23 | 0 | a Latin letter that reads as I, V or L (`ļ ĺ ľ Í Ï ɪ ı ǀ ᴠ Ĺ`) glued to a Roman numeral. It folds to a separator, so "George Vĺ" spells `george_v` but reads George VI. The cases mix lower-case, uncased and capital glyphs to probe the caps rule and the non-ASCII guard together |
| paren_affix_hedge | 16 | 0 | the slug's own prefix (non-, anti-, semi-, pre-, bi-, post-, sub-, mini-) in parentheses or brackets turns the class into an either/or between opposites |
| paren_modifier_hedge | 13 | 0 | the slug's own modifier (South, New, Northern, Lower, West, East, Old) in parentheses, so the span also names a different, well-known entity (Mongolia, Jersey, Zealand, Ireland) |
| intra_slug_coordination | 16 | 0 | `&`, `+` (spaced and glued), `=`, `≠` or `⊂` placed between two words of one slug ("Bayern & Germany"), which reads as two entities or as an extra identity or subset claim |
| hyphen_minus_sign_flip | 9 | 0 | ASCII hyphen-minus (likely allowed for hyphenated names) as a minus sign on UTC+ zones, plus one `+/-` hedge; round 3 used only `+`, U+2212 and an en dash |
| digit_punct_quantity | 8 | 0 | punctuation in a numeric slug (`1_500_m`) that changes the quantity: a range, a fraction, a decimal point, `+`, `<`, `>` or `~` |
| unit_exponent_fold | 2 | 0 | a superscript 2 or 3 after `m` makes square or cubic metres, and the superscript folds away |
| currency_sign_fold | 2 | 0 | a `$` turns "1,500 m" into 1,500 million dollars |
| digit_lookalike_letter | 5 | 0 | a Latin letter shaped like a digit (ezh ≈ 3, tone-five ≈ 5, dental click ≈ 1) in a separator slot changes the displayed number |
| genitive_initial_swap | 5 | 0 | a middle initial S (`clive_s_lewis`) written as a possessive 's ("Clive's Lewis"), which names a different referent; round 4's possessives sat only inside 40-character cap tails |
| foldable_letter_initial_to_name | 2 | 0 | an accented letter after an initial turns it into a different first name ("Aĺ Berg" reads Al Berg; "Sý Morgenstern" reads Sy) |
| latin_diacritic_homoglyph | 2 | 0 | a Latin-script V look-alike (V-tilde, V-hook) before `ästra`: the span folds to `stra_aros` (Östra Aros) but reads Västra Aros. Round 4 used Cyrillic here, so this tests a script-based guard |
| connective_word_in_slug | 4 | 0 | a slug containing `or`, `if`, `versus` or `nothing but`; a comma or lower case makes the word read as a disjunction, conditional, contrast or quantifier |
| abbreviation_reread | 1 | 0 | the slug's NO (Norway) written "No. 01", which reads as "number 01" |
| glued_logic_symbol | 2 | 0 | `~` or `¬` glued inside the object span, rather than standing alone or wrapping the sentence as in round 3 |
| split_context_new | 5 | 0 | a denial joined by separators round 4 did not use: U+2028, NEL, the Devanagari danda, an ideographic full stop followed by a Chinese denial, and an explicit "previous sentence is a lie" |
| verb_reading_new | 2 | 0 | the `-s` reading "locates" for `located_in_the_administrative_territoria` (to locate means to find) |
| F_twin | 0 | 47 | supported twins (see note 4) |

## Arguable cases (labelled U)

- **R5-DQ03** "1.500 m is a subclass of middle distance runner." In English this is 1.5 m, but continental usage writes
  1.500 for 1,500.
- **R5-DQ04** "5.000 m is a subclass of long distance running." Same separator ambiguity.
- **R5-CW02** "If chins could kill, is an instance of human specie." The comma turns the name into a conditional
  clause, but the sentence is garbled enough that a reader may see only a stray comma.

## Notes for scoring

1. **Where the falsity sits.** Every U case except VR01-VR02 uses the relation's own wording ("follows", "shares
   border with", "is a member of", "is an instance of", "is a subclass of", "is part of", "has part", "is a participant
   of", "(is) located in time zone"). The unsupported meaning sits inside a SUBJ or OBJ span, in characters or
   punctuation the store's slug rule folds away.
   - Each such case carries a fold self-check (see Files).
   - A case rejected only because its verb wording is not in the table is a correct reject, not a label problem.
   - This is why only 10 of the 60 relations are used.
2. **Split rule.** SC01-SC05 follow round 3's `split_rule`: a case is a lenient leak only if the checker keeps a span
   whose reading within the whole input is denied, hedged or framed. Report them apart from the single-sentence classes.
3. **Shared triples.**
   - Fifteen U cases (DQ01-DQ08, UX, CU, DL01-DL03) and three F twins (F23-F25) share three numeric triples (`1_500_m`, `5_000_m`,
     `10_000_m`). Each case is still a separate input.
   - Store entities that a U case's display names, but that are not gathered:
     - `university_of_paris_viii` (NX13 reads as Paris VIII);
     - `v_stra_aros`, Västra Aros (LH01, LH02).
4. **F twins.**
   - Most F twins state the U cases' triples in the correct spelling:
     - a correct minus on UTC-6/-7 zones (F01-F04);
     - appositive commas ("Bayern, Germany"; F07-F11);
     - initials with periods (F20-F22, F44);
     - plain numerals (F26-F32);
     - "Albany Lebanon OR SA" with the Oregon OR in capitals (F33).
   - Retention probes whose display equals the plain text:
     - F05/F06 (`+` sign);
     - F12 (curly apostrophe);
     - F24/F25 (narrow no-break and thin-space thousands separators);
     - F40 (no-break space);
     - F41/F42 (fi/fl ligatures);
     - F43 (zero-width non-joiner);
     - F35 (Ö);
     - F10 (æ);
     - F36/F37 (`S&P`, `A/S`);
     - F34 (`ISO 3166-2:NO-01`);
     - F39 (`e.V.`);
     - F47 (`0-6-0DH`).

     Rejecting one of these is an answer loss, not a label error.
5. **Labels that rest on world knowledge.**
   - The time-zone slugs are unsigned (`utc_3_5`, `gmt_5_45`). The U labels for HM01-HM09 rest on the fact that these
     zones are east of Greenwich (Iran, Afghanistan, Myanmar, Nepal); the gathered subjects say so.
   - CO16 ("Alicante⊂Spain") adds a claim that is true in the world but not gathered. Under the label definition, an
     extra assertion the gathered triples do not entail is U.
6. **Caps rule interplay.** Some NX cases use capital or uncased glyphs (`Í`, `Ï`, `Ĺ`, `ǀ`). If the caps rule counts
   non-ASCII capitals, it rejects them on its own. The lower-case ones (`ļ ĺ ľ ɪ ı`) and the prefix `ᴠ` do not form
   an all-capitals token.
