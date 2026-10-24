# Round-8 blind hunt: allowlist claim checker, build 6 (written 2026-10-21)

## Blindness statement

I did not open, list, grep, cat or view the git history of any of these:
- `research/runners/_moat_allowlist*.py` (all of them, including the names, caps and relations modules),
  `research/runners/_moat_claim_entailment_derisk.py`, `research/runners/rich_answer_composer.py`,
  `research/runners/brain_chat_tui.py`, or anything under `webapp/`;
- anything under `tests/` (including `tests/fixtures/`);
- `research/findings/raw/_moat_allowlist_caps/`, `_moat_allowlist_names/`, `_moat_allowlist_build6/`,
  `_moat_allowlist_verdict/`, `_moat_allowlist_hunt4/run_hunt.py`, `judge_prep.py`, `adjudicate.py`, any `results*.json`,
  `*rerun*.json` or `stub_population*.json`, or `/home/dant123/Projects/sim-data/wikidata5m/`;
- anything under `/home/dant123/Projects/sim-rev/` other than the read-only paths below and this directory (in
  particular not `preflip/`, `wt-hunt8/`, `moat-verdict/` or any other `moat-allow*/` path).

I listed no directory (this one was made with `mkdir -p`; its parent was never listed). I ran no `git` command of any
kind, and no checker, test or measurement. Nothing under `/home/dant123/Projects/sim-rev/wt-allow6/` was edited.

**What I read** (all under `/home/dant123/Projects/sim-rev/wt-allow6/`, read-only):
- `docs/drafts/moat-allowlist/SPEC.md`, in full (with the build-6 bullet).
- `research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md`, in full (Amendments 1-6 and the
  VERDICTS section).
- `research/runners/_knowledge_core_curate.py`, through two commands only: `grep -n "def sanitize" -A 30` (lines
  52-82: `sanitize` and the first part of `pick_clean_alias`) and `grep -n "_TOK_RE\s*="` (line 45).
- `research/findings/raw/_moat_allowlist_hunt3/cases.json` and `_moat_allowlist_hunt4/r4/cases.json`,
  `r5/cases.json`: every sentence, loaded by my validator for the repeat check, plus the rows (id, label, class,
  sentence) printed by one regex search for words I planned to use (the phrase wordings, "perhaps", "?", "!", and a
  few names). I did not read the hunt3, r4 or r5 `README.md` or `adjudication.json`.
- `research/findings/raw/_moat_allowlist_hunt4/r6/`: `README.md` (all of it); `adjudication.json` (`final_counts`,
  `label_crosstab`, the `relabel_to_B` set and both judges' labels, without reasons, for every expect-U row);
  `cases.json` (id, class and sentence of every expect-U row; every sentence for the repeat check).
- `research/findings/raw/_moat_allowlist_hunt4/r7/`: `README.md` (all of it); `adjudication.json` (its key list,
  `_doc`, `relabel_to_B`, `reasons`, `final_counts`, `bar3_count_note`, `label_crosstab`, `split_rule`,
  `blindness_note`, `provenance`, `backend`, `device`, and both judges' labels and reasons for the 25 relabelled
  rows); `cases.json` (sentence and gathered set of those 25 rows; every sentence for the repeat check).
- Byte sizes (`wc -c`) of the spec, the prereg and the round-3 to round-7 files above.
- The store `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json` (one `ls -la` of that file
  and queries with my own scripts, kept in my session scratchpad outside the repository).

I did not read the held-out corpora under `_moat_slug_norm_heldout/` (permitted, not needed).

**Seen by accident, disclosed.**
1. The session context held the project `CLAUDE.md`, the memory index and a git-status snapshot of
   `/home/dant123/Projects/sim`, all supplied by the system. The memory index has one-line summaries of the moat arc
   (builds 1-4 NO-GO, build 5 passing with caveats, pre-flip defects such as the joined reply and bypass paths, the
   B2h slug-aware verifier held OFF). None of it describes the matcher's code, tables or round-8 results.
2. The spec and prereg name the names table, the caps table, the relations module, `run_hunt.py`, the build-6
   golden fixture, test files and `matcher_diff_vs_build5.json`. I opened none of them.
3. The r6 and r7 adjudications show which earlier cases the judges read as faithful. That is a label record, not a
   checker verdict; the prereg already states which earlier cases each build kept.

## Files

- `cases.json`: 254 cases, each `{"id", "gathered", "sentence", "expect", "class", "why"}`; 22 add
  `"arguable": true`; the 79 reply cases add `"reply": true` and `"sentences"` (2-4 sentences, one per gathered
  triple, in order; `sentence` is their single-space join). Characters are stored as real code points (UTF-8,
  `ensure_ascii=False`), including U+00A0, U+2007, U+2009 and U+202F in 18 cases.
- `build_cases.py`: run `python3 build_cases.py`. It writes `cases.json` from its own list and checks:
  - every gathered triple outside class `taught` is an exact triple of `facts.json` (78,857 facts, 60 relations; 37
    relations used); in class `taught` every invented slug equals the store's `sanitize` of itself with underscores
    as spaces, and every relation is one of the 60;
  - 1-4 triples per case, the exact key set, ids prefixed `R8-` and unique, sentences unique;
  - reply cases: 2-4 sentences, one per triple, and `sentence` equal to their single-space join;
  - no case sentence and no reply part is identical to any of the 1,028 round-3 to round-7 sentences;
  - at least 170 U and 40 F; at least 30 expect-U reply cases (overall and within `reply_joined`); at least 20, 15,
    15 and 10 expect-U cases in `phrase_wording`, `first_second_person`, `points_back_description` and
    `known_residue`; at least 5 `taught` cases.

## Counts

By label: **U = 202, F = 52** (254 cases). Arguable: 22, all labelled U, so 180 expect-U cases are not arguable.
Expect-U reply cases: 68 (40 `reply_joined`, 19 `points_back_description`, 5 `taught`, 4 `known_residue`).

| class | U | F | arguable | what it probes |
|---|---:|---:|---:|---|
| reply_joined | 40 | 11 | 2 | 2-3 sentence replies run as one: a reversed or swapped build-6 wording in a later sentence, cross-sentence ellipsis and reference ("So is Beau Bridges", "He is also", "The latter", "both"), scope and exclusivity across sentences, a hedge or first-person guess frame inside a composer reply, and a droppable middle sentence whose removal leaves a selection with a lower-case "the band" anaphor (RJ38, RJ39) |
| phrase_wording | 32 | 0 | 0 | the nine build-6 wordings with roles reversed (all nine relations), chain hops (grandfather or grandparent taxon called "the father" / "the parent taxon"), cross swaps of two gathered triples, and one wording used for another relation (spouse for a child, sport for an instrument, father for a spouse) |
| phrase_modifier | 17 | 0 | 0 | negation, "no longer", "only", "former", "adoptive", "said to be", "may", "probably", "was once", counts and extra conjuncts wrapped around the build-6 wordings |
| first_second_person | 21 | 0 | 5 | store names that make the mouth say "I", "you", "your" or "me" (`i_need_a_beat`, `i_heart_tuesdays`, `lifting_you`, `feel_your_heart_beat`, `your_honor`, `fire_walk_with_me`, `you_don_t_arrest_voltaire`, ...), in lower case, in object position, with "plays" and "participated in", and with a display space right after "I" (FP03, FP07, FP21) or inside the name (FP04) |
| points_back_description | 19 | 0 | 4 | a later "the band" / "the duck man" whose "the" and lower-case noun are joined by a display space (so a rule that splits on U+0020 may not see "the" + a lower-case word), hypernym antecedents (the Rolling Stones, the Velvet Underground, the Jackson Five, the Who, Take That) and subset antecedents (Genesis band, Yes band), plain and title-case controls, bare surnames (no "the"), the verbatim slug, a one-word hypernym ("the monarch") |
| known_residue | 17 | 0 | 9 | Amendment 5 and 6 residue under the strict reading: "and" inside a name read as a list, quantifier or focus word at a name's start ("most Czech Republic", "only fools and horses"), restatement through the inverse relation with "a", the verbatim slug `utc_06`, capital final "S", lower-case verbatim `no_saints`, lower-case names holding a lexical verb ("steve threw up", "celtics beat la chant"), title-case hypernym "The Band", a non-initial "i", the same-slug class-noun anaphor |
| hypothesis_frame | 19 | 0 | 0 | a leading "perhaps " (exact spiking prefix, capitalised, upper case, with a comma, double space) and host-style guess frames ("My guess is that", "My best guess:", "I would guess that", "I think", "I believe", "If I had to guess,", "I do not know whether") on a mouth sentence, where they are not declared non-claims |
| terminal_punctuation | 16 | 0 | 1 | candidates with no terminal punctuation (questions, tags, dangling "or", run-ons, parenthesised hedges, "or not") that get a period added before the check; "?", "?!", "!?", " ?", "?." and "..." endings |
| taught | 21 | 4 | 1 | conversation-taught names: one-word titles that are pronouns ("me", "us", "mine", "She", "It", "Her"), first-person titles in object position ("my life", "our mutual friend", "dreams from my father", "our town", "your song"), "I"/"i" not first ("My father and I", "Trevor and I" with the past-tense conflict wording, "honey i shrunk the kids", "sometimes i lie", "Baby I love you", "and i love her"), and pronoun titles read across sentences (TA14-TA16, TA21) |
| F_twin | 0 | 37 | 0 | supported twins, most in the nine build-6 wordings (see note 3) |

## Arguable cases (labelled U)

- **R8-RJ12** "He s dead Jim" after W Churchill: the telegraphic tail may make the run read as a garbled name.
- **R8-RJ13** "Not wanted is an instance of human specie." after a membership: may read as a garbled name, not a comment.
- **R8-FP01** "I need a beat is an instance of human specie.": "is an instance of" forces the name reading.
- **R8-FP12** "Your honor is a subclass of legal scholar.": the store name is the title.
- **R8-FP14** "Electrocuting yourself ...": "you" may be read as generic.
- **R8-FP16** "All my children genre soap star.": a known title.
- **R8-FP19** "I language of work or name French language.": the slug is the letter i.
- **R8-PB10, R8-PB11** bare surnames (Cruz, Johansson) pointing back: the slugs are surname pages.
- **R8-PB12** the verbatim slug `the_band` after the Rolling Stones: underscores read as an identifier.
- **R8-PB19** "The monarch ..." after King Abdulaziz: "the monarch" is also the class name.
- **R8-KR02** "meurthe and moselle" as a list: the self-border is absurd enough to force the name reading.
- **R8-KR03** "flora and fauna" as a list: an idiom.
- **R8-KR06** "A Michael Philip Jagger ...": round 6's judges read this form (R6-DP13) as either way.
- **R8-KR07** `utc_06` written verbatim.
- **R8-KR08** "Gary Numan S": may be read as part of the name.
- **R8-KR09** `no_saints` written verbatim.
- **R8-KR13** title-case "The Band" after the Rolling Stones: reads as the proper name.
- **R8-KR15** "uk of gb and i": a recognisable short form.
- **R8-KR16** "The monarch" after Victoria: round 6's judges read this form (R6-AN01) as faithful.
- **R8-TP15** trailing "...": the claim is still made.
- **R8-TA19** "Baby I love you performer the Ronettes.": a known song title.

## Notes for scoring

1. **Where a leak can come from.** Under the spec every token outside an entity span, the relation and a/an/the
   rejects, so most U cases are expected rejects that add adjudicated U cases. The cases built to be kept are:
   - the display-space hypernym anaphors "the band" / "the duck man" (PB01-PB04, PB08, PB13, PB17, PB18, RJ38, RJ39):
     every token is a gathered name, a relation word or the period, the span equals `the_band` / `the_duck_man` once
     the display space counts as a space, the antecedent shares no word with it, and it reads as a lower-case
     description pointing back. PB05, PB07, PB15 and PB16 are the same with a subset antecedent ("Genesis band",
     "Yes band", "boy band"), which the points-back rule as specified should reject;
   - the display-space first-person names (FP03, FP04, FP07, FP21);
   - the "perhaps " prefix (HF02, HF03, HF05, HF16): the spiking-hypothesis prefix, declared a non-claim on another path;
   - the taught first-person and pronoun titles (TA01-TA07, TA11-TA16, TA20) and the Amendment 5/6 residue.
2. **Reply cases.** Sentence i is the mouth's rendering of gathered triple i (several are deliberately wrong
   renderings). Every in-order selection is meant to be judged; the `why` names the selection that matters when it
   is not the full reply (RJ38, RJ39, PB07, TA21: a hedged or tense-shifted middle sentence, if dropped, leaves an
   anaphor whose antecedent becomes the first sentence's band or person). RF11 is the order twin of the points-back
   cases: "The Band" comes first, so it points back to nothing.
3. **F twins.** Each restates gathered triples with no extra content. Retention probes whose rejection is an answer
   loss, not a label error: the orientation-swapped spouse wording (F18; true by the symmetry of marriage), store
   names that open with "the" and a lower-case word (F20 "the electric guitar", F23 "the symphony"), Roman numeral
   "I" in the father wording (F26, F27), a dropped apostrophe mid-name (F28), a mouth "!" (F37), and the taught
   title "My Fair Lady" (TAF3). Store names that are odd but are the store's own slugs (F10 "Association football
   club is a sport of Julio Iglesias") are faithful to the triple.
4. **Relation wording was guessed** for relations outside the nine build-6 wordings, from earlier rounds' READMEs:
   "is a member of", "has part", "is part of", "follows", "is followed by", "shares border with", "is an instance of",
   "is a subclass of", "is located in", "is twinned with", "is headquartered in", and telegraphic relation labels
   ("genre", "record label", "country of origin", "award received", "occupation", "place of birth", "composer",
   "cast member", "performer", "taxon rank", "given name", "language of work or name", "manner of death"). A U case
   rejected only for its wording is a correct reject.
5. **Display spaces.** U+00A0, U+2007, U+2009 and U+202F render like an ordinary space; judges should read them as
   spaces. The anonymised copy may show them as escapes.
6. **Taught slugs** are invented, follow the store's `sanitize`, and stand for names a conversation could teach
   (Elton John's memoir "Me", the film "Us", the novel "She", "Our Mutual Friend", "Dreams from My Father", ...).
   They are judged against the gathered set as given, like every other case.
7. **Store values.** Several slugs are the store's own aliases or errors (`he_s_dead_jim`, `not_wanted`,
   `the_duck_man`, `celtics_beat_la_chant`, `steve_threw_up`, `i_need_a_beat`); store family errors such as
   `maddox_jolie_pitt` were left out. In every U case the unsupported meaning sits in words the sentence adds, a reversed role, a swapped
   relation, or a reading the displayed text gives a name (an anaphor, a pronoun, a list, a clause), not in the
   alias itself.

## Regeneration (Claude, 2026-10-21)

`build_cases.py` here is the skeptic's file with one added line, `import research.runners` (the provenance door), so the module run writes `cases.json.prov.json`. The regenerated `cases.json` is byte-identical to the skeptic's (`cmp`), and `run_hunt.validate_reply_case` accepts all 79 reply cases (k = 2 for 60, k = 3 for 19).
