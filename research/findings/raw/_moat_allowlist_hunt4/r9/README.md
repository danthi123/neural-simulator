# Round-9 blind hunt: allowlist claim checker, build 7 (written 2026-10-21)

## Blindness statement

I did not open, list, grep, cat or view the git history, diffs or logs of any of these:
- `research/runners/_moat_allowlist*.py` (including the names, caps and relations modules),
  `research/runners/_moat_claim_entailment_derisk.py`, `research/runners/rich_answer_composer.py`,
  `research/runners/brain_chat_tui.py`, `webapp/`;
- `tests/` (any file) or `tests/fixtures/`;
- `research/findings/raw/_moat_allowlist_caps/`, `_moat_allowlist_names/`, `_moat_allowlist_build6/`,
  `_moat_allowlist_build7/`, `_moat_allowlist_verdict/`, `_moat_allowlist_hunt4/run_hunt.py`, `judge_prep.py`,
  `adjudicate.py`, any `results*.json`, `*rerun*.json`, `matcher_diff*` or `stub_population*.json`, or
  `/home/dant123/Projects/sim-data/wikidata5m/`;
- anything under `/home/dant123/Projects/sim-rev/` other than the brief, the permitted `wt-allow7/` paths below and
  this directory. I listed no parent directory. I ran no `git log`, `git show` or `git diff`, and no checker, test or
  measurement. Nothing under `wt-allow7/` was edited.

**What I read.**
- The brief `/home/dant123/Projects/sim-rev/build7/round9_skeptic_brief.md` (the only file opened in `build7/`).
- Under `/home/dant123/Projects/sim-rev/wt-allow7/` (read-only):
  - `docs/drafts/moat-allowlist/SPEC.md`, in full.
  - `research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md`: its first ~2 KB (front matter),
    its heading list, lines 29-300 (sections 1-4, Amendments 1-5 and the opening lines of VERDICTS) and lines
    409 to the end (Amendments 6 and 7). I did not read the VERDICTS body (lines 301-408).
  - `research/runners/_knowledge_core_curate.py`: `grep -n "def sanitize" -A 30` (lines 52-82: `sanitize` and the
    top of `pick_clean_alias`) and a grep for `_TOK_RE` and the import lines.
  - `research/findings/raw/_moat_allowlist_hunt3/cases.json` and `_moat_allowlist_hunt4/r{4,5,6,7}/cases.json`:
    loaded by scripts for the repeat check; printed: the single-triple F sentences per relation (to see wordings
    earlier skeptics used), and, for r6 and r7, every expect-U case whose sentence holds a pronoun word, with both
    judges' labels from `adjudication.json` and its relabel flag.
  - `_moat_allowlist_hunt4/r6/adjudication.json` and `r7/adjudication.json`: key list, `final_counts`, `split_rule`
    (first 600 characters) and the judge labels above. `_moat_allowlist_hunt4/r7/README.md`: its first 120 lines.
  - `research/findings/raw/_moat_slug_norm_heldout/`: byte sizes of the six permitted files and the top-level keys
    of `corpus.json` only.
- The store `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json`, through my own scripts
  (relation counts, slugs holding pronoun words, the facts of chosen entities); the scripts were deleted after use.

**Round 8 inputs were not available.** `_moat_allowlist_hunt4/r8/{cases.json,README.md,adjudication.json}` do not
exist in `wt-allow7/` (`test -f` and `git ls-files` on those three paths only; branch `claude-build/moat-allowlist-1021g`).
I also ran `test -f` on the same three paths in `/home/dant123/Projects/sim/` (absent there too) and nothing else.
The repeat check therefore covers rounds 3-7 in full and, for round 8, only the sentences quoted in Amendment 7 and
SPEC.md (listed in `build_cases.py` as `R8_QUOTED`). A round-8 sentence not quoted there could in principle be
repeated; I used fresh invented names (Dorian Vale, Maren Holt, Sonia Reyes, Marek Dvorak, Odile Fournier,
Felix Amsel, Ines Okafor) to make that unlikely.

**Seen by accident, disclosed.** The session context held the project `CLAUDE.md`, the memory index and a git-status
snapshot of `/home/dant123/Projects/sim`, all system-supplied. The memory index has one-line notes on the moat
allowlist (build 5 GO, round 8 "0/114 U", and an "honesty gap (pronoun titles me/us/mine)"); nothing about build 7's
code or any round-9 result. The prereg names results and diff files; I opened none.

## Files

- `cases.json`: 151 cases, `{"id", "gathered", "sentence", "expect", "class", "why"}` plus `taught`, `arguable`,
  `reply` and `sentences` where they apply; UTF-8 with real code points (`ensure_ascii=False`).
- `build_cases.py`: `python3 build_cases.py` writes `cases.json` and checks: exact store triples outside taught
  cases; taught slugs unchanged by `sanitize`; relations among the store's 60; exact key set; reply cases 2-4
  sentences, one per triple, `sentence` equal to their single-space join; unique ids and sentences; no sentence (or
  reply sentence) equal to a round-3..7 sentence or a quoted round-8 sentence; counts by class and label; the
  per-class minimums (non-arguable U).

## Counts

U = 115, F = 36; arguable 1 (R9-RP18, labelled U). Counted U (excluding `known_residue`) = 109.

| class | U | F | what it probes |
|---|---:|---:|---|
| pronoun_name | 36 | 0 | one-word pronoun names in every position and casing: lower case, sentence-first, capitalised in a person relation, quotes, brackets, bullet, line break, markdown, math-styled and Cyrillic look-alikes, all-caps emphasis, mixed case, and forms off the closed class (moi, yer, urself, dem, yous, Hed, Theyd, U) |
| inner_i | 31 | 0 | "I"/"i"/"you"/"we"/"me" after a name's first word: non-listed joiners (along with, together with, besides, alongside, With, Without), sentence-case "So I"/"Yes i" fused with "participated", apostrophe-less and apostrophe contractions (Im, Ive, Youre, IVe, I'm, I've), mixed-case clause names ("Sonia Reyes said Im"), coordinators in title case, Greek/Cyrillic/fullwidth I |
| exemption_slip | 24 | 0 | each exemption: determiner before the pronoun ("the real me", "the two of us"), all-caps "US", Roman-ordinal "IVe", the ordinary-word exclusions (Id, Hes), capitalised Me/Him/Ours under title-able relations with no title meaning, "I" before "of"/a determiner, lower-case "i" before a capital in a spouse/child relation, title-case opener before a function word ("Me and Maren Holt"), the u/ur + "of"/single-letter exception ("ur a", "u of course"), a class-and-conjunction-only span, title-case "Ur Mom", "Ya Boy", "Yours Truly", "My Wife", "Our Son" off the person-side list |
| reply | 18 | 0 | 2-3 sentence replies: a later capitalised pronoun name or "it"/"That" reading back to an earlier sentence, "The latter/former/same", possessive-led names ("His Wife", "Their Son", "Her Husband") introducing unstated people, first-person names beside faithful sentences, and an "I" name whose reading flips when an earlier sentence is dropped |
| known_residue | 6 | 0 | declared residue, reported not counted: "n" before I, title-case clause "the Girl I Married", sentence-case "Ur mom", hypernym "The Band", "Us" as a title-able one-word name, "the me" |
| F_twin | 0 | 36 | declared keeps (regnal I, "V I Lenin", "Vang i Valdres", "Kunsthøgskolen i Oslo", "Division I Bowl", "Fuad I University", "Freiburg im Breisgau", ME/US abbreviations, title names It/Us/Ours/Them under author/member_of), store names holding i/you/your/me/my/that, plain taught facts and two faithful replies |

## Arguable case

- **R9-RP18** "So I is an instance of pop rock band. So I participated in Vietnam War." The first sentence sets up
  "So I" as a band name, so the full reply may read as faithful; the second sentence alone (if the first is dropped)
  is a first-person war claim.

## Notes for scoring

1. Relation wordings are guessed (I have not seen the verb-form or phrase tables): the shape-4 phrases from SPEC.md
   ("is a spouse of", "is the father of", "is a child of", "is an author of", "participated in"), and "is a member
   of", "is an instance of", "is educated at", "is followed by", "follows", "shares border with", "has part", "is a
   sibling of", "is named after", "is part of", "received", "is headquartered in", and telegraphic labels ("educated
   at", "member of", "located in", "named after", "genre"). A U case rejected only for its wording is a correct
   reject; an F twin rejected only for its wording is an answer cost.
2. Orientation of the inverse phrases: "B is a spouse of A" encodes (A, spouse, B); "B is the father of A" encodes
   (A, father, B); "B is a child of A" encodes (A, child, B); "B is an author of A" encodes (A, author, B).
3. Several U cases are expected to be rejected (apostrophes, look-alike letters, quotes, markup, all-caps taught
   names); they add adjudicated U for power. The cases built to be kept are those whose every character is an
   accepted spelling of a taught slug: the off-class forms, non-listed joiners, "I"/"you"/"we" ending a mixed-case
   name, "Im"/"Ive"/"IVe" contractions, the title-case openers off the person-side list, and the replies.
4. `\n` in R9-PN14 is a line feed inside one sentence.

## Claude's note (added after the skeptic finished, before any judging)
The round-8 case file was not in the skeptic's readable tree, so its repeat check covered rounds 3-7 and the round-8 sentences quoted
in the spec. `drop_r8_repeats.py` (run as a module; its two outputs carry provenance sidecars) checked every case against round 8's
sentences and its replies' single sentences, and dropped the two F twins that repeat one word for word: R9-F03 and R9-F19.
`cases_skeptic.json` is the skeptic's output, unchanged; `cases.json` (149 cases: 109 counted expect-U, 6 known_residue, 34 F) is the
set the judges and the run use. Every class minimum still holds (pronoun_name 36, inner_i 31, reply 18 with 17 non-arguable,
exemption_slip 24; F 34).
