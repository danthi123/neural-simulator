# -*- coding: utf-8 -*-
"""Manual equivalence arguments for survivors that no witness search could separate from the original.  id -> (equivalent, why).
An entry is used only when no witness (hand, table corpus, fuzz, token soup, ambiguous fuzz) separates the mutant; when a witness exists it wins.
"verdict" below means the keep verdict of allowlist_check and the accepted verdict of allowlist_verify."""

AMB = ("Only a sentence with two or more complete parses can tell this mutant from the original, and every such sentence is rejected by an "
       "untouched check. Case analysis of `_parses`: (i) one subject option gives at most one shape-3 parse (the relation noun must be a verb_ok `_of` "
       "slug, so 'a'/'an' cannot be it) and never a shape-1/2 parse beside it ('is' has no verb reading); (ii) two object-list parses differ only where a "
       "determiner that is also a noun slug starts an NP ([D] vs [D, noun]); the noun after it must then be 'and', and from there the two parses expect "
       "an NP and a separator at alternating positions, each position needing the token 'and', so one parse ends at an odd offset and the other at an "
       "even one and they cannot both end at n; (iii) two subject options ([D] vs [D, toks[1]]) need toks[1] to be a noun and either 'is' or a verb "
       "form, and either way one of the two parses holds the entity span 'is' (as subject of the second, or as the first object of the first), always "
       "the second parse in a complete pair. A parse holding the entity 'is' is rejected by `_structure_word_violation` (a clause word, even verbatim) "
       "or by the exact check. Enumeration agrees: every sentence up to length 9 over {the, a, uses, and, x, ','} (12.1M, all triples gathered) has 0 "
       "ambiguous parses; up to length 7 over {the, uses, and, x, is, member of, ','} 10 do (all hold 'is'); up to length 6 over {the, a, is, x, y, "
       "member of, uses, and, ','} 30 do (all rejected by the original; sw48 keeps 24 of them); a 34,297-item edit fuzz of those 30 gave 0 differences for this mutant.")

NOTES = {
    # ------------------------------------------------------------------------------------------------ _bad_gap / allowlist_verify
    "bg04": ("yes", "Finding gaps with finditer on `surface` instead of `surface.strip()`: `(?<=[.!?])\\s+` cannot match at index 0, so leading "
                    "whitespace is never found by either; a trailing run after a final [.!?] is found by the unstripped version but is exactly "
                    "`surface[len(surface.rstrip()):]`, which is already gaps[1]; interior runs are identical. Same set of gap characters, same verdict. Exhaustive check of `_bad_gap` on all 5,380,840 strings of length <= 7 over {. x space LF CR TAB NBSP ! ?}: 0 differences."),
    "bg21": ("yes", "`gap` is None or a single character; a one-character string is truthy even for '\\x00', so `if gap:` is `if gap is not None:`."),
    "bg24": ("yes", "Only the character echoed in reject_reason changes ('x' instead of the offending character); the accepted flag is the same."),
    # ------------------------------------------------------------------------------------------------ _bad_space
    "bs03": ("yes", "Predicate `_bad_space(chr(cp)) is not None` compared for all 1,112,064 scalar values: identical, because every str.isspace() "
                    "character is in category Cc/Zs/Zl/Zp (checked exhaustively); _bad_space is a per-character predicate."),
    "bs05": ("yes", "Every Zs character is str.isspace() (exhaustive over Unicode), so dropping 'Zs' from the category tuple while isspace() stays changes no character's status; predicate identical on all 1,112,064 scalar values."),
    "bs06": ("yes", "U+2028 (the only Zl) is str.isspace(); predicate identical on all 1,112,064 scalar values."),
    "bs07": ("yes", "U+2029 (the only Zp) is str.isspace(); predicate identical on all 1,112,064 scalar values."),
    "bs04": ("yes", "Dropping 'Cc' leaves 55 control characters (NUL..US except the whitespace controls, DEL, C1 except U+0085) unflagged by _bad_space. "
                    "None can sit in an accepted sentence: they are not \\s, so `_TOKEN` makes each a token of its own, which no shape consumes; the slug "
                    "regex separators are [\\s\\-_]; no store name holds one. 68,145 insertion probes (every position of 37 base sentences x 55 code "
                    "points): 0 verdict differences."),
    "bs08": ("yes", "Adding 'Cf' rejects 163 format characters that the original lets through _bad_space. They are not \\s and not L/M/N, so `_TOKEN` "
                    "makes each a token no shape consumes, and no store name holds one (the names table drops any name with a Cf or Cc character): the "
                    "original rejects every sentence holding one anyway. 201,957 insertion probes: 0 verdict differences."),
    "bs21": ("yes", "`_WS = \\s+` differs from the 5-character class only on whitespace outside SPAN_SPACES; `_bad_space` has already rejected the sentence "
                    "holding such a character, so surface_form never sees it in a reachable call. (Direct surface_form('a\\tb') differs: 'a b' vs 'a\\tb' -- "
                    "the helper contract 'any other whitespace stays' is not pinned by a test.)"),
    "co05": ("yes", "`_bad_space(_prepare(sentence))`: per-character predicate identical for NFC+lower of every scalar value, and NFC/lower cannot create or "
                    "remove a whitespace character (whitespace never composes; U+2000/U+2001 map to the whitespace U+2002/U+2003). Only the index in the reason changes."),
    "co06": ("yes", "`_bad_space(NFKC(sentence))`: 11 whitespace code points (U+2000-2006, 2008, 200A, 205F, 3000) become U+0020 and pass the check, none the other way. "
                    "They are non-ASCII, not in the four display spaces: outside an entity span the non-ASCII scan of `_exact_violation` rejects them; inside "
                    "one, `surface_form` does not collapse them, so the span is not an exact store surface (no names-table row holds a bad whitespace). "
                    "13,629 insertion probes: 0 verdict differences."),
    # ------------------------------------------------------------------------------------------------ allowlist_check wiring
    "co10": ("yes", "`_exact_violation` indexes `toks[i]` only for relation spans (never the final '.'), and the preceding check has already required "
                    "`tr[1][:len(toks)] == toks`; so tr[1] and toks agree at every index used."),
    "co13": ("yes", "The guard `tr[1][:len(toks)] != toks` is defensive: `normalise_tracked` applies the same fold and the same regex substitutions in the same "
                    "order as `normalise` (a literal concatenation instead of `re.sub` replacement templates; the two differ only for a slug holding a backslash, "
                    "which can never be a `[a-z0-9_]` token and so never part of a parse). 0 mismatches over the corpus, fuzz, token soup and table inputs."),
    "co14": ("yes", "Same as co13: the token lists agree whenever the sentence parses, so comparing lengths only gives the same verdict."),
    "co15": ("yes", "Running the late checks before the exact check swaps which pure rejection fires first; each returns the same reject for the same input; only the reason text differs."),
    "co16": ("yes", "Running the exact check before the caps check swaps which pure rejection fires first; only the reason text differs."),
    "co17": ("yes", "Apostrophe check before the whitespace check: both are pure rejections of the raw sentence; only the reason text differs."),
    "co09": ("yes", AMB), "co19": ("yes", AMB), "co18": ("yes", AMB + " (co18 changes only the props field of a kept verdict; no ambiguous sentence is kept.)"),
    # ------------------------------------------------------------------------------------------------ exact
    "ex06": ("yes", "`table.surfaces(toks[i])` vs `table.surfaces(slug)`: for an 'ent' span, slug is toks[i] by construction in `_parses` (the noun token itself)."),
    "ex16": ("yes", "`ord(ch) > 128`: only U+0080 (a Cc control) differs, and `_bad_space` rejects every Cc character before `_exact_violation` runs."),
    "ex17": ("yes", "`ord(ch) >= 127`: only U+007F (DEL, a Cc control) differs, and `_bad_space` rejects it before `_exact_violation` runs."),
    "ex14": ("yes", AMB), "ex31": ("yes", AMB), "ex32": ("yes", AMB), "ex33": ("yes", AMB),
    # ------------------------------------------------------------------------------------------------ names table
    "nt09": ("yes", "Table keys are lower-case; an upper-case gathered slug can never be an entity span (tokens are lower-cased, slug patterns are case-sensitive), so `.lower()` of the looked-up slug is a no-op for every reachable slug."),
    "nt10": ("yes", "`_SIGNED_NUMBER` is applied only to rows of the committed names_table.json; scanning all 10,006 rows, no slug has a row whose sign follows a digit, so "
                    "with and without the lookbehind the same 7 slugs (utc_00_00, utc_06, utc_06_30, utc_07, utc_3_5, utc_4_30, gmt_5_45) fire."),
    "nt13": ("yes", "Over the committed table, every slug with a +/- sign row written with U+00B1 also has a row with '+' or U+2212 (utc_00_00), so dropping U+00B1 from the class changes no slug's result (scan of all rows)."),
    "nt21": ("yes", "All 10,006 rows of the committed table are already lower-case, so lower-casing at load is the identity."),
    "nt23": ("yes", "`self.entities` is assigned only after json.load and the dict comprehension both succeeded; in the except branch it still holds the initial {}, so not re-assigning it changes nothing."),
    "nt27": ("yes", "lru_cache(maxsize=0) only re-reads the file every call; the table object is rebuilt from the same file, so every verdict is the same (slower)."),
    # ------------------------------------------------------------------------------------------------ structure words
    "sw08": ("yes", "Without `if w`, `_WORD_BREAK.split` can leave empty strings only at the ends (the `+` already merges runs). An empty string is in no "
                    "word set, cannot be shown[-1] when the final-s rule is live (that rule needs no underscore in orig, and without an underscore the only "
                    "separators are display spaces, which are stripped at the ends), and no residue pair involves ''. 60,000 synthetic span texts through `_structure_word_violation`: 0 differences."),
    "sw11": ("yes", "A single-character separator split gives the same non-empty words as the run split; the `if w` filter drops the empty strings that a run of separators now yields. 60,000 synthetic span texts through `_structure_word_violation`: 0 differences."),
    "sw50": ("yes", "A span text starts or ends with a display space only for a gathered slug that itself starts/ends with a space (the slug regex starts at '_' or a word character); `surface_form` strips such a space, so the span "
                    "equals no surface of that slug and `_exact_violation` rejects it before the structure scan. For store-shaped slugs the strip is a no-op. 60,000 synthetic span texts on `_structure_word_violation` differ only for texts with edge display spaces; 120,000 edit-fuzz items: 0 verdict differences."),
    "sw51": ("yes", "`origs[i]` is a slice of the NFC source text; a slice of an NFC string is NFC, so normalising again is the identity. 60,000 synthetic span texts through `_structure_word_violation`: 0 differences."),
    "sw49": ("yes", AMB), "sw48": ("no", "witness"),
    # ------------------------------------------------------------------------------------------------ repeated
    "rp03": ("yes", AMB), "rp04": ("yes", AMB), "rp05": ("yes", AMB),
    "rp09": ("yes", "Propositions are built from tokens, which are lower-case by construction (`_TOKEN` runs over lower-cased text), so `.lower()` on the object is the identity."),
    "rp13": ("yes", "`A or B` vs `B or A` over two pure checks: the same sentences are rejected; only which reason is reported differs."),
    "rv04": ("yes", "Within one kept sentence the props are pairwise distinct (`_repeated_proposition` ran over every parse, parse 0 included), so `seen[p] != k` is always true when `p in seen`."),
    "rv07": ("yes", "`seen.setdefault(p, ...)` records p as seen exactly as `seen[p] = k` does; only the sentence number printed in the reason changes."),
    "rv08": ("yes", "`seen[p]` is the index of an earlier (or the same) sentence, so `seen[p] <= k` is always true."),
    "rv11": ("yes", "Without the inner `break` the loop keeps scanning and may overwrite `reason` with a later duplicate; `reason` stays non-None, so accepted is unchanged."),
    "rv12": ("yes", "A rejected sentence has props == [] (allowlist_check sets props only on keep), so adding rejected sentences' props to `grounded` adds nothing."),
    # ------------------------------------------------------------------------------------------------ relations
    "lx03": ("yes", "`blocked_rel` is consulted only to word the 'no shape' reason; removing the dropped-reading branch changes the reason text, not whether a parse exists."),
    "lx04": ("yes", "Reason text only (which class label is printed for a dropped reading)."),
    "lx05": ("yes", "Reason text only (which class label is printed for a dropped reading)."),
    "st02": ("yes", "`stub_third_person` is called only for a relation that has a row in RELATIONS and contains '_'; no such row ends in 'sh' (the multi-word rows end in s, ch, y, ...), so dropping 'sh' changes no reachable token. (Direct call stub_third_person('x_wash') differs.)"),
    "st04": ("yes", "No multi-word row of RELATIONS ends in 'x' (the only callers pass table rows), so dropping 'x' changes no reachable token. (Direct call stub_third_person('x_box') differs.)"),
    "st05": ("yes", "No multi-word row of RELATIONS ends in 'z' (the only callers pass table rows), so dropping 'z' changes no reachable token. (Direct call stub_third_person('x_quiz') differs.)"),
    "bs24": ("yes", "`_bad_space` returns the last instead of the first bad character; it is None under exactly the same inputs, and only the index/character quoted in the reason differs."),
    "sf07": ("yes", "`strip()` vs `strip(' ')` after the display spaces were mapped to U+0020: they differ only for another whitespace character at the edge of a span, and `_bad_space` has already rejected the sentence. (Direct surface_form('\\tx') differs: unpinned helper contract.)"),
}


def no_why(mid):
    """why-strings for survivors that a witness separates (family-generated or hand-written)."""
    import re
    m = re.fullmatch(r"pr_s_add_(\w+)", mid)
    if m: return "PRONOUNS_S gains %r: a name 'My s ...' / 'The s ...' (not a pronoun) is rejected by the mutant and kept by the original; no test pins that the set is exact." % m.group(1)
    m = re.fullmatch(r"pr_s_(\w+)", mid)
    if m: return "PRONOUNS_S loses %r: '%s s ...' (= %s's) is no longer read as a dropped apostrophe and is kept; the test parametrisation does not use this pronoun." % (m.group(1), m.group(1), m.group(1))
    m = re.fullmatch(r"pr_aux_add_(\w+)", mid)
    if m: return "PRONOUNS_AUX gains %r: 'Who ll ...' is rejected by the mutant, kept by the original." % m.group(1)
    m = re.fullmatch(r"pr_aux_(\w+)", mid)
    if m: return "PRONOUNS_AUX loses %r: '%s ll ...' (= %s'll) is kept; no test uses this pronoun with ll/re/ve/d/m." % (m.group(1), m.group(1), m.group(1))
    m = re.fullmatch(r"ar_add_(\w+)", mid)
    if m: return "AUX_RESIDUE gains %r: 'I t ...' is rejected by the mutant and kept by the original." % m.group(1)
    m = re.fullmatch(r"ar_(\w+)", mid)
    if m: return "AUX_RESIDUE loses %r: 'I %s ...' (= I'%s) is kept." % (m.group(1), m.group(1), m.group(1))
    m = re.fullmatch(r"ns_add_(\w+)", mid)
    if m: return "NEGATIVE_STEMS gains %r: '%s t ...' is rejected by the mutant, kept by the original." % (m.group(1), m.group(1).capitalize())
    m = re.fullmatch(r"ns_(\w+)", mid)
    if m: return "NEGATIVE_STEMS loses %r: '%s t ...' (= %sn't style residue) is kept; no test uses this stem." % (m.group(1), m.group(1).capitalize(), m.group(1))
    m = re.fullmatch(r"dr_(consume|consumed|consumes|consuming|devour|devoured|devouring|devours)", mid)
    if m: return "The table synonym %r of 'eat' is no longer a DROPPED_READINGS entry: 'Cat %s Mouse.' is kept; the per-reading test pins only some of the eight synonym tokens." % (m.group(1), m.group(1))
    return None


NO_NOTES = {
    "ds01": "deslug keeps empty pieces: a gathered slug with a run of underscores ('foo__bar', not a store slug) spells 'Foo bar' in the original (the docstring says a run still gives one space) and 'foo  bar' in the mutant, which rejects it. No test pins deslug on a run.",
    "ds02": "deslug uses replace('_', ' '): same witness as ds01 (slug 'foo__bar', not a store-shaped slug).",
    "ex21": "Letters are exempted from the non-ASCII scan: a determiner that is also a gathered noun slug ('a', 'the', 'an') absorbs a folded accent letter into its match ('ĳa'), but in the parse where it is the determiner (no span) the letter is outside every span; the original rejects it, the mutant keeps it. Needs the slug 'a'/'the'/'an' among the gathered nouns.",
    "ex22": "The non-ASCII scan skips the last character: a direct allowlist_check on 'Spain borders France.' + NBSP is rejected by the original and kept by the mutant. Through allowlist_verify the edge is stripped or rejected by _bad_gap, so the production path is unchanged; the public pure function is not.",
    "ex23": "The non-ASCII scan skips the first character: a direct allowlist_check on NBSP + 'Spain borders France.' is rejected by the original and kept by the mutant (the verify path rejects the edge in _bad_gap, so only the pure sentence-level function differs).",
    "ex29": "A relation span may swap a space for an underscore: 'Spain shares border_with France.' is rejected by the original (relation not written exactly) and kept by the mutant. Tests cover hyphen and the verbatim forms but not a mixed space/underscore spelling.",
    "sw12": "Hyphen counted as a word break in the structure scan: the store rows 'canton of is-sur-tille', 'ge-ge-ge no kitaro', 'not-for-profit organization' (exact store surfaces) are kept by the original and rejected by the mutant. No test names a hyphenated store name holding a clause word.",
    "sw21": "Final-s residue rule applies to a one-word span: the one-letter slug 's' written 's' is kept by the original (rule needs more than one word) and rejected by the mutant.",
    "sw27": "Final lone 't' also treated as residue: 'Saint Alban t' / store names such as 'Communaut de communes de la c te d alb t' (accent folds ending in t) are kept by the original, rejected by the mutant. Tests pin only the final 's'.",
    "sw29": "Residue loop starts at index 0 (prev wraps to the last word): 'T Don' is kept by the original and rejected by the mutant ('don' as prev of 't').",
    "sw30": "Residue loop skips the last word: 'Don T' (a negative stem plus a final lone T, the whole name) is rejected by the original and kept by the mutant; the tests pin the residue only before a further word.",
    "sw43": "The aux-residue clause tests only 'll' (AUX_RESIDUE replaced by 'll'): 'I m Dream' (= I'm) is rejected by the original, kept by the mutant; re/ve/d/m are unpinned (see ar_*).",
    "sw52": "Strip also removes edge underscores before the verbatim-slug test: a gathered slug with a leading underscore ('_is', not a store slug) written verbatim is exempt from the split in the original (kept) and split into 'is' by the mutant (rejected).",
    "rv02": "Reply-level duplicate detection compares adjacent sentences only: 'A. B. A.' (a fact stated in sentences 1 and 3) is rejected by the original and accepted by the mutant. Tests cover the adjacent repeat only.",
}
