"""STRICT (ALLOWLIST) CLAIM CHECKER for the scaffold mouth -- flag `BRAIN_MOAT_ALLOWLIST`, default OFF.

Spec: docs/drafts/moat-allowlist/SPEC.md. Bars: research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md
(filed before this file existed). This is a HOST safety filter on what the Qwen mouth may say (a scaffold check,
non-negotiable 1); it claims no brain capability, and its English shapes are a filter, not a language ability.

WHY A DIFFERENT METHOD. The slug-aware checker ("B2h", `BRAIN_MOAT_SLUG_NORM`, `_moat_claim_entailment_derisk`) accepts a
sentence unless a rule rejects it: it drops function words and hedges and attaches subjectless or verbless clauses to the
previous clause, so every round of denylist fixes was followed by a new construction that leaked (RULING moat-slug-hold-1018).
This module inverts the default: a sentence is KEPT only when ONE exact proposition shape consumes EVERY token and every
proposition it encodes is a gathered triple of this turn. Everything else is rejected.

THE SHAPES (after normalisation; N = one gathered slug, Det = a | an | the, directly before N; in shape 3 the relation noun takes only
a | an or nothing, because "X is THE member of Y" asserts that X is Y's only member, which the triple does not say):
  1. [Det] SUBJ VERB(rel) [Det] OBJ [.]                                          -> (SUBJ, rel, OBJ)
  2. [Det] SUBJ VERB(rel) [Det] OBJ1 (, [Det] OBJi)* [,] and [Det] OBJn [.]       -> (SUBJ, rel, OBJi) for EVERY i
  3. [Det] SUBJ is [a|an] REL_of [Det] OBJ [.]                                    -> (SUBJ, REL_of, OBJ)
     only for a gathered relation whose slug ENDS in `_of` (instance_of, subclass_of, member_of, part_of, participant_of,
     is_a_list_of in the store): its slug reads subject-first ("X is an instance of Y" = (X, instance_of, Y)), so the shape
     cannot reverse roles. A relation without `_of` ("capital": "Ankara is the capital of Turkey" means (turkey, capital,
     ankara)) never takes shape 3, so the reversal that sentence would need is impossible. Pinned in
     tests/test_moat_allowlist.py (test_shape3_*).
  VERB(rel) = a token whose `_classify` reading is ('verb', rel) for a relation gathered THIS turn: the same verb-form table
  `_noun_before_verb` uses (stock inflections, the tight synonyms, the guarded -es / -ies readings of commit 1b67eef9b), and ONLY for
  a relation classified VERB-SAFE or NOT-A-VERB in the reviewed table research/runners/_moat_allowlist_relations.py (prereg
  AMENDMENT 1 item 2; build 2's MEANING test: "SUBJ <reading> OBJ" must paraphrase back to exactly the triple). A ROLE-REVERSING
  relation ("The book authors Moses." for (book, author, moses)), a MEANING-DRIFT relation ("X sports Y" = X wears Y) and a relation
  with no row take no verb reading and no shape 3, so every sentence on them is rejected (reason names the class).
  SUBJ / OBJ = a span that spells a gathered subject/object slug by THE STORE'S OWN SLUG RULE (`_store_fold` below; the rule is
  `sanitize` in research/runners/_knowledge_core_curate.py, which built wikidata_100k: lowercase, every run of characters outside
  [a-z0-9] -> "_", leading/trailing "_" stripped). After NFC and lowercasing, an accented Latin letter (or a combining mark) is a
  separator, and one at the START or END of a name vanishes from the slug as the store's strip("_") made it vanish: "Île de
  France" spells le_de_france, "Bodø" spells bod, "İstanbul" spells i_stanbul, and "Universität des Saarlandes" spells
  universit_t_des_saarlandes (the store's universitat_des_saarlandes was made from the ASCII alias "Universitat des Saarlandes" of
  Q700758, so the accented spelling is NOT that slug by the store's rule). Spaces, hyphens and underscores separate words as before. Other characters the store also treated as separators
  (ASCII punctuation, apostrophes, en dashes, the 40-character cut) are NOT folded: they keep their structural role here or
  reject the sentence. A token made only of underscores is never a noun.

NO INVENTED ACRONYMS (build 3, controller ruling 2026-10-20: case fidelity of entity spans). The store's slug rule lowercases, so a span
can spell a gathered slug in a casing that names a different thing: "X is a member of the WHO." spells the_who (the band) while all-caps
"WHO" reads as the World Health Organization (round-4 case R4-DT02, kept by build 2). So, after a sentence parses and every proposition
is gathered: inside every SUBJ/OBJ span a token written entirely in capitals with 2 or more letters (`caps_tokens`: "WHO", "FC", "MP3";
not "Who", a one-letter "R") is accepted only if that same token appears in all capitals in one of that entity's own store names (its
canonical label or aliases in wikidata5m, the source wikidata_100k was built from). Otherwise the sentence is rejected with a reason
naming the token and the entity. The same rule covers the relation span (the verb of shapes 1/2, the REL_of of shape 3) against the
relation's own label; only two store labels write capitals ("IUCN conservation status", "original language of film or TV show"), so
in practice an all-caps relation word rejects except IUCN / TV in those two. "The Who" and
"the who" pass; "the WHO" and "THE WHO" fail; "FC Aberdeen" passes because the store's names for fc_aberdeen write "FC". The names are a
compact committed table (research/findings/raw/_moat_allowlist_caps/caps_table.json, generated by
research/runners/_moat_allowlist_caps_build.py, which first reproduces the store's facts from the source to pin the entity ids), read
once per process. A slug with no row (an entity outside the store, a conversation-taught name) allows no all-caps token; a missing or
unreadable table allows none either (fail closed, said in the reason). Tokens outside the spans (a determiner before the span, the
copula "is", "and") are not read.

FOLD TO FIND, EXACT TO ACCEPT (build 4, after round 5: build 3 kept 24 of 89 U cases, every one ACCEPTED by fold-equality, e.g. "George V"
+ U+013A reads as George VI but folds to george_v; "1-500 m" folds to 1_500_m). The store fold above now only FINDS which gathered slug a
span could be. After a sentence parses, every proposition is gathered and the case rule passes, `_exact_violation` requires, for every
parse: each SUBJ/OBJ span's `surface_form` (NFC, lowercase, runs of U+0020 / U+00A0 / U+2007 / U+2009 / U+202F -> one space, stripped;
nothing else) is exactly one of the slug's canonical surfaces (the slug verbatim, as the stub mouth renders it; its de-slugged form; or a
row of the committed exceptions table research/findings/raw/_moat_allowlist_names/names_table.json made by
research/runners/_moat_allowlist_names_build.py); each relation span is written exactly as the token the relation table read (verbatim or
with underscores as spaces: no hyphen, no fold); and no non-ASCII character appears outside the accepted SUBJ/OBJ spans. Any difference
rejects, naming the span and the slug. A missing or unreadable table accepts only the slug and its de-slugged form (fail closed, said in
the reason). The table (review rulings F1, F2, F4): a slug has rows only when the store facts under it come from EXACTLY ONE Wikidata id
(by the generator's reproduction of the store's 78,857 facts), and then only that id's names whose sanitize() is the slug, minus any name
holding a Unicode format (Cf) or control (Cc) character; a name with ", " also gives the same name with every ", " written " " (a table
row, compared exactly; nothing is normalised at run time). A slug merged from two or more ids (j_n: Jan and Jon; saint_cloud: the French
commune and the Minnesota city) has no row, so only its de-slugged form and the slug itself are accepted. Store names holding an
apostrophe or other ASCII punctuation are in the table but unreachable: the finder keeps that punctuation structural.
WHITESPACE (F3): between tokens only U+0020; inside an entity span U+0020 and the four display-identical spaces above; every other
whitespace or control character (TAB, VT, FF, U+001C-U+001F, U+0085, U+1680, U+2028, U+2029, U+3000, ...) anywhere rejects the sentence
before matching (`_bad_space`). Sentence splitting (`split_sentences`) is unchanged.
NO CLAUSE WORD INSIDE A SPACED NAME; A LIST NAMES DISTINCT THINGS (build 5, after round 6: build 4 kept 5 of 17 U cases). The relation
table accepts telegraphic readings ("X located in time zone Y"), so a parse can hold no finite verb, and a reader then takes one from
inside a name: "Solomon is located in time zone Kosrae time" spells solomon_is (Solomon Is., the islands) but reads "Solomon is located
in". After the exact check, `_structure_word_violation` rejects a SUBJ/OBJ span whose displayed text, split on spaces, holds a finite
auxiliary (is, are, was, were, has, have, had, does, did) or a NEG_MARKERS word; the slug written verbatim ("solomon_is") is one word and
passes. Store cost: entities with such a word are rare (9 slugs with "is", 5 with "not", 8 with "no"), and each stays sayable as its
slug. `_repeated_proposition` rejects a parse that lists one proposition twice: "X has part A and a A" asserts a second A.
Review fixes (2026-10-20, before any build-5 measurement): (1) a multi-word relation keeps only its slug and the stub's own +s/+es token
("has_parted" reads "has parted"; `reading_dropped` in the relation table); (2) `allowlist_verify` rejects a reply in which two sentences
state the same proposition (the list rule across sentences); (3) a CR between sentences only before an LF (a bare CR overprints the line
on a terminal); (4) the slug verbatim in lower case is one identifier, but any other casing of an underscore spelling ("Marshall_Is",
"No_Saints") is prose and its underscores split words; (5) "be" joins the clause words, and a lone lower-case "s" ending a name or "t"
after a negative stem ("Saint Alban s", "don t") is the residue of a dropped apostrophe and rejects; (6) when every table row of a slug
writes a sign before a digit ("utc-06"), its de-slugged form ("utc 06", read as UTC+06) is not accepted. A one-word slug that is itself
a clause word is rejected even verbatim (the 100k store has none).

NOTHING IS DROPPED. A token outside the shape (pronoun, adverb, "just", "also", hedge, negation, subordinator, a comma that
opens a clause, a digit not inside a gathered slug, "!", "?", ":", "-", a second ".") leaves the shape unmatched and the
sentence is rejected. Any apostrophe (possessive 's, contraction) rejects the sentence BEFORE normalisation, because the B2h
slug regex would otherwise swallow a possessive 's; so does non-ASCII text other than accented Latin letters (or combining marks)
attached to a Latin word (`_foreign_text`: a free-standing non-Latin word between two slug words would fold into a separator). A negation word
rejects even when it is also a gathered slug.
ONE SENTENCE, ONE SHAPE. The surface is split into sentences first; each sentence must match a shape on its own. If two parses
of one sentence exist, every parse must encode only gathered triples (an ambiguous sentence with any ungathered reading is
rejected). The production verdict on a rendered candidate (`allowlist_verify`) keeps it only if EVERY sentence is kept: the
composer emits candidates whole, so a candidate holding a rejected sentence is not emitted (the turn keeps its OTHER
candidates, each verified on its own). Partial emission would change the composer's output path and is out of scope.

NO EDIT TO THE B2h PARSER PATH. The B2h helpers (`_base_verb`, `_classify`) are REUSED unchanged by binding them onto a small
lexicon object (build 2 replaces only the use of B2h's `_normalise_slugs` by `_store_fold`, which reproduces the store's slug rule at
the edges of a name where B2h's did not; B2h itself is untouched); no `ClaimEntailmentVerifier` is constructed, so its role-permutation guard cannot make a
mutual pair unbuildable (board #265), and neither the stock clause parser nor `_noun_is_before` (the 3-letter predicate match,
board #266) is consulted. The substrate role parser is not used either: roles come from the shape, and the proposition is
adjudicated by set membership.

Wiring (flag ON): `ChatBrain._build_claim_verifier` returns an `AllowlistClaimVerifier`, `ChatBrain._verify_claim_set` uses it
even when `BRAIN_CLAIM_MOAT=0`, and `RichAnswerComposer._verify_rendered` routes EVERY turn (one fact or many) to it with no
fallback to the single-triple `chat._verify`. Flag OFF: both call sites are byte-identical to before.
"""
from __future__ import annotations

import functools
import json
import os
import re
import unicodedata

from research.runners._moat_allowlist_relations import MEANING_DRIFT, has_verb_reading, reading_dropped, relation_class
from research.runners._moat_claim_entailment_derisk import NEG_MARKERS, VERB_SYNONYMS, ClaimEntailmentVerifier, _build_inflection_map

ALLOWLIST_ENV = "BRAIN_MOAT_ALLOWLIST"
# build 3: the compact committed case-fidelity table (store slug -> the all-caps tokens of that entity's own store names), generated by
# `python -m research.runners._moat_allowlist_caps_build` from the source wikidata_100k was built from (research/runners/_knowledge_core_curate.py)
CAPS_TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "research", "findings", "raw", "_moat_allowlist_caps", "caps_table.json")
# build 4: the compact committed exact-surface table (store slug -> the surface forms of that entity's own store names that spell the slug by
# the store's rule and differ from its de-slugged form), generated by `python -m research.runners._moat_allowlist_names_build`
NAMES_TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "research", "findings", "raw", "_moat_allowlist_names", "names_table.json")
DETERMINERS = frozenset(("a", "an", "the"))
REL_NOUN_DETERMINERS = frozenset(("a", "an"))           # shape 3, before REL_of: "the" would assert uniqueness
COPULA = "is"
# straight, curly, modifier-letter and prime apostrophes, and the backtick: any of them rejects the sentence (possessive /
# contraction). Checked on the RAW sentence, before folding could read "X's" as the slug X.
_APOSTROPHES = re.compile(r"['`‘’‛ʼ′]")
# after normalisation: a word is a run of [a-z0-9_]; every other non-space character is a token of its own (so it can never be
# consumed by a shape except "," and the one sentence-final ".")
_TOKEN = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]")
_HAS_ALNUM = re.compile(r"[a-z0-9]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ASCII_ALNUM = re.compile(r"[a-z0-9]")
# build 4 F3: the only whitespace a sentence may hold is U+0020 between tokens; inside an entity span U+0020 and exactly these four
# display-identical spaces (no-break, figure, thin, narrow no-break) count as one space. Every other whitespace or control character
# (TAB, VT, FF, U+001C-U+001F, U+0085, U+1680, U+2028, U+2029, U+3000, ...) anywhere rejects the sentence (`_bad_space`).
SPAN_SPACES = " \u00a0\u2007\u2009\u202f"
# build 5: a sign before a digit (hyphen-minus, U+2212 minus, plus, plus-minus) that is not a range or joiner between two digits
_WORD_BREAK = re.compile("[_%s]+" % SPAN_SPACES)
_SIGNED_NUMBER = re.compile("(?<![0-9])[-\u2212+\u00b1][0-9]")
_WS = re.compile("[%s]+" % SPAN_SPACES)


def _bad_space(text):
    """-> (index, char) of the first whitespace or control character of `text` that is not U+0020 or one of the four span spaces
    (str.isspace, or Unicode category Cc / Zs / Zl / Zp), else None."""
    for k, ch in enumerate(text):
        if ch not in SPAN_SPACES and (ch.isspace() or unicodedata.category(ch) in ("Cc", "Zs", "Zl", "Zp")):
            return k, ch
    return None


def _nonascii_word_runs(t):
    """Maximal runs of non-ASCII letters, marks and digits (Unicode categories L*, M*, N*) in `t` -> [(start, end)]."""
    runs, start = [], None
    for i, ch in enumerate(t):
        word = ord(ch) > 127 and unicodedata.category(ch)[0] in "LMN"
        if word and start is None:
            start = i
        elif not word and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(t)))
    return runs


def _prepare(text):
    """NFC, then lowercase: the order the store's alias -> slug step sees an alias (wikidata5m aliases are NFC; `sanitize`
    lowercases)."""
    return unicodedata.normalize("NFC", text).lower()


def _foreign_text(text):
    r"""The store's slug rule makes every non-ASCII letter a separator ("Montréal" -> montr_al), and slug words are joined by
    [\s\-_]+, so a free-standing non-ASCII word BETWEEN two slug words ("shares 不 border with", 不 = not) would fold into a
    separator and vanish. Allowed here: a run of LATIN letters and combining marks attached to an ASCII letter/digit (an accented
    letter inside or at the edge of a Latin word: Montréal, República, Île, Bodø, the dot of a lowercased İ). Anything else
    (non-Latin script, a free-standing accented word such as "à", a non-ASCII digit) -> that run, so the sentence is rejected
    before folding."""
    t = _prepare(text)
    for a, b in _nonascii_word_runs(t):
        run = t[a:b]
        latin = all(unicodedata.name(ch, "").startswith("LATIN ") or unicodedata.category(ch) == "Mn" for ch in run)
        before = t[a - 1] if a > 0 else ""
        after = t[b] if b < len(t) else ""
        if not latin or not (_ASCII_ALNUM.match(before) or _ASCII_ALNUM.match(after)):
            return run
    return None


def _store_fold(text):
    """The store's slug rule on the characters it folds (research/runners/_knowledge_core_curate.py `sanitize`: lowercase, every
    run of characters outside [a-z0-9] -> "_"): NFC + lowercase, then every run of non-ASCII letters / marks / digits -> "_". The
    leading/trailing strip("_") of `sanitize` is applied per slug by `_store_slug_re`. Spaces, hyphens and underscores stay (they
    separate slug words); ASCII punctuation is left alone (it is structure here: ",", ".", or a rejecting token)."""
    t = _prepare(text)
    out, last = [], 0
    for a, b in _nonascii_word_runs(t):
        out.append(t[last:a]); out.append("_"); last = b
    out.append(t[last:])
    return "".join(out)


def _tracked_fold(text):
    """`_store_fold(text)` with, for every character of the folded string, the (start, end) of the NFC source characters it came from
    -> (nfc_source, folded, spans), or None when the per-character lowercase does not reproduce `_prepare` (a context-dependent case
    mapping; never for text `_foreign_text` lets through): the case check then fails closed."""
    src = unicodedata.normalize("NFC", text)
    chars, spans = [], []
    for i, ch in enumerate(src):
        for c in ch.lower():
            chars.append(c); spans.append((i, i + 1))
    low = "".join(chars)
    if low != _prepare(text):
        return None
    out, osp, last = [], [], 0
    for a, b in _nonascii_word_runs(low):
        out.append(low[last:a]); osp.extend(spans[last:a])
        out.append("_"); osp.append((spans[a][0], spans[b - 1][1]))
        last = b
    out.append(low[last:]); osp.extend(spans[last:])
    folded = "".join(out)
    if folded != _store_fold(text):
        return None
    return src, folded, osp


# ---- build 3: case fidelity of entity spans (no invented acronyms) ---------------------------------------------------------
def _word_runs(text):
    """Maximal runs of letters, marks and digits (Unicode L*, M*, N*) in `text`: the words whose case the rule reads."""
    out, cur = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in "LMN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def caps_tokens(text):
    """The ALL-CAPS tokens of `text` (NFC): a word (`_word_runs`) with 2 or more letters, every letter of which is upper case
    ("WHO", "FC", "MP3"; not "Who", "McDONALD", a one-letter "R" or "U"). Shared by the matcher and the table generator
    (research/runners/_moat_allowlist_caps_build.py), so both read case by one rule."""
    out = []
    for w in _word_runs(unicodedata.normalize("NFC", text)):
        letters = [c for c in w if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            out.append(w)
    return out


class _CapsTable:
    """The committed table: entity slug -> frozenset of the all-caps tokens of that entity's store names, relation slug -> the same
    for its label. A slug with no row (an entity outside the store, a conversation-taught name) allows none: every all-caps token
    in its span rejects. A missing or unreadable table also allows none (fail closed) and says so in the reason."""

    def __init__(self, path):
        self.path, self.entities, self.relations, self.error = path, {}, {}, None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            self.entities = {k: frozenset(v) for k, v in doc["entities"].items()}
            self.relations = {k: frozenset(v) for k, v in doc["relations"].items()}
        except Exception as exc:                               # fail closed: no all-caps token is allowed anywhere
            self.error = "%s: %s" % (type(exc).__name__, exc)

    def allowed(self, kind, slug):
        return (self.entities if kind == "ent" else self.relations).get(slug, frozenset())


@functools.lru_cache(maxsize=1)
def caps_table():
    """The case-fidelity table, read once per process (like the per-turn lexicon, it is cached)."""
    return _CapsTable(CAPS_TABLE_PATH)


def _case_violation(parses, origs):
    """-> (kind, slug, token, span_text) for the first all-caps token, inside a SUBJ/OBJ span (kind 'ent') or a relation span (kind
    'rel', the verb of shapes 1/2 or the REL_of of shape 3), that no store name of that entity / no label of that relation writes in
    capitals; None when every such token is allowed. Every parse is checked (an ambiguous sentence is kept only if all parses pass)."""
    table = caps_table()
    for _shape, _props, spans in parses:
        for kind, i, slug in spans:
            allowed = table.allowed(kind, slug)
            for tok in caps_tokens(origs[i]):
                if tok not in allowed:
                    return kind, slug, tok, origs[i]
    return None


# ---- build 4: fold to find, exact to accept ----------------------------------------------------------------------------------
def surface_form(text):
    """The ONLY normalisation an entity span's displayed text gets before the exact comparison (build 4): NFC, then `str.lower()`,
    then every run of U+0020 / U+00A0 / U+2007 / U+2009 / U+202F (F3: the display-identical spaces) -> one ASCII space,
    leading/trailing space stripped. Any other whitespace stays (and `_bad_space` has already rejected a sentence holding it); a
    zero-width character is not whitespace and stays. No hyphen/space equivalence, no NFKC, no
    accent stripping, no confusable mapping. Shared by the matcher and the table generator
    (research/runners/_moat_allowlist_names_build.py), so both compare by one rule."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text).lower()).strip(" ")


def deslug(slug):
    """The de-slugged canonical surface of a store slug: underscores -> single spaces (a slug has no run of underscores; a run, if
    any, still gives one space)."""
    return " ".join(w for w in slug.split("_") if w)


class _NamesTable:
    """The committed exact-surface table: entity slug -> frozenset of the surface forms of that entity's store names that spell the
    slug by the store's rule and differ from its de-slugged form (only those exceptions are stored). A slug with no row accepts only
    its de-slugged form (and the slug verbatim). A missing or unreadable table accepts only those two everywhere (fail closed) and
    says so."""

    def __init__(self, path):
        self.path, self.entities, self.error = path, {}, None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            self.entities = {k: frozenset(v) for k, v in doc["entities"].items()}
        except Exception as exc:                               # fail closed: only de-slugged forms are accepted
            self.entities, self.error = {}, "%s: %s" % (type(exc).__name__, exc)

    def surfaces(self, slug):
        """The slug's canonical surfaces: the slug itself as written (the stub mouth renders slugs verbatim; it is the store's own
        identifier, [a-z0-9_] only, so no fold is involved), its de-slugged form, and its table rows. Build 5 (review 2026-10-20): when
        EVERY table row of the slug writes a sign before a digit (a minus or plus not preceded by a digit: "utc-06", "utc−06"), the
        de-slugged form, which lost the sign ("utc 06" reads as UTC+06), is not accepted."""
        rows = self.entities.get(slug, frozenset())
        if rows and all(_SIGNED_NUMBER.search(r) for r in rows):
            return frozenset((slug,)) | rows
        return frozenset((slug, deslug(slug))) | rows


@functools.lru_cache(maxsize=1)
def names_table():
    """The exact-surface table, read once per process."""
    return _NamesTable(NAMES_TABLE_PATH)


def _exact_violation(parses, toks, origs, offs, src):
    """Build 4 (fold to find, exact to accept) -> a reason string for the first violation over every parse, or None:
      - a SUBJ/OBJ span whose `surface_form` is not one of its slug's canonical surfaces (`names_table().surfaces`);
      - a relation span whose displayed text, lowercased with whitespace collapsed, is not the token the relation table read
        (a single verb form as written, or the relation slug verbatim or with underscores as single spaces): no fold of any kind;
      - a non-ASCII character anywhere outside the accepted SUBJ/OBJ spans (determiners, relation words, "is", "and", commas, the
        final period, the space between words)."""
    table = names_table()
    for _shape, _props, spans in parses:
        covered = set()
        for kind, i, slug in spans:
            shown = surface_form(origs[i])
            if kind == "ent":
                ok = table.surfaces(slug)
                if shown not in ok:
                    return ("entity span %r is not an exact store surface of %s (fold to find, exact to accept: accepted %s)%s"
                            % (origs[i], slug, " | ".join(repr(x) for x in sorted(ok)),
                               " (names table unreadable: %s; only the slug and its de-slugged form accepted)" % table.error
                               if table.error else ""))
                covered.update(range(*offs[i]))
            elif shown not in (toks[i], deslug(toks[i])):
                return ("relation span %r is not written exactly as the relation reading %r of %s (no fold on relation words)"
                        % (origs[i], deslug(toks[i]), slug))
        for k, ch in enumerate(src):
            if ord(ch) > 127 and k not in covered:
                return ("non-ASCII character %r (U+%04X) at %d outside every accepted SUBJ/OBJ span"
                        % (ch, ord(ch), k))
    return None


# ---- build 5: no clause word inside a spaced name; a list names distinct things --------------------------------------------------
# A finite auxiliary (the copula and the other finite be / have / do forms) or a negation word (B2h's NEG_MARKERS, which the matcher
# already refuses as a whole token) inside an entity span written with spaces. English reads such a word as the clause's verb or its
# negation, so the name is cut there: "Solomon is located in time zone Kosrae time" is read as "Solomon" + "is located in", not as
# solomon_is (Solomon Is., the islands) + the telegraphic relation label (round-6 cases R6-CF01, R6-CF02, kept by build 4). Non-finite
# forms ("be", "am", "being", "been", "do") are not in the set: a reader does not take them as a third-person verb, and they are
# common inside real names ("Offenbach am Main", "Rio Grande do Sul", "Biel BE").
STRUCTURE_WORDS = frozenset(("is", "are", "was", "were", "be", "has", "have", "had", "does", "did")) | frozenset(NEG_MARKERS)
# "be" (review 2026-10-20): before a participial label ("Thun be located in ...") a name-final "be" supplies the missing verb; the store
# has two such slugs (biel_be, thun_be), so the cost is small. "am", "being", "been" and "do" stay out.
# The residue of a dropped apostrophe, compared on the text as written (lower case only, so a capital initial such as "Harry S Truman"
# is not residue): a lone "s" that ENDS the name ("Saint Alban s located in ..." reads "Saint Alban's located" = "Saint Alban is
# located") and a lone "t" right after a negative stem ("You don t arrest Voltaire" reads "don't"). A lone "s" or "t" elsewhere is the
# residue of a folded accent or period ("universit t zu berlin", "u s of a") and reads as no word.
CONTRACTION_RESIDUE = frozenset(("s", "t"))
NEGATIVE_STEMS = frozenset(("don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "can", "couldn", "wouldn", "shouldn",
                            "hasn", "haven", "hadn", "ain", "mustn", "needn"))


def _structure_word_violation(parses, origs):
    """Build 5 -> a reason string for the first SUBJ/OBJ span, over every parse, whose displayed text holds a word of
    STRUCTURE_WORDS once split on spaces (`surface_form`, so the four display-identical spaces split too), else None. The slug written
    verbatim in lower case ("solomon_is", as the stub mouth renders it) is one identifier and is not split; any other casing of an
    underscore spelling is split on underscores too. Also rejects the residue of a dropped apostrophe (CONTRACTION_RESIDUE)."""
    for _shape, _props, spans in parses:
        for kind, i, slug in spans:
            if kind != "ent":
                continue
            orig = unicodedata.normalize("NFC", origs[i]).strip(SPAN_SPACES)
            # the slug written verbatim in lower case (as the stub mouth renders it) is one identifier; any other casing of an
            # underscore spelling is prose ("Marshall_Is", "No_Saints": review 2026-10-20), so its underscores split words like spaces
            shown = [orig] if orig == slug else [w for w in _WORD_BREAK.split(orig) if w]
            hit = [w for w in shown if w.lower() in STRUCTURE_WORDS]
            if hit:
                return ("entity span %r of %s holds the clause word %r: English reads it as the verb or a negation and cuts the name "
                        "there" % (origs[i], slug, hit[0]))
            res = None
            if len(shown) > 1 and shown[-1] == "s":
                res = "s"
            for k in range(1, len(shown)):
                if shown[k] == "t" and shown[k - 1].lower() in NEGATIVE_STEMS:
                    res = "t"
            if res:
                return ("entity span %r of %s holds a lone lower-case %r, the residue of a dropped apostrophe: English reads it as a "
                        "contraction (\"'s\" = is, \"n't\" = not)" % (origs[i], slug, res))
    return None


def _repeated_proposition(parses):
    """Build 5 -> a reason string when a parse lists one proposition twice, else None. A coordinated list ("A and B", shape 2) asserts
    distinct things, so "Take That has part Robbie Williams and a Robbie Williams" asserts a second Robbie Williams that the one
    gathered triple does not (round-6 cases R6-S201, R6-S203, R6-S210, kept by build 4)."""
    for shape, props, _spans in parses:
        seen = set()
        for p in props:
            if p in seen:
                return ("%s lists (%s, %s, %s) twice: a coordinated list asserts distinct things" % ((shape,) + tuple(p)))
            seen.add(p)
    return None


def _store_slug_re(slug):
    """Any space / hyphen / underscore spelling of `slug`'s words, with the store's strip("_") at both ends: a folded accented
    letter at the start or end of the name ("_le de france", "bod_") is part of the span, because `sanitize` dropped it from the
    slug. Bounded so it never matches inside a longer word or an already-substituted slug token."""
    words = [w for w in slug.split("_") if w]
    body = r"[\s\-_]+".join(re.escape(w) for w in words)
    return re.compile(r"(?<![a-z0-9_])_*" + body + r"_*(?![a-z0-9_])")


def allowlist_enabled():
    """`BRAIN_MOAT_ALLOWLIST` truthy (1/true/on/yes) -> the production claim check is this module's matcher. Default OFF
    (unset/0/false/off/no/anything else). Same truthy/falsy contract as `slug_norm_enabled`; the TUI and composer readers are
    pinned to this one in tests/test_moat_allowlist.py."""
    return os.environ.get(ALLOWLIST_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


class _Lexicon:
    """This turn's gathered vocabulary, read by the B2h helpers REUSED UNCHANGED (bound here as plain functions)."""

    _base_verb = ClaimEntailmentVerifier._base_verb
    _classify = ClaimEntailmentVerifier._classify

    def __init__(self, key):
        self.gated = sorted(key)
        self.gated_set = frozenset(key)
        self.nouns = {t for f in self.gated for t in (f[0], f[2])}
        self.verbs = {f[1] for f in self.gated}
        self.synonyms = dict(VERB_SYNONYMS)
        self.inflect = _build_inflection_map(sorted(self.verbs))
        self.slug_norm = True                       # the B2h readings of `_base_verb` / `_classify`
        # every gathered slug (one word or several) is spelled by the store's rule (`_store_slug_re`). A slug with no letter or
        # digit, e.g. a degenerate "_", would compile to a pattern that matches bare underscores; it can never be a SUBJ/OBJ/VERB
        # here, so it is left out of the spellings.
        slugs = {t for f in self.gated for t in f if _HAS_ALNUM.search(t)}
        self._slug_pats = [(s, _store_slug_re(s)) for s in sorted(slugs, key=lambda x: (-len(x), x))]
        # the gathered relations allowed a verb reading / shape 3 (the reviewed table: VERB-SAFE or NOT-A-VERB; a ROLE-REVERSING or
        # unclassified relation gets none)
        self.verb_ok = frozenset(r for r in self.verbs if has_verb_reading(r))
        # shape-3 relation nouns: a gathered, verb-reading relation slug ending in "_of" with a real noun before it
        self.of_relations = frozenset(r for r in self.verb_ok if r.endswith("_of") and _HAS_ALNUM.search(r[:-3]))

    def normalise(self, sentence):
        """The store's fold, then every spelling of a gathered slug -> the slug token itself."""
        t = _store_fold(sentence)
        for slug, rx in self._slug_pats:
            t = rx.sub(" " + slug + " ", t)
        return t

    def normalise_tracked(self, sentence):
        """`normalise`, tracking where each token came from -> (normalised, tokens, original_texts, offsets, nfc_source) with
        original_texts[i] the NFC text of the sentence that token i was made from (the whole spelled span for a gathered slug: "the
        WHO" for the_who) and offsets[i] its (start, end) in nfc_source, or None when the fold cannot be tracked (the case check and
        build 4's exact check then fail closed). Same substitutions, same order as `normalise`."""
        tr = _tracked_fold(sentence)
        if tr is None:
            return None
        src, t, sp = tr
        for slug, rx in self._slug_pats:
            rep = " " + slug + " "
            out, osp, pos = [], [], 0
            for m in rx.finditer(t):
                a, b = m.span()
                out.append(t[pos:a]); osp.extend(sp[pos:a])
                region = (min(s for s, _e in sp[a:b]), max(e for _s, e in sp[a:b]))
                out.append(rep); osp.extend([region] * len(rep))
                pos = b
            out.append(t[pos:]); osp.extend(sp[pos:])
            t, sp = "".join(out), osp
        toks, origs, offs = [], [], []
        for m in _TOKEN.finditer(t):
            a, b = m.span()
            toks.append(m.group())
            lo, hi = min(s for s, _e in sp[a:b]), max(e for _s, e in sp[a:b])
            origs.append(src[lo:hi]); offs.append((lo, hi))
        return t, toks, origs, offs, src

    def is_noun(self, tok):
        return tok in self.nouns and tok not in NEG_MARKERS and _HAS_ALNUM.search(tok) is not None

    def verb_rel(self, tok):
        """-> the gathered relation `tok` is a verb form of, or None (also None for a relation with no verb reading)."""
        if _HAS_ALNUM.search(tok) is None:
            return None
        kind, val = self._classify(tok)
        return val if kind == "verb" and val in self.verb_ok and not reading_dropped(val, tok) else None

    def blocked_rel(self, tok):
        """-> (relation, class) when `tok` is a reading of a GATHERED relation that the table denies a verb reading, else None."""
        if _HAS_ALNUM.search(tok) is None:
            return None
        kind, val = self._classify(tok)
        if kind == "verb" and val in self.verbs and val not in self.verb_ok:
            return val, relation_class(val)
        if kind == "verb" and val in self.verb_ok and reading_dropped(val, tok):      # a reading that failed the per-reading test
            return val, MEANING_DRIFT
        if tok in self.verbs and tok not in self.verb_ok:                  # e.g. an unclassified `_of` slug after "is"
            return tok, relation_class(tok)
        return None


def _key_of(gathered):
    return frozenset(tuple(f) for f in (gathered or ())
                     if isinstance(f, (list, tuple)) and len(f) == 3 and all(isinstance(x, str) for x in f))


@functools.lru_cache(maxsize=1024)
def _lexicon(key):
    return _Lexicon(key)


def split_sentences(surface):
    """Split a rendered surface into sentences on . ! ? followed by whitespace. Empty pieces are dropped."""
    if not isinstance(surface, str):
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(surface.strip()) if s.strip()]


# ---- the shapes ----------------------------------------------------------------------------------------------------------
class _Fail:
    """Furthest-failure record: what each shape expected at the furthest token any partial parse reached."""

    def __init__(self):
        self.pos, self.expected = -1, set()

    def at(self, pos, what):
        if pos > self.pos:
            self.pos, self.expected = pos, {what}
        elif pos == self.pos:
            self.expected.add(what)


def _np(toks, i, lex, fail):
    """NP = [Det] N at token i -> [(noun, next_i)]."""
    out = []
    n = len(toks)
    if i < n and lex.is_noun(toks[i]):
        out.append((toks[i], i + 1))
    if i < n and toks[i] in DETERMINERS:
        if i + 1 < n and lex.is_noun(toks[i + 1]):
            out.append((toks[i + 1], i + 2))
        else:
            fail.at(i + 1, "a gathered slug after the determiner")
    if not out:
        fail.at(i, "a gathered slug (SUBJ/OBJ)")
    return out


def _object_lists(toks, k, lex, fail):
    """OBJ (shape 1) or OBJ1 (, OBJi)* [,] and OBJn (shape 2), starting at token k -> [(objs, end)], objs = [(noun, token_index)]
    (an NP's noun is its last token, so its index is the NP's end - 1)."""
    n = len(toks)
    done = []
    for o1, e1 in _np(toks, k, lex, fail):
        done.append(([(o1, e1 - 1)], e1))
        stack = [([(o1, e1 - 1)], e1)]
        while stack:
            objs, e = stack.pop()
            if e < n and toks[e] == ",":
                if e + 1 < n and toks[e + 1] == "and":
                    done.extend((objs + [(o, e2 - 1)], e2) for o, e2 in _np(toks, e + 2, lex, fail))
                else:
                    stack.extend((objs + [(o, e2 - 1)], e2) for o, e2 in _np(toks, e + 1, lex, fail))
            elif e < n and toks[e] == "and":
                done.extend((objs + [(o, e2 - 1)], e2) for o, e2 in _np(toks, e + 1, lex, fail))
            elif len(objs) > 1:
                fail.at(e, "', and OBJ' or 'and OBJ' (shape 2 ends with 'and OBJn')")
    return done


def _no_reading(tok, blocked):
    rel, cls = blocked
    return ("a relation with a verb reading (%r reads %s, classified %s in _moat_allowlist_relations: no verb reading)"
            % (tok, rel, cls))


def _parses(toks, lex, fail):
    """Every complete parse of `toks` -> [(shape, [(s, r, o), ...], spans)], spans = [(kind, token_index, slug)]: each SUBJ/OBJ
    noun ('ent') and the relation token ('rel': the verb of shapes 1/2, the REL_of of shape 3) the parse consumed (build 3's case
    check reads the original text of exactly these tokens)."""
    n = len(toks)
    out = []
    for subj, j in _np(toks, 0, lex, fail):
        if j >= n:
            fail.at(j, "a verb form of a gathered relation (shapes 1/2) or 'is' (shape 3)")
            continue
        rel = lex.verb_rel(toks[j])
        if rel is not None:                                               # shapes 1 and 2
            for objs, end in _object_lists(toks, j + 1, lex, fail):
                if end == n:
                    out.append(("shape1" if len(objs) == 1 else "shape2", [(subj, rel, o) for o, _oi in objs],
                                [("ent", j - 1, subj), ("rel", j, rel)] + [("ent", oi, o) for o, oi in objs]))
                else:
                    fail.at(end, "end of sentence, or ', OBJ' / 'and OBJ' (shape 2)")
        if toks[j] == COPULA:                                             # shape 3
            # the relation noun takes "a" / "an" or nothing: "X is THE member of Y" asserts uniqueness
            r_at = [j + 1] + ([j + 2] if j + 1 < n and toks[j + 1] in REL_NOUN_DETERMINERS else [])
            if j + 1 < n and toks[j + 1] in DETERMINERS and toks[j + 1] not in REL_NOUN_DETERMINERS:
                fail.at(j + 1, "'a' / 'an' or no determiner before the relation noun (shape 3: %r asserts uniqueness)" % toks[j + 1])
            for ri in r_at:
                if ri < n and toks[ri] in lex.of_relations:
                    for obj, end in _np(toks, ri + 1, lex, fail):
                        if end == n:
                            out.append(("shape3", [(subj, toks[ri], obj)],
                                        [("ent", j - 1, subj), ("rel", ri, toks[ri]), ("ent", end - 1, obj)]))
                        else:
                            fail.at(end, "end of sentence (shape 3)")
                else:
                    blocked = lex.blocked_rel(toks[ri]) if ri < n else None
                    fail.at(ri, _no_reading(toks[ri], blocked) if blocked else "a gathered relation slug ending in _of (shape 3)")
        elif rel is None:
            blocked = lex.blocked_rel(toks[j])
            fail.at(j, _no_reading(toks[j], blocked) if blocked
                    else "a verb form of a gathered relation (shapes 1/2) or 'is' (shape 3)")
    return out


def allowlist_check(sentence, gathered):
    """ONE sentence against this turn's gathered triples -> dict(keep, reason, shape, props, tokens). Pure (no substrate,
    no parser seed): the same input gives the same verdict at every parser seed."""
    # every input is filtered to well-formed (str, str, str) triples, a frozenset too: a malformed frozenset passed directly used to
    # raise IndexError (mutation witness m96); a filtered key comes back equal, so the lexicon cache is unchanged
    key = _key_of(gathered)
    res = {"sentence": sentence, "keep": False, "reason": None, "shape": None, "props": [], "tokens": None}
    if not key:
        res["reason"] = "allowlist: nothing gathered"
        return res
    if not isinstance(sentence, str) or not sentence.strip():
        res["reason"] = "allowlist: empty sentence"
        return res
    bad_space = _bad_space(sentence)
    if bad_space is not None:            # build 4 F3: only U+0020 between words (and four display-identical spaces inside a name)
        k, ch = bad_space
        res["reason"] = ("allowlist: whitespace or control character %r (U+%04X) at %d: only U+0020 separates words (U+00A0, U+2007, "
                         "U+2009, U+202F also inside an entity name)" % (ch, ord(ch), k))
        return res
    if _APOSTROPHES.search(sentence):
        res["reason"] = "allowlist: apostrophe (possessive or contraction) in %r" % sentence
        return res
    foreign = _foreign_text(sentence)
    if foreign is not None:
        res["reason"] = "allowlist: non-Latin or free-standing non-ASCII text %r" % foreign
        return res
    lex = _lexicon(key)
    toks = _TOKEN.findall(lex.normalise(sentence))
    if toks and toks[-1] == ".":
        toks = toks[:-1]
    res["tokens"] = list(toks)
    if not toks:
        res["reason"] = "allowlist: empty sentence"
        return res
    fail = _Fail()
    parses = _parses(toks, lex, fail)
    if not parses:
        got = toks[fail.pos] if 0 <= fail.pos < len(toks) else "<end of sentence>"
        res["reason"] = "allowlist: no shape: at token %d %r expected %s" % (fail.pos, got, " or ".join(sorted(fail.expected)))
        return res
    bad = sorted({p for _shape, props, _spans in parses for p in props if p not in lex.gated_set})
    if bad:
        res["reason"] = "allowlist: not gathered: %s" % ", ".join("(%s, %s, %s)" % p for p in bad)
        res["shape"] = parses[0][0]
        return res
    # build 3: no invented acronyms (case fidelity of entity spans; controller ruling 2026-10-20)
    tr = lex.normalise_tracked(sentence)
    if tr is None or tr[1][:len(toks)] != toks:
        res["reason"] = "allowlist: all-caps check: cannot track the original text of %r (fail closed)" % sentence
        res["shape"] = parses[0][0]
        return res
    viol = _case_violation(parses, tr[2])
    if viol is not None:
        kind, slug, tok, span = viol
        table = caps_table()
        res["reason"] = ("allowlist: all-caps token %r in the %s span %r of %s: no store %s of %s writes %r in capitals%s"
                         % (tok, "SUBJ/OBJ" if kind == "ent" else "relation", span, slug,
                            "name" if kind == "ent" else "label", slug, tok,
                            " (caps table unreadable: %s)" % table.error if table.error else ""))
        res["shape"] = parses[0][0]
        return res
    # build 4: fold to find, exact to accept (every entity span is an exact store surface; no fold on relation words; ASCII outside)
    exact = _exact_violation(parses, tr[1], tr[2], tr[3], tr[4])
    if exact is not None:
        res["reason"] = "allowlist: " + exact
        res["shape"] = parses[0][0]
        return res
    # build 5: no finite auxiliary or negation word inside a spaced entity name; a list names distinct things
    late = _structure_word_violation(parses, tr[2]) or _repeated_proposition(parses)
    if late is not None:
        res["reason"] = "allowlist: " + late
        res["shape"] = parses[0][0]
        return res
    shape, props, _spans = parses[0]
    res.update(keep=True, reason="ok", shape=shape, props=[list(p) for p in props])
    return res


def allowlist_keep(sentence, gathered):
    """The spec's small pure function: ONE sentence -> (keep: bool, reason: str)."""
    r = allowlist_check(sentence, gathered)
    return bool(r["keep"]), r["reason"]


def _bad_gap(surface):
    """build 4 F3 (Claude ruling after the correctness re-review): the whitespace `split_sentences` consumes (before the first
    sentence, after the last, and between sentences) may only be U+0020, LF or CR -> the first other character there, else None.
    Inside a sentence `_bad_space` rules."""
    gaps = [surface[:len(surface) - len(surface.lstrip())], surface[len(surface.rstrip()):]]
    gaps.extend(m.group(0) for m in _SENTENCE_SPLIT.finditer(surface.strip()))
    for gap in gaps:
        for k, ch in enumerate(gap):
            if ch not in _GAP_OK:
                return ch
            if ch == "\r" and gap[k + 1:k + 2] != "\n":      # build 5 (review): a bare CR overprints the line on a terminal
                return ch
    return None


_GAP_OK = " \n\r"


def allowlist_verify(surface, gathered):
    """A rendered candidate (one or more sentences) -> the verifier result dict `ChatBrain._verify_claim_set` returns
    (accepted, reject_reason, grounded, hypotheses, trace). Accepted iff it has a sentence and EVERY sentence is kept."""
    key = _key_of(gathered)
    gap = _bad_gap(surface) if isinstance(surface, str) else None
    if gap is not None:
        return {"accepted": False, "grounded": [], "hypotheses": [], "trace": [], "allowlist": True,
                "reject_reason": "allowlist: whitespace or control character U+%04X between sentences or at an edge of the "
                                 "reply (only U+0020, LF and CR are allowed there)" % ord(gap)}
    sentences = split_sentences(surface)
    trace = [allowlist_check(s, key) for s in sentences]
    grounded = [[a, v, p, [a, v, p]] for t in trace if t["keep"] for a, v, p in t["props"]]
    if not sentences:
        reason = "allowlist: empty surface"
    else:
        first_bad = next((i for i, t in enumerate(trace) if not t["keep"]), None)
        reason = None if first_bad is None else "%s [sentence %d of %d]" % (trace[first_bad]["reason"], first_bad + 1, len(trace))
        if reason is None:                   # build 5 (review): the list rule across sentences ("... Robbie Williams. ... a Robbie Williams.")
            seen = {}
            for k, t in enumerate(trace):
                for p in map(tuple, t["props"]):
                    if p in seen:
                        reason = ("allowlist: sentences %d and %d both state (%s, %s, %s): a reply that states one fact twice can "
                                  "read as two distinct things" % ((seen[p] + 1, k + 1) + p))
                        break
                    seen[p] = k
                if reason is not None:
                    break
    return {"accepted": reason is None, "reject_reason": reason, "grounded": grounded, "hypotheses": [], "trace": trace,
            "allowlist": True}


class AllowlistClaimVerifier:
    """Drop-in for `ClaimEntailmentVerifier` in `ChatBrain._build_claim_verifier` when the flag is ON. Always buildable for a
    non-empty gathered set (no role-permutation guard: a mutual `shares_border_with` pair is fine, board #265)."""

    allowlist = True
    slug_norm = False

    def __init__(self, gathered):
        self.key = _key_of(gathered)

    def verify(self, prose):
        return allowlist_verify(prose, self.key)
