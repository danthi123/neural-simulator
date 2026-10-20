"""STRICT (ALLOWLIST) CLAIM CHECKER for the scaffold mouth -- flag `BRAIN_MOAT_ALLOWLIST`, default OFF.

Spec: docs/drafts/moat-allowlist/SPEC.md. Bars: research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md
(filed before this file existed). This is a HOST safety filter on what the Qwen mouth may say (a scaffold check,
non-negotiable 1); it claims no brain capability, and its English shapes are a filter, not a language ability.

WHY A DIFFERENT METHOD. The slug-aware checker ("B2h", `BRAIN_MOAT_SLUG_NORM`, `_moat_claim_entailment_derisk`) accepts a
sentence unless a rule rejects it: it drops function words and hedges and attaches subjectless or verbless clauses to the
previous clause, so every round of denylist fixes was followed by a new construction that leaked (RULING moat-slug-hold-1018).
This module inverts the default: a sentence is KEPT only when ONE exact proposition shape consumes EVERY token and every
proposition it encodes is a gathered triple of this turn. Everything else is rejected.

THE SHAPES (after normalisation; N = one gathered slug, Det = a | an | the, directly before N or, in shape 3, the relation noun):
  1. [Det] SUBJ VERB(rel) [Det] OBJ [.]                                          -> (SUBJ, rel, OBJ)
  2. [Det] SUBJ VERB(rel) [Det] OBJ1 (, [Det] OBJi)* [,] and [Det] OBJn [.]       -> (SUBJ, rel, OBJi) for EVERY i
  3. [Det] SUBJ is [Det] REL_of [Det] OBJ [.]                                    -> (SUBJ, REL_of, OBJ)
     only for a gathered relation whose slug ENDS in `_of` (instance_of, subclass_of, member_of, part_of, participant_of,
     is_a_list_of in the store): its slug reads subject-first ("X is an instance of Y" = (X, instance_of, Y)), so the shape
     cannot reverse roles. A relation without `_of` ("capital": "Ankara is the capital of Turkey" means (turkey, capital,
     ankara)) never takes shape 3, so the reversal that sentence would need is impossible. Pinned in
     tests/test_moat_allowlist.py (test_shape3_*).
  VERB(rel) = a token whose `_classify` reading is ('verb', rel) for a relation gathered THIS turn: the same verb-form table
  `_noun_before_verb` uses (stock inflections, the tight synonyms, the guarded -es / -ies readings of commit 1b67eef9b).
  SUBJ / OBJ = a token equal to a gathered subject/object slug after B2h normalisation (`_normalise_slugs`: case, NFC, spaces /
  hyphens / underscores, the store's "_" for accented letters). A token made only of underscores is never a noun.

NOTHING IS DROPPED. A token outside the shape (pronoun, adverb, "just", "also", hedge, negation, subordinator, a comma that
opens a clause, a digit not inside a gathered slug, "!", "?", ":", "-", a second ".") leaves the shape unmatched and the
sentence is rejected. Any apostrophe (possessive 's, contraction) rejects the sentence BEFORE normalisation, because the B2h
slug regex would otherwise swallow a possessive 's; so does non-ASCII text other than an accented Latin letter inside a Latin
word (`_foreign_text`: B2h would read a free-standing non-Latin word between two slug words as a separator). A negation word
rejects even when it is also a gathered slug.
ONE SENTENCE, ONE SHAPE. The surface is split into sentences first; each sentence must match a shape on its own. If two parses
of one sentence exist, every parse must encode only gathered triples (an ambiguous sentence with any ungathered reading is
rejected). The production verdict on a rendered candidate (`allowlist_verify`) keeps it only if EVERY sentence is kept: the
composer emits candidates whole, so a candidate holding a rejected sentence is not emitted (the turn keeps its OTHER
candidates, each verified on its own). Partial emission would change the composer's output path and is out of scope.

NO EDIT TO THE B2h PARSER PATH. The B2h helpers (`_normalise_slugs`, `_base_verb`, `_classify`) are REUSED unchanged by binding
them onto a small lexicon object; no `ClaimEntailmentVerifier` is constructed, so its role-permutation guard cannot make a
mutual pair unbuildable (board #265), and neither the stock clause parser nor `_noun_is_before` (the 3-letter predicate match,
board #266) is consulted. The substrate role parser is not used either: roles come from the shape, and the proposition is
adjudicated by set membership.

Wiring (flag ON): `ChatBrain._build_claim_verifier` returns an `AllowlistClaimVerifier`, `ChatBrain._verify_claim_set` uses it
even when `BRAIN_CLAIM_MOAT=0`, and `RichAnswerComposer._verify_rendered` routes EVERY turn (one fact or many) to it with no
fallback to the single-triple `chat._verify`. Flag OFF: both call sites are byte-identical to before.
"""
from __future__ import annotations

import functools
import os
import re
import unicodedata

from research.runners._moat_claim_entailment_derisk import (
    _NONASCII_ALNUM_RUN, NEG_MARKERS, VERB_SYNONYMS, ClaimEntailmentVerifier, _build_inflection_map, _slug_phrase_re)

ALLOWLIST_ENV = "BRAIN_MOAT_ALLOWLIST"
DETERMINERS = frozenset(("a", "an", "the"))
COPULA = "is"
# straight, curly, modifier-letter and prime apostrophes, and the backtick: any of them rejects the sentence (possessive /
# contraction). Checked on the RAW sentence, before `_normalise_slugs` could read "X's" as the slug X.
_APOSTROPHES = re.compile(r"['`‘’‛ʼ′]")
# after normalisation: a word is a run of [a-z0-9_]; every other non-space character is a token of its own (so it can never be
# consumed by a shape except "," and the one sentence-final ".")
_TOKEN = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]")
_HAS_ALNUM = re.compile(r"[a-z0-9]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ASCII_ALNUM = re.compile(r"[a-z0-9]")


def _foreign_text(text):
    r"""B2h maps every non-ASCII letter/digit run to "_" (the store's slug form: "Montréal" -> montr_al), and its slug spelling
    joins slug words with [\s\-_]+, so a free-standing non-ASCII word BETWEEN two slug words ("shares 不 border with", 不 = not)
    would be swallowed as a separator. Allowed here: a run of LATIN letters attached to an ASCII letter/digit (an accented letter
    inside a Latin word: Montréal, República, José). Anything else (non-Latin script, a free-standing accented word such as "à",
    a non-ASCII digit) -> that run, so the sentence is rejected before normalisation."""
    t = unicodedata.normalize("NFC", text).lower()
    for m in _NONASCII_ALNUM_RUN.finditer(t):
        run = m.group(0)
        latin = all(unicodedata.name(ch, "").startswith("LATIN ") for ch in run)
        before = t[m.start() - 1] if m.start() > 0 else ""
        after = t[m.end()] if m.end() < len(t) else ""
        if not latin or not (_ASCII_ALNUM.match(before) or _ASCII_ALNUM.match(after)):
            return run
    return None


def allowlist_enabled():
    """`BRAIN_MOAT_ALLOWLIST` truthy (1/true/on/yes) -> the production claim check is this module's matcher. Default OFF
    (unset/0/false/off/no/anything else). Same truthy/falsy contract as `slug_norm_enabled`; the TUI and composer readers are
    pinned to this one in tests/test_moat_allowlist.py."""
    return os.environ.get(ALLOWLIST_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


class _Lexicon:
    """This turn's gathered vocabulary, read by the B2h helpers REUSED UNCHANGED (bound here as plain functions)."""

    _normalise_slugs = ClaimEntailmentVerifier._normalise_slugs
    _base_verb = ClaimEntailmentVerifier._base_verb
    _classify = ClaimEntailmentVerifier._classify

    def __init__(self, key):
        self.gated = sorted(key)
        self.gated_set = frozenset(key)
        self.nouns = {t for f in self.gated for t in (f[0], f[2])}
        self.verbs = {f[1] for f in self.gated}
        self.synonyms = dict(VERB_SYNONYMS)
        self.inflect = _build_inflection_map(sorted(self.verbs))
        self.slug_norm = True                       # the B2h readings of `_base_verb` / `_classify` / `_normalise_slugs`
        # (a slug with no letter or digit, e.g. a degenerate "_", would compile to an empty pattern that matches everywhere; it
        # can never be a SUBJ/OBJ/VERB here, so it is left out of the spellings)
        slugs = {t for f in self.gated for t in f if "_" in t and _HAS_ALNUM.search(t)}
        self._slug_pats = [(s, _slug_phrase_re(s)) for s in sorted(slugs, key=lambda x: (-len(x), x))]
        # shape-3 relation nouns: a gathered relation slug ending in "_of" with a real noun before it
        self.of_relations = frozenset(r for r in self.verbs if r.endswith("_of") and _HAS_ALNUM.search(r[:-3]))

    def is_noun(self, tok):
        return tok in self.nouns and tok not in NEG_MARKERS and _HAS_ALNUM.search(tok) is not None

    def verb_rel(self, tok):
        """-> the gathered relation `tok` is a verb form of, or None."""
        if _HAS_ALNUM.search(tok) is None:
            return None
        kind, val = self._classify(tok)
        return val if kind == "verb" and val in self.verbs else None


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
    """OBJ (shape 1) or OBJ1 (, OBJi)* [,] and OBJn (shape 2), starting at token k -> [(objs, end)]."""
    n = len(toks)
    done = []
    for o1, e1 in _np(toks, k, lex, fail):
        done.append(([o1], e1))
        stack = [([o1], e1)]
        while stack:
            objs, e = stack.pop()
            if e < n and toks[e] == ",":
                if e + 1 < n and toks[e + 1] == "and":
                    done.extend((objs + [o], e2) for o, e2 in _np(toks, e + 2, lex, fail))
                else:
                    stack.extend((objs + [o], e2) for o, e2 in _np(toks, e + 1, lex, fail))
            elif e < n and toks[e] == "and":
                done.extend((objs + [o], e2) for o, e2 in _np(toks, e + 1, lex, fail))
            elif len(objs) > 1:
                fail.at(e, "', and OBJ' or 'and OBJ' (shape 2 ends with 'and OBJn')")
    return done


def _parses(toks, lex, fail):
    """Every complete parse of `toks` -> [(shape, [(s, r, o), ...])]."""
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
                    out.append(("shape1" if len(objs) == 1 else "shape2", [(subj, rel, o) for o in objs]))
                else:
                    fail.at(end, "end of sentence, or ', OBJ' / 'and OBJ' (shape 2)")
        if toks[j] == COPULA:                                             # shape 3
            r_at = [j + 1] + ([j + 2] if j + 1 < n and toks[j + 1] in DETERMINERS else [])
            for ri in r_at:
                if ri < n and toks[ri] in lex.of_relations:
                    for obj, end in _np(toks, ri + 1, lex, fail):
                        if end == n:
                            out.append(("shape3", [(subj, toks[ri], obj)]))
                        else:
                            fail.at(end, "end of sentence (shape 3)")
                else:
                    fail.at(ri, "a gathered relation slug ending in _of (shape 3)")
        elif rel is None:
            fail.at(j, "a verb form of a gathered relation (shapes 1/2) or 'is' (shape 3)")
    return out


def allowlist_check(sentence, gathered):
    """ONE sentence against this turn's gathered triples -> dict(keep, reason, shape, props, tokens). Pure (no substrate,
    no parser seed): the same input gives the same verdict at every parser seed."""
    key = gathered if isinstance(gathered, frozenset) else _key_of(gathered)
    res = {"sentence": sentence, "keep": False, "reason": None, "shape": None, "props": [], "tokens": None}
    if not key:
        res["reason"] = "allowlist: nothing gathered"
        return res
    if not isinstance(sentence, str) or not sentence.strip():
        res["reason"] = "allowlist: empty sentence"
        return res
    if _APOSTROPHES.search(sentence):
        res["reason"] = "allowlist: apostrophe (possessive or contraction) in %r" % sentence
        return res
    foreign = _foreign_text(sentence)
    if foreign is not None:
        res["reason"] = "allowlist: non-Latin or free-standing non-ASCII text %r" % foreign
        return res
    lex = _lexicon(key)
    toks = _TOKEN.findall(lex._normalise_slugs(sentence))
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
    bad = sorted({p for _shape, props in parses for p in props if p not in lex.gated_set})
    if bad:
        res["reason"] = "allowlist: not gathered: %s" % ", ".join("(%s, %s, %s)" % p for p in bad)
        res["shape"] = parses[0][0]
        return res
    shape, props = parses[0]
    res.update(keep=True, reason="ok", shape=shape, props=[list(p) for p in props])
    return res


def allowlist_keep(sentence, gathered):
    """The spec's small pure function: ONE sentence -> (keep: bool, reason: str)."""
    r = allowlist_check(sentence, gathered)
    return bool(r["keep"]), r["reason"]


def allowlist_verify(surface, gathered):
    """A rendered candidate (one or more sentences) -> the verifier result dict `ChatBrain._verify_claim_set` returns
    (accepted, reject_reason, grounded, hypotheses, trace). Accepted iff it has a sentence and EVERY sentence is kept."""
    key = _key_of(gathered)
    sentences = split_sentences(surface)
    trace = [allowlist_check(s, key) for s in sentences]
    grounded = [[a, v, p, [a, v, p]] for t in trace if t["keep"] for a, v, p in t["props"]]
    if not sentences:
        reason = "allowlist: empty surface"
    else:
        first_bad = next((i for i, t in enumerate(trace) if not t["keep"]), None)
        reason = None if first_bad is None else "%s [sentence %d of %d]" % (trace[first_bad]["reason"], first_bad + 1, len(trace))
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
