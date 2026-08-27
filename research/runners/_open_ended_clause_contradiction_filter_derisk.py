"""CLAUSE-granularity repair of the known-topic contradiction filter's SAME-SENTENCE residual. (2026-08-27)

CONTEXT (Vikunja #112, the open-ended honesty gap). `webapp/open_ended_chat.py`'s wired post-filter
(2026-08-27-known-supplement-contradiction-filter-wired-into-open-ended-chat-postfilter-GO.md) closed the
KNOWN-topic wrong-supplement gap: `sentence_contradicts` (this arc's 2026-08-21 de-risk,
`_open_ended_known_supplement_filter_derisk.py`) catches 10/10 wrong supplements across the 3 saved
known-topic replies with 0 leaks. But that check works at SENTENCE granularity: it returns a bool per
sentence, so when a CORRECT detail and a WRONG detail land in the SAME sentence, the whole sentence is
dropped and the correct detail is lost with it -- the wiring finding's own disclosed "Honest scope":
canada's saved reply put "bordered by the United States [correct] ... and Mexico [wrong]" in one sentence,
and "Ottawa [correct], which was founded in 1867 [wrong, unsupported]" in another.

THIS FILE closes that residual with `clause_filter_sentence(sent, topic, facts)`: given a sentence
`sentence_contradicts` already flagged, it tries two SAFE, declared, span-level repairs and re-verifies the
result against the UNCHANGED `sentence_contradicts` before ever returning edited text -- so it can only
ever keep MORE correct content than today, never less, and it NEVER returns text that still contradicts.

  Repair 1 -- APPOSITIVE / RELATIVE CLAUSE: a wrong supplement riding a trailing ", which ..." / ", who
  ..." / ", that ..." clause (the unsupported-date shape: "Ottawa, which was founded in 1867") is dropped
  by removing just that embedded clause, keeping the head clause ("Ottawa") intact.

  Repair 2 -- COORDINATED RELATION-OBJECT LIST: a wrong supplement riding a same-relation conjunct list
  (the wrong-border shape: "bordered by the United States ... and Mexico") is dropped by removing only the
  STORE-WRONG list item(s) (with their connecting comma/"and" glue and any directional modifier), keeping
  the store-correct item(s) intact. `_bad_relation_tokens` LOCATES which tokens are wrong by mirroring
  `sentence_contradicts`'s OWN border/continent detection (the SAME imported gazetteers `COUNTRIES` /
  `CONTINENTS` and the SAME `_obj` helper, unchanged) -- `sentence_contradicts` itself only returns a
  reason STRING, not a span, so locating the span is new; the DECISION of what counts as wrong is not.

FALLBACK (never less safe than today). If neither repair changes the sentence, if what survives is empty
or a single word, if it ends/starts on a dangling function word (a preposition/conjunction with nothing
left to attach to -- e.g. all of a border-list's items were wrong, so nothing survives to attach to
"bordered by"), or if the edited text STILL contradicts on re-check, `clause_filter_sentence` returns None
and the caller drops the whole sentence -- byte-identical to the pre-existing sentence-granularity
behavior. This is the "wrong capital" reason (a garbled non-list assertion) and the bare "number/date not
in store" reason outside a relative clause (e.g. "a population of around 35 million people, making it one
of the largest countries") -- SCOPE (honest): v1 only makes the two OBSERVED same-sentence residual shapes
(appositive-date, coordinated-list) clause-safe; a bare unsupported number with no relative-clause
boundary has no declared-safe repair here and keeps falling back to whole-sentence removal, same as today.

  python -m research.runners._open_ended_clause_contradiction_filter_derisk
"""
from __future__ import annotations
import re

from research.runners._open_ended_known_supplement_filter_derisk import (
    sentence_contradicts, _obj, COUNTRIES, CONTINENTS,
)

# ---------------------------------------------------------------------------------------------------------
# Repair 1: appositive / relative clause carrying the wrong supplement. Non-greedy: stops at the next comma
# (or end of sentence) so a relative clause followed by MORE independent content is only partially eaten --
# a declared simplification (no case in the observed data needs the general form).
# ---------------------------------------------------------------------------------------------------------
_RELATIVE_CLAUSE_RE = re.compile(r",\s*(?:which|who|that)\b[^,]*", re.IGNORECASE)

# Directional modifier that rides a border-list conjunct ("to the west") -- removed together with its token.
_MODIFIER = r"(?:\s+to\s+the\s+(?:north|south|east|west|northeast|northwest|southeast|southwest))?"

# A cleaned sentence that still starts or ends on one of these is not standing on its own -- a dangling
# preposition/conjunction/copula with nothing left to attach to (e.g. every border-list item was wrong).
_DANGLING_END_RE = re.compile(
    r"\b(?:by|with|including|to|and|or|of|the|a|an|for|in|on|at|as|from|is|are|was|were)\s*$",
    re.IGNORECASE)
_DANGLING_START_RE = re.compile(r"^\s*(?:and|but|or|which|who|that)\b", re.IGNORECASE)


def _bad_relation_tokens(sent, topic, facts):
    """The SPAN-locatable subset of `sentence_contradicts`'s border/continent branches: returns the actual
    set of wrong tokens (not just a bool), by re-running the SAME detection (`COUNTRIES`/`CONTINENTS`
    gazetteers + `_obj`, imported unchanged from the GO de-risk) against `sent`. The "wrong capital" and
    bare number/date branches have no removable span here by design (see module docstring SCOPE) and are
    not reproduced -- a sentence flagged only for those returns an empty set, so Repair 2 is a no-op and
    the caller falls through to Repair 1 / the whole-sentence fallback."""
    s = sent.lower()
    bad = set()
    bord = _obj(facts, "borders")
    if bord and re.search(r"\bborder", s):
        mentioned = {c for c in COUNTRIES if re.search(r"\b" + re.escape(c) + r"\b", s)} - {topic}
        bad |= (mentioned - bord - {w for b in bord for w in b.split()})
    cont = _obj(facts, "continent")
    if cont:
        for c in CONTINENTS:
            if re.search(r"\b" + re.escape(c) + r"\b", s) and c not in cont:
                bad.add(c)
    return bad


def _remove_bad_tokens(s, bad_tokens):
    """Remove each bad token (case-insensitive whole-word match) plus its connecting glue: a LEADING
    comma/"and" if one immediately precedes the token (mid-/end-of-list position), else a TRAILING
    comma/"and" if one immediately follows (start-of-list position), else nothing (the token was the
    sentence's only list item). Never both sides, so a middle list item's OWN separators are not
    double-consumed. Longest-token-first so a multi-word bad token (e.g. "north america") is matched whole
    before any single-word substring of it could be."""
    out = s
    lead_alt = r"(?P<lead>,\s*and\s+|\s+and\s+|,\s*)?"
    trail_alt = r"(?:,\s*and\s+|\s+and\s+|,\s*)?"
    for tok in sorted(bad_tokens, key=len, reverse=True):
        pattern = re.compile(
            lead_alt + r"\b" + re.escape(tok) + r"\b" + _MODIFIER + r"(?(lead)|" + trail_alt + r")",
            re.IGNORECASE)
        out = pattern.sub("", out, count=0)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    return out


def clause_filter_sentence(sent, topic, facts):
    """Returns the sentence to KEEP (possibly edited, dropping only the store-wrong span(s)), or None to
    drop it entirely -- the caller's existing contract (`s for s in sents if clause_filter_sentence(...)`
    style is NOT used; the caller must use the RETURNED text, since it may differ from `sent`).

    `facts` is the (relation, object) pair list `sentence_contradicts` itself expects (the SAME shape
    `_facts_as_relation_pairs` already produces in `webapp/open_ended_chat.py`)."""
    original = sent.strip()
    reason = sentence_contradicts(original, topic, facts)
    if reason is None:
        return original                                          # nothing wrong -- unchanged, byte-identical
    candidate = _RELATIVE_CLAUSE_RE.sub("", original)
    bad = _bad_relation_tokens(candidate, topic, facts)
    if bad:
        candidate = _remove_bad_tokens(candidate, bad)
    candidate = re.sub(r"\s{2,}", " ", candidate).strip(" ,").strip()
    if not candidate or candidate == original:
        return None                                                # neither repair changed anything
    if len(candidate.split()) < 2:
        return None                                                 # nothing meaningful survived
    if _DANGLING_END_RE.search(candidate) or _DANGLING_START_RE.search(candidate):
        return None                                                 # a dangling function word, not grammatical
    if sentence_contradicts(candidate, topic, facts) is not None:
        return None                                                 # defense-in-depth: still contradicts, don't leak
    return candidate


if __name__ == "__main__":
    # A tiny smoke demo (not the scored verify -- see
    # research/runners/_open_ended_clause_contradiction_filter_verify.py for the GO-gated measurement).
    FACTS = {"canada": [("borders", "united states"), ("capital", "ottawa"), ("continent", "north america")]}
    demo = [
        "Canada is bordered by the United States to the south and Mexico to the west",
        "The capital city of Canada is Ottawa, which was founded in 1867",
        "It has a population of around 35 million people, making it one of the largest countries in the world",
    ]
    for s in demo:
        print(repr(s), "->", repr(clause_filter_sentence(s, "canada", FACTS["canada"])))
