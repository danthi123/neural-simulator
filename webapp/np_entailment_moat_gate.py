"""NP-ENTAILMENT MOAT GATE (2026-09-01, Vikunja board frontier row) -- wires NPHeadBinder (spiking
NP-boundary binding, `research/runners/_spiking_np_boundary_extraction_derisk.py`) + entailment
classification (`FactStore`/`classify_claim`, `research/runners/_open_text_moat_verifier_derisk.py`)
into the LIVE open-text moat verifier: `webapp.open_ended_chat.post_filter`'s KNOWN-topic path.

THE GAP THIS CLOSES (not a hollow checkbox flip -- see the finding for the measured proof). The
live known-topic filter (`_clause_filter_sentence` -> `sentence_contradicts`,
`research/runners/_open_ended_known_supplement_filter_derisk.py`) is a HOST GAZETTEER: it only
recognizes THREE relation shapes -- borders / continent / capital -- plus a bare number/year regex.
A fabricated supplement on ANY OTHER relation has no matching branch, so `sentence_contradicts`
returns None and the sentence survives unedited: "Mercury discovered Neptune" (store only holds
mercury/orbits/sun) trips zero gazetteer branches and leaks. This gate closes that class
GENERICALLY: NPHeadBinder + BridgeParser (both spiking, vocabulary-agnostic position x voice role
assignment -- BOTH reused UNCHANGED, no `sim/` edit) extract an (agent, action, patient) triple for
any clause `segment_clause` can segment (the exact-3-content-word fast path is fully general over
the VERB -- "discovered" needs no lexicon entry, unlike the gazetteer's fixed relation set), then
`classify_claim`/`FactStore` (the SAME entailment semantics production's single-triple
`ask_yes_no` already uses, reused UNCHANGED) checks it against the SAME retrieved facts the live
turn already holds. Any relation the extractor can parse becomes checkable, not just the three
gazetteer branches.

MONOTONIC / ADDITIVE-ONLY SCOPE (bounds the false-reject risk deliberately). This gate can only
ever DROP a sentence the earlier stages already kept -- it never restores or edits text an earlier
stage removed, and it is a no-op (passes the sentence through unchanged) whenever:
  (a) no clause in the sentence reduces to a frame `segment_clause` can close (most compound /
      listy / long descriptive sentences fall here -- an honest coverage limit, not a defect: it
      means this gate can UNDER-catch, never OVER-reject content it cannot confidently parse);
  (b) the parsed clause's subject does not normalize to the retrieved topic (the fact store handed
      to this gate holds ONLY that topic's facts, so a claim about a different entity is out of
      its adjudicable scope -- left to whatever else in the pipeline handles it);
  (c) the clause is a hedge/opinion (`is_opinion`, reused unchanged -- entailment does not apply to
      an explicitly flagged guess, the same carve-out `_moat_claim_entailment_derisk.py` documents);
  (d) the parsed verb is a COPULA ("is"/"are"/"was"/"were"). A copula's object is typically a rich,
      elaborative predicate nominal ("a vast country located in North America") that a single
      store object cannot be expected to exact-match -- checking it with the SAME strict
      `classify_claim` used for a discrete relation produces FALSE REJECTS on true, merely
      elaborated content (measured directly against the real saved known-topic Qwen replies during
      this wiring's own verify -- see the finding). Excluding copula keeps the gate's value-add
      (catching SPECIFIC wrong-relation/wrong-entity confabulations on non-copula verbs, the
      observed hallucination shape) without regressing genuinely-grounded descriptive prose.
This is a scope decision, not a weakening: it is the SAME "when unsure, don't touch it" posture the
gazetteer filter and the base post-filter already take on constructions outside their own coverage.

FLAG (default OFF -- `BRAIN_OPEN_ENDED_NP_ENTAILMENT`, see `webapp.open_ended_chat.
np_entailment_enabled`). Additive: `webapp.open_ended_chat.post_filter` only imports THIS module
inside that flag's truthy branch (mirroring the existing `wkv_mouth_generator` / gen-time-honesty
lazy-import pattern in the same file), so with the flag off this module -- and the heavier
`research.runners._spiking_np_boundary_extraction_derisk` / `brain_conversational_agent` imports
it in turn pulls in, including their own `os.environ.setdefault("SIM_BACKEND", "numpy")` -- is
NEVER imported and `post_filter` is BYTE-IDENTICAL to before this file existed.

WIDENED COVERAGE (2026-09-02 follow-on, board frontier row, `BRAIN_OPEN_ENDED_NP_ENTAILMENT_
COPULA_COVERAGE`, default OFF). The 2026-09-01 real-traffic moat-safety soak (`research/findings/
2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`) measured that scope-(d)'s
copula exclusion is the dominant real-traffic miss: NP-entailment changed ZERO of 12 real known-
topic Qwen replies, because free prose is dominated by copula ("is a professional football club")
that scope (d) declines to touch. The concrete miss: `castleford_f_c` gets called a "professional
football club" when the store's only sport fact is `rugby_leauge` -- a confident wrong-type
fabrication that leaked through unedited. This flag adds a SEPARATE, NARROW, ADDITIVE-ONLY check
(`_copula_wide_extract` + `_copula_category_conflict` below) that fires ONLY when ALL of the
following hold, keeping the false-reject risk that motivated scope (d) bounded rather than
reopening the whole copula construction to the strict single-object `classify_claim` match that
would false-reject genuinely elaborated true content (the SAME risk scope (d) was written to
avoid):
  1. a copula auxiliary (is/are/was/were) is found anywhere in the sentence, with the SUBJECT
     read as the sentence's leading comma-segment (tolerates a between-subject-and-copula
     appositive -- "Castleford FC, commonly known as Castleford F.C., is ..." -- which
     `split_clauses`'s blanket comma-split (used by the ORIGINAL per-clause loop below) severs,
     losing the subject before the per-clause path ever sees it);
  2. that subject SEPARATOR-INVARIANTLY matches the retrieved topic (`_normalize_loose`: collapses
     underscores/hyphens/periods/whitespace and drops a leading article before comparing, so a
     Wikidata-slug topic like `castleford_f_c` matches the human-readable subject text extraction
     pulls from real prose, `castleford fc` -- still an EXACT-identity match after the collapse,
     not a fuzzy one, same precision the pre-existing item (b) already relies on);
  3. the predicate is not a negation, present-participle ("is playing ...", progressive aspect,
     not an identity predicate), or passive past-participle ("is built ...", a different
     construction already handled by `segment_clause`'s own passive pass) -- these are left
     untouched, an honest residual, not a guess;
  4. the predicate names a CATEGORY WORD from a small, explicit, extensible lexicon
     (`_CATEGORY_WORDS`, today: common sport names -- the concrete measured miss's own domain) that
     CONFLICTS with a DIFFERENT, also-recognized, category word in a store fact for the SAME
     topic (`_copula_category_conflict`) -- e.g. predicate "football" vs stored `sport=rugby_
     leauge` (substring-matched, so the store's own "leauge" typo does not block the "rugby" hit).
An unrecognized predicate, or a store with no matching-family fact, never trips this (silently
returns False) -- it cannot false-reject content this lexicon has no opinion about, the same
"when unsure, don't touch it" posture scope (a)-(d) already take. This check can only ever
ADDITIONALLY drop a sentence the rest of the gate already kept (monotonic, same contract as the
gate's existing scope) -- it never restores text an earlier stage removed, and every other
construction (participial, pronoun-referent, and any copula clause the new checks above do not
positively conflict) is untouched, a named residual, not attempted here (see the finding).
With the flag OFF, `copula_coverage_enabled()` short-circuits before any of the above runs and
`gate_sentence` is BYTE-IDENTICAL to before this addition (verified directly, not assumed -- see
`research/runners/_np_entailment_copula_coverage_verify.py`).

Run the wiring verify: `python -m research.runners._np_entailment_moat_gate_wiring_verify`
Run the copula-coverage verify: `python -m research.runners._np_entailment_copula_coverage_verify`
"""
from __future__ import annotations

import os
import re
import threading

_COPULA_ACTIONS = {"is", "are", "was", "were"}

_LOCK = threading.Lock()
_PARSER = None
_NP_BINDER = None

# ---------------------------------------------------------------------------------------------
# WIDENED COVERAGE (flag-gated, additive-only) -- see module docstring for the full rationale.
# ---------------------------------------------------------------------------------------------

_FLAG_COPULA_COVERAGE = "BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE"


def copula_coverage_enabled() -> bool:
    """Truthy iff `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` is set to a non-empty,
    non-'0'/'false'/'no' value. Default OFF, matching the parent gate's own default-off
    convention (`webapp.open_ended_chat.np_entailment_enabled`)."""
    return os.environ.get(_FLAG_COPULA_COVERAGE, "").strip().lower() not in ("", "0", "false", "no")


_COPULA_RE = re.compile(r"\b(?:is|are|was|were)\b", re.IGNORECASE)
_CLAUSE_BOUNDARY_RE = re.compile(
    r",|\b(?:and|but|because|so|although|while|which)\b", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NEGATION_WORDS = {"not", "never", "n't"}
# Reuses the SAME passive-participle lexicon `_spiking_np_boundary_extraction_derisk.segment_clause`
# already uses for its own passive pass -- a predicate opening on one of these is a passive
# ("is built ..."), a different construction already handled elsewhere, not a predicate nominal.
_PASSIVE_PARTICIPLES = {"built", "discovered", "designed", "founded", "created", "constructed"}


def _normalize_loose(s) -> str:
    """Separator-invariant normalization for the WIDENED-COVERAGE subject/topic comparison ONLY
    (never used by the pre-existing per-clause path's `_normalize`, which stays byte-identical).
    Strips a leading article, then collapses underscores/hyphens/periods/apostrophes/parens/
    whitespace, so a Wikidata-slug topic ('castleford_f_c') can match the human-readable subject
    text extraction pulls from real prose ('Castleford FC' -> 'castleford fc'). Still an
    EXACT-identity comparison after the collapse -- not a fuzzy/partial match, the same precision
    the gate's existing `_normalize` already relies on for item (b)."""
    s = str(s).strip()
    s = _LEADING_ARTICLE_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"[\s_\-.'()]+", "", s)
    return s


def _has_negation(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", str(text).lower())
    return any(w in _NEGATION_WORDS or w.endswith("n't") for w in words)


def _copula_wide_extract(sent: str):
    """Whole-sentence copula extraction that tolerates a leading comma-appositive between the
    subject and the copula ('X, known as Y, is Z') -- which `split_clauses`'s blanket comma-split
    (used by the pre-existing per-clause loop below) would otherwise sever, losing the subject
    before that loop ever sees the copula clause. Host lexical segmentation only -- never decides
    which word plays which grammatical role (same boundary the rest of this module already draws
    around `segment_clause`/NPHeadBinder/BridgeParser).

    Returns (subject_text, predicate_text) using the FIRST copula auxiliary found, or None when:
    no copula is present; the subject or predicate span is empty; or the predicate opens on a
    negation, a present participle (progressive aspect, e.g. 'is playing football' -- not an
    identity predicate), or a passive past participle (a different construction, already handled
    by `segment_clause`'s own passive pass)."""
    m = _COPULA_RE.search(sent)
    if not m:
        return None
    before, after = sent[:m.start()], sent[m.end():]
    subject_text = before.split(",", 1)[0].strip()
    predicate_text = _CLAUSE_BOUNDARY_RE.split(after, maxsplit=1)[0].strip()
    if not subject_text or not predicate_text:
        return None
    if _has_negation(predicate_text):
        return None
    first_word = re.findall(r"[a-zA-Z']+", predicate_text.lower())
    first_word = first_word[0] if first_word else ""
    if first_word.endswith("ing") or first_word in _PASSIVE_PARTICIPLES:
        return None
    return subject_text, predicate_text


# Canonical CATEGORY word -> family name. Deliberately narrow + extensible: today only the
# concrete 2026-09-01 measured miss's own domain (sport). A conflict below fires ONLY when the
# copula predicate contains one of these words AND a store fact for the SAME topic contains a
# DIFFERENT word from the SAME family -- both recognized, both different -- never on an
# unrecognized word (an honest coverage limit, not a guess).
_CATEGORY_WORDS = {
    "football": "sport", "soccer": "sport", "rugby": "sport", "basketball": "sport",
    "baseball": "sport", "cricket": "sport", "hockey": "sport", "tennis": "sport",
    "golf": "sport", "boxing": "sport", "volleyball": "sport", "handball": "sport",
    "athletics": "sport", "swimming": "sport", "gymnastics": "sport", "netball": "sport",
}


def _category_hits(text: str) -> set:
    """Every (family, word) `_CATEGORY_WORDS` recognizes in `text`, word-boundary matched against
    a plain letters-only token stream so e.g. 'rugby_leauge' (the real store typo the 2026-09-01
    soak's data has) still hits 'rugby' (the token 'leauge' just does not hit anything else)."""
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return {(_CATEGORY_WORDS[w], w) for w in words if w in _CATEGORY_WORDS}


def _copula_category_conflict(predicate_text: str, facts: list, topic_norm_loose: str) -> bool:
    """True iff the copula predicate names a category (e.g. a sport) that CONFLICTS with a store
    fact for this topic naming a DIFFERENT word in the SAME family -- the concrete 2026-09-01 miss
    (Qwen: '... a professional football club'; store: sport=rugby_leauge). Conservative by
    construction: False unless BOTH sides hit a recognized, DIFFERENT word in the same family, so
    an unrecognized predicate or a store with no matching-family fact never trips this."""
    pred_hits = _category_hits(predicate_text)
    if not pred_hits:
        return False
    for (a, _v, p) in facts:
        if _normalize_loose(a) != topic_norm_loose:
            continue
        for fam, word in pred_hits:
            for fam2, word2 in _category_hits(p):
                if fam == fam2 and word != word2:
                    return True
    return False


def _get_spiking_pair(seed: int = 42):
    """Process-shared (BridgeParser, NPHeadBinder), built ONCE under a lock -- mirrors
    `webapp.open_ended_chat.get_generator`'s one-shared-object pattern. Both classes are reused
    UNCHANGED by import; nothing here re-implements or retrains either spiking mechanism."""
    global _PARSER, _NP_BINDER
    if _PARSER is not None and _NP_BINDER is not None:
        return _PARSER, _NP_BINDER
    with _LOCK:
        if _PARSER is None or _NP_BINDER is None:
            from research.runners.brain_conversational_agent import BridgeParser
            from research.runners._spiking_np_boundary_extraction_derisk import NPHeadBinder
            _PARSER = BridgeParser(seed=seed)
            _NP_BINDER = NPHeadBinder(seed=seed)
        return _PARSER, _NP_BINDER


def reset_spiking_pair_for_test():
    """Test-only: force the next `_get_spiking_pair()` call to rebuild (used by the wiring verify
    to exercise the lazy-build path deterministically; never called from the live server)."""
    global _PARSER, _NP_BINDER
    with _LOCK:
        _PARSER = None
        _NP_BINDER = None


def _normalize(s) -> str:
    return " ".join(str(s).strip().lower().split())


def gate_sentence(sent: str, topic: str, facts: list, parser=None, np_binder=None):
    """Screen `sent` (a sentence the existing known-topic string filter already KEPT) with the
    spiking NPHeadBinder-extraction + FactStore-entailment gate.

    Returns `sent` unchanged when the gate does not apply to (any clause of) it -- see the module
    docstring's scope list (a)-(d) -- or every adjudicable clause is grounded. Returns None (drop
    the WHOLE sentence) the first time an adjudicable clause is classified ungrounded/contradicted
    by the SAME `classify_claim` semantics production's single-triple moat already uses.

    `facts` is the SAME (agent, action, patient) triple list `webapp.open_ended_chat.answer_turn`
    already retrieved for this turn (one shared shape end to end -- no new adapter). `parser` /
    `np_binder` default to the process-shared pair (built lazily, once); passed explicitly by the
    wiring verify so it can reuse ONE built pair across many cases without rebuilding."""
    from research.runners._open_text_moat_verifier_derisk import (
        FactStore, Claim, classify_claim, split_clauses, is_opinion,
    )
    from research.runners._spiking_np_boundary_extraction_derisk import extract_svo_npbind

    if parser is None or np_binder is None:
        parser, np_binder = _get_spiking_pair()

    store = FactStore()
    for (a, v, p) in facts:
        store.store(_normalize(a), _normalize(v), _normalize(p))
    topic_norm = _normalize(topic)

    # ---- WIDENED COVERAGE (flag-gated, additive-only; byte-identical when off -- see module
    # docstring). Runs BEFORE the per-clause loop below because `split_clauses`'s blanket
    # comma-split (which that loop depends on) severs the subject from a copula predicate whenever
    # an appositive sits between them ("X, known as Y, is Z") -- this whole-sentence extraction is
    # the only path that ever sees that subject. Can only ADDITIONALLY drop; never restores.
    if copula_coverage_enabled():
        wide = _copula_wide_extract(sent)
        if wide is not None:
            subject_text, predicate_text = wide
            if _normalize_loose(subject_text) == _normalize_loose(topic):
                if _copula_category_conflict(predicate_text, facts, _normalize_loose(topic)):
                    return None                                   # widened-coverage: copula category conflict

    for clause in split_clauses(sent):
        if is_opinion(clause.lower()):
            continue                                         # (c) hedge/opinion -- out of entailment scope
        parsed, _meta = extract_svo_npbind(clause, parser, np_binder)
        if parsed is None:
            continue                                         # (a) not confidently parseable -- no-op, safe
        agent, action, patient, negated = parsed
        if _normalize(agent) != topic_norm:
            continue                                         # (b) not a claim about the retrieved topic
        if _normalize(action) in _COPULA_ACTIONS:
            continue                                         # (d) copula predicate nominal -- out of scope
        claim = Claim(text=clause, kind="assertion", agent=_normalize(agent), action=_normalize(action),
                      patient=_normalize(patient), negated=negated)
        verdict = classify_claim(claim, store)
        if verdict != "grounded":
            return None                                       # NPHeadBinder-extracted, entailment-ungrounded
    return sent
