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

Run the wiring verify: `python -m research.runners._np_entailment_moat_gate_wiring_verify`
"""
from __future__ import annotations

import threading

_COPULA_ACTIONS = {"is", "are", "was", "were"}

_LOCK = threading.Lock()
_PARSER = None
_NP_BINDER = None


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
