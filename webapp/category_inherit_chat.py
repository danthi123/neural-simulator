"""CATEGORY INHERITANCE on the strict route's recall miss (Stage-1 C1; default-OFF, `BRAIN_CATEGORY_INHERIT`).

Wires `research/runners/category_inherit_organ.CategoryInheritOrgan` (rung-3c's learned member->context pathway,
generalized to runtime vocabulary) into the live `/api/brain-chat` turn:

  * FEED: every fact the chat ACQUIRES this session (the `('done', svo)` acquisition branch of
    `ChatBrain.gate_extract`, AFFIRM polarity only) is observed by the organ (a within-turn Hebbian rehearsal).
  * READ: when the production gate stack returns None for a ROUTABLE, NON-ANAPHORA strict recall (`gate_extract`
    returned `('route', q, agent, action, False)` this turn: the substrate bus vetoed, the honest abstain), the
    organ is asked which object `agent` reaches for `action` through the learned pathway. A reached object is
    returned as an `InheritedSVO` -- a `HypothesisSVO` subclass, so every surface renders it as a FLAGGED guess
    ("Maybe the canary eats the worm -- that's a guess"), the rich path reports it as `hypothesis_svo` (never as
    `recalled_svo`), and it is never spoken as a recalled fact (the honesty boundary). No reached object -> None
    (the abstain is unchanged).

Installed by `webapp/server.py::brain_reply` OUTSIDE every other gate wrapper (after the multistep gate, before the
default-off open-ended-gated trace), so it acts only on the stack's final verdict and never changes a committed
answer. It wraps the INSTANCE's `gate_extract` to see each turn's route (every production bus calls
`chat.gate_extract`), and `gate` to act on the miss. Idempotent. Flag OFF (the default): `brain_reply` never imports
this module and nothing is wrapped -> byte-identical. The host-escape path (`BRAIN_GNW_BUS_HOST=1`, the original
`ChatBrain.gate`, which never calls `gate_extract`) is NOT covered: declared.

Lesion: `BRAIN_CATEGORY_INHERIT_LESION=1` (read-time transmission lesion inside the organ; see its module doc).
Prereg: research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md.
"""
from __future__ import annotations

import os

# PRODUCTION DEFAULT -- OFF. A flip needs the prereg's six-seed C1 PASS (flag on) plus the flip checks named in its
# HANDOFF; flipping is never done on this research branch.
_CATEGORY_INHERIT_DEFAULT_ON = False


def category_inherit_enabled() -> bool:
    v = os.environ.get("BRAIN_CATEGORY_INHERIT")
    if v is None:
        return _CATEGORY_INHERIT_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def _hypothesis_cls():
    from research.runners.brain_chat_tui import HypothesisSVO
    return HypothesisSVO


_INHERITED_CLS = None


def _inherited_cls():
    """`InheritedSVO`, built once: a `HypothesisSVO` subclass (so every surface renders it as a flagged guess and
    the rich path reports it as `hypothesis_svo`), carrying the organ read that produced it as `inherited_via`."""
    global _INHERITED_CLS
    if _INHERITED_CLS is None:
        HypothesisSVO = _hypothesis_cls()

        class InheritedSVO(HypothesisSVO):
            """[agent, action, patient] reached through the learned category pathway (an inference, a guess)."""
        _INHERITED_CLS = InheritedSVO
    return _INHERITED_CLS


def _make_inherited(agent, action, patient, trace):
    s = _inherited_cls()([agent, action, patient])
    s.inherited_via = trace
    return s


def _affirm_polarity(question) -> bool:
    """True unless the production B3 extractor reads the heard assertion as a NEGATION (a negated fact must not
    teach the organ that the agent does the thing). Mirrors `ChatBrain._maybe_acquire`'s own polarity read."""
    try:
        import research.runners.b3_noncontradiction_production_organ as _b3nc
        if not _b3nc.noncontradiction_enabled():
            return True
        parsed = _b3nc.extract_polar_assertion(str(question).strip())
        return parsed is None or str(parsed[3]).upper() == "AFFIRM"
    except Exception:
        return True


def get_organ(chat, seed):
    org = getattr(chat, "_category_inherit_organ", None)
    if org is None:
        from research.runners.category_inherit_organ import CategoryInheritOrgan
        org = CategoryInheritOrgan(seed=int(seed))
        chat._category_inherit_organ = org
    return org


def install(chat, *, seed: int = 42) -> bool:
    """Wrap `chat.gate_extract` (feed + route record) and `chat.gate` (act on the routable miss). Idempotent.
    Returns True if installed now."""
    if getattr(chat, "_category_inherit_installed", False):
        return False
    HypothesisSVO = _hypothesis_cls()
    orig_extract = chat.gate_extract
    orig_gate = chat.gate

    def _ci_gate_extract(question):
        res = orig_extract(question)
        chat._ci_last_extract = res
        try:
            if (category_inherit_enabled() and isinstance(res, tuple) and res and res[0] == "done"
                    and res[1] is not None and not isinstance(res[1], HypothesisSVO)
                    and len(res[1]) == 3 and all(isinstance(x, str) for x in res[1])
                    and _affirm_polarity(question)):
                a, v, p = res[1]
                tr = get_organ(chat, seed).observe(a, v, p)
                chat._ci_last_observe = dict(tr, svo=[a, v, p])
        except Exception as e:                          # never let the organ crash a turn
            chat._ci_last_observe = {"error": "%s: %s" % (type(e).__name__, e)}
        return res

    def _ci_gate(question):
        chat._ci_last_extract = None
        chat._ci_last_observe = None
        chat._last_category_inherit = None
        svo = orig_gate(question)
        if svo is not None or not category_inherit_enabled():
            return svo
        ext = getattr(chat, "_ci_last_extract", None)
        if not (isinstance(ext, tuple) and len(ext) == 5 and ext[0] == "route" and ext[4] is False):
            return svo                                   # not a direct routable strict recall: untouched
        _, _q, agent, action, _anaph = ext
        org = getattr(chat, "_category_inherit_organ", None)
        if org is None:                                  # nothing was ever taught this session -> nothing to reach
            chat._last_category_inherit = {"agent": agent, "action": action, "patient": None,
                                           "reason": "organ_not_built"}
            return svo
        try:
            out = org.infer(agent, action)
        except Exception as e:
            chat._last_category_inherit = {"agent": agent, "action": action, "patient": None,
                                           "reason": "error: %s: %s" % (type(e).__name__, e)}
            return svo
        out = dict(out, agent=agent, action=action)
        chat._last_category_inherit = out
        if out.get("patient"):
            return _make_inherited(agent, action, out["patient"], out)
        return svo

    chat.gate_extract = _ci_gate_extract
    chat.gate = _ci_gate
    chat._category_inherit_installed = True
    return True


def turn_trace(chat):
    """The per-turn trace attached to the response as `category_inherit` (flag on only)."""
    return {"observe": getattr(chat, "_ci_last_observe", None),
            "read": getattr(chat, "_last_category_inherit", None)}


def turn_stamp(chat):
    """RULING c1-stamp-fix: the trace only when the organ engaged this turn (an observe or read
    record exists), None otherwise — the server attaches `category_inherit` exactly when the
    pathway fired, so a MISSING stamp always means it never fired."""
    tr = turn_trace(chat)
    if tr.get("observe") is not None or tr.get("read") is not None:
        return tr
    return None
