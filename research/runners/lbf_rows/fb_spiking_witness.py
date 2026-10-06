"""LBF rows: packet fb-spiking-witness (session-2 packet factory, 2026-10-01).

Prereg: research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md. HANDOFF: docs/drafts/fb-spiking-witness/.

The false-belief chat with its witnessing decided by the spiking `presence` flip-flop
(research/runners/tom_spiking_witness_organ.py) instead of the host's `_Scenario.present` dict. Both rows need
BRAIN_FALSE_BELIEF_CHAT=1 and BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1 in BOTH arms (REQUIRED_ENV; set by the step-R job
lines with --extra-env, together with BRAIN_FALSE_BELIEF_WITNESS_OP=<the operating point step 1 selected>), so at
production defaults they read not-exercised and never join a headline battery until packet P4 flips the anchor.

  fb-spiking-witness            the classic unwitnessed move. Intact: "basket" (the belief the presence population
                                kept from being overwritten). Lesion (BRAIN_FALSE_BELIEF_WITNESS_LESION=1: presence's
                                influence removed, every move written): "box".
  fb-spiking-true-belief-ctl    Sally never leaves, so presence is ON at the move: intact names the TRUE location
                                ("box") and the lesion predicts the same "box" (reply unchanged).

EXTRA_TURNS must stay a literal list (the battery reads it with ast, never imports).
"""
from __future__ import annotations

# Attribution discipline (tools.lab; gate class AT): `expectation_holds` below makes the lesion's MOVE an executed check
# (`lever`) beside each row's directional expectation, the pattern lbf_rows/iw2_controls.py uses.
from tools.lab import lever

REQUIRED_ENV = {"BRAIN_FALSE_BELIEF_CHAT": "1", "BRAIN_FALSE_BELIEF_SPIKING_WITNESS": "1"}

EXTRA_TURNS = [
    ("s2fb_fb", "Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
                "Where will Sally look for the marble?", "s2fbfb", True, None, False),
    ("s2fb_tb", "Sally puts the marble in the basket. Anne moves the marble to the box. "
                "Where will Sally look for the marble?", "s2fbtb", True, None, False),
]

EXTRA_LESIONS = {
    "fb-spiking-witness": dict(
        flag="BRAIN_FALSE_BELIEF_WITNESS_LESION", value="1", kind="neural-lesion",
        note="removes the spiking presence population's influence on the belief write "
             "(research/runners/tom_spiking_witness_organ.py: relay mode builds without presence->relay and drives "
             "the relays tonically; readout mode ignores the presence read): every move is written, so the belief "
             "store tracks reality. Needs BRAIN_FALSE_BELIEF_CHAT=1 and BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1 in both "
             "arms. Residuals declared in the prereg: the sentence classifier, agent-name routing, the location "
             "argmax and (readout mode) the threshold read that sets the W3 write gate."),
    "fb-spiking-true-belief-ctl": dict(
        flag="BRAIN_FALSE_BELIEF_WITNESS_LESION", value="1", kind="neural-lesion",
        note="CONTROL for fb-spiking-witness: a witnessed move (Sally never leaves). Predicted: intact names the true "
             "location (box); the lesion leaves the reply unchanged."),
}

EXTRA_PROBES = [
    ("fb-spiking-witness", "s2fb_fb", ["answer", "abstained", "recalled_svo"], False),
    ("fb-spiking-true-belief-ctl", "s2fb_tb", ["answer", "abstained", "recalled_svo"], False),
]


# the predictions (tools/s2p_score.py PACKETS["fb-spiking-witness"] holds the registered copy)
EXPECT = {
    "fb-spiking-witness": {"abstains": False, "must": ["basket"], "must_not": ["box"], "lesion_moves": True},
    "fb-spiking-true-belief-ctl": {"abstains": False, "must": ["box"], "must_not": ["basket"], "lesion_moves": False},
}


def expectation_holds(key: str, intact: dict, lesion: dict) -> dict:
    """Given a row's intact and lesion reply dicts, return {"held", "moved", "detail"}. The capability row holds when
    the intact reply states its content AND the lesion MOVED the reply; the control holds when the intact reply meets
    its prediction AND the lesion did NOT move it. `lever(required=False)` makes the move an executed check (gate class
    AT, the lbf_rows/iw2_controls.py pattern). The registered verdict is tools/s2p_score.py R; this is a data check
    the HANDOFF's dev reads may call. Never raises."""
    try:
        e = EXPECT[key]
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("%s reply" % key, ia, la, required=False)
        ok = (intact.get("abstained") is True) if e["abstains"] else (intact.get("abstained") is not True and bool(ia))
        ok = ok and all(w in ia.lower() for w in e["must"]) and not any(w in ia.lower() for w in e["must_not"])
        held = ok and (bool(moved) if e["lesion_moves"] else not moved)
        return {"held": bool(held), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as ex:  # noqa: BLE001 -- reported, never swallowed as held
        return {"held": False, "moved": None, "detail": "unreadable arm: %s" % ex}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from webapp.false_belief_chat import _split_sentences, classify
    kinds = {t[0]: [(classify(s) or {}).get("kind") for s in _split_sentences(t[1])] for t in EXTRA_TURNS}
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "every probe turn is defined here": all(r[1] in {t[0] for t in EXTRA_TURNS} for r in EXTRA_PROBES),
        "each probe is one fresh session": all(t[3] is True for t in EXTRA_TURNS),
        "capability story is PLACE, LEAVE, MOVE, QUERY": kinds["s2fb_fb"] == ["place", "leave", "move", "query"],
        "control story is PLACE, MOVE, QUERY": kinds["s2fb_tb"] == ["place", "move", "query"],
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
    }
    return {"ok": all(checks.values()), "checks": checks}
