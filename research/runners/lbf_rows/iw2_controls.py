"""LBF rows: the integration-wave-2 PERMUTED CONTROLS (research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md).

Each wave-2 packet's reply-level gate (step R) pairs its capability row with one control row run in the SAME shard
job file, at the SAME pin, with the SAME packet flag on. A control changes one thing about the probe so that the
capability's answer must NOT appear, and predicts that the packet's lesion leaves the reply unchanged (S2 `pass`):

  tom-true-belief-ctl      (packet false-belief-chat)  the same story, but Sally never leaves the room, so she SEES the
                           move: the intact reply must name the TRUE location ("box") -- an organ that always named
                           the first location, or never tracked witnessing, fails here. The lesion (witnessing gate
                           forced open) predicts the same "box": reply unchanged.
  transitive-reversed-ctl  (packet transitive-chat)    the chain taught in REVERSE (river -> worm -> ball -> cat),
                           the same question "does cat precede river?": not derivable, so the intact reply must abstain -- a
                           route that answered "yes" to the question's shape (a memorized/hardcoded pair) fails here.
                           The lesion predicts the same abstain.
  selfid-nofact-ctl        (packet neural-selfid)      a self question with no stored self fact ("what do you
                           drink?"): the substrate route must abstain (the no-confab moat; the 2026-09-05 seed-43
                           defect was a self-alias over-fire that confabulated a self fact). The lesion predicts the
                           same abstain.

These rows are NEVER part of a production headline battery: REQUIRED_ENV needs LB_IW2_CONTROLS=1 (an LB_* probe flag,
which the job guards allow, set only by the wave-2 step-R job lines), and tools/integration_wave2_flip.py never lifts it.
The packet flag each control needs is set by the same job line (--extra-env). Scored by tools/iw2_score.py R (the
`control` entry of each packet). EXTRA_TURNS must stay a literal list (the battery reads it with ast, never imports).
"""
from __future__ import annotations

# Attribution discipline (tools.lab; gate class AT): `expectation_holds` below reads a control row's intact and lesion
# replies and makes the (predicted: absent) MOVE an executed check (`lever`) beside the control's own expectation, the
# pattern research/runners/lbf_rows/live_organs.py uses. The dev-seed read in the packet HANDOFF calls it; the
# registered verdict is tools/iw2_score.py's control_cell.
from tools.lab import lever

REQUIRED_ENV = {"LB_IW2_CONTROLS": "1"}

EXTRA_TURNS = [
    ("iw2ctl_tb", "Sally puts the marble in the basket. Anne moves the marble to the box. "
                  "Where will Sally look for the marble?", "iw2ctltb", True, None, False),
    ("iw2ctl_rv1", "the river precedes the worm", "iw2ctlrv", True, None, False),
    ("iw2ctl_rv2", "the worm precedes the ball", "iw2ctlrv", False, None, False),
    ("iw2ctl_rv3", "the ball precedes the cat", "iw2ctlrv", False, None, False),
    ("iw2ctl_rv", "does cat precede river?", "iw2ctlrv", False, None, False),
    ("iw2ctl_nf", "what do you drink?", "iw2ctlnf", True, None, False),
]

EXTRA_LESIONS = {
    "tom-true-belief-ctl": dict(
        flag="BRAIN_FALSE_BELIEF_LESION", value="1", kind="neural-lesion",
        note="CONTROL for tom-false-belief (iw2): a WITNESSED move (Sally never leaves). Needs BRAIN_FALSE_BELIEF_CHAT=1 "
             "in both arms. Predicted: intact names the true location (box); the witnessing-gate lesion leaves it."),
    "transitive-reversed-ctl": dict(
        flag="BRAIN_TRANSITIVE_LESION", value="1", kind="neural-lesion",
        note="CONTROL for reasoning-transitive-chat (iw2): the chain taught in reverse; 'does cat precede river?' is not "
             "derivable. Needs BRAIN_TRANSITIVE_CHAT=1 in both arms. Predicted: intact abstains; the lesion leaves it."),
    "selfid-nofact-ctl": dict(
        flag="BRAIN_SUBSTRATE_PARSE_LESION", value="1", kind="neural-lesion",
        note="CONTROL for neural-selfid (iw2): a self question with no stored self fact. Needs BRAIN_NEURAL_SELFID=1 in "
             "both arms. Predicted: intact abstains (the moat); the parse lesion leaves it."),
}

EXTRA_PROBES = [
    ("tom-true-belief-ctl", "iw2ctl_tb", ["answer", "abstained", "recalled_svo"], False),
    ("transitive-reversed-ctl", "iw2ctl_rv", ["answer", "abstained", "recalled_svo"], False),
    ("selfid-nofact-ctl", "iw2ctl_nf", ["answer", "abstained", "recalled_svo"], False),
]


# the control predictions (tools/iw2_score.py PACKETS[...]["control"] holds the registered copy; a test pins agreement)
EXPECT = {
    "tom-true-belief-ctl": {"abstains": False, "must": ["box"], "must_not": ["basket"]},
    "transitive-reversed-ctl": {"abstains": True, "must": [], "must_not": []},
    "selfid-nofact-ctl": {"abstains": True, "must": [], "must_not": []},
}


def expectation_holds(key: str, intact: dict, lesion: dict) -> dict:
    """Given a control row's intact and lesion reply dicts, return {"held", "moved", "detail"}: held = the intact reply
    meets the control's prediction (EXPECT[key]) AND the lesion did NOT move the reply. `lever(required=False)` records
    the move. A data check, never a claim; never raises."""
    try:
        e = EXPECT[key]
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("%s reply" % key, ia, la, required=False)
        ok = (intact.get("abstained") is True) if e["abstains"] else (intact.get("abstained") is not True and bool(ia))
        ok = ok and all(w in ia.lower() for w in e["must"]) and not any(w in ia.lower() for w in e["must_not"])
        return {"held": bool(ok and not moved), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as ex:  # noqa: BLE001 -- reported, never swallowed as held
        return {"held": False, "moved": None, "detail": "unreadable arm: %s" % ex}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    from webapp.false_belief_chat import _split_sentences, classify
    from webapp.reasoning_transitive_chat import parse_transitive_question
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    by = {t[0]: t for t in EXTRA_TURNS}
    tb_kinds = [(classify(s) or {}).get("kind") for s in _split_sentences(by["iw2ctl_tb"][1])]
    rv_taught = {t[1] for t in EXTRA_TURNS if t[2] == "iw2ctlrv" and t[0] != "iw2ctl_rv"}
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turns reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "each session's first turn resets, no other does": all(
            t[3] == (t[0] == min((u for u in EXTRA_TURNS if u[2] == t[2]), key=EXTRA_TURNS.index)[0])
            for t in EXTRA_TURNS),
        "true-belief story is PLACE, MOVE, QUERY (the agent never leaves)": tb_kinds == ["place", "move", "query"],
        "reversed chain: the probe is a transitive question": parse_transitive_question(by["iw2ctl_rv"][1]) is not None,
        "reversed chain: never teaches a fact from cat": not any(s.startswith("the cat ") for s in rv_taught),
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
        "opt-in by an LB_* probe flag only": set(REQUIRED_ENV) == {"LB_IW2_CONTROLS"},
    }
    return {"ok": all(checks.values()), "checks": checks}
