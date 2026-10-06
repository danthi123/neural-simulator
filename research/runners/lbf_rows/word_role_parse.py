"""LBF rows: word-role-parse -- the question parser's (agent, action) is read from the WORDS on spiking neurons
(packet word-role-parse, session 2; organ research/runners/word_role_parser_organ.py).

WHAT IT MEASURES, AT REPLY LEVEL. The integration-wave-2 review ruled the self-ID faculty (BRAIN_NEURAL_SELFID)
"the same positional role map reaching self questions": `ChatBrain._neural_question_parse` presents
(content[0], content[1], "__q__") to `BridgeParser.role_of(pos)`, a (position, voice) -> role conjunction that never
sees a word. With
BRAIN_WORD_ROLE_PARSE=1 the roles come from the words: word and position assemblies project onto agent/action/patient
ensembles through Hebbian-trained synapses (lateral inhibition between the ensembles), so a modifier the brain never
heard drops out and the next word keeps its role.

THE PROBE. One turn, "what do you usually use?", in a fresh session. `_extract_route` strips what/do (its _STOP set) and
resolves the self-alias, so the content words are [brain, usually, use]. "usually" SURVIVES the strip (it is not a
stop-word). The positional map reads (agent, action) = (brain, usually) and misses the self fact; the word-reading organ
gives "usually" no role, reads (brain, use) and the reply states the built-in self fact (brain, use, spikes). The lesion
arm adds BRAIN_WORD_ROLE_LESION=1, which closes the word -> role transmission gate: position alone is subthreshold, the
parse declines and the chat abstains. Load-bearing = the lesion changes the REPLY (docs/TERMS.md `load-bearing`).

THE CONTROL (`word-role-nofact-ctl`). "what do you usually drink?": no stored self fact has a 'drink' action, and the
brain never heard the word, so the intact organ must decline and the chat abstain (the no-confab moat); the lesion
predicts the same abstain, so the reply is unchanged.

KIND. neural-lesion (the cut is a spiking read: the word -> role synapses). Compared fields: the primary reply fields
`answer`, `abstained`, `recalled_svo`. OPT-IN until the flip (REQUIRED_ENV, the convention of
lbf_rows/neural_selfid.py): both flags are set in BOTH arms of a shard (lb_shard --extra-env). EXTRA_TURNS are
read by onebrain_regression_battery with ast (never imported), so they must stay a literal list.

NAMED RESIDUALS. The organ's training stimuli are transcribed from the composer's fact index (host transcription); the
self-alias -> 'brain' resolution (QuestionRouter._resolve_self) and the stop-word strip stay host.
"""
from __future__ import annotations

# Attribution discipline (tools.lab; gate class AT): `expectation_holds` below makes the MOVE an executed check
# (`lever`) beside the row's own directional expectation, the pattern lbf_rows/neural_selfid.py uses.
from tools.lab import lever

# Opt-in until the packet flips: comment this line out then (the rows join the headline set).
REQUIRED_ENV = {"BRAIN_NEURAL_SELFID": "1", "BRAIN_WORD_ROLE_PARSE": "1"}

EXTRA_TURNS = [
    ("s2wr_use", "what do you usually use?", "s2wr", True, None, False),
    ("s2wr_nf", "what do you usually drink?", "s2wrnf", True, None, False),
]

EXTRA_LESIONS = {
    "word-role-selfid": dict(
        flag="BRAIN_WORD_ROLE_LESION", value="1", kind="neural-lesion",
        note="research/runners/brain_chat_tui.py `_word_role_lesioned()`: BRAIN_WORD_ROLE_LESION=1 closes the word -> "
             "role transmission gate of the spiking word-role organ (research/runners/word_role_parser_organ.py) read "
             "by ChatBrain._neural_question_parse. With BRAIN_NEURAL_SELFID=1 and BRAIN_WORD_ROLE_PARSE=1 in both "
             "arms, 'what do you usually use?' parses to (brain, use): intact states the self fact (brain use "
             "spikes), lesioned abstains. Probe = one fresh-session turn (session 's2wr'). Reply fields only."),
    "word-role-nofact-ctl": dict(
        flag="BRAIN_WORD_ROLE_LESION", value="1", kind="neural-lesion",
        note="CONTROL for word-role-selfid: a self question with an unheard modifier and an unheard action ('what do "
             "you usually drink?'). Needs BRAIN_NEURAL_SELFID=1 and BRAIN_WORD_ROLE_PARSE=1 in both arms. Predicted: "
             "intact abstains (the moat); the lesion leaves it."),
}

EXTRA_PROBES = [
    ("word-role-selfid", "s2wr_use", ["answer", "abstained", "recalled_svo"], False),
    ("word-role-nofact-ctl", "s2wr_nf", ["answer", "abstained", "recalled_svo"], False),
]

# the control prediction (the same shape as lbf_rows/iw2_controls.py EXPECT)
EXPECT = {
    "word-role-nofact-ctl": {"abstains": True, "must": [], "must_not": []},
}


def expectation_holds(key: str, intact: dict, lesion: dict) -> dict:
    """Given a row's intact and lesion reply dicts ({answer, abstained, recalled_svo}), return {"held", "moved",
    "detail"}. Capability row (`word-role-selfid`): held = the intact reply states the self fact (answer names 'spikes',
    not abstained) AND the lesioned reply abstains. Control row: held = the intact reply meets EXPECT AND the lesion did
    NOT move the reply. `lever(required=False)` records the move. A data check, never a claim; never raises."""
    try:
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("%s reply" % key, ia, la, required=False)
        if key in EXPECT:
            e = EXPECT[key]
            ok = (intact.get("abstained") is True) if e["abstains"] else (
                intact.get("abstained") is not True and bool(ia))
            ok = ok and all(w in ia.lower() for w in e["must"]) and not any(w in ia.lower() for w in e["must_not"])
            return {"held": bool(ok and not moved), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
        held = ("spikes" in ia.lower() and intact.get("abstained") is not True and lesion.get("abstained") is True)
        return {"held": bool(held), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed as held
        return {"held": False, "moved": None, "detail": "unreadable arm: %s" % e}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    import os
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    by = {t[0]: t for t in EXTRA_TURNS}
    # the stop-word strip of ChatBrain._extract_route (a copy for the shape check; the strip itself stays host)
    stop = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
            "to", "it", "that", "this", "they", "them", "of", "about"}
    content = {k: [w for w in (t.lower().strip(".,!?") for t in by[k][1].split()) if w and w not in stop] for k in by}
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_chat_tui.py")
    try:
        text = open(src, encoding="utf8").read()
    except OSError:
        text = ""
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turns reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "each session's first turn resets, no other does": all(
            t[3] == (t[0] == min((u for u in EXTRA_TURNS if u[2] == t[2]), key=EXTRA_TURNS.index)[0])
            for t in EXTRA_TURNS),
        "new turns and sessions carry the s2wr prefix": all(t[0].startswith("s2wr") and t[2].startswith("s2wr")
                                                            for t in EXTRA_TURNS),
        "every probe names the brain itself": all(" you " in " %s " % t[1].rstrip("?") for t in EXTRA_TURNS),
        "the modifier 'usually' survives the stop-word strip": all("usually" in c for c in content.values()),
        "the self probe reads (brain, usually, use) after the self-alias resolves": content["s2wr_use"] ==
        ["you", "usually", "use"],
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
        "lesion is a neural cut, not a parse flag": all(v["flag"] == "BRAIN_WORD_ROLE_LESION"
                                                        and v["kind"] == "neural-lesion"
                                                        for v in EXTRA_LESIONS.values()),
        "opt-in by the selfid + word-role flags": REQUIRED_ENV == {"BRAIN_NEURAL_SELFID": "1",
                                                                   "BRAIN_WORD_ROLE_PARSE": "1"},
        "the lesion flag resolves in the chat source": "BRAIN_WORD_ROLE_LESION" in text,
        "the control is the only EXPECT row": set(EXPECT) == {"word-role-nofact-ctl"},
    }
    return {"ok": all(checks.values()), "checks": checks}
