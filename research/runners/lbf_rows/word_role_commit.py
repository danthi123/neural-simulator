"""LBF rows: word-role-commit -- the word-role parse is decided by a SPIKING COMMIT stage, not a host count rule
(packet word-role-commit, plan docs/drafts/spiking-commit/PLAN.md application A1; organ
research/runners/word_role_parser_organ.py built with COMMIT_GRID params).

WHAT IT MEASURES, AT REPLY LEVEL. With BRAIN_WORD_ROLE_PARSE=1 alone, the word-role organ computes role-ensemble
spike counts and a host rule (ROLE_MIN, a 2x margin, then a per-role argmax over words) turns them into the parse
(RULING word-role-verify-go: credit PARTIAL). With BRAIN_WORD_ROLE_COMMIT=1 the same organ carries, per role, a
mutually exclusive commit gate and a bistable role-slot latch; a word's role is the latch that turned on while it was
presented, and the host rule is not called. The lesion BRAIN_WORD_ROLE_COMMIT_LESION=1 closes the role -> commit-gate
transmission only: the role ensembles still fire as in the intact arm, nothing commits, the parse declines and the
chat abstains. So a reply change under this lesion is caused by the commit stage, the part that replaced the host
rule. Load-bearing = the lesion changes the REPLY (docs/TERMS.md `load-bearing`).

THE PROBE. One turn, "what do you usually use?", in a fresh session (content words [brain, usually, use] after the
host stop-word strip and self-alias resolution): intact commits brain -> agent and use -> action ("usually", never
heard, latches nothing) and states the built-in self fact (brain, use, spikes); the commit lesion abstains.

THE CONTROL (`word-role-commit-nofact-ctl`). "what do you usually drink?": 'drink' was never heard, so no action latch
turns on, the intact parse declines and the chat abstains (the no-confab moat); the lesion predicts the same abstain.

KIND. neural-lesion (the cut is the role -> commit-gate synapses). Compared fields: `answer`, `abstained`,
`recalled_svo`. OPT-IN (REQUIRED_ENV): the three flags are set in BOTH arms of a shard (lb_shard --extra-env).
EXTRA_TURNS are read by onebrain_regression_battery with ast (never imported), so they must stay a literal list.

NAMED RESIDUALS. As word-role-parse (training stimuli transcribed from the composer's fact index; self-alias
resolution and the stop-word strip stay host; word form -> assembly dictionary), plus: the read of WHICH slot latch is
on is a host level read of spike counts (registered threshold-invariant at x0.5 / x2), and the inter-word reset keeps
the latches (a declared compression of the inter-word interval).
"""
from __future__ import annotations

from tools.lab import lever

# Opt-in until a flip: the rows join the headline set only by a later packet.
# REQUIRED_ENV lifted by the iw2 flip (tools/integration_wave2_flip.py): REQUIRED_ENV = {"BRAIN_NEURAL_SELFID": "1", "BRAIN_WORD_ROLE_PARSE": "1", "BRAIN_WORD_ROLE_COMMIT": "1"}

EXTRA_TURNS = [
    ("s2wc_use", "what do you usually use?", "s2wc", True, None, False),
    ("s2wc_nf", "what do you usually drink?", "s2wcnf", True, None, False),
]

EXTRA_LESIONS = {
    "word-role-commit-selfid": dict(
        flag="BRAIN_WORD_ROLE_COMMIT_LESION", value="1", kind="neural-lesion",
        note="research/runners/brain_chat_tui.py `_word_role_commit_lesioned()`: BRAIN_WORD_ROLE_COMMIT_LESION=1 "
             "closes the role -> commit-gate transmission of the word-role organ's spiking commit stage "
             "(research/runners/word_role_parser_organ.py, COMMIT_GRID). With BRAIN_NEURAL_SELFID=1, "
             "BRAIN_WORD_ROLE_PARSE=1 and BRAIN_WORD_ROLE_COMMIT=1 in both arms, 'what do you usually use?' commits "
             "(brain, use): intact states the self fact (brain use spikes), lesioned abstains. Probe = one "
             "fresh-session turn (session 's2wc'). Reply fields only."),
    "word-role-commit-nofact-ctl": dict(
        flag="BRAIN_WORD_ROLE_COMMIT_LESION", value="1", kind="neural-lesion",
        note="CONTROL for word-role-commit-selfid: 'what do you usually drink?' (unheard modifier and action). Needs "
             "the three flags in both arms. Predicted: intact abstains (the moat); the lesion leaves it."),
}

EXTRA_PROBES = [
    ("word-role-commit-selfid", "s2wc_use", ["answer", "abstained", "recalled_svo"], False),
    ("word-role-commit-nofact-ctl", "s2wc_nf", ["answer", "abstained", "recalled_svo"], False),
]

EXPECT = {
    "word-role-commit-nofact-ctl": {"abstains": True, "must": [], "must_not": []},
}


def expectation_holds(key: str, intact: dict, lesion: dict) -> dict:
    """{"held", "moved", "detail"} for a row's intact and lesion reply dicts. Capability row: held = the intact reply
    states the self fact ('spikes', not abstained) AND the lesioned reply abstains. Control row: held = the intact reply
    meets EXPECT AND the lesion did NOT move it. A data check, never a claim; never raises."""
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
        "new turns and sessions carry the s2wc prefix": all(t[0].startswith("s2wc") and t[2].startswith("s2wc")
                                                            for t in EXTRA_TURNS),
        "the self probe reads (you, usually, use) before the self-alias resolves": content["s2wc_use"] ==
        ["you", "usually", "use"],
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
        "lesion is the commit-stage cut": all(v["flag"] == "BRAIN_WORD_ROLE_COMMIT_LESION"
                                              and v["kind"] == "neural-lesion" for v in EXTRA_LESIONS.values()),
        "row is default-on (REQUIRED_ENV lifted at the flip)": "REQUIRED_ENV" not in globals(),
        "the lesion flag resolves in the chat source": "BRAIN_WORD_ROLE_COMMIT_LESION" in text,
        "the control is the only EXPECT row": set(EXPECT) == {"word-role-commit-nofact-ctl"},
    }
    return {"ok": all(checks.values()), "checks": checks}
