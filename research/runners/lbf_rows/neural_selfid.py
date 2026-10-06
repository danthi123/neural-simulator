"""LBF row: neural-selfid -- a self-referential factual question is comprehended by the on-brain parser, not the host
router (rank-13 BRAIN_NEURAL_SELFID; integration wave 2).

Packet: research/packets/integration-wave2/neural-selfid/HANDOFF.md. Pre-registration:
research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md. Ledger row: `content-selection` (its declared
residual: "still the self/identity + noisy-anaphora fallback" -- the host QuestionRouter answers every question that
names the brain itself). Mechanism de-risk: research/findings/2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md.

WHAT IT MEASURES, AT REPLY LEVEL. One turn, "what do you use?", in a fresh session. With BRAIN_NEURAL_SELFID=1 the
self-alias resolves to 'brain' before the self-alias block in `_extract_route`, so the question goes through the SAME
on-brain BridgeParser (`ChatBrain._neural_question_parse`, parser.role_of on Izhikevich neurons) and substrate recall
every other factual question uses; the reply states the built-in self fact (brain, use, spikes). The lesion arm adds
BRAIN_SUBSTRATE_PARSE_LESION=1, the content-selection row's own cut of that role-conjunction read: a declined parse
abstains honestly instead of falling to the router (the 2026-08-12 CHOOSE-1 contract). Load-bearing = the lesion
changes the REPLY. Without the flag the router owns this question and the same lesion leaves the reply unchanged (the
row is then not exercised: it is OPT-IN, below).

KIND. neural-lesion (the cut is a spiking read: the parser's role conjunction). Compared fields: the primary reply
fields `answer`, `abstained`, `recalled_svo` (docs/TERMS.md `load-bearing`).

OPT-IN until the flip (REQUIRED_ENV; the convention of research/runners/lbf_rows/reasoning_transitive_chat.py).
EXTRA_TURNS are read by onebrain_regression_battery with ast (never imported), so they must stay a literal list.
"""
from __future__ import annotations

# Attribution discipline (tools.lab; gate class AT): `expectation_holds` below reads the intact and lesion replies of
# this row's probe turn and makes the MOVE an executed check (`lever`) beside the row's own directional expectation,
# the pattern research/runners/lbf_rows/live_organs.py uses. The dev-seed read in the packet HANDOFF calls it.
from tools.lab import lever

# tools/integration_wave2_flip.py comments this line out when the packet flips (the row then joins the headline set).
# REQUIRED_ENV lifted by the iw2 flip (tools/integration_wave2_flip.py): REQUIRED_ENV = {"BRAIN_NEURAL_SELFID": "1"}

EXTRA_TURNS = [
    ("iw2si_use", "what do you use?", "iw2si", True, None, False),
]

EXTRA_LESIONS = {
    "neural-selfid": dict(
        flag="BRAIN_SUBSTRATE_PARSE_LESION", value="1", kind="neural-lesion",
        note="research/runners/brain_chat_tui.py `_substrate_parse_lesioned()`: BRAIN_SUBSTRATE_PARSE_LESION=1 cuts "
             "the on-brain BridgeParser's (position, voice)->role read in ChatBrain._neural_question_parse (the "
             "content-selection row's own cut). With BRAIN_NEURAL_SELFID=1 in both arms, 'what do you use?' is "
             "comprehended by that parser: intact states the self fact (brain use spikes), lesioned abstains. "
             "Probe = one fresh-session turn (session 'iw2si'). Reply fields only."),
}

EXTRA_PROBES = [
    ("neural-selfid", "iw2si_use", ["answer", "abstained", "recalled_svo"], False),
]


def expectation_holds(intact: dict, lesion: dict) -> dict:
    """Given the probe turn's intact and lesion reply dicts ({answer, abstained, recalled_svo}), return {"held",
    "moved", "detail"}: held = the intact reply states the self fact (answer names 'spikes', not abstained) AND the
    parse-lesioned reply abstains. `lever(required=False)` records whether the reply moved at all (a move in the wrong
    direction reads moved=True, held=False). A data check, never a claim; never raises."""
    try:
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("neural-selfid reply", ia, la, required=False)
        held = ("spikes" in ia.lower() and intact.get("abstained") is not True and lesion.get("abstained") is True)
        return {"held": bool(held), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed as held
        return {"held": False, "moved": None, "detail": "unreadable arm: %s" % e}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "one fresh session": len(EXTRA_TURNS) == 1 and EXTRA_TURNS[0][3] is True,
        "the probe names the brain itself": " you " in " %s " % EXTRA_TURNS[0][1].rstrip("?"),
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
        "lesion is a neural cut, not the flag itself": EXTRA_LESIONS["neural-selfid"]["flag"] != "BRAIN_NEURAL_SELFID",
    }
    return {"ok": all(checks.values()), "checks": checks}
