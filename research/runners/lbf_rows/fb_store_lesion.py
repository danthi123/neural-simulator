"""LBF row: does the false-belief chat's spiking belief STORE carry its reply? (cloud Claude session 7, 2026-10-05.)

Prereg: research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md. The registered `tom-false-belief` row's lesion
(BRAIN_FALSE_BELIEF_LESION) forces the host witnessing gate open: it overrides a HOST decision and never cuts the store
(review of research/findings/2026-10-05-iw2-false-belief-chat-default-on-flip-validated-PROVISIONAL.md). This row cuts
the store itself: BRAIN_FALSE_BELIEF_STORE_LESION=1 builds the belief store with its assembly self-recurrence set to 0
(world and self stores untouched; webapp/false_belief_chat.py `_new_organ`). The reply reads `answer`; an empty or tied
store now abstains (webapp/false_belief_chat.py `belief_store_read_ok`) instead of answering the host argmax's slot-0
default, so a store that no longer holds the belief is visible at reply level.

OPT-IN (REQUIRED_ENV): a measurement row, outside the headline set until Claude rules on its six-seed result.
Engagement: the lesion arm's `false_belief_tom.store_lesioned` is True (absent in the intact arms).
EXTRA_TURNS: none; the probe turn `tom_fb` is the one research/runners/lbf_rows/tom_false_belief_chat.py declares.
"""
from __future__ import annotations

# Attribution discipline (tools.lab; gate class AT): `expectation_holds` makes the lesion's MOVE an executed check
# (`lever`) beside the prereg's reading, the pattern research/runners/lbf_rows/neural_selfid.py uses.
from tools.lab import lever

REQUIRED_ENV = {"LB_FB_STORE_LESION_PROBE": "1"}

EXTRA_LESIONS = {
    "tom-false-belief-store": dict(
        flag="BRAIN_FALSE_BELIEF_STORE_LESION", value="1", kind="neural-lesion",
        note="cuts the belief store's own assembly self-recurrence (research/runners/_false_belief_register_derisk.py "
             "build_tom_bridge belief_attractor_weight=0.0); the witnessing gate, the world and self stores and the "
             "host write are untouched. Same probe as tom-false-belief (the classic unwitnessed move). Engagement: "
             "false_belief_tom.store_lesioned is True on the lesion arm. Prereg "
             "research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md."),
}

EXTRA_PROBES = [
    ("tom-false-belief-store", "tom_fb", ["answer", "abstained", "recalled_svo"], False),
]


def expectation_holds(intact: dict, lesion: dict) -> dict:
    """Given the probe turn's intact and lesion reply dicts, return {"store_needed", "moved", "detail"}: store_needed =
    the intact reply names the stale location ("basket") AND the store-lesioned reply no longer does (abstains or
    names another location). `lever(required=False)` records whether the reply moved at all. The prereg predicts
    store_needed False on this probe. A data check, never a claim; never raises."""
    try:
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("fb store-lesion reply", ia, la, required=False)
        needed = ("basket" in ia.lower() and intact.get("abstained") is not True
                  and (lesion.get("abstained") is True or "basket" not in la.lower()))
        return {"store_needed": bool(needed), "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed as a result
        return {"store_needed": None, "moved": None, "detail": "unreadable arm: %s" % e}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "the lesion is not the gate override": EXTRA_LESIONS["tom-false-belief-store"]["flag"]
        != "BRAIN_FALSE_BELIEF_LESION",
        "opt-in": bool(REQUIRED_ENV),
    }
    return {"ok": all(checks.values()), "checks": checks}
