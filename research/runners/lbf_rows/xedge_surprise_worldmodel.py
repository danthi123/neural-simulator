"""LBF row: xedge-surprise-worldmodel -- cross-edge C1, D2 surprise gates an online update of the E2 world model.

Packet: research/packets/integration-wave1/xedge-surprise-worldmodel/HANDOFF.md. Pre-registration:
research/findings/2026-09-29-iw1-xedge-surprise-worldmodel-flip-PREREGISTRATION.md. Ledger row: `worldmodel-forward`
(its declared residual: the affective forward model is TRAINED then FROZEN at build, "teacher-driven, not
self-organized from conversation"; the C1 edge lets the model's own spiking prediction-error unit gate one Hebbian
co-fire step of its state->pred transition toward the observed valence on each surprising turn).

WHAT IT MEASURES, AT REPLY LEVEL. The C1 wiring finding (research/findings/2026-09-02-c1-surprise-worldmodel-wired-flip-
ready.md) showed the edge GROWS on real alternating-valence traffic and the prediction margin shifts, but never
measured a reply. Here: a positive turn, then eight alternating negative/positive turns (each violates the
persistence expectation the previous turn left, so the gate can open turn after turn -- the C1 verify's own ARM B
design and texts), then "what do you expect?". The reply to that query is the model's expectation read-out, which
states the predicted sign and the pool-rate margin (worldmodel_production_organ.expectation_readout). Compared
fields: the primary reply fields `answer`, `abstained`, `recalled_svo` (docs/TERMS.md `load-bearing`).

KIND. "whether-disable": the lesion arm sets BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=0, so the two arms differ ONLY
in whether the gated update runs; the frozen organ, the D2 notice and every other read are the same code in both.
The module's neural lesion (BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION=1) zeroes the obs->surprise edges of the
circuit the D2 notice also reads, so it would silence the notice on the eight sequence turns as well as the update:
it cannot attribute a reply change to the learning alone, so it is not used here.

OPT-IN until the flip (REQUIRED_ENV; see research/runners/lbf_rows/d6_hebbian_write.py for the convention).
EXTRA_TURNS are read by onebrain_regression_battery with ast (never imported), so they must stay a literal list.
"""
from __future__ import annotations

REQUIRED_ENV = {"BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL": "1"}

EXTRA_TURNS = [
    ("iw1swm_p0", "that was wonderful, I am so happy", "iw1swm", True, None, False),
    ("iw1swm_n1", "that was terrible, I am so sad", "iw1swm", False, None, False),
    ("iw1swm_p1", "that was wonderful, I am so happy", "iw1swm", False, None, False),
    ("iw1swm_n2", "that was terrible, I am so sad", "iw1swm", False, None, False),
    ("iw1swm_p2", "that was wonderful, I am so happy", "iw1swm", False, None, False),
    ("iw1swm_n3", "that was terrible, I am so sad", "iw1swm", False, None, False),
    ("iw1swm_p3", "that was wonderful, I am so happy", "iw1swm", False, None, False),
    ("iw1swm_n4", "that was terrible, I am so sad", "iw1swm", False, None, False),
    ("iw1swm_p4", "that was wonderful, I am so happy", "iw1swm", False, None, False),
    ("iw1swm_ask", "what do you expect?", "iw1swm", False, None, False),
]

EXTRA_LESIONS = {
    "xedge-surprise-worldmodel": dict(
        flag="BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL", value="0", kind="whether-disable",
        note="research/runners/onebrain_xedge_surprise_worldmodel_production.py xedge_surprise_worldmodel_enabled(): "
             "0 turns the D2-gated online update of the live E2 organ's state->pred transition off; nothing else "
             "differs between the arms. Probe = +, then 8 alternating -/+ turns, then 'what do you expect?' (session "
             "'iw1swm'); the expectation read-out states the predicted sign and pool-rate margin. Reply fields only."),
}

EXTRA_PROBES = [
    ("xedge-surprise-worldmodel", "iw1swm_ask", ["answer", "abstained", "recalled_svo"], False),
]


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    from research.runners.worldmodel_production_organ import is_expectation_query
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "turns share one session, first resets": (len({t[2] for t in EXTRA_TURNS}) == 1 and EXTRA_TURNS[0][3]
                                                  and not any(t[3] for t in EXTRA_TURNS[1:])),
        "the probe turn is an expectation query": is_expectation_query(EXTRA_TURNS[-1][1]),
        "no sequence turn is an expectation query": not any(is_expectation_query(t[1]) for t in EXTRA_TURNS[:-1]),
        "reply fields only": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} for r in EXTRA_PROBES),
        "lesion is the master switch at 0 (whether-disable)":
            EXTRA_LESIONS["xedge-surprise-worldmodel"]["value"] == "0"
            and EXTRA_LESIONS["xedge-surprise-worldmodel"]["kind"] == "whether-disable",
    }
    return {"ok": all(checks.values()), "checks": checks}
