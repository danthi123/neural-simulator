"""LBF row: d6-hebbian-write -- the in-conversation fact WRITE by the local Hebbian rule (integration wave 1).

Packet: research/packets/integration-wave1/d6-hebbian-write/HANDOFF.md. Pre-registration:
research/findings/2026-09-29-iw1-d6-hebbian-write-flip-PREREGISTRATION.md. Ledger row: `in-loop-learning` (its named
residual: the live chat-teach write is a host copy into the weights; research/runners/d6_hebbian_store.py replaces it).

WHAT IT MEASURES, AT REPLY LEVEL. The D6 capability gate v3 protocol (research/runners/d6_learn_through_use_lb.py,
GO 6/6, research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md) moved into the standard
load-bearing battery: teach "the wolf hunts the deer" -> two intervening recalls of boot facts -> "what does the wolf
hunt". The compared fields are the three PRIMARY REPLY fields (docs/TERMS.md `load-bearing`): `answer`, `abstained`,
`recalled_svo`. Intact = the three anchored D6 flags on; lesion = the same flags plus BRAIN_D6_HEBBIAN_FREEZE=1 (eta=0
for in-conversation writes only: same input, same host code, the encoding episode runs, no synapse changes -- K6 of
the capability gate verified the lesion is write-only). The capability gate predicts intact recalls deer, lesion
abstains.

KIND. "neural-lesion": BRAIN_D6_HEBBIAN_FREEZE is a dedicated lesion knob that cuts the neural WRITE while the organ
stays installed and the flags stay on -- not a master-switch disable (the whether-disable class would be
BRAIN_D6_HEBBIAN_STORE=0). Its name lacks the usual `_LESION` suffix; the semantics are the lesion class's.

OPT-IN until the flip. REQUIRED_ENV keeps this row out of every production-default battery while the anchor
`research/runners/d6_hebbian_store._D6_LEARN_DEFAULT_ON` is False; a capability shard sets the three flags with
`tools/lb_shard.py jobs ... --extra-env BRAIN_D6_HEBBIAN_STORE=1 BRAIN_D6_ENGRAM_VOCAB=1 BRAIN_D6_ENGRAM_READTIME=1`.
The flip script (tools/integration_wave1_flip.py) deletes the REQUIRED_ENV line in the same commit that flips the
anchor, so the row joins the headline registry exactly when the capability becomes the default.

EXTRA_TURNS are read by onebrain_regression_battery with ast (never imported), so they must stay a literal list.
"""
from __future__ import annotations

REQUIRED_ENV = {"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1"}

EXTRA_TURNS = [
    ("iw1d6_teach", "the wolf hunts the deer", "iw1d6", True, None, False),
    ("iw1d6_d1", "what does the cat eat", "iw1d6", False, None, False),
    ("iw1d6_d2", "what does the dog chase", "iw1d6", False, None, False),
    ("iw1d6_probe", "what does the wolf hunt", "iw1d6", False, None, False),
]

EXTRA_LESIONS = {
    "d6-hebbian-write": dict(
        flag="BRAIN_D6_HEBBIAN_FREEZE", value="1", kind="neural-lesion",
        note="research/runners/d6_hebbian_store.py hebbian_freeze_lesioned(): eta=0 for IN-CONVERSATION writes only "
             "(ChatBrain._maybe_acquire's conversation_write context); build-time facts and the read path are "
             "untouched. Probe = the capability gate v3 protocol (teach -> d1 -> d2 -> probe, session 'iw1d6'); "
             "with the write the probe recalls the taught fact, with the freeze it abstains (K3 of the 6/6 gate); "
             "the load-bearing runner attributes the difference. Reply fields only."),
}

EXTRA_PROBES = [
    ("d6-hebbian-write", "iw1d6_probe", ["answer", "abstained", "recalled_svo"], False),
]


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "turns share one session, first resets": (len({t[2] for t in EXTRA_TURNS}) == 1 and EXTRA_TURNS[0][3]
                                                  and not any(t[3] for t in EXTRA_TURNS[1:])),
        "reply fields only": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} for r in EXTRA_PROBES),
        "lesion is the freeze knob": EXTRA_LESIONS["d6-hebbian-write"]["flag"] == "BRAIN_D6_HEBBIAN_FREEZE",
    }
    return {"ok": all(checks.values()), "checks": checks}
