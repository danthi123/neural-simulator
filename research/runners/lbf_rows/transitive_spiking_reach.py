"""LBF rows for the spiking-reachability packet `transitive-spiking-reach` (session-2 packet factory, 2026-10-01).
See research/runners/transitive_spiking_reach_organ.py and webapp/reasoning_transitive_chat.py (the default-OFF
`BRAIN_TRANSITIVE_SPIKING_REACH` route).

Two rows, both discovered by the lbf_rows import hook (no registry edit):
  transitive-spiking-reach        the capability row: teach the 3-link 'precede' chain cat -> ball -> worm -> river, ask the
                                  never-told "does cat precede river?". With the route on, the "yes" is the spike count of
                                  the organ's `detect` population, not a host `target in path` test. The lesion
                                  (BRAIN_TRANSITIVE_REACH_LESION=1) closes the organ's `reach_hop` transmission gate, so
                                  the wave cannot leave the start assembly and the reply flips to an honest abstain.
  transitive-spiking-reversed-ctl the control row: the chain taught in REVERSE (river -> worm -> ball -> cat), the same
                                  question. Not derivable, so the intact reply must abstain, and the lesion leaves it.

Turn labels and session names are prefixed `s2tr` so nothing collides with reasoning-transitive-chat or iw2_controls.
OPT-IN rows (REQUIRED_ENV): they enter the registry only when BRAIN_TRANSITIVE_CHAT=1 and
BRAIN_TRANSITIVE_SPIKING_REACH=1 are set in both arms (lb_shard --extra-env). EXTRA_TURNS must stay a LITERAL list (the
battery reads it with ast and never imports this module).
"""
from __future__ import annotations

REQUIRED_ENV = {"BRAIN_TRANSITIVE_CHAT": "1", "BRAIN_TRANSITIVE_SPIKING_REACH": "1"}

# (label, message, session, reset, percept, rich) -- the PROBE_TURNS shape. Two sessions; each first turn resets.
EXTRA_TURNS = [
    ("s2tr_teach1", "the cat precedes the ball", "s2trchain", True, None, False),
    ("s2tr_teach2", "the ball precedes the worm", "s2trchain", False, None, False),
    ("s2tr_teach3", "the worm precedes the river", "s2trchain", False, None, False),
    ("s2tr_probe", "does cat precede river?", "s2trchain", False, None, False),
    ("s2tr_rv1", "the river precedes the worm", "s2trrev", True, None, False),
    ("s2tr_rv2", "the worm precedes the ball", "s2trrev", False, None, False),
    ("s2tr_rv3", "the ball precedes the cat", "s2trrev", False, None, False),
    ("s2tr_rv", "does cat precede river?", "s2trrev", False, None, False),
]

EXTRA_LESIONS = {
    "transitive-spiking-reach": dict(
        flag="BRAIN_TRANSITIVE_REACH_LESION", value="1", kind="neural-lesion",
        note="webapp/reasoning_transitive_chat.py: BRAIN_TRANSITIVE_REACH_LESION=1 closes the `reach_hop` transmission "
             "gate of research/runners/transitive_spiking_reach_organ.py, so the activity wave cannot leave the start "
             "assembly, match[target] never sees wave input, `detect` is silent and a derivable non-adjacent 'does A "
             "precede D?' flips from derived=True to an honest abstain. Needs BRAIN_TRANSITIVE_CHAT=1 and "
             "BRAIN_TRANSITIVE_SPIKING_REACH=1 in both arms. The adjacent-pair short-circuit (a plain recall) is "
             "unaffected."),
    "transitive-spiking-reversed-ctl": dict(
        flag="BRAIN_TRANSITIVE_REACH_LESION", value="1", kind="neural-lesion",
        note="CONTROL for transitive-spiking-reach: the chain taught in reverse; 'does cat precede river?' is not "
             "derivable. Needs BRAIN_TRANSITIVE_CHAT=1 and BRAIN_TRANSITIVE_SPIKING_REACH=1 in both arms. Predicted: "
             "intact abstains; the lesion leaves it."),
}

EXTRA_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin) -- reply fields only (docs/TERMS.md `load-bearing`)
    ("transitive-spiking-reach", "s2tr_probe", ["answer", "abstained", "recalled_svo"], False),
    ("transitive-spiking-reversed-ctl", "s2tr_rv", ["answer", "abstained", "recalled_svo"], False),
]

# the control prediction (the reversed chain is not derivable: abstain, lesion leaves it)
EXPECT = {
    "transitive-spiking-reach": {"abstains": False},
    "transitive-spiking-reversed-ctl": {"abstains": True},
}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    from webapp.reasoning_transitive_chat import parse_transitive_question
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    by = {t[0]: t for t in EXTRA_TURNS}
    sessions = {t[2] for t in EXTRA_TURNS}
    taught = {s: [t for t in EXTRA_TURNS if t[2] == s and not parse_transitive_question(t[1])] for s in sessions}
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turns reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "labels and sessions are s2tr-prefixed": (all(t[0].startswith("s2tr") for t in EXTRA_TURNS)
                                                  and all(s.startswith("s2tr") for s in sessions)),
        "each session's first turn resets, no other does": all(
            t[3] == (t is next(u for u in EXTRA_TURNS if u[2] == t[2])) for t in EXTRA_TURNS),
        "both probes are the same transitive question": (parse_transitive_question(by["s2tr_probe"][1])
                                                         == parse_transitive_question(by["s2tr_rv"][1])
                                                         == ("cat", "precede", "river")),
        "no teach turn is a transitive question": all(len(v) == 3 for v in taught.values()),
        "forward chain is never taught directly": not any("cat precedes the river" in t[1] for t in EXTRA_TURNS),
        "reversed chain never teaches a fact from cat": not any(t[1].startswith("the cat ") for t in taught["s2trrev"]),
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
        "opt-in by both packet flags": set(REQUIRED_ENV) == {"BRAIN_TRANSITIVE_CHAT", "BRAIN_TRANSITIVE_SPIKING_REACH"},
        "lesion flag is the reach lesion": all(v["flag"] == "BRAIN_TRANSITIVE_REACH_LESION"
                                               for v in EXTRA_LESIONS.values()),
    }
    return {"ok": all(checks.values()), "checks": checks}
