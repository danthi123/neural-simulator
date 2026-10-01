"""LBF row for A6 / reasoning-transitive-chat (midnight-plan 2026-09-24, S15(b)).
See research/findings/2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md and webapp/reasoning_transitive_chat.py.

INTERFACE: `EXTRA_LESIONS`/`EXTRA_PROBES` follow exactly the entry shapes of `FACULTY_LESIONS`/`FACULTY_PROBES` in
research/runners/load_bearing_fraction.py. Per the LBF-ROW-INTERFACE rule, this module does NOT edit those
registries directly -- the AG-REG import hook merges it in.

NON-THIN SINCE INTEGRATION WAVE 2 (2026-09-30, packet research/packets/integration-wave2/transitive-chat/, prereg
research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md). The row's turns are now declared as a
literal EXTRA_TURNS list (read by research/runners/onebrain_regression_battery.py with ast, never imported), so the
battery can play the whole same-session group (teach a 3-link 'precede' chain cat -> ball -> worm -> river, then ask
the never-told non-adjacent "does cat precede river?") and compare the probe turn's REPLY fields. The 2026-09-24 turns
used the tokens e0..e4; through the real handler (dev seed 7, 2026-09-30) the acquisition path splits them to the
unknown word 'e' ("I don't know the words 'e' or 'e' yet") and stores nothing, so every probe abstained in every arm.
The chain is now made of words the tiny-demo brain has, alternating animate and inanimate nouns: two animate nouns
around a verb ("the cat chases the bird") are held back by the comprehension monitor's other-repair ("which of them is
the 'chase' done to?") and never stored. Before this the turns lived in a `_PROPOSED_PROBE_TURNS` list the hook could not reach, so
the row was PARKED (lbf_rows/__init__.py). The mechanism
is unchanged; its standalone gate runner (research/runners/reasoning_transitive_chat_gate.py) drives the same
production `webapp.server.brain_chat` handler through the World-A/World-B/lesion battery (its 2026-09-24 seed-7 run
was stopped before completion; the wave-2 dev-seed-7 evidence is in the packet HANDOFF).
"""
from __future__ import annotations

# ── the lesion knob (reuses the keystone's own already-verified assembly-self-recurrence lesion; see
#    webapp/reasoning_transitive_chat.py::transitive_lesion_on -> multistep_chase(..., lesion=True)) ──
EXTRA_LESIONS = {
    "reasoning-transitive-chat": dict(
        flag="BRAIN_TRANSITIVE_LESION", value="1", kind="neural-lesion",
        note="webapp/reasoning_transitive_chat.py: BRAIN_TRANSITIVE_LESION=1 forces the reused keystone chase "
             "(webapp.gnw_multistep_deliberation.multistep_chase) to run on the GNW workspace built with its "
             "assembly self-recurrence ZEROED (webapp/gnw_deliberation.py::_get_bridge(seed, lesion=True)) -- the "
             "SAME lesion mechanism `gnw-multistep-deliberation` already uses. Collapses ignition at hop 0, so a "
             "genuinely-derivable non-adjacent 'does A precede D?' can no longer chain past the direct-fact check "
             "and flips from derived=True to an honest abstain. The ADJACENT-pair short-circuit "
             "(query_patient(agent,relation)==target, a plain recall, not the chase) is UNAFFECTED by this lesion "
             "-- isolating the cut to the multi-hop contribution specifically, not the underlying recall "
             "primitive. Confirmed present in source (webapp/reasoning_transitive_chat.py)."),
}

# ── the probe ──
# OPT-IN row (2026-09-24): it measures a default-OFF capability, so it enters the registry only when BRAIN_TRANSITIVE_CHAT=1 is
# set in both arms, i.e. in a capability battery launched with tools/lb_shard.py ... --extra-env BRAIN_TRANSITIVE_CHAT=1.
# tools/integration_wave2_flip.py comments this line out when the packet flips (the row then joins the headline set).
REQUIRED_ENV = {'BRAIN_TRANSITIVE_CHAT': "1"}

# (label, message, session, reset, percept, rich) -- the PROBE_TURNS shape. One session; the first turn resets it.
# Must stay a LITERAL list (the battery reads it with ast).
EXTRA_TURNS = [
    ("transitive_teach1", "the cat precedes the ball", "transchain", True, None, False),
    ("transitive_teach2", "the ball precedes the worm", "transchain", False, None, False),
    ("transitive_teach3", "the worm precedes the river", "transchain", False, None, False),
    ("transitive_nonadjacent", "does cat precede river?", "transchain", False, None, False),
]

EXTRA_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin) -- reply fields only (docs/TERMS.md `load-bearing`)
    ("reasoning-transitive-chat", "transitive_nonadjacent", ["answer", "abstained", "recalled_svo"], False),
]


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    from webapp.reasoning_transitive_chat import parse_transitive_question
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "turns share one session, first resets": (len({t[2] for t in EXTRA_TURNS}) == 1 and EXTRA_TURNS[0][3]
                                                  and not any(t[3] for t in EXTRA_TURNS[1:])),
        "the probe turn is a transitive yes/no question": parse_transitive_question(EXTRA_TURNS[-1][1]) is not None,
        "no teach turn is a transitive question": not any(parse_transitive_question(t[1]) for t in EXTRA_TURNS[:-1]),
        "the probe pair is never taught directly": not any(t[1].startswith("the cat precedes the river")
                                                           for t in EXTRA_TURNS[:-1]),
        "the probe needs 3 hops": parse_transitive_question(EXTRA_TURNS[-1][1]) == ("cat", "precede", "river"),
        "reply fields only, not thin": all(set(r[2]) == {"answer", "abstained", "recalled_svo"} and r[3] is False
                                           for r in EXTRA_PROBES),
    }
    return {"ok": all(checks.values()), "checks": checks}
