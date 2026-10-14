"""LBF row for Stage-1 C1 category-inheritance (the flip, RULING c1-flip-go, Claude 2026-10-11).

Measures whether the recall-miss organ is LOAD-BEARING for the flagged one-hop guess: the C1 context
(stage1_exit_gate._CAT_CONTEXT -- shared eat-patient + motion contexts, so the trout/salmon share the
bug and the lake) plus a bat that eats the bug and flies in the sky; `what does the bat swim` may then
come back as a FLAGGED guess `lake` (one hop: the bat's own told eat-bug is shared with the fish, so the
guess is the C1 inference itself, not a hop -- dev seeds 1-8: 8/8). With the organ's read-time lesion the
read admits nothing and the turn falls back to an honest abstain, so the probe's reply fields change.

FLIP STATUS: default-ON since the flip (webapp/category_inherit_chat.py::_CATEGORY_INHERIT_DEFAULT_ON,
hop attenuation likewise default-ON), so this row carries NO REQUIRED_ENV -- it joins the headline set
the moment research/c1-flip merges. The explicit BRAIN_CATEGORY_INHERIT=0 / HOP=0 escapes are the
byte-identical pre-flip paths (tests/test_category_inherit_organ.py pins them).

INTERFACE: `EXTRA_LESIONS`/`EXTRA_TURNS`/`EXTRA_PROBES` follow exactly the entry shapes of
`FACULTY_LESIONS`/`PROBE_TURNS`/`FACULTY_PROBES` (research/runners/load_bearing_fraction.py and
research/runners/onebrain_regression_battery.py). Per the LBF-ROW-INTERFACE rule this module does NOT
edit those registries directly -- the AG-REG import hook merges it in. EXTRA_TURNS must stay a LITERAL
list (the battery reads it with ast, never imported), hence the 11 _CAT_CONTEXT teaches spelled out here
instead of imported.
"""
from __future__ import annotations

# ── the lesion knob (the organ's own read-time transmission lesion; see its module docstring) ──
EXTRA_LESIONS = {
    "category-inherit": dict(
        flag="BRAIN_CATEGORY_INHERIT_LESION", value="1", kind="neural-lesion",
        note="webapp/category_inherit_chat.py wires BRAIN_CATEGORY_INHERIT_LESION into "
             "research/runners/category_inherit_organ.py, whose read-time transmission lesion (mode "
             "'all') admits no member->frame / frame->frame content, so a routable strict recall miss "
             "that would carry a FLAGGED guess falls back to an honest abstain. Flag-on only: with the "
             "wiring off (BRAIN_CATEGORY_INHERIT=0) the lesion knob changes nothing, exactly as before "
             "the flip. Confirmed present in source (2026-10-11)."),
}

# (label, message, session, reset, percept, rich) -- the PROBE_TURNS shape, one session, first resets.
# Must stay a LITERAL list (the battery reads it with ast).
EXTRA_TURNS = [
    # the C1 context, lifted verbatim from stage1_exit_gate._CAT_CONTEXT (two categories from shared
    # contexts only: eat-patient + motion context)
    ("t_robin_eat", "the robin eats the worm", "c1bridge", True, None, False),
    ("t_trout_eat", "the trout eats the bug", "c1bridge", False, None, False),
    ("t_robin_fly", "the robin flies in the sky", "c1bridge", False, None, False),
    ("t_trout_swim", "the trout swims in the lake", "c1bridge", False, None, False),
    ("t_sparrow_eat", "the sparrow eats the worm", "c1bridge", False, None, False),
    ("t_salmon_eat", "the salmon eats the bug", "c1bridge", False, None, False),
    ("t_sparrow_fly", "the sparrow flies in the sky", "c1bridge", False, None, False),
    ("t_salmon_swim", "the salmon swims in the lake", "c1bridge", False, None, False),
    ("t_finch_eat", "the finch eats the worm", "c1bridge", False, None, False),
    ("t_finch_fly", "the finch flies in the sky", "c1bridge", False, None, False),
    # the bridging member (one hop from the fish: shared told eat-bug)
    ("t_bat_eat", "the bat eats the bug", "c1bridge", False, None, False),
    ("t_bat_fly", "the bat flies in the sky", "c1bridge", False, None, False),
    ("t_canary_fly", "the canary flies in the sky", "c1bridge", False, None, False),
    ("p_canary_eat", "what does the canary eat", "c1bridge", False, None, False),
    ("c_canary_swim", "what does the canary swim", "c1bridge", False, None, False),
    ("c_bat_swim", "what does the bat swim", "c1bridge", False, None, False),
    ("c_robin_swim", "what does the robin swim", "c1bridge", False, None, False),
]

EXTRA_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin) -- reply fields only (docs/TERMS.md
    # `load-bearing`). The flagged guess rides hypothesis_svo (a HypothesisSVO subclass) with abstained
    # False, so all four reply fields are compared intact vs lesioned.
    ("category-inherit", "c_bat_swim", ["answer", "abstained", "recalled_svo", "hypothesis_svo"], False),
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
        "the probe turn is the one-hop bat swim": EXTRA_TURNS[-2][0] == "c_bat_swim"
                                                  and EXTRA_TURNS[-2][1] == "what does the bat swim",
        "context carries the shared bug-eaters and the lake": all(
            any(m == t[1] for t in EXTRA_TURNS) for m in
            ("the trout eats the bug", "the salmon eats the bug",
             "the trout swims in the lake", "the bat eats the bug")),
        "reply fields only, not thin": (set(EXTRA_PROBES[0][2]) ==
                                        {"answer", "abstained", "recalled_svo", "hypothesis_svo"}
                                        and EXTRA_PROBES[0][3] is False),
        "row is default-on (no REQUIRED_ENV)": "REQUIRED_ENV" not in globals(),
    }
    return {"ok": all(checks.values()), "checks": checks}
