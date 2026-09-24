"""LBF row for A5 (theory-of-mind false-belief chat wire, 2026-09-24 midnight plan S15a).

Exposes `EXTRA_LESIONS`/`EXTRA_PROBES` in the exact shapes
`research.runners.load_bearing_fraction.FACULTY_LESIONS`/`research.runners.onebrain_regression_battery.
FACULTY_PROBES` use, for a future "AG-REG import hook" to merge (NOT YET WRITTEN anywhere in the repo as of
2026-09-24 -- see `research/runners/lbf_rows/__init__.py`) — this module never edits those registries directly,
so this row is inert/unregistered until that hook lands.

Also exposes `EXTRA_TURNS` (same shape as `onebrain_regression_battery._EXTRA_TURNS`): the plan's row contract
names only EXTRA_LESIONS/EXTRA_PROBES, but this faculty's probe needs a turn label
(`onebrain_regression_battery.PROBE_TURNS`/`_TURN_BY_LABEL` do not have) that no other lane defines, and
`_EXTRA_TURNS` is the codebase's own existing precedent for exactly this need (labels reachable BY LABEL ONLY,
kept OUT of the default `PROBE_TURNS` roster so the regression battery and every flip-verify harness stay
byte-identical). Whoever wires the import hook should merge `EXTRA_TURNS` into `_TURN_BY_LABEL` the same way.

THE PROBE. One turn narrates the full classic Sally-Anne change-of-location script in a single message (the
organ's sentence-splitter folds each clause in order): Sally places the marble in the basket (witnessed) ->
Sally leaves the room -> Anne moves the marble to the box (UNwitnessed by Sally) -> "where will Sally look for
the marble?". Intact: the belief store answers "basket" (the stale, pre-move location) because the tracked
agent (Sally) did not witness the move. Lesioned (`BRAIN_FALSE_BELIEF_LESION=1`): the witnessing gate is
forced open, so the belief store tracks reality and answers "box" instead — the load-bearing diff this row
measures (`false_belief_tom.belief_location` flips `basket` (intact) -> `box` (lesion)).

THIS IS A CONDITIONAL, DEFAULT-OFF CAPABILITY (not yet a shipped faculty): `BRAIN_FALSE_BELIEF_CHAT=1` must be
present in BOTH arms for the row to be exercised at all (a prerequisite, not the lesion difference), the same
shape `LB_DA_TAG_CAPTURE_PROBE`'s `--extra-env BRAIN_DA_TAG_CAPTURE=1` uses in `load_bearing_fraction.py`'s own
docstring for another conditional flag. Run at production defaults (the flag unset), this row reads
not-exercised (`false_belief_tom` absent on both arms) — honest, since the capability has not shipped. Its
6-seed capability gate is deferred to B2b at the frozen SHA F per the midnight plan; this file only registers
the row shape.
"""
from __future__ import annotations

EXTRA_LESIONS = {
    "tom-false-belief": dict(
        flag="BRAIN_FALSE_BELIEF_LESION", value="1", kind="neural-lesion",
        note="forces the W3 register's witnessing gate open at write AND query "
             "(research/runners/tom_false_belief_chat_organ.py), collapsing the belief store onto reality. "
             "REQUIRES BRAIN_FALSE_BELIEF_CHAT=1 as a shared prerequisite in BOTH arms (a conditional "
             "default-OFF capability, not yet shipped default-on); at production defaults (flag unset) this "
             "row is not-exercised. Host residuals declared in the PRE-REGISTRATION: witnessing/presence is "
             "host-parsed (webapp/false_belief_chat.py's leave/return regex), the belief-location action read "
             "is a host argmax."),
}

# (faculty_key, turn_label, decision_field_paths, thin)
EXTRA_PROBES = [
    ("tom-false-belief", "tom_fb",
     ["false_belief_tom.acted", "false_belief_tom.belief_location", "false_belief_tom.reality_location"],
     False),
]

# (label, message, session, reset, percept, rich) -- see onebrain_regression_battery.PROBE_TURNS for the shape.
EXTRA_TURNS = [
    ("tom_fb",
     "Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
     "Where will Sally look for the marble?",
     "tomfb", True, None, False),
]
