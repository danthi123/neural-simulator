"""LBF row for A6 / reasoning-transitive-chat (midnight-plan 2026-09-24, S15(b)).
See research/findings/2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md and webapp/reasoning_transitive_chat.py.

INTERFACE: `EXTRA_LESIONS`/`EXTRA_PROBES` follow exactly the entry shapes of `FACULTY_LESIONS`/`FACULTY_PROBES` in
research/runners/load_bearing_fraction.py. Per the LBF-ROW-INTERFACE rule, this module does NOT edit those
registries directly -- the AG-REG import hook merges it in.

HONEST GAP (thin, named, not hidden): the turn this row needs (`transitive_nonadjacent`, a "does e0 precede e3?"
query run AFTER teaching the 4-link chain e0->e1->e2->e3->e4) is NOT YET a member of the shared PROBE_TURNS /
_TURN_BY_LABEL registry in `research/runners/onebrain_regression_battery.py` -- this lane's mandate is scoped to
`lbf_rows/*` only, never that shared file. `_PROPOSED_PROBE_TURNS` below is the EXACT tuple set (same 6-tuple
shape: label, message, session, reset, percept, rich) a future AG-REG pass can copy verbatim into `PROBE_TURNS` to
make this row non-thin. Until then this row is registered `thin=True`, and the seed-7 de-risk evidence for the
mechanism lives in the row's own standalone gate runner (`research/runners/reasoning_transitive_chat_gate.py`),
which drives the SAME production `webapp.server.brain_chat` handler directly (not through this shared harness).
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

# ── the probe (thin until the shared turn registry carries `transitive_nonadjacent`; see the module docstring) ──
EXTRA_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin)
    ("reasoning-transitive-chat", "transitive_nonadjacent",
     ["derived", "abstained", "derived_from"], True),
]

# The exact PROBE_TURNS-shape tuples (label, message, session, reset, percept, rich) a future AG-REG pass can copy
# into research/runners/onebrain_regression_battery.py::PROBE_TURNS (consecutively, same session, in this order)
# to make the row above non-thin. Requires BRAIN_TRANSITIVE_CHAT=1 in the arm's env (this row is an OPT-IN
# capability probe, like open-ended-generation/prospective-memory, not a default-ON faculty).
_PROPOSED_PROBE_TURNS = [
    ("transitive_teach1", "e0 precede e1",       "transchain", True,  None, False),
    ("transitive_teach2", "e1 precede e2",       "transchain", False, None, False),
    ("transitive_teach3", "e2 precede e3",       "transchain", False, None, False),
    ("transitive_teach4", "e3 precede e4",       "transchain", False, None, False),
    ("transitive_nonadjacent", "does e0 precede e3?", "transchain", False, None, False),
]
