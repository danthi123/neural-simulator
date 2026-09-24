"""A10 LBF row (midnight plan S15c, 2026-09-24): reward-value-spiking-afferent.

`BRAIN_REWARD_VALUE_AFFERENT` (master) drives the SNc reward/context afferent that `da-mode-drives-response`
folds into its engagement EMA from the surprise organ's confirm/violate rate instead of the host
`engagement_of()` scalar -- see `webapp/reward_value_afferent_chat.py` and
`research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md`.

This is an OPT-IN capability row: it reads NOTHING unless BOTH `BRAIN_DA_DRIVES=1` (the pre-existing
production anchor; usually already default-on) AND `BRAIN_REWARD_VALUE_AFFERENT=1` are set, so it must run
with `--extra-env BRAIN_REWARD_VALUE_AFFERENT=1` (plus `BRAIN_DA_DRIVES=1` if the anchor is off) -- NEVER
inside the default `adequate`/`thin` probe sets, exactly like the other S15 conditional flags
(BRAIN_AFFECT_MARKER_SETTLE, BRAIN_OPEN_ENDED_GATED, ...). The probe turn reuses the EXISTING `contra` turn
("the dog chase the fish", session "surp2") already declared in `onebrain_regression_battery.PROBE_TURNS` for
`surprise-monitor` -- no new turn needed; this row only adds a NEW field path on that same turn.

Per the LBF ROW INTERFACE (`research/runners/lbf_rows/__init__.py`): this module is inert until AG-REG's
import hook in `load_bearing_fraction.py` merges it. It does not edit FACULTY_LESIONS/FACULTY_PROBES.
"""
from __future__ import annotations

EXTRA_LESIONS = {
    "reward-value-spiking-afferent": dict(
        flag="BRAIN_REWARD_VALUE_LESION", value="1", kind="neural-lesion",
        note="cuts the spiking read (surprise-organ prediction edges, via the organ's OWN per-call lesioned "
             "twin) that BRAIN_REWARD_VALUE_AFFERENT=1 drives the SNc reward/context afferent from -- a "
             "CONFIRM and a CONTRADICT turn read the SAME (elevated, undifferentiated) rate under this lesion. "
             "OPT-IN: requires --extra-env BRAIN_DA_DRIVES=1 BRAIN_REWARD_VALUE_AFFERENT=1 (unset otherwise, "
             "so the default adequate/thin probe sets never exercise this row). See "
             "webapp/reward_value_afferent_chat.py."),
}

# (key, turn_label, fields, thin) -- exactly FACULTY_PROBES' 4-tuple shape (onebrain_regression_battery.py).
# `contra` ("the dog chase the fish", session "surp2") is an EXISTING PROBE_TURNS entry (surprise-monitor's own
# turn); this row reads NEW fields nested under the SAME response's da_drives key that turn already populates
# once BRAIN_DA_DRIVES is on.
EXTRA_PROBES = [
    ("reward-value-spiking-afferent", "contra",
     ["da_drives.reward_value.source", "da_drives.reward_value.normalized", "da_drives.reward_value.surprised",
      "da_drives.mode"],
     False),
]
