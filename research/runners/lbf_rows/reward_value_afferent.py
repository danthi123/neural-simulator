"""A10 LBF row (midnight plan S15c, 2026-09-24; fix round): surprise-salience-snc-afferent.

`BRAIN_REWARD_VALUE_AFFERENT=1` replaces da-mode-drives-response's per-turn engagement mix with the surprise organ's
spiking mismatch read on expectation-bearing assertion turns (an UNSIGNED prediction-error salience, not a reward
value -- hence the row key). See webapp/reward_value_afferent_chat.py and
research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md (+ AMENDMENT-1).

PROBE TURN: `confirm` ("the dog chase the cat", session "surp", already in onebrain_regression_battery.PROBE_TURNS).
Not `contra`: the finding 2026-09-20-hollow-surprise-monitor-confirm-probe established that the surprise lesion only
bites on a CONFIRM trial (asserted == stored, the shared block the prediction inhibits). On CONTRADICT the asserted
block is un-inhibited in the intact arm too, so the lesion changes nothing there (seed 7, first A10 run: contra
normalized 0.958 intact vs 0.969 lesioned, `surprised` True in both arms).

COMPARED FIELDS are decision fields only: `reward_value.source`, `reward_value.surprised`, `da_drives.mode` and
`da_drives.lead` (the reply suffix the mode selects). The continuous `reward_value.normalized` is NOT compared: it is
not in the battery's `_NOISE_FIELDS`, `compare()` uses exact `!=`, and a ~0.01 float jitter would read as a change
while every decision field is equal.

OPT-IN, IMPLEMENTED HERE (not assumed). The row can only exercise anything when the harness process runs with
`BRAIN_REWARD_VALUE_AFFERENT` truthy (the master flag must be on in BOTH the intact and lesion arms; arms inherit the
harness env, see onebrain_regression_battery._spawn_arm). The flag is read at import time:
  * set   -> kind "neural-lesion" (measured: intact vs `BRAIN_REWARD_VALUE_LESION=1`);
  * unset -> kind "thin" (load_bearing_fraction reports `not-covered:thin` and builds no arm), so the DEFAULT
             registry never scores this row as hollow because its master flag was off.
The way to set it for a sharded run is `tools/lb_shard.py jobs --extra-env BRAIN_REWARD_VALUE_AFFERENT=1
--faculties surprise-salience-snc-afferent` (that argparse option exists on main and on
research/lbf-row-registry-hook; `load_bearing_fraction.py` itself has no --extra-env). Restrict the job set with
--faculties: the flag is process-wide, so any other faculty measured in the same job would also run with it on.
`BRAIN_DA_DRIVES` is default-ON in production (server `_DA_DRIVES_DEFAULT_ON`), so it needs no flag.

INTERFACE. `EXTRA_PROBES` is a list of 4-tuples, FACULTY_PROBES' own shape. The AG-REG hook merged on main
(338d9ecee, research/runners/lbf_rows/__init__.py `merge_lbf_rows`) accepts a list through its
`isinstance(extra_probes, (list, tuple))` branch as well as a dict, so a plain list is enough there. `_RowList` is a
list subclass (it takes that same list branch) that also answers `.items()` as (faculty_key, row) pairs; it was
written against the pre-merge branch of the hook, which read only `.items()`. It is harmless on main and kept so
the row loads under either reading (pinned by tests/test_reward_value_afferent.py).

WHAT THE LESION ARM MEASURES (review of 7d5c2743d). `BRAIN_REWARD_VALUE_LESION=1` does not cut the afferent this
row is named for. It SWAPS the source organ's read for the read of the organ's standalone, prediction-edges-zeroed
twin (a different bridge, no homeostat). So intact-vs-lesion measures whether the surprise PREDICTION reaches the
DA mode through A10, not whether the afferent itself is load-bearing. The kind stays "neural-lesion" because the
twin's cut is synaptic and checked at read time, but the note below says what it is. Since fix round 2 neither arm
perturbs the production surprise read (the A10 read restores the state it touches).

This module does not edit FACULTY_LESIONS / FACULTY_PROBES.
"""
from __future__ import annotations

import os

_KEY = "surprise-salience-snc-afferent"
_MASTER_ON = os.environ.get("BRAIN_REWARD_VALUE_AFFERENT", "0").strip().lower() in ("1", "true", "on", "yes")


class _RowList(list):
    """A list of FACULTY_PROBES 4-tuples that also answers `.items()` as {faculty_key: row} pairs (see the module
    docstring: main's merged hook takes the list branch; `.items()` is kept for the pre-merge reading)."""

    def items(self):
        return [(row[0], row) for row in self]


EXTRA_LESIONS = {
    _KEY: dict(
        flag="BRAIN_REWARD_VALUE_LESION", value="1",
        kind=("neural-lesion" if _MASTER_ON else "thin"),
        note=("measures whether the surprise PREDICTION reaches the DA mode through A10: the lesion swaps the "
              "surprise organ's read for its OWN prediction-edges-zeroed twin's read (a standalone bridge, no "
              "homeostat; not a substrate-matched cut and not a cut of the afferent itself; the read-time cut "
              "check is recorded under da_drives.reward_value.lesion_cut). Probed on the CONFIRM turn, where the "
              "prediction cancels the surprise pool's response and the lesion removes that cancellation. The A10 "
              "read never perturbs the production surprise read (fix round 2). OPT-IN: measured only when the "
              "harness runs with BRAIN_REWARD_VALUE_AFFERENT=1 (e.g. tools/lb_shard.py jobs --extra-env "
              "BRAIN_REWARD_VALUE_AFFERENT=1); otherwise reported not-covered:thin, never scored hollow. "
              "See webapp/reward_value_afferent_chat.py."
              + ("" if _MASTER_ON else " THIS PROCESS: master flag unset -> thin.")),
    ),
}

# (faculty_key, turn_label, decision_field_paths, thin) -- FACULTY_PROBES' 4-tuple shape.
EXTRA_PROBES = _RowList([
    (_KEY, "confirm",
     ["da_drives.reward_value.source", "da_drives.reward_value.surprised", "da_drives.mode", "da_drives.lead"],
     not _MASTER_ON),
])
