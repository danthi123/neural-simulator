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

INTERFACE. `EXTRA_PROBES` is a list of 4-tuples, as main's research/runners/lbf_rows/__init__.py documents, AND it
answers `.items()` as (faculty_key, row) pairs, because the AG-REG hook on research/lbf-row-registry-hook iterates
`EXTRA_PROBES.items()` (a dict keyed by faculty). A plain list crashes that hook with AttributeError at
load_bearing_fraction import time; `_RowList` works under both readings (pinned by
tests/test_reward_value_afferent.py). The sibling rows' list-vs-dict mismatch with that hook is systemic and is
reconciled at integration, not here.

This module does not edit FACULTY_LESIONS / FACULTY_PROBES.
"""
from __future__ import annotations

import os

_KEY = "surprise-salience-snc-afferent"
_MASTER_ON = os.environ.get("BRAIN_REWARD_VALUE_AFFERENT", "0").strip().lower() in ("1", "true", "on", "yes")


class _RowList(list):
    """A list of FACULTY_PROBES 4-tuples that also answers `.items()` as {faculty_key: row} pairs (see the module
    docstring: main documents a list, the AG-REG hook reads a dict)."""

    def items(self):
        return [(row[0], row) for row in self]


EXTRA_LESIONS = {
    _KEY: dict(
        flag="BRAIN_REWARD_VALUE_LESION", value="1",
        kind=("neural-lesion" if _MASTER_ON else "thin"),
        note=("reads the surprise organ's OWN prediction-edges-zeroed twin (a standalone bridge, not a "
              "substrate-matched cut of the normal read; the read-time cut check is recorded under "
              "da_drives.reward_value.lesion_cut). Probed on the CONFIRM turn, where the prediction cancels the "
              "surprise pool's response and the lesion removes that cancellation. OPT-IN: measured only when the "
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
