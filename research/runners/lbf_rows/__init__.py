"""Per-lane load-bearing-fraction row modules (2026-09-24 midnight plan).

Each sibling module exposes module-level `EXTRA_LESIONS` (dict, same entry shape as
`research.runners.load_bearing_fraction.FACULTY_LESIONS`) and `EXTRA_PROBES` (list of tuples, same shape as
`research.runners.onebrain_regression_battery.FACULTY_PROBES`). This package's modules never edit
`FACULTY_LESIONS`/`FACULTY_PROBES` directly.

NOT YET WIRED (2026-09-24 review finding D): no "AG-REG import hook" exists anywhere in the repo yet --
`research/runners/load_bearing_fraction.py` and `research/runners/onebrain_regression_battery.py` do not import
this package or merge these dicts/lists in. Every row here is inert (unregistered, not exercised by either
battery) until a follow-on integrator adds that hook (grep the repo for "AG-REG" before assuming otherwise). A
module needing a NEW probe turn (one `PROBE_TURNS`/`_TURN_BY_LABEL` does not already have) also exposes
`EXTRA_TURNS` (list of tuples, same shape as `onebrain_regression_battery._EXTRA_TURNS`) for that same
still-to-be-written hook to merge into `_TURN_BY_LABEL`.
"""
