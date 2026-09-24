"""Per-lane load-bearing-fraction row modules (2026-09-24 midnight plan).

Each sibling module exposes module-level `EXTRA_LESIONS` (dict, same entry shape as
`research.runners.load_bearing_fraction.FACULTY_LESIONS`) and `EXTRA_PROBES` (list of tuples, same shape as
`research.runners.onebrain_regression_battery.FACULTY_PROBES`). The AG-REG import hook in
`research/runners/load_bearing_fraction.py` merges these into the shared registries; this package's modules
never edit `FACULTY_LESIONS`/`FACULTY_PROBES` directly. A module needing a NEW probe turn (one
`PROBE_TURNS`/`_TURN_BY_LABEL` does not already have) also exposes `EXTRA_TURNS` (list of tuples, same shape as
`onebrain_regression_battery._EXTRA_TURNS`) for the same hook (or a follow-on integrator) to merge into
`_TURN_BY_LABEL`.
"""
