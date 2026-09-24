"""LBF ROW INTERFACE package (midnight plan, S15/S28). Each sibling module exposes module-level dicts
`EXTRA_LESIONS` and `EXTRA_PROBES` with EXACTLY the same entry shapes as `FACULTY_LESIONS` /
`FACULTY_PROBES` in `research/runners/load_bearing_fraction.py` / `onebrain_regression_battery.py`.

This package does NOT import or edit those literals directly (AG-REG's import hook in
`load_bearing_fraction.py` is the single place that merges every module here into the live registry -- see
that runner's docstring). A row landing here before the hook exists is inert until the hook lands; it is not
run by anything on its own.
"""
from __future__ import annotations
