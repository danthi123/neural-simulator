"""LBF ROW INTERFACE (2026-09-24 midnight plan, AG-REG lane). Each sibling module here exposes module-level
`EXTRA_LESIONS` (dict, same entry shape as `research.runners.load_bearing_fraction.FACULTY_LESIONS`) and
`EXTRA_PROBES` (list of `(faculty_key, turn_label, fields, thin)` tuples, same shape as
`research.runners.onebrain_regression_battery.FACULTY_PROBES`). AG-REG's import hook in
`load_bearing_fraction.py` merges every sibling module's `EXTRA_LESIONS`/`EXTRA_PROBES` into the two literals;
this package intentionally holds NO merge logic itself (that hook is AG-REG's, not a lane's) and no lane edits
`FACULTY_LESIONS`/`FACULTY_PROBES` directly.
"""
