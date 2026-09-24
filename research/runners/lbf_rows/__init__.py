"""LBF ROW INTERFACE (research/lbf-rows-*, 2026-09-24 midnight plan, AG-REG).

Each sibling module here exposes two module-level objects, in EXACTLY the shapes
`research.runners.load_bearing_fraction.FACULTY_LESIONS` / `research.runners.onebrain_regression_battery.
FACULTY_PROBES` already use:

    EXTRA_LESIONS: dict[str, dict]   # faculty_key -> dict(flag=..., value=..., kind=..., note=...)
    EXTRA_PROBES:  list[tuple]       # (faculty_key, turn_label, [decision_field_paths], thin: bool)

AG-REG's import hook (research/runners/load_bearing_fraction.py) merges every sibling module's EXTRA_LESIONS/
EXTRA_PROBES into the two live registries at import time. This package deliberately does NOT import those two
registry modules itself (no circular import; a sibling module's own turn labels/lesion flags are self-contained
strings, not objects it needs to import to define).

Until the merge hook lands, a sibling module's own smoke script merges its rows in-process (documented per
module) so its rows can be validated end-to-end without editing FACULTY_LESIONS/FACULTY_PROBES directly.
"""
from __future__ import annotations
