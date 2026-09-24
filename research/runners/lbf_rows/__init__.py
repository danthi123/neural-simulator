"""LBF ROW MODULES — per-lane load-bearing-fraction rows, merged into `FACULTY_LESIONS`/`FACULTY_PROBES` by
AG-REG's import hook (research/runners/load_bearing_fraction.py, S08/S26) rather than by editing those literal
registries directly. See docs/plans/2026-09-24 midnight plan, S12/S13.

Each sibling module (e.g. `live_organs.py`, `learning.py`) exposes two module-level names:
  EXTRA_LESIONS: dict[str, dict] -- SAME entry shape as `load_bearing_fraction.FACULTY_LESIONS`:
      {faculty_key: dict(flag=<BRAIN_*_LESION str or None>, value=<'1' or '0'>, kind=<one of neural-lesion /
       whether-disable / in-process / thin / mechanism-only / proposed>, note=<str>)}
  EXTRA_PROBES: list[tuple] -- SAME entry shape as `load_bearing_fraction.FACULTY_PROBES` (via
      `onebrain_regression_battery.FACULTY_PROBES`): (faculty_key, turn_label, [decision_field_paths], thin_bool)

A turn_label used by EXTRA_PROBES that is not already in `onebrain_regression_battery.PROBE_TURNS` must be added
to that module's `_EXTRA_TURNS` (label-only, kept OUT of the default `PROBE_TURNS` roster so no existing runner's
default turn set grows) by AG-REG's merge step -- see each row module's own docstring for the exact turn tuple.

This package intentionally has NO import-time side effects (no eager registry mutation): merging is the import
hook's job (S08), not this package's.
"""
