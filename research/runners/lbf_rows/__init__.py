"""research/runners/lbf_rows/ -- the LBF ROW REGISTRY IMPORT HOOK (research/lbf-row-registry-hook, 2026-09-24,
plan step S08 / lane AG-REG, docs/plans/... midnight_plan.json).

WHY. Several parallel build lanes (S12 live-organ lesion rows, S13 spiking-learning rows, S14 proposed-lesion knobs,
S15 new-capability wire-ins, ...) each add a faculty row to the load-bearing-fraction registry (research/runners/
load_bearing_fraction.py's FACULTY_LESIONS dict + the FACULTY_PROBES list it shares with
research/runners/onebrain_regression_battery.py). If every lane edited those two literals directly, N parallel
agent worktrees would all touch the SAME lines and collide on merge. Instead, each lane drops exactly ONE module in
THIS package declaring its own rows; this __init__ discovers every such module and merges its rows into the two
registries at import time (research/runners/load_bearing_fraction.py calls merge_lbf_rows() once, right after its
own FACULTY_LESIONS dict literal closes).

Nothing here computes a lesion or a probe decision -- this module only ASSEMBLES the registry (host bookkeeping over
rows the brain-based organ + its dedicated BRAIN_<X>_LESION knob already supply). The actual mechanism a row names
lives in the organ's own module; see that row's `note` / the row lane's own commit for the brain-based lesion itself.

INTERFACE a row module (research/runners/lbf_rows/<your_module>.py) implements -- both dicts are OPTIONAL (a
lesion-only or probe-only module is fine), but a faculty missing from either side will read `lesion-knob-missing`
from load_bearing_fraction's own _flag_resolves() guard, or fail the "every battery faculty is mapped" self-test, so
keep the two dicts in lockstep for any faculty you want actually measured:

    EXTRA_LESIONS = {
        "<faculty_key>": dict(flag="BRAIN_<X>_LESION", value="1", kind="neural-lesion", note="..."),
        ...   # EXACT shape as a FACULTY_LESIONS value in load_bearing_fraction.py: flag / value / kind / note.
        #       kind is one of neural-lesion / whether-disable / in-process / thin / mechanism-only / proposed
        #       (see load_bearing_fraction.py's module docstring for what each means).
    }
    EXTRA_PROBES = {      # or a LIST of the same 4-tuples (FACULTY_PROBES' own shape); both are accepted
        "<faculty_key>": ("<faculty_key>", "<turn_label>", ["<dotted.decision.field>", ...], False),
        ...   # EXACT shape as a FACULTY_PROBES row (research/runners/onebrain_regression_battery.py): a 4-tuple
        #       (faculty_key, turn_label, decision_field_paths, thin). Keyed by faculty_key here (a dict, not the
        #       battery's own list) so merge_lbf_rows() can O(1)-detect a collision before it ever reaches the
        #       shared list; the key is dropped once merged (FACULTY_PROBES stays the battery's original 4-tuple
        #       list shape, unaware rows ever arrived through this hook).
    }

A row that needs a genuinely NEW probe turn/session (not one of PROBE_TURNS' existing turns) adds it to
onebrain_regression_battery.PROBE_TURNS itself in the row lane's own commit -- out of scope for this hook, which
only merges the two dicts above.

COLLISION POLICY. A faculty_key already present in the BASE registry (FACULTY_LESIONS as literally written in
load_bearing_fraction.py, or the battery's own FACULTY_PROBES) or already claimed by an earlier-loaded lbf_rows
module is a collision: it is reported in the merge report and the LATER module's entry for that key is DROPPED --
this hook never silently overwrites an existing row. Row modules are discovered in sorted filename order, so
collisions are deterministic run to run, not dependent on filesystem iteration order.

WHY FACULTY_PROBES IS MUTATED IN PLACE, NOT REBOUND. load_bearing_fraction.py imports FACULTY_PROBES BY REFERENCE
from onebrain_regression_battery.py (`from research.runners.onebrain_regression_battery import ... FACULTY_PROBES,
...`); both module-level names then point at the SAME list object. That module's own `faculty_list()` helper (used
by load_bearing_fraction.py's self-test: "every FACULTY_LESIONS key is a real battery faculty" / "every battery
faculty is mapped") reads `FACULTY_PROBES` as ITS OWN module global, not load_bearing_fraction's -- so if this hook
REBOUND the name in load_bearing_fraction.py (`FACULTY_PROBES = FACULTY_PROBES + extra`), faculty_list() would never
see the merged rows and both self-test invariants would break for any row a lbf_rows module adds. Mutating the
existing list object in place (`.append()`, never reassignment) keeps both modules' view of "FACULTY_PROBES"
consistent by construction, because there was only ever one list object underneath.

An EMPTY research/runners/lbf_rows/ package (no modules besides this __init__.py -- true as of this commit; S12-S20
add the first row modules later, landing on separate branches merged in after this one) is a NO-OP: merge_lbf_rows()
finds zero submodules and returns immediately, so both registries' key sets stay BYTE-IDENTICAL to what they were
before this hook existed. tests/test_lbf_row_registry_hook.py pins this.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List

# Idempotency guard: merge_lbf_rows() must be safe to call more than once in one process (e.g. a test that re-
# imports research.runners.load_bearing_fraction) without DOUBLE-appending every row to the shared FACULTY_PROBES
# list. Only the FIRST call in a process actually merges; later calls report `"skipped": "..."` and touch nothing.
_MERGED = False


def _discover_row_modules():
    """Import every submodule of this package except this __init__ itself, in sorted (deterministic) name order.
    An import error in one lane's row module is COLLECTED, never silently swallowed and never allowed to abort the
    other lanes' rows (one broken module must not zero out the whole registry mid-battery)."""
    mods = []
    errors = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.ispkg:
            continue
        try:
            mods.append(importlib.import_module(__name__ + "." + info.name))
        except Exception as exc:  # noqa: BLE001 -- reported in the merge report, not raised (see docstring above)
            errors.append("%s: %r" % (info.name, exc))
    return mods, errors


def merge_lbf_rows(faculty_lesions: Dict[str, Any], faculty_probes: List[tuple], _mods_override=None) -> Dict[str, Any]:
    """Merge every research/runners/lbf_rows/*.py module's EXTRA_LESIONS/EXTRA_PROBES into the two registries GIVEN
    BY REFERENCE, in place:
      faculty_lesions.update(...)   -- a dict; safe to mutate in place.
      faculty_probes.append(...)    -- mutates the LIST OBJECT ITSELF (never rebinds the name the caller passed) --
          see the module docstring for why this is what keeps onebrain_regression_battery.faculty_list() (and the
          two self-test invariants that depend on it) consistent with the merged registry.

    Returns a report dict: {"modules": [row-module names that contributed >=1 row], "import_errors": [...],
    "keys_added": [...], "collisions": [...]}. An empty research/runners/lbf_rows/ package (no submodules) returns
    a report with every list empty and mutates NEITHER argument -- the no-op this hook's own unit test pins.

    `_mods_override`: TEST SEAM ONLY -- a list of already-imported fake modules to merge instead of discovering the
    real package contents. Never passed by load_bearing_fraction.py's own call.
    """
    global _MERGED
    report: Dict[str, Any] = {"modules": [], "import_errors": [], "keys_added": [], "collisions": [], "parked": []}
    try:
        from research.runners.onebrain_regression_battery import PROBE_TURNS as _probe_turns
        known_turns = {t[0] for t in _probe_turns}
    except Exception:                       # the battery module unavailable: skip the turn check, never crash
        known_turns = None
    if _MERGED and _mods_override is None:
        report["skipped"] = "merge_lbf_rows() already ran once in this process; no-op to avoid double-appending rows"
        return report

    if _mods_override is not None:
        mods, errors = list(_mods_override), []
    else:
        mods, errors = _discover_row_modules()
    report["import_errors"] = errors
    existing_probe_keys = {row[0] for row in faculty_probes}

    for mod in mods:
        extra_lesions = getattr(mod, "EXTRA_LESIONS", {}) or {}
        extra_probes = getattr(mod, "EXTRA_PROBES", {}) or {}
        short_name = mod.__name__.rsplit(".", 1)[-1]
        if isinstance(extra_probes, (list, tuple)):
            # A LIST of 4-tuples is accepted too: it is FACULTY_PROBES' own shape, and row modules merged before this
            # hook (reasoning_transitive_chat) declare it that way. Key each row by its own faculty_key so the
            # collision and validity checks below apply unchanged; a key listed twice is reported, never overwritten.
            as_dict = {}
            for row in extra_probes:
                k = row[0] if isinstance(row, (tuple, list)) and row else repr(row)
                if k in as_dict:
                    report["collisions"].append(
                        "%s: EXTRA_PROBES lists %r more than once -- the later row dropped" % (short_name, k))
                    continue
                as_dict[k] = row
            extra_probes = as_dict
        # A row whose probe turn is not in the battery's PROBE_TURNS would enter the registry as a faculty that can
        # NEVER complete (every battery would then fail its "no incomplete faculty" criterion). Such a row is PARKED
        # -- both its lesion and its probe are left out and the report says why -- until its turn is registered.
        # (First case, 2026-09-24 merge: reasoning_transitive_chat probes `transitive_nonadjacent`, which its own
        # module keeps in _PROPOSED_PROBE_TURNS for a later pass.)
        parked = set()
        if known_turns is not None:
            for key, row in extra_probes.items():
                if isinstance(row, (tuple, list)) and len(row) == 4 and row[1] not in known_turns:
                    parked.add(key)
                    report["parked"].append(
                        "%s: %r probes turn %r, which is not in onebrain_regression_battery.PROBE_TURNS -- row parked "
                        "(lesion and probe both left out)" % (short_name, key, row[1]))
        extra_lesions = {k: v for k, v in extra_lesions.items() if k not in parked}
        extra_probes = {k: v for k, v in extra_probes.items() if k not in parked}
        touched = False
        for key, spec in extra_lesions.items():
            if key in faculty_lesions:
                report["collisions"].append(
                    "%s: EXTRA_LESIONS[%r] collides with an existing FACULTY_LESIONS row -- dropped" % (short_name, key))
                continue
            faculty_lesions[key] = spec
            report["keys_added"].append(key)
            touched = True
        for key, row in extra_probes.items():
            if key in existing_probe_keys:
                report["collisions"].append(
                    "%s: EXTRA_PROBES[%r] collides with an existing FACULTY_PROBES row -- dropped" % (short_name, key))
                continue
            if not isinstance(row, (tuple, list)) or len(row) != 4 or row[0] != key:
                report["collisions"].append(
                    "%s: EXTRA_PROBES[%r] is not a valid 4-tuple (key, turn_label, fields, thin) matching its own "
                    "dict key -- dropped" % (short_name, key))
                continue
            faculty_probes.append(tuple(row))
            existing_probe_keys.add(key)
            touched = True
        if touched:
            report["modules"].append(short_name)

    _MERGED = True
    return report
