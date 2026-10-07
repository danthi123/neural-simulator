#!/usr/bin/env python3
"""registry_fingerprint.py -- the B2d registry of the tree this runs in, as canonical JSON (no brain build).

  cd <a tree> && SIM_NO_PROVENANCE=1 SIM_BACKEND=numpy <python> <this file> > fingerprint.json

What B2d's re-pin rule (pre-registration, "Frozen revision and registry") asks to be compared between H and a new pin:
every registry row (key, kind, lesion flag, lesion value, REQUIRED_ENV) from research/runners/load_bearing_fraction.py,
every FACULTY_PROBES row (probe turn, decision fields, thin), every probe-turn definition (label, message, session,
reset, percept, rich), the noise-field list, the world-step labels, the row-merge report, the adequate probe flags of
tools/lb_shard.py and the kinds it shards. Read by import from the CURRENT DIRECTORY's tree only.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in [k for k in os.environ if k.startswith("LB_") or k.startswith("BRAIN_")]:
    os.environ.pop(k)          # the production default: no probe flag, no BRAIN_* override at import

from research.runners import load_bearing_fraction as lbf  # noqa: E402
from research.runners import onebrain_regression_battery as orb  # noqa: E402
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import lb_shard  # noqa: E402


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def fingerprint():
    rows = {k: {f: spec.get(f) for f in ("kind", "flag", "value", "REQUIRED_ENV", "required_env")}
            for k, spec in lbf.FACULTY_LESIONS.items()}
    return _plain({
        "n_rows": len(lbf.FACULTY_LESIONS),
        "rows": rows,
        "faculty_probes": [list(r) for r in orb.FACULTY_PROBES],
        "turns": {lab: list(t) for lab, t in orb._TURN_BY_LABEL.items()},
        "noise_fields": sorted(orb._NOISE_FIELDS),
        "world_steps": sorted(orb._WORLD_STEPS),
        "row_merge_report": lbf.LBF_ROW_MERGE_REPORT,
        "adequate_probe_flags": lb_shard.PROBE_SETS["adequate"],
        "measurable_kinds": list(lb_shard.MEASURABLE_KINDS),
        "coverable_kinds": list(lb_shard.COVERABLE_KINDS),
        "sharded_keys": lb_shard.faculty_keys(),
    })


if __name__ == "__main__":
    json.dump(fingerprint(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
