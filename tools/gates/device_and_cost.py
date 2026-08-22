"""CLASS DC — AN ARTIFACT CANNOT SAY WHAT DEVICE IT RAN ON, OR BURNED HOURS WITHOUT EVER PROJECTING ITS COST.

TWO DEFECTS, both measured 2026-07-31, both mine, and both invisible to every other gate because they are
properties of the RUN rather than of any claim it makes.

(a) THE DEVICE. Runners do `os.environ.setdefault("SIM_BACKEND", "numpy")`, so a caller who does not set it
    explicitly silently gets the CPU path — documented in CLAUDE.md as costing months once, and it still got
    me. I launched a four-cell GPU precondition test that ran on CPU for 30 minutes at 10-50x the intended
    cost. I had "verified" the GPU by finding nvidia mappings in /proc/PID/maps, which only proves CuPy is
    IMPORTABLE. The runner's own first log line said "this run is on the CPU" in plain English.
    A run on the wrong device is not a slow run. It is a different experiment.

(b) THE COST. The gap#4 crux was planned at ~6h45m per cell and was actually ~23h: after printing its arm
    result each cell trained THREE MORE FULL NETS as anti-cheats, each the same cost as the arm. Nobody
    counted them. Eight cells ran nine hours toward a ~136 GPU-hour tail that could not have changed the
    verdict — and the information needed to catch it existed at the 5h47m mark, where the runner printed
    `(20539s)` for arm one. Nothing multiplied it by the units remaining.

WHAT THIS GATE ENFORCES, on newly-added artifacts only:
  1. the artifact (or its provenance sidecar) must RECORD the backend it ran on — you cannot audit a result
     whose device is unknown, and the provenance door already captures `SIM_BACKEND` for free;
  2. an artifact recording a long elapsed time must ALSO record a cost projection or an explicit
     acknowledgement, so that "we knew and proceeded" is distinguishable from "nobody looked".

The runtime halves are `tools.lab.assert_backend` (raises on mismatch, by importing the backend rather than
inspecting the process) and `tools.lab.project_cost` (projects the total from ONE finished unit — measured,
not estimated, because a config parser gets it wrong exactly when the runner does something unusual, which
is the case that hurts).

WHAT IT CANNOT CATCH: a run on the RIGHT device that is simply mis-configured, and a projection that is
recorded and then ignored. It closes "nobody can tell" and "nobody looked", not "somebody looked and chose
badly" — that last one is judgement and is left as judgement.
"""
from __future__ import annotations

import json
import os
import tempfile

NAME = "device-and-cost"
CLASS_ID = "DC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_KEYS = ("sim_backend", "backend", "device")
ELAPSED_KEYS = ("elapsed_seconds", "elapsed_s", "elapsed", "runtime_seconds", "wall_seconds")
COST_KEYS = ("cost_projection", "projected_total_hours", "cost_acknowledged", "projected_seconds")
LONG_RUN_S = 8 * 3600.0


def _is_structural_record(obj):
    """True for create-only commands/configs that record no completed run."""
    schema = obj.get("schema")
    # Operational state is not a scientific result.  The persistent coordinator
    # deliberately records lanes, agents, and resource observations separately;
    # forcing it to pretend it has a completed backend/cost receipt would make
    # the provenance gate less precise, not more protective.
    if schema in ("sim-autonomous-workboard-v1", "board-sync-v1", "tool-health-v1"):
        # board-sync-v1 (tools/vikunja.sh receipt) + tool-health-v1 (tools/tool_health.py smoke) are
        # coordination/state files under research/coordination/, not scientific runs; a backend/cost receipt
        # would be meaningless for them (the same rationale as the workboard above).
        return True
    if obj.get("execution") == "not_executed" and isinstance(obj.get("argv"), list):
        return True
    return (
        obj.get("status") == "frozen"
        and isinstance(schema, str)
        and "controller-config" in schema
    )


def _find(obj, keys, depth=0):
    """Shallow search — top level and one nested level (config/provenance/meta blocks live there)."""
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if k.lower() in keys and v is not None and not isinstance(v, (dict, list)):
            return v
    if depth < 2:
        for v in obj.values():
            if isinstance(v, dict):
                got = _find(v, keys, depth + 1)
                if got is not None:
                    return got
    return None


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    if rel.endswith(".prov.json") or rel.endswith(".cmd.json"):
        return []                                          # sidecars are the evidence, not the subject
    try:
        obj = json.load(open(path, errors="ignore"))
    except (OSError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []
    if _is_structural_record(obj):
        return []
    problems = []

    backend = _find(obj, BACKEND_KEYS)
    if backend is None:                                    # fall back to the provenance sidecar
        for sib in (path + ".prov.json", os.path.splitext(path)[0] + ".prov.json"):
            if os.path.exists(sib):
                try:
                    backend = _find(json.load(open(sib, errors="ignore")), BACKEND_KEYS)
                except (OSError, ValueError):
                    backend = None
                if backend is not None:
                    break
    if backend is None:
        problems.append(
            "%s: records NO backend/device, and no provenance sidecar supplies one. A result whose device "
            "is unknown cannot be audited — and `SIM_BACKEND` defaults to numpy via setdefault, so 'not "
            "recorded' most often means 'ran on CPU without anyone intending it'. Use "
            "tools.lab.assert_backend, or let research/runners/__init__ stamp the sidecar." % rel)

    elapsed = _find(obj, ELAPSED_KEYS)
    try:
        elapsed = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed = None
    if elapsed is not None and elapsed > LONG_RUN_S and _find(obj, COST_KEYS) is None:
        problems.append(
            "%s: records %.1fh of compute but NO cost projection or acknowledgement. Use "
            "tools.lab.project_cost after the first unit — on 2026-07-31 a 9-hour run was heading for ~23h "
            "per cell and the arithmetic was available after the first arm." % (rel, elapsed / 3600.0))
    return problems


def check(paths):
    if paths is None or len(paths) == 0:
        return []                                          # legacy corpus predates this; audited on touch
    problems = []
    for p in [x for x in paths if x.endswith(".json")]:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full, p)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: both defects, then the controls that keep the gate usable."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            json.dump(obj, open(p, "w"))
            return p

        # 1. no backend recorded anywhere
        if not any("NO backend" in x for x in _check_one(w("a.json", {"means": {"acc": 0.5}}), "raw/a.json")):
            bad.append("did NOT catch an artifact recording no backend")
        # 2. a long run with no cost projection
        p = w("b.json", {"backend": "cupy", "elapsed_seconds": 9 * 3600})
        if not any("cost projection" in x for x in _check_one(p, "raw/b.json")):
            bad.append("did NOT catch a long run with no cost projection")
        # 3. NEGATIVE CONTROL — backend at top level is enough.
        if _check_one(w("c.json", {"sim_backend": "cupy"}), "raw/c.json"):
            bad.append("FALSE POSITIVE: flagged an artifact that records its backend")
        # 4. NEGATIVE CONTROL — nested in a config/provenance block counts too.
        if _check_one(w("d.json", {"provenance": {"SIM_BACKEND": "numpy"}}), "raw/d.json"):
            bad.append("FALSE POSITIVE: flagged a backend recorded in a nested provenance block")
        # 5. NEGATIVE CONTROL — a SHORT run needs no projection.
        if _check_one(w("e.json", {"backend": "cupy", "elapsed_seconds": 120}), "raw/e.json"):
            bad.append("FALSE POSITIVE: flagged a short run for having no cost projection")
        # 6. NEGATIVE CONTROL — a long run that DID project must pass, else nobody can satisfy it.
        p = w("f.json", {"backend": "cupy", "elapsed_seconds": 9 * 3600, "projected_total_hours": 22.8})
        if _check_one(p, "raw/f.json"):
            bad.append("FALSE POSITIVE: flagged a long run that recorded its projection")
        # 7. NEGATIVE CONTROL — a sidecar supplies the backend for an artifact lacking one.
        p = w("g.json", {"means": {"acc": 0.5}})
        json.dump({"env": {"SIM_BACKEND": "cupy"}}, open(p + ".prov.json", "w"))
        if any("NO backend" in x for x in _check_one(p, "raw/g.json")):
            bad.append("FALSE POSITIVE: ignored a provenance sidecar that records the backend")
        # 8. NEGATIVE CONTROL — sidecars themselves are not the subject.
        if _check_one(p + ".prov.json", "raw/g.json.prov.json"):
            bad.append("FALSE POSITIVE: audited a provenance sidecar as if it were a result")
        # 9. SCOPING — standalone/empty scans nothing.
        if check(None) or check([]):
            bad.append("SCOPE LEAK: standalone/empty mode must not scan the legacy corpus")
        # 10. NEGATIVE CONTROL — a frozen command/config describes future execution, not a run.
        structural = (
            w("command.json", {"execution": "not_executed", "argv": ["python", "runner.py"]}),
            w("config.json", {"schema": "v13-stage0-controller-config-v3", "status": "frozen"}),
        )
        if any(_check_one(p, os.path.basename(p)) for p in structural):
            bad.append("FALSE POSITIVE: treated a frozen command/config as a completed result")
        # 11. NEGATIVE CONTROL — operational workboard state is not a completed result.
        if _check_one(w("workboard.json", {"schema": "sim-autonomous-workboard-v1", "lanes": {}}),
                      "research/coordination/workboard.json"):
            bad.append("FALSE POSITIVE: treated coordinator state as a completed result")
        # 11b. NEGATIVE CONTROL — the board-sync receipt + tool-health smoke are coordination state, not runs.
        if _check_one(w("board_sync.json", {"schema": "board-sync-v1", "entries": []}),
                      "research/coordination/board_sync.json"):
            bad.append("FALSE POSITIVE: treated the board-sync receipt as a completed result")
        if _check_one(w("tool_health.json", {"schema": "tool-health-v1", "results": []}),
                      "research/coordination/tool_health.json"):
            bad.append("FALSE POSITIVE: treated the tool-health smoke as a completed result")
    return bad
