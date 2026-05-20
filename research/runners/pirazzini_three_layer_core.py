"""Pre-registered fixed-bar three-state verdict instrument for the
Pirazzini-reference three-layer stage (PFC working-memory + CA3
auto-associative + CA1 hetero-associative + external theta generator).

The constants below are FROZEN and NEVER tuned by results. This module
mirrors the project's frozen-verdict discipline exactly (same shape as
the prior Stage-1 compose_retrieval_core and SPEAR spear_conversational_core
modules): instrument-validity is checked FIRST; malformed input maps to a
safe "cannot conclude" VOID (strictly distinct from FAIL); every return
dict carries the literal frozen-bar block so the verdict is reproducible
and the bars are not movable by results.

Imports are restricted to the Python standard library + typing (math,
typing). No numpy, no project modules, no verdict modules, no I/O. The
decisive built-in control is `theta_disabled_acc`: it MUST collapse to
<= the convergent Stage-1 + SPEAR ceiling so that any capability observed
is attributable to the Pirazzini disinhibition-based theta mechanism. A
degenerate / broken run (always-abstain, always-answer, rhythm-artifact,
etc.) cannot score PASS under these bars.
"""

from __future__ import annotations

import math
from typing import Any, Dict

_PZ_FULL_MIN = 0.80
_PZ_CONVERGENT_CEILING_MAX = 0.10
_PZ_ABSTAIN_MIN = 0.90
_PZ_SCALE_TOL = 0.10
_PZ_LADDER = (2, 3, 5)
_PZ_MIN_SEEDS = 3

REQUIRED_KEYS = (
    "N",
    "n_seeds",
    "full_acc",
    "theta_disabled_acc",
    "abstain_correct_theta_disabled",
)
_ACC_KEYS = (
    "full_acc",
    "theta_disabled_acc",
    "abstain_correct_theta_disabled",
)


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _frozen_bars() -> Dict[str, Any]:
    return {
        "full_min": _PZ_FULL_MIN,
        "convergent_ceiling_max": _PZ_CONVERGENT_CEILING_MAX,
        "abstain_min": _PZ_ABSTAIN_MIN,
        "scale_tol": _PZ_SCALE_TOL,
        "ladder": _PZ_LADDER,
        "min_seeds": _PZ_MIN_SEEDS,
    }


def _void(reason: str) -> Dict[str, Any]:
    return {"gate": "VOID", "reason": reason, "frozen_bars": _frozen_bars()}


def pirazzini_three_layer_verdict(rungs: Any) -> Dict[str, Any]:
    # ---- INSTRUMENT-VALIDITY FIRST (never raise) ----
    if not isinstance(rungs, list) or len(rungs) == 0:
        return _void("rungs must be a non-empty list")

    seen_N = []
    for r in rungs:
        if not isinstance(r, dict):
            return _void("rung must be a dict")
        for k in REQUIRED_KEYS:
            if k not in r:
                return _void("missing required key: " + str(k))

        N = r["N"]
        if isinstance(N, bool) or not isinstance(N, int):
            return _void("N must be int")
        if N not in _PZ_LADDER:
            return _void("N not in frozen ladder")

        n_seeds = r["n_seeds"]
        if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
            return _void("n_seeds must be int")
        if not _finite_number(n_seeds):
            return _void("n_seeds not finite")
        if n_seeds < _PZ_MIN_SEEDS:
            return _void("n_seeds below min_seeds")

        for k in _ACC_KEYS:
            v = r[k]
            if not _finite_number(v):
                return _void("accuracy not finite: " + k)
            if v < 0.0 or v > 1.0:
                return _void("accuracy out of [0,1]: " + k)

        seen_N.append(N)

    if len(set(seen_N)) != len(seen_N):
        return _void("duplicate N rungs")

    sorted_N = sorted(seen_N)
    expected_prefix = list(_PZ_LADDER[: len(rungs)])
    if sorted_N != expected_prefix:
        return _void("rung N set must be a prefix of the frozen ladder")

    # ---- SCORING ----
    rungs_sorted = sorted(rungs, key=lambda r: r["N"])

    def ok(r: Dict[str, Any]) -> bool:
        return (
            r["full_acc"] >= _PZ_FULL_MIN
            and r["theta_disabled_acc"] <= _PZ_CONVERGENT_CEILING_MAX
            and r["abstain_correct_theta_disabled"] >= _PZ_ABSTAIN_MIN
        )

    r0 = rungs_sorted[0]
    if not ok(r0):
        return {
            "gate": "FAIL",
            "reason": "smallest-N rung does not clear all three bars",
            "frozen_bars": _frozen_bars(),
        }

    for r in rungs_sorted[1:]:
        if not ok(r):
            return {
                "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
                "reason": "larger-N rung failed a frozen bar",
                "frozen_bars": _frozen_bars(),
            }
        if r["full_acc"] < r0["full_acc"] - _PZ_SCALE_TOL:
            return {
                "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
                "reason": "full_acc dropped beyond scale_tol vs smallest-N",
                "frozen_bars": _frozen_bars(),
            }

    return {
        "gate": "PASS",
        "reason": "all rungs cleared frozen bars and held within scale_tol",
        "frozen_bars": _frozen_bars(),
    }
