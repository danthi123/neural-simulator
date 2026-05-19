"""Pre-registered fixed-bar three-state verdict instrument for
regime-correct compositional retrieval.

This module is the LOAD-BEARING pre-registered verdict instrument that
scores the "regime-correct compositional retrieval" capability. Its
numeric thresholds are FROZEN: set in advance and NEVER tuned to a
result. It mirrors the discipline of the project's existing frozen
verdict modules EXACTLY:

  * instrument-validity is checked FIRST;
  * malformed / degenerate / under-powered input -> a safe
    "cannot conclude" VOID, NEVER a crash;
  * VOID is strictly distinct from FAIL;
  * a precomputed/caller-supplied verdict is ignored -- the gate is
    always recomputed purely from the raw numbers, so the bars cannot
    be moved by results and a broken run cannot be scored PASS.

Imports ONLY the Python standard library + typing. It does NOT import
or modify any existing verdict module, the no-confabulation moat, or
numpy. ASCII only.
"""

from __future__ import annotations

import math
from typing import Any, Dict

# --- FROZEN CONSTANTS (pre-registered; NEVER tuned to a result) -------
_CR_FULL_MIN = 0.80
_CR_ABLATION_MAX = 0.40
_CR_ABSTAIN_MIN = 0.90
_CR_SCALE_TOL = 0.10
_CR_LADDER = (2, 4, 8)
_CR_MIN_SEEDS = 3
# ----------------------------------------------------------------------

REQUIRED_KEYS = (
    "N",
    "n_seeds",
    "full_acc",
    "recent_only_acc",
    "remote_only_acc",
    "abstain_correct_recent_only",
    "abstain_correct_remote_only",
)

_ACC_KEYS = (
    "full_acc",
    "recent_only_acc",
    "remote_only_acc",
    "abstain_correct_recent_only",
    "abstain_correct_remote_only",
)


def _frozen_bars() -> Dict[str, Any]:
    return {
        "full_min": _CR_FULL_MIN,
        "ablation_max": _CR_ABLATION_MAX,
        "abstain_min": _CR_ABSTAIN_MIN,
        "scale_tol": _CR_SCALE_TOL,
        "ladder": _CR_LADDER,
        "min_seeds": _CR_MIN_SEEDS,
    }


def _finite_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def _void(reason: str) -> Dict[str, Any]:
    return {"gate": "VOID", "reason": reason, "frozen_bars": _frozen_bars()}


def _fail(reason: str) -> Dict[str, Any]:
    return {"gate": "FAIL", "reason": reason, "frozen_bars": _frozen_bars()}


def compose_retrieval_verdict(rungs: Any) -> Dict[str, Any]:
    """Score a compositional-retrieval scaling ladder.

    Returns a dict whose ``gate`` is exactly one of:
    ``"VOID"``, ``"FAIL"``, ``"WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"``,
    ``"PASS"``. Every return dict also carries ``reason`` and
    ``frozen_bars``. Never raises.
    """
    # --- INSTRUMENT-VALIDITY FIRST ------------------------------------
    if not isinstance(rungs, list) or len(rungs) == 0:
        return _void("rungs must be a non-empty list")

    seen_n = []
    for r in rungs:
        if not isinstance(r, dict):
            return _void("each rung must be a dict")
        for k in REQUIRED_KEYS:
            if k not in r:
                return _void("rung missing required key: " + str(k))

        n = r["N"]
        if isinstance(n, bool) or not isinstance(n, int):
            return _void("rung N must be an int")
        if n not in _CR_LADDER:
            return _void("rung N not in frozen ladder")

        ns = r["n_seeds"]
        if isinstance(ns, bool) or not isinstance(ns, int):
            return _void("rung n_seeds must be an int")
        if not _finite_number(ns):
            return _void("rung n_seeds not finite")
        if ns < _CR_MIN_SEEDS:
            return _void("rung n_seeds below frozen minimum")

        for k in _ACC_KEYS:
            v = r[k]
            if not _finite_number(v):
                return _void("rung field not a finite number: " + str(k))
            if v < 0.0 or v > 1.0:
                return _void("rung field out of [0,1]: " + str(k))

        seen_n.append(n)

    if len(set(seen_n)) != len(seen_n):
        return _void("duplicate rung N")

    sorted_n = sorted(seen_n)
    expected_prefix = list(_CR_LADDER[: len(rungs)])
    if sorted_n != expected_prefix:
        return _void("rung N set is not a prefix of the frozen ladder")

    # --- SCORING (recomputed purely from raw numbers) -----------------
    ordered = sorted(rungs, key=lambda r: r["N"])

    def ok(r: Dict[str, Any]) -> bool:
        return (
            r["full_acc"] >= _CR_FULL_MIN
            and r["recent_only_acc"] <= _CR_ABLATION_MAX
            and r["remote_only_acc"] <= _CR_ABLATION_MAX
            and r["abstain_correct_recent_only"] >= _CR_ABSTAIN_MIN
            and r["abstain_correct_remote_only"] >= _CR_ABSTAIN_MIN
        )

    r0 = ordered[0]
    if not ok(r0):
        return _fail("smallest-load rung fails the frozen regime bars")

    for r in ordered[1:]:
        if not ok(r):
            return {
                "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
                "reason": "larger-load rung fails the frozen regime bars",
                "frozen_bars": _frozen_bars(),
            }
        if r["full_acc"] < r0["full_acc"] - _CR_SCALE_TOL:
            return {
                "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
                "reason": "larger-load full_acc dropped beyond scale tolerance",
                "frozen_bars": _frozen_bars(),
            }

    return {
        "gate": "PASS",
        "reason": "all rungs meet the frozen regime bars and scale within tolerance",
        "frozen_bars": _frozen_bars(),
    }
