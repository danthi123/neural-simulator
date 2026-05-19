"""Pre-registered fixed-bar three-state verdict instrument for the shared
theta-gamma SPEAR (Separate Phases of Encoding And Retrieval) + generative-
replay conversational stage.

This module is a LOAD-BEARING pre-registered verdict instrument. It mirrors
the project's frozen-verdict discipline EXACTLY:

  * Numeric thresholds are set in advance and are NEVER tuned to results.
  * Instrument-validity is checked FIRST; any malformed / degenerate input
    yields a safe "cannot conclude" VOID rather than a crash.
  * VOID (cannot conclude) is strictly distinct from FAIL (concluded
    negative).
  * Imports are restricted to the Python standard library + typing only.
  * It does NOT import or modify any existing verdict module, the
    abstention gate, or the no-confabulation moat.

Anti-cheat: the bars are not movable by results, and a degenerate or
broken run must NOT be scorable as PASS. The decisive built-in control is
``rhythm_removed_acc``: a faithful ``rhythm_removed`` ablation reduces the
system to the Stage-1 static composition (which empirically scored ~0.00),
so the rhythm-removed arm MUST collapse below ``_SP_STATIC_CTRL_MAX`` --
the conversational capability must be attributable to the shared rhythm.
"""

from __future__ import annotations

import math
from typing import Any, Dict

# --- Frozen constants (pre-registered; NEVER tuned to results) -------------
_SP_FULL_MIN = 0.80
_SP_STATIC_CTRL_MAX = 0.40
_SP_ABSTAIN_MIN = 0.90
_SP_SCALE_TOL = 0.10
_SP_LADDER = (2, 4, 8)
_SP_MIN_SEEDS = 3

REQUIRED_KEYS = (
    "N",
    "n_seeds",
    "full_acc",
    "rhythm_removed_acc",
    "abstain_correct_rhythm_removed",
)
_ACC_KEYS = (
    "full_acc",
    "rhythm_removed_acc",
    "abstain_correct_rhythm_removed",
)


def _finite_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def _frozen_bars() -> Dict[str, Any]:
    return {
        "full_min": _SP_FULL_MIN,
        "static_ctrl_max": _SP_STATIC_CTRL_MAX,
        "abstain_min": _SP_ABSTAIN_MIN,
        "scale_tol": _SP_SCALE_TOL,
        "ladder": _SP_LADDER,
        "min_seeds": _SP_MIN_SEEDS,
    }


def _void(reason: str) -> Dict[str, Any]:
    return {"gate": "VOID", "reason": reason, "frozen_bars": _frozen_bars()}


def spear_conversational_verdict(rungs: Any) -> Dict[str, Any]:
    """Score a shared-rhythm SPEAR conversational ladder.

    Returns a dict with ``gate`` in exactly one of ``"VOID"``, ``"FAIL"``,
    ``"WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE"``, ``"PASS"``, plus
    ``reason`` and ``frozen_bars``. Never raises.
    """
    # --- INSTRUMENT-VALIDITY FIRST ----------------------------------------
    if not isinstance(rungs, list) or len(rungs) == 0:
        return _void("rungs must be a non-empty list")

    seen_n = []
    for r in rungs:
        if not isinstance(r, dict):
            return _void("each rung must be a dict")
        for k in REQUIRED_KEYS:
            if k not in r:
                return _void("rung missing required key: %s" % k)
        n = r["N"]
        if not isinstance(n, int) or isinstance(n, bool) or n not in _SP_LADDER:
            return _void("rung N must be an int in the frozen ladder")
        ns = r["n_seeds"]
        if (
            not isinstance(ns, int)
            or isinstance(ns, bool)
            or not _finite_number(ns)
            or ns < _SP_MIN_SEEDS
        ):
            return _void("rung n_seeds must be an int >= min_seeds")
        for ak in _ACC_KEYS:
            v = r[ak]
            if not _finite_number(v) or v < 0.0 or v > 1.0:
                return _void("rung %s must be a finite number in [0, 1]" % ak)
        seen_n.append(n)

    if len(set(seen_n)) != len(seen_n):
        return _void("duplicate N across rungs")
    if sorted(seen_n) != list(_SP_LADDER[: len(rungs)]):
        return _void("rung N set must be a frozen-ladder prefix")

    # --- SCORING (recompute from raw only; ignore any extra keys) ---------
    ordered = sorted(rungs, key=lambda r: r["N"])

    def ok(r: Dict[str, Any]) -> bool:
        return (
            r["full_acc"] >= _SP_FULL_MIN
            and r["rhythm_removed_acc"] <= _SP_STATIC_CTRL_MAX
            and r["abstain_correct_rhythm_removed"] >= _SP_ABSTAIN_MIN
        )

    r0 = ordered[0]
    if not ok(r0):
        return {
            "gate": "FAIL",
            "reason": (
                "smallest-load rung fails a frozen bar "
                "(full/rhythm-removed/abstain)"
            ),
            "frozen_bars": _frozen_bars(),
        }

    for r in ordered[1:]:
        if not ok(r) or r["full_acc"] < r0["full_acc"] - _SP_SCALE_TOL:
            return {
                "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
                "reason": (
                    "smallest-load rung passes but a larger-load rung "
                    "fails a bar or drops beyond scale tolerance"
                ),
                "frozen_bars": _frozen_bars(),
            }

    return {
        "gate": "PASS",
        "reason": (
            "all rungs clear the frozen bars; capability holds across "
            "load and collapses under rhythm removal"
        ),
        "frozen_bars": _frozen_bars(),
    }
