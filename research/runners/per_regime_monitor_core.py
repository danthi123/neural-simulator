"""Per-regime metacognitive-monitor stage: pre-registered fixed-bar verdict instrument.

This module is the LOAD-BEARING pre-registered verdict instrument for the
per-regime metacognitive-monitor architecture (Miyamoto 2017 doubly-dissociable
parallel metamemory streams). All numeric thresholds are FROZEN constants set
in advance and MUST NEVER be tuned in response to results. It mirrors the
frozen-verdict discipline of the prior Stage-1 compose_retrieval_core,
SPEAR spear_conversational_core, and Pirazzini pirazzini_three_layer_core.

Imports are restricted to the Python standard library (math) + typing (+
__future__). No imports of any other *_core module or abstention_gate*; no
numpy. Malformed input -> safe "cannot conclude" (VOID), never a crash.
VOID is strictly distinct from FAIL.

THREE decisive built-in anti-cheat controls (bars not movable by results):
- uniform_ctrl_acc <= 0.10: capability MUST be attributable to per-regime
  separation (the triple-convergent ceiling means per-regime separation is
  the differentiator; a uniform monitor should collapse).
- direct_retain_acc >= 0.80: direct retrieval MUST NOT degrade (existing
  abstention_gate.py stays byte-unchanged; the new compositional monitor
  sits alongside).
- abstain_correct >= 0.90: the trustworthy property MUST hold (a degenerate
  always-abstain or always-answer run cannot be scorable PASS).
"""

from __future__ import annotations

import math
from typing import Any, Dict


_PR_FULL_MIN = 0.80
_PR_UNIFORM_CTRL_MAX = 0.10
_PR_DIRECT_RETAIN_MIN = 0.80
_PR_ABSTAIN_CORRECT_MIN = 0.90
_PR_SCALE_TOL = 0.10
_PR_LADDER = (2, 3, 5)
_PR_MIN_SEEDS = 3


REQUIRED_KEYS = (
    "N",
    "n_seeds",
    "full_acc",
    "uniform_ctrl_acc",
    "direct_retain_acc",
    "abstain_correct",
)

_ACC_KEYS = (
    "full_acc",
    "uniform_ctrl_acc",
    "direct_retain_acc",
    "abstain_correct",
)


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _frozen_bars() -> Dict[str, Any]:
    return {
        "full_min": _PR_FULL_MIN,
        "uniform_ctrl_max": _PR_UNIFORM_CTRL_MAX,
        "direct_retain_min": _PR_DIRECT_RETAIN_MIN,
        "abstain_correct_min": _PR_ABSTAIN_CORRECT_MIN,
        "scale_tol": _PR_SCALE_TOL,
        "ladder": _PR_LADDER,
        "min_seeds": _PR_MIN_SEEDS,
    }


def _void(reason: str) -> Dict[str, Any]:
    return {"gate": "VOID", "reason": reason, "frozen_bars": _frozen_bars()}


def _fail(reason: str) -> Dict[str, Any]:
    return {"gate": "FAIL", "reason": reason, "frozen_bars": _frozen_bars()}


def _works_small(reason: str) -> Dict[str, Any]:
    return {
        "gate": "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
        "reason": reason,
        "frozen_bars": _frozen_bars(),
    }


def _pass(reason: str) -> Dict[str, Any]:
    return {"gate": "PASS", "reason": reason, "frozen_bars": _frozen_bars()}


def per_regime_monitor_verdict(rungs: Any) -> Dict[str, Any]:
    # INSTRUMENT-VALIDITY FIRST: any failure -> VOID (never raise).
    if not isinstance(rungs, list) or len(rungs) == 0:
        return _void("rungs must be a non-empty list")

    for r in rungs:
        if not isinstance(r, dict):
            return _void("rung must be a dict")
        for k in REQUIRED_KEYS:
            if k not in r:
                return _void("missing required key: " + str(k))
        N = r["N"]
        if isinstance(N, bool) or not isinstance(N, int):
            return _void("N must be int")
        if N not in _PR_LADDER:
            return _void("N not in frozen ladder")
        n_seeds = r["n_seeds"]
        if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
            return _void("n_seeds must be int")
        if not _finite_number(n_seeds):
            return _void("n_seeds not finite")
        if n_seeds < _PR_MIN_SEEDS:
            return _void("n_seeds below min")
        for ak in _ACC_KEYS:
            v = r[ak]
            if not _finite_number(v):
                return _void("non-finite or non-numeric acc: " + ak)
            if v < 0.0 or v > 1.0:
                return _void("acc out of [0,1]: " + ak)

    ns = [r["N"] for r in rungs]
    if len(set(ns)) != len(ns):
        return _void("duplicate N values")
    expected_prefix = list(_PR_LADDER[: len(rungs)])
    if sorted(ns) != expected_prefix:
        return _void("ladder prefix mismatch")

    sorted_rungs = sorted(rungs, key=lambda r: r["N"])

    def ok(r: Dict[str, Any]) -> bool:
        return (
            r["full_acc"] >= _PR_FULL_MIN
            and r["uniform_ctrl_acc"] <= _PR_UNIFORM_CTRL_MAX
            and r["direct_retain_acc"] >= _PR_DIRECT_RETAIN_MIN
            and r["abstain_correct"] >= _PR_ABSTAIN_CORRECT_MIN
        )

    r0 = sorted_rungs[0]
    if not ok(r0):
        return _fail("smallest-N rung does not meet frozen bars")

    for r in sorted_rungs[1:]:
        if not ok(r):
            return _works_small("larger-N rung does not meet frozen bars")
        if r["full_acc"] < r0["full_acc"] - _PR_SCALE_TOL:
            return _works_small("full_acc drop beyond scale tolerance at larger N")

    return _pass("all rungs meet frozen bars within scale tolerance")
