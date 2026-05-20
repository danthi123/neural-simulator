"""Generative replay + PFC-held compositional frame stage: pre-registered fixed-bar verdict instrument.

This module is the LOAD-BEARING pre-registered verdict instrument for the
6th architecture in the gating-based composition design line: generative
replay (proposing-and-pattern-completing compositional hypotheses during
NREM-equivalent cycles via the already-validated SWR replay subsystem) +
PFC-held compositional frame (prefrontal working memory holding the
ordered compositional structure via NMDA-bistable attractors in the
already-validated dlpfc_verb region). See
docs/plans/2026-05-20-generative-replay-PFC-frame-design.md. The 6th arc
sits downstream of the empirically grounded 5-architecture convergent
ceiling (Stage-1 + SPEAR + Pirazzini + Unified per-regime monitor +
Theta-gamma all failed decisively at biological scale with different
mechanism-level signatures); the theta-gamma finding established that
cue-suppression-during-retrieve violates the encoding-specificity
principle, so this 6th arc REMOVES cue-suppression and instead adds
generative replay + PFC-frame as augmenting mechanisms that respect
encoding-specificity. All numeric thresholds are FROZEN constants set
in advance and MUST NEVER be tuned in response to results. It mirrors
the frozen-verdict discipline of the prior Stage-1 compose_retrieval_core,
SPEAR spear_conversational_core, Pirazzini pirazzini_three_layer_core,
per-regime per_regime_monitor_core, and theta-gamma
theta_gamma_mode_unification_core.

Imports are restricted to the Python standard library (math) + typing (+
__future__). No imports of any other *_core module or abstention_gate*; no
numpy. Malformed input -> safe "cannot conclude" (VOID), never a crash.
VOID is strictly distinct from FAIL.

THREE decisive built-in anti-cheat controls (bars not movable by results):
- uniform_ctrl_acc <= 0.10: capability MUST be attributable to the
  generative-replay + PFC-frame augmenting mechanisms (these are the
  differentiator; a uniform run with replay disabled and PFC-frame
  disabled should collapse to the documented FAIL behaviour).
- direct_retain_acc >= 0.80: direct retrieval MUST NOT degrade (existing
  abstention_gate.py stays byte-unchanged; the new compositional
  retrieve sits alongside).
- abstain_correct >= 0.90: the trustworthy property MUST hold (a
  degenerate always-abstain or always-answer run cannot be scorable
  PASS).
"""

from __future__ import annotations

import math
from typing import Any, Dict


_GR_FULL_MIN = 0.80
_GR_UNIFORM_CTRL_MAX = 0.10
_GR_DIRECT_RETAIN_MIN = 0.80
_GR_ABSTAIN_CORRECT_MIN = 0.90
_GR_SCALE_TOL = 0.10
_GR_LADDER = (2, 3, 5)
_GR_MIN_SEEDS = 3


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
        "full_min": _GR_FULL_MIN,
        "uniform_ctrl_max": _GR_UNIFORM_CTRL_MAX,
        "direct_retain_min": _GR_DIRECT_RETAIN_MIN,
        "abstain_correct_min": _GR_ABSTAIN_CORRECT_MIN,
        "scale_tol": _GR_SCALE_TOL,
        "ladder": _GR_LADDER,
        "min_seeds": _GR_MIN_SEEDS,
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


def generative_replay_pfc_frame_verdict(rungs: Any) -> Dict[str, Any]:
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
        if N not in _GR_LADDER:
            return _void("N not in frozen ladder")
        n_seeds = r["n_seeds"]
        if isinstance(n_seeds, bool) or not isinstance(n_seeds, int):
            return _void("n_seeds must be int")
        if not _finite_number(n_seeds):
            return _void("n_seeds not finite")
        if n_seeds < _GR_MIN_SEEDS:
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
    expected_prefix = list(_GR_LADDER[: len(rungs)])
    if sorted(ns) != expected_prefix:
        return _void("ladder prefix mismatch")

    sorted_rungs = sorted(rungs, key=lambda r: r["N"])

    def ok(r: Dict[str, Any]) -> bool:
        return (
            r["full_acc"] >= _GR_FULL_MIN
            and r["uniform_ctrl_acc"] <= _GR_UNIFORM_CTRL_MAX
            and r["direct_retain_acc"] >= _GR_DIRECT_RETAIN_MIN
            and r["abstain_correct"] >= _GR_ABSTAIN_CORRECT_MIN
        )

    r0 = sorted_rungs[0]
    if not ok(r0):
        return _fail("smallest-N rung does not meet frozen bars")

    for r in sorted_rungs[1:]:
        if not ok(r):
            return _works_small("larger-N rung does not meet frozen bars")
        if r["full_acc"] < r0["full_acc"] - _GR_SCALE_TOL:
            return _works_small("full_acc drop beyond scale tolerance at larger N")

    return _pass("all rungs meet frozen bars within scale tolerance")
