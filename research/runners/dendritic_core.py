"""Pure FIXED-bar verdict for the Dendritic credit-assignment gate.
Mirrors the generator_h_core / subword_lm_gate_core DISCIPLINE (fixed
bars, never tuned, multi-seed, fail-closed) but holds Dendritic's OWN
frozen constants -- does NOT import/modify any other core or
abstention_gate. The LOAD-BEARING criteria: (a) the local rule LEARNS
the hidden-credit task, (b) the no-hidden-credit floor genuinely fails
it (the task really requires hidden credit), (c) the 2026-05-03
permuted-label control does NOT clear (a result not beating its own
permuted control is NOT real), (d) the local update develops emergent
alignment with the true gradient, (e) biologically-local by
construction (no weight transport / no autograd). Pure stdlib;
CPU-unit-testable."""
from __future__ import annotations
import math
from typing import Dict

_DEND_GRAD_COSINE_MIN = 0.30      # emergent local-vs-true-grad cosine
_DEND_HIDDEN_CREDIT_MIN = 0.90    # local rule must LEARN the task
_DEND_NOHIDDEN_FLOOR_MAX = 0.70   # no-hidden-credit floor must FAIL it
_DEND_PERMUTED_MAX = 0.70         # permuted-label control must NOT clear
_DEND_MIN_SEEDS = 3


def _finite(*xs):
    for x in xs:
        try:
            if not math.isfinite(float(x)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _all_fail_dict() -> Dict:
    return {
        "GATE": "FAIL",
        "finite": False,
        "biologically_local": False,
        "has_permuted_control": False,
        "task_learned": False,
        "nohidden_floor_fails": False,
        "permuted_control_not_cleared": False,
        "emergent_grad_alignment": False,
        "hidden_credit": None,
        "nohidden_floor": None,
        "permuted": None,
        "grad_cosine": None,
        "bars": {"grad_cosine_min": _DEND_GRAD_COSINE_MIN,
                 "hidden_credit_min": _DEND_HIDDEN_CREDIT_MIN,
                 "nohidden_floor_max": _DEND_NOHIDDEN_FLOOR_MAX,
                 "permuted_max": _DEND_PERMUTED_MAX},
    }


def dend_verdict(hidden_credit, nohidden_floor, permuted,
                 grad_cosine, biologically_local,
                 has_permuted_control) -> Dict:
    # Fail-closed BEFORE any comparison: a non-numeric / str / None
    # numeric arg must NOT raise AND must NOT be silently coerced --
    # it returns the all-FAIL dict. (float('0.9') succeeds, so a bare
    # try/float() would let '0.9' slip through to PASS; instead any
    # arg that is not a genuine real number -- str, None, bool, etc.
    # -- fails closed deterministically.) bool is rejected because it
    # is an int subclass and True >= 0.90 would silently evaluate.
    for _v in (hidden_credit, nohidden_floor, permuted, grad_cosine):
        if isinstance(_v, bool) or not isinstance(_v, (int, float)):
            return _all_fail_dict()
    finite = _finite(hidden_credit, nohidden_floor, permuted,
                      grad_cosine)
    # Strict-bool: only the literal True passes (a truthy non-True
    # string like 'false' must NOT satisfy the gate).
    bio_local = (biologically_local is True)
    has_ctrl = (has_permuted_control is True)
    learned = finite and hidden_credit >= _DEND_HIDDEN_CREDIT_MIN
    floor_fails = finite and nohidden_floor <= _DEND_NOHIDDEN_FLOOR_MAX
    permuted_ok = finite and permuted <= _DEND_PERMUTED_MAX
    aligned = finite and grad_cosine >= _DEND_GRAD_COSINE_MIN
    gate = bool(finite and bio_local and has_ctrl and learned
                and floor_fails and permuted_ok and aligned)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "finite": bool(finite),
        "biologically_local": bio_local,
        "has_permuted_control": has_ctrl,
        "task_learned": bool(learned),
        "nohidden_floor_fails": bool(floor_fails),
        "permuted_control_not_cleared": bool(permuted_ok),
        "emergent_grad_alignment": bool(aligned),
        "hidden_credit": float(hidden_credit) if finite else None,
        "nohidden_floor": float(nohidden_floor) if finite else None,
        "permuted": float(permuted) if finite else None,
        "grad_cosine": float(grad_cosine) if finite else None,
        "bars": {"grad_cosine_min": _DEND_GRAD_COSINE_MIN,
                 "hidden_credit_min": _DEND_HIDDEN_CREDIT_MIN,
                 "nohidden_floor_max": _DEND_NOHIDDEN_FLOOR_MAX,
                 "permuted_max": _DEND_PERMUTED_MAX},
    }


def dend_aggregate_multiseed(per_seed_verdicts,
                             min_seeds: int = _DEND_MIN_SEEDS) -> Dict:
    n = len(per_seed_verdicts)
    eff_min = max(int(min_seeds), _DEND_MIN_SEEDS)
    n_pass = sum(1 for v in per_seed_verdicts
                 if v.get("GATE") == "PASS")
    all_have_ctrl = (n > 0 and all(
        v.get("has_permuted_control") is True
        for v in per_seed_verdicts))
    gate = bool(n >= eff_min and n_pass == n and n > 0
                and all_have_ctrl)
    return {"GATE": "PASS" if gate else "FAIL", "n_seeds": n,
            "min_seeds": eff_min, "n_pass": n_pass,
            "all_have_permuted_control": all_have_ctrl,
            "all_pass": (n > 0 and n_pass == n)}
