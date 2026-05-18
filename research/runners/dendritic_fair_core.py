"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
owner-authorized fair-scale dendritic GLR-2017 run. Own frozen
constants; does NOT import/modify any other core or abstention_gate.
INSTRUMENT-VALIDITY FIRST: a non-discriminating or oracle-broken run
is VOID (instrument not sound) -- explicitly NOT a science PASS/FAIL.
Pure stdlib; CPU-unit-testable."""
from __future__ import annotations
import math
from typing import Dict

_DFAIR_ORACLE_MIN = 0.95
_DFAIR_WRONGSIGN_MAX = 0.30
_DFAIR_CORRECT_MIN = 0.90
_DFAIR_GLOBALSCALAR_MAX = 0.30
_DFAIR_PERMUTED_MAX = 0.30
_DFAIR_ALIGN_MIN = 0.30
_DFAIR_MIN_SEEDS = 3


def _nums_ok(*xs):
    # Fail-closed BEFORE coercion: a str / None numeric-looking arg
    # must NOT slip through. float('0.93') succeeds, so a bare
    # try/float() would let the string '0.93' reach PASS. Reject any
    # arg that is not a genuine real number (str, None, etc.). bool is
    # rejected because it is an int subclass and True would silently
    # satisfy a >= comparison. Mirrors dendritic_core's strict guard.
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        try:
            if not math.isfinite(float(x)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _bars():
    return {"oracle_min": _DFAIR_ORACLE_MIN,
            "wrongsign_max": _DFAIR_WRONGSIGN_MAX,
            "correct_min": _DFAIR_CORRECT_MIN,
            "globalscalar_max": _DFAIR_GLOBALSCALAR_MAX,
            "permuted_max": _DFAIR_PERMUTED_MAX,
            "align_min": _DFAIR_ALIGN_MIN}


def dfair_verdict(oracle_heldout, correct_heldout, wrongsign_heldout,
                  globalscalar_heldout, permuted_heldout,
                  end_align_cos, biologically_local,
                  has_controls) -> Dict:
    finite = _nums_ok(oracle_heldout, correct_heldout,
                      wrongsign_heldout, globalscalar_heldout,
                      permuted_heldout, end_align_cos)
    bio = (biologically_local is True)
    ctrl = (has_controls is True)
    valid = bool(
        finite and bio and ctrl
        and float(oracle_heldout) >= _DFAIR_ORACLE_MIN
        and float(wrongsign_heldout) <= _DFAIR_WRONGSIGN_MAX)
    if not valid:
        return {"GATE": "VOID", "instrument_valid": False,
                "biologically_local": bio, "has_controls": ctrl,
                "finite": bool(finite),
                "reason": "V1/V2 instrument-validity unmet "
                          "(need finite + bio-local + controls + "
                          "oracle>=%.2f + wrongsign<=%.2f)"
                          % (_DFAIR_ORACLE_MIN, _DFAIR_WRONGSIGN_MAX),
                "bars": _bars()}
    learned = float(correct_heldout) >= _DFAIR_CORRECT_MIN
    gs_fail = float(globalscalar_heldout) <= _DFAIR_GLOBALSCALAR_MAX
    pm_fail = float(permuted_heldout) <= _DFAIR_PERMUTED_MAX
    aligned = float(end_align_cos) >= _DFAIR_ALIGN_MIN
    gate = bool(learned and gs_fail and pm_fail and aligned)
    return {"GATE": "PASS" if gate else "FAIL",
            "instrument_valid": True,
            "task_learned": bool(learned),
            "globalscalar_fails": bool(gs_fail),
            "permuted_fails": bool(pm_fail),
            "emergent_alignment": bool(aligned),
            "oracle_heldout": float(oracle_heldout),
            "correct_heldout": float(correct_heldout),
            "wrongsign_heldout": float(wrongsign_heldout),
            "globalscalar_heldout": float(globalscalar_heldout),
            "permuted_heldout": float(permuted_heldout),
            "end_align_cos": float(end_align_cos),
            "bars": _bars()}


def dfair_aggregate_multiseed(per_seed, min_seeds=_DFAIR_MIN_SEEDS):
    n = len(per_seed)
    eff = max(int(min_seeds), _DFAIR_MIN_SEEDS)
    gates = [v.get("GATE") for v in per_seed]
    if n < eff or n == 0:
        return {"GATE": "FAIL", "n_seeds": n, "min_seeds": eff,
                "reason": "fewer than %d seeds" % eff}
    if any(g == "VOID" for g in gates):
        return {"GATE": "VOID", "n_seeds": n, "min_seeds": eff,
                "n_void": sum(g == "VOID" for g in gates),
                "reason": "instrument VOID in >=1 seed"}
    n_pass = sum(g == "PASS" for g in gates)
    return {"GATE": "PASS" if n_pass == n else "FAIL",
            "n_seeds": n, "min_seeds": eff, "n_pass": n_pass}
