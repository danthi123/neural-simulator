"""Pure FIXED-bar TREND-PRIMARY scale-confidence verdict for Q2R --
a FRESH larger-KB experiment of the VALIDATED Q2 constrained-decoding
mechanism. Mirrors the adversarial-hardened Q2-constrained-decode-core
and compose-bridge-core DISCIPLINE (fixed bars NEVER tuned,
fail-closed, VOID strictly distinct from FAIL, malformed/junk ->
VOID-not-raise). Holds its OWN frozen _Q2R_*; does NOT import/mutate
the Q2 constrained-decode core or any *_core sibling module. Pure
stdlib+typing; NO torch, NO autograd. ASCII.

A-PRIORI justification of the frozen criterion (defensible WITHOUT any
reference to Q2's observed numbers):
- _Q2R_LADDER = (12,24,48,96): scale-confidence is definitionally
  about behaviour as capacity SCALES UP toward a useful target. The
  ladder must START at a non-toy size (K=12 = the smallest KB a
  "grounded conversational agent" claim could even be ABOUT; a
  6-proposition KB is a toy below the floor of the question) and
  EXTEND UPWARD geometrically (x2 per rung) to where scale-confidence
  actually lives (K=96). The K=6 omission is a principled non-toy
  floor decided by what the QUESTION means, not by any observed value.
- _Q2R_TOP_MIN = 0.50: DELIBERATELY the SAME absolute non-vacuity
  value as the validated Q2 core's _CDC_MIN_GROUNDED_ANSWER_RATE
  (0.50). It is NOT a softened bar. The ONLY methodological change vs
  Q2 is WHERE the absolute floor + the trend are applied (at the
  LARGEST scale where scale-confidence is claimed + a monotone trend),
  NOT WHAT the value is. This identity is the strongest structural
  defense against a goalpost-move.
- _Q2R_SCALE_TOL = 0.10: a stochastic 5-seed non-vacuity rate has a
  noise floor; 0.10 is a defensible max permitted DROP between
  ascending rungs (same magnitude family as the validated _CDC
  tolerances). The TREND being non-decreasing-up-to-tol is the PRIMARY
  scale-confidence signal.
These values are pre-registered HERE, BEFORE any Q2R run, and NEVER
tuned to a result."""
from __future__ import annotations
from typing import Dict

_Q2R_LADDER = (12, 24, 48, 96)
_Q2R_SCALE_TOL = 0.10
_Q2R_TOP_MIN = 0.50
_Q2R_MIN_SEEDS = 3


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def q2r_scale_confidence(rungs) -> Dict:
    """Pure, deterministic, fail-closed. rungs: list of {"K",
    "verdict":{"GATE":...}, "constrained_nonvac_rate_mean"}. Recomputed
    from the single recorded JSON; NEVER raises.

    SCALE-CONFIDENT-PASS iff: the ordered-by-K key tuple == _Q2R_LADDER
    EXACTLY (guards padding/duplication/mismatch) AND every rung
    verdict GATE == "PASS" AND constrained_nonvac_rate_mean is
    non-decreasing up to _Q2R_SCALE_TOL across the ascending ladder AND
    the LARGEST rung (K=96) constrained_nonvac_rate_mean >=
    _Q2R_TOP_MIN. Else: any rung GATE VOID/missing/unknown -> VOID
    (precedence); any rung GATE FAIL -> FAIL; otherwise (all PASS but
    trend breaks or top below floor) ->
    WORKS-SMALL-NO-SCALE-CONFIDENCE. Non-numeric/unorderable/malformed
    -> VOID."""
    bars = {"LADDER": list(_Q2R_LADDER), "SCALE_TOL": _Q2R_SCALE_TOL,
            "TOP_MIN": _Q2R_TOP_MIN, "MIN_SEEDS": _Q2R_MIN_SEEDS}
    try:
        ordered = sorted(rungs, key=lambda r: r["K"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by K", "frozen_bars": bars}
    try:
        ladder = tuple(int(r["K"]) for r in ordered)
    except (TypeError, ValueError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rung K not integer-coercible",
                "frozen_bars": bars}
    if ladder != _Q2R_LADDER:
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder %s != pre-registered %s "
                          "(padding/mismatch guard)"
                          % (ladder, _Q2R_LADDER),
                "frozen_bars": bars}
    gates = []
    for r in ordered:
        v = r.get("verdict")
        gates.append(v.get("GATE") if isinstance(v, dict) else None)
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is VOID/missing",
                "frozen_bars": bars}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE is FAIL", "frozen_bars": bars}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is not PASS/FAIL/VOID",
                "frozen_bars": bars}
    nv = []
    for r in ordered:
        f = _num(r.get("constrained_nonvac_rate_mean"))
        if f is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric constrained_nonvac_rate_mean",
                    "frozen_bars": bars}
        nv.append(f)
    monotone = all(nv[i + 1] >= nv[i] - _Q2R_SCALE_TOL
                   for i in range(len(nv) - 1))
    top_ok = nv[-1] >= _Q2R_TOP_MIN
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "every rung PASS; non-vacuity non-decreasing "
                          "up to tol across the ascending ladder; "
                          "K=96 clears the 0.50 floor (same value as "
                          "Q2's bar, applied at the largest scale)",
                "nonvac_by_rung": nv, "frozen_bars": bars}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "every rung PASS but %s%s"
                      % ("" if monotone else "non-vacuity drops > "
                         "_Q2R_SCALE_TOL between ascending rungs; ",
                         "" if top_ok else "K=96 non-vacuity below "
                         "_Q2R_TOP_MIN=0.50"),
            "nonvac_by_rung": nv, "frozen_bars": bars}
