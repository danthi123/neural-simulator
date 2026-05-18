"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the TD
value-function critic. Instrument-validity FIRST, fail-closed: a
V1-broken or non-discriminating run is VOID -- explicitly NOT a science
PASS/FAIL. Frozen _TDC_* are pre-registered and NEVER tuned. Mirrors
the hardened dendritic_fair_core discipline. ASCII only."""
from __future__ import annotations
import math

_TDC_V1_VALUE_RMSE_MAX = 0.05
_TDC_TRANSFER_MIN = 0.90
_TDC_US_DECAY_MAX = 0.15
_TDC_MIN_SEEDS = 3

_CONTROLS = ("no_bootstrap", "permuted", "wrongsign")


def _finite(x):
    """Strict: reject non-numeric (a bare float('0.9') would let the
    string '0.9' through). Return finite float or None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _passes_signature(vr, tr, ud):
    """The SAME criterion a science PASS uses. A control 'genuinely
    fails' iff it does NOT reproduce this valid finite signature; a
    diverged/non-finite control = correctly failed (NOT
    non-discriminating)."""
    vrf, trf, udf = _finite(vr), _finite(tr), _finite(ud)
    if vrf is None or trf is None or udf is None:
        return False
    return (vrf <= _TDC_V1_VALUE_RMSE_MAX and trf >= _TDC_TRANSFER_MIN
            and udf <= _TDC_US_DECAY_MAX)


def tdc_verdict(per_seed: dict) -> dict:
    bars = {"V1_VALUE_RMSE_MAX": _TDC_V1_VALUE_RMSE_MAX,
            "TRANSFER_MIN": _TDC_TRANSFER_MIN,
            "US_DECAY_MAX": _TDC_US_DECAY_MAX,
            "MIN_SEEDS": _TDC_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        # malformed harness (non-orderable seed keys): the instrument
        # did not soundly measure -> VOID, never raise (mirrors the
        # hardened dendritic_fair_core coerce-don't-raise doctrine).
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable (instrument did "
                          "not soundly measure)",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars, "per_seed": {str(s): per_seed[s]
                                              for s in seeds}}
    if len(seeds) < _TDC_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _TDC_MIN_SEEDS, **base}
    v1_ok = True
    science_ok = True
    controls_fail = True
    metrics_finite = True
    for s in seeds:
        d = per_seed[s]
        vr, tr, ud = (_finite(d.get("vrmse")), _finite(d.get("transfer")),
                      _finite(d.get("us_decay")))
        if vr is None or tr is None or ud is None:
            # a required science metric was non-numeric/non-finite ->
            # the instrument did NOT soundly produce a measurement ->
            # VOID (instrument-invalid), never a fabricated science
            # FAIL/PASS. Mirrors the hardened dendritic_fair_core
            # numeric-coercion -> VOID-not-raise doctrine.
            metrics_finite = False
        if vr is None or vr > _TDC_V1_VALUE_RMSE_MAX:
            v1_ok = False
        if not (tr is not None and ud is not None
                and tr >= _TDC_TRANSFER_MIN and ud <= _TDC_US_DECAY_MAX):
            science_ok = False
        ctrls = d.get("controls", {})
        for name in _CONTROLS:
            tup = ctrls.get(name)
            if (tup is None or not isinstance(tup, (tuple, list))
                    or len(tup) != 3):
                # missing control == cannot certify discrimination
                controls_fail = controls_fail and False
                continue
            if _passes_signature(*tup):
                controls_fail = False
    instrument_valid = bool(v1_ok and controls_fail and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: critic did NOT converge to analytic "
                       "V* (instrument unsound)")
        if not controls_fail:
            why.append("a discriminating control passed the signature "
                       "(instrument non-discriminating)")
        if not metrics_finite:
            why.append("a required science metric was non-numeric/"
                       "non-finite (instrument did not soundly measure)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
