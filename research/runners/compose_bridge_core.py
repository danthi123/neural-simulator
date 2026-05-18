"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
compose temporal-credit SPIKING-BRIDGE integration gate. Instrument-
validity FIRST, fail-closed: a V1-broken or non-discriminating in-sim
run is VOID -- explicitly NOT a science PASS/FAIL. Frozen _CBR_*
pre-registered and NEVER tuned. EXACT mirror of the adversarial-
hardened compose_bind_core discipline (strict numeric, malformed/junk
-> VOID-not-raise, diverged numeric control = correctly-failed, VOID
strictly distinct from FAIL). Owns its OWN bars; imports no other
*_core. ASCII only."""
from __future__ import annotations
import math

_CBR_V1_ACC_MIN = 0.80
_CBR_SCI_ACC_MIN = 0.80
_CBR_CTRL_ACC_MAX = 0.35
_CBR_MIN_SEEDS = 3

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")


def _finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _control_failed(x):
    """Genuinely diverged NUMERIC (nan/inf) = correctly failed (good).
    Non-numeric junk (str/bool/None) is NOT a certified failure (a
    learned control serialized as "0.9"/True must force VOID, not pass
    as good). A finite value above the bar means the control LEARNED
    -> non-discriminating."""
    if x is None:
        return False
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    f = _finite(x)
    if f is None:
        return True
    return f <= _CBR_CTRL_ACC_MAX


def cbr_verdict(per_seed: dict) -> dict:
    bars = {"V1_ACC_MIN": _CBR_V1_ACC_MIN,
            "SCI_ACC_MIN": _CBR_SCI_ACC_MIN,
            "CTRL_ACC_MAX": _CBR_CTRL_ACC_MAX,
            "MIN_SEEDS": _CBR_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable (instrument did "
                          "not soundly measure)",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CBR_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CBR_MIN_SEEDS,
                **base}
    v1_ok = True
    science_ok = True
    controls_fail = True
    metrics_finite = True
    for s in seeds:
        d = per_seed[s]
        nogap = _finite(d.get("nogap_td"))
        sci = _finite(d.get("td"))
        if nogap is None or sci is None:
            metrics_finite = False
        if nogap is None or nogap < _CBR_V1_ACC_MIN:
            v1_ok = False
        if sci is None or sci < _CBR_SCI_ACC_MIN:
            science_ok = False
        ctrls = d.get("controls", {})
        if not isinstance(ctrls, dict):
            controls_fail = False
            continue
        for name in _CONTROLS:
            if name not in ctrls:
                controls_fail = False
            elif not _control_failed(ctrls.get(name)):
                controls_fail = False
    instrument_valid = bool(v1_ok and controls_fail and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: in-bridge TD did NOT learn the "
                       "no-gap verb->motor bind (instrument unsound)")
        if not controls_fail:
            why.append("a control learned / is missing -> temporal "
                       "credit is NOT the in-bridge discriminator "
                       "(non-discriminating)")
        if not metrics_finite:
            why.append("a required metric was non-numeric/non-finite "
                       "(instrument did not soundly measure)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
