"""Pure FIXED-bar THREE-STATE (VOID/PASS/FAIL) verdict for the
compose x temporal-credit gate. Instrument-validity FIRST,
fail-closed: a V1-broken or non-discriminating run is VOID --
explicitly NOT a science PASS/FAIL. Frozen _CTB_* pre-registered and
NEVER tuned. Mirrors the adversarial-hardened td_critic_core
discipline (strict numeric, malformed -> VOID-not-raise, diverged
control = correctly-failed, VOID strictly distinct from FAIL). Owns
its OWN bars; imports no other *_core. ASCII only."""
from __future__ import annotations
import math

_CTB_V1_ACC_MIN = 0.90
_CTB_SCIENCE_ACC_MIN = 0.90
_CTB_CONTROL_ACC_MAX = 0.35
_CTB_MIN_SEEDS = 3

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")


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


def _control_failed(x):
    """A control 'genuinely fails' (good) iff it did NOT learn. A
    genuinely diverged NUMERIC value (nan/inf) = correctly failed
    (good). Non-numeric junk (str/bool/None) is NOT a certified
    failure: a control that learned but was serialized as "0.99"/True
    must not pass as 'good' -> force VOID (fail-closed; mirrors the
    science-path and the hardened td_critic_core discipline; _finite
    already rejects these). A finite value ABOVE the chance bar means
    the control LEARNED -> the instrument is non-discriminating."""
    if x is None:
        return False
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False                      # non-numeric junk -> NOT good -> VOID
    f = _finite(x)
    if f is None:
        return True                       # numeric but nan/inf = diverged = good
    return f <= _CTB_CONTROL_ACC_MAX


def ctb_verdict(per_seed: dict) -> dict:
    bars = {"V1_ACC_MIN": _CTB_V1_ACC_MIN,
            "SCIENCE_ACC_MIN": _CTB_SCIENCE_ACC_MIN,
            "CONTROL_ACC_MAX": _CTB_CONTROL_ACC_MAX,
            "MIN_SEEDS": _CTB_MIN_SEEDS}
    try:
        seeds = sorted(per_seed.keys())
    except TypeError:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "per_seed keys not orderable (instrument did "
                          "not soundly measure)",
                "frozen_bars": bars, "per_seed": {}}
    base = {"frozen_bars": bars,
            "per_seed": {str(s): per_seed[s] for s in seeds}}
    if len(seeds) < _CTB_MIN_SEEDS:
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "fewer than %d seeds" % _CTB_MIN_SEEDS,
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
        if nogap is None or nogap < _CTB_V1_ACC_MIN:
            v1_ok = False
        if sci is None or sci < _CTB_SCIENCE_ACC_MIN:
            science_ok = False
        ctrls = d.get("controls", {})
        if not isinstance(ctrls, dict):
            controls_fail = False
            continue
        for name in _CONTROLS:
            if name not in ctrls:
                controls_fail = False     # cannot certify discrimination
            elif not _control_failed(ctrls.get(name)):
                controls_fail = False     # a control LEARNED
    instrument_valid = bool(v1_ok and controls_fail and metrics_finite)
    if not instrument_valid:
        why = []
        if not v1_ok:
            why.append("V1 unmet: TD harness did NOT learn the no-gap "
                       "bijection (instrument unsound)")
        if not controls_fail:
            why.append("a control learned / is missing -> temporal "
                       "credit is NOT the discriminator (instrument "
                       "non-discriminating)")
        if not metrics_finite:
            why.append("a required science metric was non-numeric/"
                       "non-finite (instrument did not soundly measure)")
        return {"GATE": "VOID", "instrument_valid": False,
                "reason": "; ".join(why), **base}
    return {"GATE": "PASS" if science_ok else "FAIL",
            "instrument_valid": True, "science_ok": bool(science_ok),
            **base}
