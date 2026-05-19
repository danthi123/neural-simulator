"""Pure FIXED-bar success-criteria + necessity-verdict for the full
spiking-network integrated-loop test. Scores the lesion study: the
full closed loop must succeed on BOTH the working-memory query and the
episodic-sequence recall at every load; each of the three SHARED
systems (the combinatorial binding step, the one shared theta-gamma
timing rhythm, the fast hippocampal store) must collapse BOTH readouts
when lesioned (the decisive emergent-from-integration signature,
already robustly confirmed by the cheap preliminary simulation); each
HELPER system must collapse the readout it is responsible for.

Mirrors the adversarial-hardened frozen-verdict discipline EXACTLY:
instrument-validity FIRST; fail-closed; fixed bars pre-registered HERE
and NEVER tuned to a result; "cannot conclude" (VOID) strictly
distinct from "fails" (FAIL); malformed / non-numeric / unorderable
input -> VOID, never an exception. Owns its OWN frozen bars; imports
no other verdict module; standard library + typing only; no torch,
no autograd. ASCII only.

A-priori justification of every frozen value (defensible WITHOUT
reference to any observed run):
- _IL_LADDER = (2, 4, 8): compositional load = number of role-filler
  bindings held and composed simultaneously. Two is the smallest
  non-trivial composition; the ladder doubles to a load where a
  scale-confidence claim actually lives. Same geometric-doubling
  shape as the other scale-confidence ladders in this project.
- _IL_V1_MIN = 0.90: the full loop, on a NO-GAP trivial single bind,
  must nearly perfectly learn the bijection, or the instrument cannot
  even measure composition (this is soundness, not science).
- _IL_SCI_MIN = 0.80: the full loop must clear a clear-majority bar
  on the genuine compositional task on BOTH readouts. Same value
  family as the project's other validated science bars.
- _IL_LESION_MAX = 0.40: a lesioned readout has "collapsed" iff it
  is at/near chance. For a 1-of-N readout chance is <= 0.5 (N=2) and
  lower for larger N; 0.40 is a defensible at/near-chance ceiling and
  is the SAME value the cheap preliminary simulation used for its
  ablation ceiling. A lesion that does NOT collapse its responsibility
  means the capability is not genuinely emergent-from-integration ->
  the instrument cannot discriminate emergence from a wiring artifact
  -> VOID (NOT a science PASS/FAIL), exactly the compose-bridge-core
  "a control learned -> VOID" discipline.
- _IL_SCALE_TOL = 0.10: a stochastic multi-seed accuracy has a noise
  floor; 0.10 is a defensible max permitted DROP between ascending
  rungs, same magnitude family as the other validated tolerances.
- _IL_MIN_SEEDS = 3: below three seeds a multi-seed claim is not
  supportable.
These are pre-registered in this file, BEFORE any full-model run, and
NEVER tuned to an outcome."""
from __future__ import annotations
import math
from typing import Dict

_IL_LADDER = (2, 4, 8)
_IL_V1_MIN = 0.90
_IL_SCI_MIN = 0.80
_IL_LESION_MAX = 0.40
_IL_SCALE_TOL = 0.10
_IL_MIN_SEEDS = 3

# Pre-registered lesion partition. SHARED systems must collapse BOTH
# readouts (non-separability). HELPER systems must collapse the
# readout each is responsible for. This partition is itself frozen.
_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")
_HELPER_WM = ("no_bg_gate",)
_HELPER_EP = ("no_sequencing", "no_cls_replay")
_HELPER_BOTH = ("no_neuromod_timing",)

_ALL_LESIONS = _SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH


def _num(x):
    """Strict finite real or None. bool is NOT a number here; a
    control/metric serialized as a string or bool must force VOID,
    never silently pass."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair(d):
    """Return (wm, ep) as finite floats, or None if malformed."""
    if not isinstance(d, dict):
        return None
    wm = _num(d.get("wm"))
    ep = _num(d.get("ep"))
    if wm is None or ep is None:
        return None
    return (wm, ep)


def integrated_loop_verdict(rungs) -> Dict:
    """Pure, deterministic, fail-closed. Recomputed from the single
    recorded JSON; NEVER raises.

    rungs: list of per-load dicts, each:
      {"N": int, "n_seeds": int,
       "v1":   {"wm": float, "ep": float},   # no-gap trivial bind
       "full": {"wm": float, "ep": float},   # genuine composition
       "lesions": {<name>: {"wm": float, "ep": float}, ...}}

    Precedence (fail-closed, self-consistent): any soundness or
    discrimination defect -> VOID; else if every load meets the
    science bar AND composition is non-decreasing up to tolerance AND
    the largest load holds -> SCALE-CONFIDENT-PASS; else if the
    smallest (minimal-composition) load meets the science bar but
    scale confidence fails (a larger load drops below the bar, or the
    trend breaks, or the top is below the bar) -> GATE FAIL with
    classification WORKS-SMALL-NO-SCALE-CONFIDENCE (an honest
    non-success: works small, does not scale); else (the loop fails
    the science bar even at the smallest load) -> GATE FAIL with
    classification FAIL (it does not perform the capability at all)."""
    bars = {"LADDER": list(_IL_LADDER), "V1_MIN": _IL_V1_MIN,
            "SCI_MIN": _IL_SCI_MIN, "LESION_MAX": _IL_LESION_MAX,
            "SCALE_TOL": _IL_SCALE_TOL, "MIN_SEEDS": _IL_MIN_SEEDS}

    def void(reason):
        return {"GATE": "VOID", "instrument_valid": False,
                "classification": "VOID", "reason": reason,
                "frozen_bars": bars}

    if not isinstance(rungs, list) or not rungs:
        return void("rungs not a non-empty list")
    try:
        ordered = sorted(rungs, key=lambda r: r["N"])
    except (TypeError, KeyError):
        return void("rungs not orderable by N")
    try:
        ladder = tuple(int(r["N"]) for r in ordered)
    except (TypeError, ValueError, KeyError):
        return void("rung N not integer-coercible")
    if ladder != _IL_LADDER:
        return void("ladder %s != pre-registered %s (padding/"
                    "duplicate/missing-rung guard)"
                    % (ladder, _IL_LADDER))

    # ---- instrument validity FIRST (soundness + discrimination) ----
    full_min = []
    for r in ordered:
        ns = _num(r.get("n_seeds"))
        if ns is None or ns < _IL_MIN_SEEDS:
            return void("a rung has fewer than %d seeds"
                        % _IL_MIN_SEEDS)
        v1 = _pair(r.get("v1"))
        fu = _pair(r.get("full"))
        if v1 is None or fu is None:
            return void("v1/full readout pair missing or non-numeric")
        if v1[0] < _IL_V1_MIN or v1[1] < _IL_V1_MIN:
            return void("instrument unsound: the full loop did NOT "
                        "learn the no-gap trivial bind on both "
                        "readouts (>= %.2f) at N=%d"
                        % (_IL_V1_MIN, int(r["N"])))
        les = r.get("lesions")
        if not isinstance(les, dict):
            return void("lesions block missing/not a dict at N=%d"
                        % int(r["N"]))
        for name in _ALL_LESIONS:
            lp = _pair(les.get(name))
            if lp is None:
                return void("lesion '%s' missing/non-numeric at N=%d"
                            % (name, int(r["N"])))
            lw, le = lp
            if name in _SHARED or name in _HELPER_BOTH:
                if not (lw <= _IL_LESION_MAX and le <= _IL_LESION_MAX):
                    return void("non-discriminating: lesion '%s' did "
                                "NOT collapse BOTH readouts at N=%d "
                                "(wm=%.3f ep=%.3f, bar<=%.2f) -- the "
                                "capability is not emergent-from-"
                                "integration here / wiring artifact"
                                % (name, int(r["N"]), lw, le,
                                   _IL_LESION_MAX))
            elif name in _HELPER_WM:
                if not (lw <= _IL_LESION_MAX):
                    return void("non-discriminating: helper lesion "
                                "'%s' did NOT collapse the working-"
                                "memory readout at N=%d (wm=%.3f, "
                                "bar<=%.2f)"
                                % (name, int(r["N"]), lw,
                                   _IL_LESION_MAX))
            else:  # _HELPER_EP
                if not (le <= _IL_LESION_MAX):
                    return void("non-discriminating: helper lesion "
                                "'%s' did NOT collapse the episodic-"
                                "recall readout at N=%d (ep=%.3f, "
                                "bar<=%.2f)"
                                % (name, int(r["N"]), le,
                                   _IL_LESION_MAX))
        full_min.append(min(fu[0], fu[1]))

    # Instrument is sound + discriminating. Now the science verdict.
    # Precedence is ordered so every pre-registered classification is
    # reachable and means exactly what the design defines:
    #   PASS              -> every load meets the bar AND scales
    #   WORKS-SMALL(FAIL) -> minimal load works, but does not scale
    #   FAIL              -> does not even work at the minimal load
    base = {"instrument_valid": True, "frozen_bars": bars,
            "full_min_by_rung": full_min}
    all_science_ok = all(fm >= _IL_SCI_MIN for fm in full_min)
    monotone = all(full_min[i + 1] >= full_min[i] - _IL_SCALE_TOL
                   for i in range(len(full_min) - 1))
    top_ok = full_min[-1] >= _IL_SCI_MIN
    if all_science_ok and monotone and top_ok:
        return {"GATE": "PASS",
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "the full integrated loop succeeds on both "
                          "readouts at every load; every single-system "
                          "lesion collapses the capability it is "
                          "responsible for (the three shared systems "
                          "collapse both readouts together); "
                          "composition is non-decreasing up to "
                          "tolerance across the ascending load ladder "
                          "and holds at the largest load", **base}
    if full_min[0] >= _IL_SCI_MIN:
        why = []
        if not all_science_ok:
            why.append("a larger load falls below the science bar")
        if not monotone:
            why.append("composition drops > tolerance between "
                       "ascending loads")
        if not top_ok:
            why.append("the largest load is below the science bar")
        return {"GATE": "FAIL",
                "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
                "reason": "the loop performs the minimal composition "
                          "(smallest load >= the science bar) but is "
                          "NOT scale-confident: %s -- an honest "
                          "non-success (works small, does not scale)"
                          % "; ".join(why), **base}
    return {"GATE": "FAIL", "classification": "FAIL",
            "reason": "instrument sound+discriminating but the full "
                      "loop is below the science bar even at the "
                      "smallest (minimal-composition) load -- it does "
                      "not perform the capability at all", **base}
