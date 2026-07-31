#!/usr/bin/env python3
"""Executable experiment discipline. Import these instead of REMEMBERING the rules.

WHY THIS EXISTS. On 2026-07-28 a single session produced NINE retractions. Three were terminology
(docs/TERMS.md now covers those). SIX were INSTRUMENT failures, and every one shared one shape:
**a check that could not fail.** The rules to prevent them were written down — several in
.claude/skills/verify-go/SKILL.md the same day — and violated anyway, because a rule you must remember
is not a mechanism. These helpers make the rule execute.

    from tools.lab import lever, void_if, before_after, undefined_if_empty

THE SIX INSTRUMENT FAILURES AND THE HELPER THAT CATCHES EACH:
  1. an A/B whose flag was ALREADY set by the config -> lever()           (two identical arms)
  2. a measurement taken BEFORE the manipulation      -> before_after()   (zero delta is visible)
  3. a lesion that did not persist (weights regrew)   -> before_after()   (re-read at measure time)
  4. a broken getattr printing None for a whole A/B   -> lever()          (prints the actual values)
  5. a type error reported as a scientific null       -> undefined_if_empty()
  6. a metric too coarse to resolve a real effect     -> lever(continuous=)
"""
from __future__ import annotations


class LeverError(AssertionError):
    pass


def lever(name, before, after, required=True, continuous=None):
    """Assert a manipulation actually CHANGED something, and print what.

    `required=True` raises if the value did not move — the arms would be identical and the A/B void.
    `continuous` is an optional second, finer-grained quantity: a coarse metric (an argmax, a 0-3 count)
    can be blind to a real effect, so pass a continuous reading alongside it when you have one.

    Earned: --weighted-coincidence was a no-op because comp_dendritic already set the flag, and a whole
    6-run sweep compared two identical configurations.
    """
    moved = before != after
    tag = "MOVED" if moved else "UNCHANGED"
    extra = "" if continuous is None else "  continuous=%s" % (continuous,)
    print("  LEVER %-28s %s -> %s  [%s]%s" % (name, before, after, tag, extra))
    if required and not moved:
        raise LeverError(
            "lever %r did not move (%r -> %r): both arms are identical, so any A/B over this is VOID. "
            "If the value is already set by the config, set it EXPLICITLY in both directions." % (name, before, after))
    return moved


def before_after(name, read_fn, apply_fn, settle_fn=None, tol=0.0):
    """Read a quantity, apply a manipulation, re-read it, and (optionally) re-read AFTER settling.

    Returns (before, after, settled). Prints the deltas. Use for any lesion or write:
      * a ZERO delta means the measurement sat UPSTREAM of the manipulation (retraction #9: the weight
        table ran before the replay it claimed to characterise);
      * a settled value that reverts means the manipulation DID NOT PERSIST (a zeroed weight regrew to
        0.05 within five steps, during the very read meant to measure its absence).
    """
    b = read_fn()
    apply_fn()
    a = read_fn()
    s = None
    print("  %-28s before=%s after=%s  delta=%s" % (name, b, a, _delta(b, a)))
    if settle_fn is not None:
        settle_fn()
        s = read_fn()
        print("  %-28s after settling=%s  drift=%s %s"
              % ("", s, _delta(a, s), "<- MANIPULATION DID NOT PERSIST" if _abs(_delta(a, s)) > tol else ""))
    if _abs(_delta(b, a)) <= tol:
        print("  ⚠️  %s: zero delta — is this measurement UPSTREAM of the manipulation?" % name)
    return b, a, s


def undefined_if_empty(label, n_evaluable, score, total):
    """Print UNDEFINED rather than a score when nothing was evaluable.

    Earned: a CuPy-vs-NumPy type error was swallowed by a bare except, reported as "no engram core",
    and the verdict line then printed "own-is-max 0/3" — a NEGATIVE fabricated from a type bug, across
    three seeds, on the exact measurement meant to adjudicate a banked GO.
    """
    if not n_evaluable:
        print("  => %s: UNDEFINED — nothing was evaluable. This is NOT a score of 0/%s; reporting one "
              "would fabricate a negative out of an instrument failure." % (label, total))
        return None
    print("  => %s: %s/%s evaluable" % (label, score, n_evaluable))
    return score


def bound_check(rule, bound, weight, strict=True):
    """Assert a plasticity BOUND sits above the weights it governs. Raises by default.

    THE FIFTH INSTANCE EARNED THIS. CLAUDE.md has documented this trap for STDP (`stdp_w_max`), BDSP
    (`bdsp_w_max`), BTSP (`btsp_w_max`) and Hebbian (`hebbian_max_weight`), and states the pre-flight in
    plain words: "compare its bound against the ACTUAL weight". It was prose, so it was skipped a fifth
    time -- gap#5's tuned operating point ran `w_max=150` against an initial weight `W0=250`, so the clamp
    dragged every weight DOWN and **97% of the measured weight change was the clamp, identical in the
    `lr=0` control**. The tuning then walked DEEPER in (w_max 110 -> 150 -> 220, 150 chosen as "optimal"),
    because what the metric rewarded was clamp depth.

    A bound below the weights does not merely fail to learn: it destroys weights uniformly, which reads as a
    substrate limitation. Call this where the bound is chosen, not where it is used.

        bound_check("btsp_w_max", cfg.btsp_w_max, W0)
    """
    try:
        b, w = float(bound), float(weight)
    except (TypeError, ValueError):
        print("  ⚠️  bound_check(%s): non-numeric bound=%r weight=%r — NOT checked" % (rule, bound, weight))
        return None
    if b <= w:
        msg = ("BOUND TRAP: %s=%g is AT OR BELOW the weight it governs (%g). The clamp will drag weights "
               "DOWN, every increment goes negative, and the 'learning' arm becomes its own control. "
               "Raise the bound above the weights, or state explicitly why destruction is intended." % (rule, b, w))
        if strict:
            raise LeverError(msg)
        print("  ⛔ %s" % msg)
        return False
    print("  ✔ bound_check %s: %g > weight %g (headroom %.2fx)" % (rule, b, w, b / w if w else float("inf")))
    return True


def sign_budget(label, dW):
    """Report what FRACTION of a weight change is positive, before any rectifying metric hides it.

    Earned the same day as bound_check: `circ_resultant` clips negatives internally and returns 0.0 when the
    clipped sum is <= 0, so an `lr=0` control read "circ_dW exactly 0.000000" at every seed and was quoted as
    a clean control -- while that same arm's mean |dW| was 21.94. The zero meant EVERY increment was negative.
    Any metric built on a rectified quantity must report this alongside, or it silently scores the residual of
    a destructive process.
    """
    try:
        import numpy as _np
        a = _np.asarray(dW, dtype=float).ravel()
        if not a.size:
            print("  ⚠️  sign_budget(%s): empty — UNDEFINED, not 0" % label)
            return None
        pos = float((a > 0).mean()); neg = float((a < 0).mean())
        raw_tot = float(_np.abs(a).sum())
        # NO CHANGE AT ALL is the IDEAL lr=0 control, not destruction. Flagging it "MOSTLY DESTRUCTIVE" (which a
        # naive pos_mass<0.5 test does, since 0/0 -> 0) is a false alarm, and a false alarm is as corrosive as a
        # missed one -- it trains the reader to skim past the flag that matters. Caught on this helper's own
        # first real run, where the CORRECT w_max=2500 config printed it for a genuinely all-zero control.
        if raw_tot <= 0.0:
            print("  sign_budget %s: dW is EXACTLY ZERO everywhere — a clean no-change control, not destruction"
                  % label)
            return dict(frac_pos=0.0, frac_neg=0.0, pos_mass_frac=None, all_zero=True)
        pos_mass = float(a[a > 0].sum()) / raw_tot
        print("  sign_budget %s: %.1f%% of synapses positive, %.1f%% negative | %.1f%% of |dW| mass is positive%s"
              % (label, 100 * pos, 100 * neg, 100 * pos_mass,
                 "   ⛔ MOSTLY DESTRUCTIVE — a rectifying metric will hide this" if pos_mass < 0.5 else ""))
        return dict(frac_pos=pos, frac_neg=neg, pos_mass_frac=pos_mass, all_zero=False)
    except Exception as e:                                  # narrow enough to see, never silent
        print("  ⚠️  sign_budget(%s) failed: %s: %s" % (label, type(e).__name__, e))
        return None


def void_if(condition, reason):
    """Mark an arm VOID and say why, instead of letting its numbers be read as a result."""
    if condition:
        print("  ⛔ VOID ARM — %s. Do not interpret this arm's metrics." % reason)
    return bool(condition)


def _delta(a, b):
    try:
        return "%+.6g" % (b - a)
    except Exception:
        return "changed" if a != b else "0"


def _abs(d):
    try:
        return abs(float(d))
    except Exception:
        return 0.0 if d in ("0", "changed") else 1.0
