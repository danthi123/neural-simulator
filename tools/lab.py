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

THE SEVENTH SHAPE (2026-07-31) IS NOT AN INSTRUMENT FAILURE, AND IS THE STRONGEST LEVER WE HAVE FOUND.
Biology runs INTERACTING processes -- potentiation AND heterosynaptic depression AND synaptic scaling AND
competition, each holding the others in a viable regime. We implement ONE and substitute a static proxy
(almost always a hard bound) for the rest. Then **the proxy DOMINATES and nothing says so**: at gap#5, 97%
of the measured weight change was the CLAMP, identical in the `lr=0` control, and four honest tuning steps
walked DEEPER into it because what the metric rewarded was clamp depth. The measurement was not wrong; it
simply belonged to something other than the mechanism under test.
  7. a static proxy owning most of the measured change -> term_budget()  /  attributable_to()
See research/findings/2026-07-31-why-we-hit-walls-the-missing-companion-process.md.
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


def term_budget(label, total_change, terms, dominance=0.8):
    """Split an observed change into NAMED terms and say, out loud, which one OWNS it.

    The general form of `sign_budget` above: that one asks *what sign* the change had, this one asks *whose
    change it was*. Pass the total you measured and a dict of the contributions you can name --
    `{"clamp": ..., "learning": ..., "decay": ...}` -- and every term is reported as a fraction of the total,
    with a loud flag when any single one exceeds `dominance` (default 0.8) and a line for whatever the named
    terms do NOT account for.

    EARNED BY gap#5 (2026-07-31). `w_max=150` sat below the initial weight `W0=250`, so the clamp dragged every
    weight down: mean|dW| was 22.6098 in the BTSP arm and 21.9393 in the `lr=0` control -- **97.0% of the
    measured weight change was the clamp**, 6/6 seeds, and nothing in the pipeline said so for weeks. The
    tuning then walked `w_max` 110 -> 150 -> 220 and selected 150 as an interior optimum, because what the
    metric rewarded was clamp depth. A clamp is not homeostasis: it is a SCALAR standing where biology runs a
    PROCESS, and a scalar proxy that owns 97% of your signal is the result you are actually publishing.

        term_budget("mean|dW| @ tuned point", 22.6098, {"clamp": 21.9393, "learning": 0.6705})

    THE DEGENERATE CASES ARE HANDLED HONESTLY, because normalising them away is how this hides:
      * total ~0 and all terms ~0 -> **NO CHANGE**. Nothing happened, so there is nothing to attribute. This is
        a clean null, NOT "every term dominates" (dividing ~0 by ~0 would report exactly that).
      * total ~0 while the terms are LARGE -> **CANCELLATION**, shares UNDEFINED. Earned by lane D, where
        potentiation-only toward a ceiling with no heterosynaptic depression made ON and OFF converge and the
        signed receptive field cancel -- `on_mean 9.189755`, `off_mean 9.16872`, net `0.021035`, i.e. 0.11% of
        the gross (raw/laneD_norm/base_42.json, whose own diagnosis field reads "COMMON-MODE CONVERGENCE").
        Two large processes, a net of nothing. Reporting a share there divides by ~0 and manufactures enormous
        fractions out of a null; and even ABOVE the zero threshold such a budget has no owner, so the dominance
        verdict is withheld once less than half the gross survives into the net (`cancelling`, `net_frac`).
      * total < 0 -> **NET DESTRUCTION**, said in those words. Shares are shares OF THE LOSS (a positive share
        means the term DROVE it, a negative share means it opposed it); the sign is never normalised away.

    Returns a dict: label, status (OK / NO_CHANGE / CANCELLATION / NET_DESTRUCTION / UNDEFINED), total, terms,
    shares, dominant (the owning term, or None -- also None when the budget is CANCELLING, where "share of the
    net" is not ownership), unexplained (the part the named terms do not account for), cancelling, net_frac
    (|total| / Σ|term|: how much of the gross activity survives into the net).
    """
    who = "term_budget(%s)" % label
    t = _scalar(who, "total_change", total_change)
    vals, bad = {}, (t is None)
    for k, v in (terms or {}).items():
        s = _scalar(who, "terms[%r]" % (k,), v)
        bad = bad or s is None
        vals[k] = s
    if not terms:
        print("  ⚠️  term_budget %s: NO TERMS named — a budget with no terms attributes nothing. UNDEFINED."
              % label)
        return dict(label=label, status="UNDEFINED", total=t, terms={}, shares={}, dominant=None,
                    unexplained=None, cancelling=None, net_frac=None)
    if bad:
        print("  ⚠️  term_budget %s: UNDEFINED — a value was refused above; reporting shares over it would "
              "invent numbers." % label)
        return dict(label=label, status="UNDEFINED", total=t, terms=vals, shares={}, dominant=None,
                    unexplained=None, cancelling=None, net_frac=None)

    mass = sum(abs(v) for v in vals.values())
    tol = 1e-9 * max(mass, abs(t))
    print("  term_budget %s: total=%+.6g   (%d term%s, Σ|term|=%.6g)"
          % (label, t, len(vals), "" if len(vals) == 1 else "s", mass))

    if abs(t) <= tol:
        if mass <= tol:
            print("    => NO CHANGE: the total AND every term are ~0. Nothing happened, so there is nothing "
                  "to attribute — a clean null, NOT 'every term dominates'.")
            status = "NO_CHANGE"
        else:
            print("    => ⛔ CANCELLATION: the total is ~0 while the terms carry Σ|term|=%.6g. Shares are "
                  "UNDEFINED (dividing by ~0 manufactures enormous fractions). LARGE OPPOSING PROCESSES "
                  "CANCELLED — report that, do not normalise it away." % mass)
            for k, v in vals.items():
                print("       %-24s %+.6g" % (k, v))
            status = "CANCELLATION"
        return dict(label=label, status=status, total=t, terms=vals, shares={}, dominant=None,
                    unexplained=None, cancelling=(status == "CANCELLATION"),
                    net_frac=(abs(t) / mass if mass > 0 else 0.0))

    status = "OK"
    if t < 0:
        print("    ⛔ NET DESTRUCTION: the observed total is NEGATIVE (%+.6g). The shares below are shares OF "
              "THAT LOSS — a POSITIVE share means the term DROVE the destruction, a negative share means it "
              "opposed it. The magnitude is reported, not normalised away." % t)
        status = "NET_DESTRUCTION"

    # A budget whose net is a small residue of large opposing terms has shares that are arithmetically correct
    # and meaningless as OWNERSHIP: lane D's ON/OFF (9.19, -9.17, net 0.02) yields +45950% / -45850%, and every
    # term trips a dominance test that way. Flagging all of them "DOMINANT" is a false alarm, and this file
    # already records that a false alarm is as corrosive as a missed one (see sign_budget). So below half the
    # gross surviving, the dominance verdict is WITHHELD and the cancellation is reported instead.
    net_frac = abs(t) / mass if mass > 0 else 1.0
    cancelling = net_frac < 0.5

    shares, dominant = {}, None
    for k, v in vals.items():
        f = v / t
        shares[k] = f
        flag = ""
        if abs(f) > dominance and not cancelling:
            flag = "   ⛔ DOMINANT (>%.0f%%)" % (100 * dominance)
            if dominant is None or abs(f) > abs(shares[dominant]):
                dominant = k
        print("    %-24s %+.6g   %+.1f%% of total%s" % (k, v, 100 * f, flag))
    unexplained = 1.0 - sum(shares.values())
    _rest = unexplained * t
    print("    %-24s %+.6g   %+.1f%% of total%s"
          % ("(unexplained)", 0.0 if _rest == 0 else _rest, 100 * unexplained,
             "   ⚠️  the named terms do not account for the change" if abs(unexplained) > 0.05 else ""))
    if cancelling:
        print("    ⚠️  CANCELLING BUDGET: the net is only %.2f%% of Σ|term| — a small residue of LARGE OPPOSING "
              "terms. Shares are not ownership here (every one of them exceeds 100%%), so the dominance verdict "
              "is WITHHELD: read the magnitudes. This shape is lane D — potentiation with no heterosynaptic "
              "depression, ON and OFF converging, the signed receptive field cancelling."
              % (100 * net_frac))
    elif max(abs(f) for f in shares.values()) > 1.0 + 1e-12:
        print("    ⚠️  a share exceeds 100%: some terms partly OPPOSE each other, so the net is smaller than "
              "its parts. Read the magnitudes alongside the shares.")
    if dominant is not None:
        print("    ⛔ DOMINATED BY %r: %.1f%% of the observed change is ONE term. If that term is a static "
              "proxy for a process biology runs ALONGSIDE the mechanism (a clamp standing in for homeostasis), "
              "then what you measured is the proxy — not the mechanism under test."
              % (dominant, 100 * shares[dominant]))
    return dict(label=label, status=status, total=t, terms=vals, shares=shares, dominant=dominant,
                unexplained=unexplained, cancelling=cancelling, net_frac=net_frac)


def attributable_to(label, treatment_value, control_value, warn_below=0.5):
    """What FRACTION of an effect is NOT present in its control: `(treatment - control) / treatment`.

    The gap#5 97%-clamp calculation in one call. Both arms had already been measured and BOTH numbers were
    sitting in the same JSON object, one key apart, for weeks -- nobody subtracted them:

        attributable_to("lr @ tuned point", 22.6098, 21.9393)   # => 0.0297: the lever moves 3% of the change

    The rest -- 97.0% -- was the clamp, running identically in both arms. Note what that exposes about the
    control itself: `lr=0` holds the LEARNING fixed, it does not hold the PROXY fixed, so "the control is
    clean" was never the right reading. A control only bounds the terms it actually varies.

    DEGENERATE CASES, reported rather than papered over:
      * treatment ~0 and control ~0 -> UNDEFINED, returns None. No effect in either arm is a NULL; calling it
        0% or 100% attributable would fabricate an attribution out of an absence. (`lr=0` reading "circ_dW
        exactly 0.000000" was quoted as instrument validation when it meant every increment was negative.)
      * treatment ~0 with a non-zero control -> UNDEFINED, returns None: the denominator is ~0, and the control
        showing an effect the treatment does not is itself the finding.
      * a fraction > 1 means the control moved OPPOSITE the treatment; a fraction < 0 means the control
        EXCEEDED the treatment (the manipulation reduced the effect). Both are printed as such.
    """
    who = "attributable_to(%s)" % label
    t = _scalar(who, "treatment_value", treatment_value)
    c = _scalar(who, "control_value", control_value)
    if t is None or c is None:
        print("  ⚠️  attributable_to %s: UNDEFINED — a value was refused above." % label)
        return None
    tol = 1e-9 * max(abs(t), abs(c))
    print("  attributable_to %s: treatment=%+.6g  control=%+.6g  diff=%+.6g" % (label, t, c, t - c))
    if abs(t) <= tol:
        if abs(c) <= tol:
            print("    => UNDEFINED: BOTH arms are ~0. There is no effect to attribute — this is a null, NOT "
                  "0% or 100% attributable.")
        else:
            print("    => UNDEFINED: the treatment is ~0 while the control reads %+.6g. The denominator is ~0, "
                  "and a control showing an effect the treatment does not IS the finding." % c)
        return None
    f = (t - c) / t
    print("    => %.1f%% of the effect is attributable to the manipulation; %.1f%% is ALSO PRESENT IN THE "
          "CONTROL" % (100 * f, 100 * (1 - f)))
    if f < 0:
        print("    ⛔ NEGATIVE: the control EXCEEDS the treatment — the manipulation REDUCED the effect.")
    elif f > 1:
        print("    ⚠️  ABOVE 100%: the control moved OPPOSITE the treatment, so the difference exceeds the "
              "treatment's own magnitude.")
    elif f < warn_below:
        print("    ⛔ MOST OF THIS EFFECT IS IN THE CONTROL (%.1f%%). Whatever produces it is running "
              "identically in both arms — a clamp, a bound, a drive, an initialisation. The lever you are "
              "testing owns %.1f%% of it, and any conclusion drawn from the total belongs to the other term."
              % (100 * (1 - f), 100 * f))
    return f


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


def _scalar(who, name, x):
    """Coerce to a float scalar, or REFUSE loudly. Never silently reduces an array.

    The reduction is a scientific choice, not a formatting detail: `dW.sum()` and `np.abs(dW).mean()` answer
    different questions, and picking the rectified one is exactly how a destructive process came to read as a
    clean 0.000000 across six seeds. So an array is refused with instructions rather than reduced for you.
    """
    size = getattr(x, "size", None)                    # numpy scalars and 0-d arrays report size == 1
    if size is not None and size != 1:
        print("  ⚠️  %s: %s is an ARRAY of %s elements — REFUSED. Choose the reduction yourself and say which: "
              "float(dW.sum()) is the net, float(np.abs(dW).mean()) is the magnitude, and they can disagree "
              "completely (a cancelling array has a net of ~0 and a large magnitude)." % (who, name, size))
        return None
    if size is None and not isinstance(x, str) and (hasattr(x, "__len__") or hasattr(x, "__iter__")):
        print("  ⚠️  %s: %s is a sequence — REFUSED. Pass a scalar summary you chose deliberately." % (who, name))
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        print("  ⚠️  %s: %s=%r is not numeric — REFUSED (UNDEFINED, not 0)." % (who, name, x))
        return None


# --------------------------------------------------------------------------------------------------------
# SELF-CHECK.  `python tools/lab.py`  /  `python -m tools.lab`
# A check that cannot fail is this project's most expensive failure class (nine incidents), so this one both
# asserts real numbers AND proves at the end that it is capable of failing.
# --------------------------------------------------------------------------------------------------------
def _check(cond, msg):
    if not cond:
        raise AssertionError("SELF-CHECK FAILED: %s" % msg)
    print("     ✔ %s" % msg)


def _selfcheck():
    print("=" * 108)
    print("(1) ONE DOMINANT TERM — the real gap#5 numbers, seed 400 (AGG_clamp_budget.json)")
    print("=" * 108)
    r = term_budget("mean|dW| @ tuned point", 22.6098, {"clamp": 21.9393, "learning": 0.6705})
    _check(r["status"] == "OK", "status OK")
    _check(r["dominant"] == "clamp", "the DOMINANT term is named: 'clamp'")
    _check(abs(r["shares"]["clamp"] - 0.9703) < 1e-3, "clamp share == 97.0% (finding says 97.0)")
    _check(abs(r["shares"]["learning"] - 0.0297) < 1e-3, "learning share == 3.0%")
    _check(abs(r["unexplained"]) < 1e-9, "nothing unexplained (the two terms are exhaustive)")

    print("\n" + "=" * 108)
    print("(2) BALANCED TERMS — no single owner, so nothing must be flagged DOMINANT")
    print("=" * 108)
    r = term_budget("balanced", 10.0, {"a": 3.4, "b": 3.3, "c": 3.3})
    _check(r["status"] == "OK" and r["dominant"] is None, "no term flagged dominant")
    _check(abs(sum(r["shares"].values()) - 1.0) < 1e-12, "shares sum to 1.0")

    print("\n" + "=" * 108)
    print("(3a) ZERO TOTAL, ZERO TERMS — NO CHANGE, not 'everything dominates'")
    print("=" * 108)
    r = term_budget("clean null", 0.0, {"clamp": 0.0, "learning": 0.0})
    _check(r["status"] == "NO_CHANGE", "status NO_CHANGE")
    _check(r["dominant"] is None and r["shares"] == {}, "no shares invented from 0/0")

    print("\n" + "=" * 108)
    print("(3b) ZERO TOTAL, LARGE TERMS — CANCELLATION (real lane D numbers, raw/laneD_norm/base_42.json)")
    print("=" * 108)
    r = term_budget("laneD signed RF", 0.021035, {"ON": 9.189755, "OFF": -9.16872})
    _check(r["cancelling"] is True and r["net_frac"] < 0.01,
           "a 0.021035 net out of Σ|term|=18.36 is CANCELLING (0.11% of the gross survives)")
    _check(r["dominant"] is None, "NO false DOMINANT alarm: a share of tens of thousands of percent is not "
                                  "ownership, so the verdict is withheld instead of firing on every term")
    _check(abs(r["shares"]["ON"]) > 1.0, "the shares are still reported (>100%), the magnitudes are readable")
    r = term_budget("laneD exact cancel", 0.0, {"ON": 9.189755, "OFF": -9.189755})
    _check(r["status"] == "CANCELLATION", "status CANCELLATION")
    _check(r["shares"] == {}, "shares UNDEFINED, not manufactured by dividing by ~0")
    r = term_budget("mild opposition", 10.0, {"a": 12.0, "b": -2.0})
    _check(r["cancelling"] is False and r["dominant"] == "a",
           "mild opposition (71% of the gross survives) still names its owner — the withholding is scoped")

    print("\n" + "=" * 108)
    print("(4) NEGATIVE TOTAL — net destruction, reported as such and NOT normalised away")
    print("=" * 108)
    r = term_budget("net destruction", -10.0, {"clamp": -9.7, "learning": -0.3})
    _check(r["status"] == "NET_DESTRUCTION", "status NET_DESTRUCTION")
    _check(r["total"] < 0, "the negative total survives into the returned dict")
    _check(r["dominant"] == "clamp" and abs(r["shares"]["clamp"] - 0.97) < 1e-9,
           "clamp drove 97% of the LOSS (positive share of a negative total)")

    print("\n" + "=" * 108)
    print("(5) UNEXPLAINED RESIDUAL + REFUSED ARRAY")
    print("=" * 108)
    r = term_budget("partial budget", 10.0, {"named": 4.0})
    _check(abs(r["unexplained"] - 0.6) < 1e-12, "60% unexplained is stated, not hidden")
    r = term_budget("array", 10.0, {"dW": [1.0, 2.0, 3.0]})
    _check(r["status"] == "UNDEFINED", "an array term is REFUSED (the reduction is the caller's choice)")
    r = term_budget("no terms", 10.0, {})
    _check(r["status"] == "UNDEFINED", "an empty budget attributes nothing")

    print("\n" + "=" * 108)
    print("(6) attributable_to — the 97%-clamp calculation in one call, all four gap#5 arms")
    print("=" * 108)
    f = attributable_to("lr @ tuned point", 22.6098, 21.9393)
    _check(abs(f - 0.02966) < 1e-4, "3.0% attributable to lr => 97.0% is in the control (matches the finding)")
    f2 = attributable_to("field-quality config (bound ABOVE the weights)", 22.6098, 1.0)
    _check(f2 > 0.9, "a genuinely clean control leaves >90% attributable — the flag does NOT fire")
    _check(attributable_to("both arms null", 0.0, 0.0) is None, "both-arms-zero is UNDEFINED, not 0% or 100%")
    _check(attributable_to("treatment null", 0.0, 5.0) is None, "treatment ~0 is UNDEFINED (denominator ~0)")
    _check(abs(attributable_to("both negative", -10.0, -9.7) - 0.03) < 1e-9,
           "signs cancel correctly when both arms are negative: 3% attributable")
    _check(attributable_to("control opposes", 10.0, -10.0) == 2.0, "an opposing control reads >100%, flagged")
    _check(attributable_to("control exceeds treatment", 10.0, 12.0) == -0.2, "a negative fraction is flagged")
    _check(attributable_to("array", [1.0, 2.0], 1.0) is None, "an array arm is REFUSED, returns None")

    print("\n" + "=" * 108)
    print("(7) PROVE THE SELF-CHECK CAN FAIL — a self-check that only confirms good input is vacuous")
    print("=" * 108)
    try:
        _check(term_budget("meta", 22.6098, {"clamp": 21.9393, "learning": 0.6705})["dominant"] == "learning",
               "(deliberately wrong expectation: 'learning' is the dominant term)")
    except AssertionError as e:
        print("     ✔ the wrong expectation RAISED, as it must: %s" % e)
    else:
        raise SystemExit("⛔ THE SELF-CHECK CANNOT FAIL — it passed a deliberately wrong expectation.")

    print("\nSELF-CHECK PASSED: term_budget (dominant / balanced / zero / cancelling / negative / unexplained "
          "/ refused) + attributable_to (real gap#5 numbers, clean control, both degenerate zeros, negative "
          "arms, opposing control, refused array), and the checker was shown to be capable of failing.")


if __name__ == "__main__":
    _selfcheck()


# ---------------------------------------------------------------------------------------------------
# A verdict you have to EARN. Re-exported here because `tools.lab` is where this project looks for the
# experiment-hygiene helpers, but the implementation lives in tools/verdict.py (it is large enough to
# deserve its own file, and its selftest replays the five 2026-07-31 misses through one vocabulary).
#
#   from tools.lab import Verdict
#   v = Verdict("my probe", chance=1/k)
#   v.floor("held-out vs chance", acc, 1/k); v.require("depth-separating", sep, expect=True)
#   v.control("lesion", treatment=arm, control=lesion); v.reaches("lesion lands", before=x0, after=x1)
#   v.knob("lr", requested=a.lr, applied=cfg.lr); v.disabled("STP", why="isolation")
#   result = v.decide(go=...)          # GO | NO-GO | UNDEFINED, and UNDEFINED is the default
#   json.dump({**payload, **result}, f)   # emits the `preconditions` block gates/verdict_preconditions needs
# ---------------------------------------------------------------------------------------------------
from tools.verdict import Verdict, Check, GO, NO_GO, UNDEFINED  # noqa: E402,F401


def project_cost(label, unit_index, n_units, elapsed_s, warn_hours=8.0):
    """After unit 1 finishes, PROJECT the total from what it actually cost. Measured, not estimated.

    EARNED 2026-07-31, at a price. The gap#4 crux was planned at ~6h45m per cell and was actually ~23h:
    after printing its arm result each cell trained THREE MORE FULL NETS as anti-cheats, each the same cost
    as the arm. Nobody counted them, so 8 cells ran 9 hours toward a ~136 GPU-hour tail that could not have
    changed the verdict. The information needed to catch it existed at the 5h47m mark — the runner printed
    `(20539s)` for its first arm — and nothing multiplied it by the units remaining.

    This is deliberately NOT a config-parsing estimator. A parser has to know what each runner does, gets it
    wrong exactly when the runner does something unusual, and that is precisely the case that hurts. One
    finished unit is ground truth.

    Returns projected total seconds; prints, and flags loudly past `warn_hours`.
    """
    if not elapsed_s or unit_index < 1 or n_units < 1:
        print("  cost projection UNDEFINED for %s (unit_index=%s n_units=%s elapsed=%s)"
              % (label, unit_index, n_units, elapsed_s))
        return None
    per_unit = float(elapsed_s) / float(unit_index)
    total = per_unit * float(n_units)
    remaining = max(0.0, total - float(elapsed_s))
    print("  COST %s: unit %d/%d took %.0fs -> projected total %.1fh (%.1fh remaining)"
          % (label, unit_index, n_units, per_unit, total / 3600.0, remaining / 3600.0))
    if total / 3600.0 > warn_hours:
        print("  ⛔ PROJECTED %.1fh EXCEEDS %.1fh. Decide NOW whether the remaining %.1fh can change the "
              "verdict — on 2026-07-31 it could not, and the answer was already readable from unit 1."
              % (total / 3600.0, warn_hours, remaining / 3600.0))
    return total


def assert_backend(expected, note=""):
    """Assert the backend ACTUALLY in use, by importing it — not by inspecting the process.

    EARNED 2026-07-31, twice in one project. Runners do `os.environ.setdefault("SIM_BACKEND","numpy")`, so
    a caller who does not set it explicitly gets the CPU path silently, 10-50x slower. I then "verified" a
    GPU run by finding nvidia mappings in /proc/PID/maps — which only proves CuPy is IMPORTABLE, not used —
    while the runner's own first log line said "this run is on the CPU" in plain English. Checking a proxy
    for the thing instead of the thing.

    Raises on mismatch, because a run on the wrong device is not a slow run, it is a different experiment.
    """
    import os as _os
    actual = _os.environ.get("SIM_BACKEND", "numpy")
    on_gpu = False
    try:
        import cupy as _cp                                  # noqa: F401
        on_gpu = (actual == "cupy")
    except ImportError:
        on_gpu = False
    resolved = "cupy" if on_gpu else "numpy"
    print("  BACKEND declared=%s resolved=%s %s" % (expected, resolved, note))
    if resolved != expected:
        raise AssertionError(
            "backend mismatch: expected %r, actually running on %r (SIM_BACKEND=%r). A run on the wrong "
            "device is a different experiment, not a slower one. Set SIM_BACKEND explicitly at the call "
            "site — runners default it to numpy via setdefault, which silently wins." % (expected, resolved, actual))
    return resolved
