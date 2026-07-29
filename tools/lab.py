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
