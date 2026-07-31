"""The gate registry: one file per failure class, one interface, auto-discovered.

WHY AN ARCHITECTURE AND NOT MORE SCRIPTS (owner, 2026-07-31): "I want them all addressed... a strong, consistent,
anti-hallucinatory workflow that genuinely prevents all failure modes we've come across, and that we can easily
and without bloat continue to develop as new failure modes arise."

Bespoke scripts do not compose: each one needs its own wiring into the hook, its own output format, its own
test, and its own place in someone's memory. The measured consequence of that pattern here is stark -- 1330
runners, `tools/lab.py` imported by 2, `tools/experiment.py` by 0. Tools that must be remembered are not used.

THE CONTRACT. A gate is a module in this package exposing:

    NAME        : str          short id, e.g. "single-seed"
    CLASS_ID    : str          the failure class from docs/FAILURE_GATE_MATRIX.md, e.g. "9"
    BLOCKING    : bool         True => a violation blocks the commit. False => reported only.
    def check(paths: list[str]) -> list[str]     problems; empty means pass
    def selftest() -> list[str]                  MUST demonstrate the gate FAILING on a case it should catch

`selftest` is not optional and not a formality. Failure class 3 -- "check-that-cannot-fail-or-was-bypassed", 9
incidents -- is the class where the anti-drift mechanism is itself the defect: a `;` where `&&` was meant, a pipe
eating an exit status, a relevance count that made a gate unfailable, a nonsense query scoring 18 hits and
PASSING. Every one of those gates looked healthy while checking nothing. So the registry REFUSES to run a gate
whose selftest does not itself fail in the failing direction, and `run_all` reports that refusal loudly.

Adding a failure class is therefore: write one file here, implement two functions. No hook edit, no wiring.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def discover():
    """Every gate module in this package, sorted by CLASS_ID for stable output."""
    out = []
    for m in pkgutil.iter_modules([_PKG_DIR]):
        if m.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module("tools.gates.%s" % m.name)
        except Exception as e:                       # a broken gate must be LOUD, never silently absent
            out.append(("!" + m.name, None, "import failed: %s: %s" % (type(e).__name__, e)))
            continue
        if not all(hasattr(mod, a) for a in ("NAME", "CLASS_ID", "BLOCKING", "check", "selftest")):
            out.append(("!" + m.name, None, "does not implement the gate contract"))
            continue
        out.append((mod.NAME, mod, None))
    return sorted(out, key=lambda t: (t[2] is None, getattr(t[1], "CLASS_ID", "zz") if t[1] else "zz"))


def run_all(paths, verbose=True, selftest_first=True, budget_s=None):
    """Run every gate. Returns (blocking_problems, report_lines).

    A gate whose selftest does not FAIL in the failing direction is treated as BROKEN and its verdict is not
    trusted -- because a gate that cannot fail is indistinguishable from no gate, and this project has shipped
    four of those.
    """
    # PER-GATE TIME BUDGET. The registry timed out TWICE committing 390 staged files -- a real scalability
    # defect in the checking layer itself, and one that would block the Tier-2 audit outright (it stages far
    # more). A gate that exceeds its budget is reported as SKIPPED-ON-TIME rather than silently dropped: an
    # unbounded check that stalls a commit gets bypassed with --no-verify, which disables every OTHER gate too.
    import os as _os, time as _t
    budget_s = budget_s if budget_s is not None else float(_os.environ.get("GATE_BUDGET_S", "12"))
    blocking, report = [], []
    for name, mod, err in discover():
        if err:
            blocking.append("GATE %s IS BROKEN: %s" % (name, err))
            report.append("  ⛔ %-22s %s" % (name, err))
            continue
        if selftest_first:
            st = mod.selftest()
            if st:
                blocking.append("GATE %s FAILED ITS OWN SELFTEST: %s" % (name, "; ".join(st)))
                report.append("  ⛔ %-22s selftest FAILED: %s" % (name, "; ".join(st)[:80]))
                continue
        _t0 = _t.time()
        try:
            probs = mod.check(paths)
        except Exception as e:                       # a crashing gate must be LOUD, never silently absent
            blocking.append("GATE %s CRASHED: %s: %s" % (name, type(e).__name__, e))
            report.append("  ⛔ %-22s CRASHED: %s" % (name, type(e).__name__))
            continue
        _el = _t.time() - _t0
        if _el > budget_s:
            report.append("  ⏱  %-22s took %.1fs (budget %.0fs) — verdict kept, but this gate needs scoping"
                          % (name, _el, budget_s))
        tag = "BLOCK" if mod.BLOCKING else "warn "
        if probs:
            report.append("  %s %-22s %d problem(s)" % ("⛔" if mod.BLOCKING else "⚠️ ", name, len(probs)))
            for p in probs[:8]:
                report.append("        %s" % p)
            if len(probs) > 8:
                report.append("        ... and %d more" % (len(probs) - 8))
            if mod.BLOCKING:
                blocking += probs
        elif verbose:
            report.append("  ✔  %-22s [class %-2s %s]" % (name, mod.CLASS_ID, tag))
    return blocking, report
