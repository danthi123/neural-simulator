"""CLASS CCT — claim_check's own selftest, wired into the gate registry.

WHY (round 6, 2026-09-25). GATE 2 in tools/githooks/pre-commit shells out to `tools/claim_check.py` against staged
findings but never runs `claim_check.py --selftest`, so the registry's founding rule -- refuse to trust a gate whose
selftest cannot FAIL in the failing direction -- never covered claim_check at all.

ROUND 7 (issue 11 of round 6's review). Round 6's version put the REAL `claim_check.selftest()` inside THIS gate's
`selftest()` ("the unmodified registry must report clean"). So when claim_check itself regressed, the registry saw
THIS WRAPPER fail its selftest, labelled the regression "problems against the UNMODIFIED registry (false
positive)", skipped `check()`, and the one message that says what broke never printed. Now:
  * `check()` runs claim_check's selftest and passes every problem through VERBATIM, each labelled
    "BROKEN INSTRUMENT", so a regression blocks the commit with its own words;
  * `selftest()` proves only the WRAPPER's mechanics, with injected fake instruments (a broken one must be
    reported verbatim, a healthy one must be clean, a crashing one must be reported) -- it never depends on the
    real claim_check's current health, which is `check()`'s job.
The same behaviour is pinned from the other side by claim_check's own SELFTEST_CASES entry
`cct_gate_reports_broken_instrument_verbatim`, re-derived against round 6's copy of this file from git.
"""
from __future__ import annotations

import types

import tools.claim_check as claim_check

NAME = "claim-check-selftest"
CLASS_ID = "CCT"
BLOCKING = True
_LABEL = ("BROKEN INSTRUMENT: tools/claim_check.py's own selftest fails, so GATE 2's verdicts on findings cannot "
          "be trusted until it is fixed -- ")


def check(paths, _cc=None):
    """Ignores `paths`: a registry-level health check on claim_check ITSELF. Returns its selftest problems
    verbatim, labelled, which BLOCKS the commit."""
    cc = _cc if _cc is not None else claim_check
    try:
        problems = list(cc.selftest())
    except Exception as e:                     # a crashing instrument is as broken as a failing one -- and LOUD
        problems = ["claim_check.selftest() CRASHED: %s: %s" % (type(e).__name__, e)]
    return [_LABEL + p for p in problems]


def selftest():
    """The registry's contract (demonstrate FAILING in the failing direction), on the wrapper alone."""
    problems = []
    msg = "SELFTEST BROKEN: case demo expected FAIL, got PASS (demo)"
    broken = types.SimpleNamespace(selftest=lambda: [msg])
    healthy = types.SimpleNamespace(selftest=lambda: [])

    def _boom():
        raise RuntimeError("instrument exploded")
    crashing = types.SimpleNamespace(selftest=_boom)

    got = check(None, _cc=broken)
    if not got or not all(p.startswith("BROKEN INSTRUMENT") for p in got) or not any(msg in p for p in got):
        problems.append("did NOT pass a broken instrument's selftest problem through verbatim with the BROKEN "
                        "INSTRUMENT label: %s" % got)
    if check(None, _cc=healthy):
        problems.append("reported problems for a HEALTHY fake instrument")
    crashed = check(None, _cc=crashing)
    if not crashed or "CRASHED" not in crashed[0]:
        problems.append("did not report a CRASHING instrument: %s" % crashed)
    if not hasattr(claim_check, "selftest") or not getattr(claim_check, "SELFTEST_CASES", None):
        problems.append("tools/claim_check.py exposes no selftest()/SELFTEST_CASES to wrap")
    return problems
