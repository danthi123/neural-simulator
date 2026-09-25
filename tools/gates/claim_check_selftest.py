"""CLASS CCT -- claim_check's own selftest, wired into the gate registry.

WHY. GATE 2 in tools/githooks/pre-commit runs `tools/claim_check.py` on staged findings but never its `--selftest`,
so the registry's founding rule -- refuse to trust a gate whose selftest cannot FAIL in the failing direction --
never covered the claims checker itself. Rounds 5-7 each shipped a claim_check that a later review showed was
passing wrong numbers; its SELFTEST_CASES (tools/claim_check_cases.py) now hold one repro per hole, so running them
on every commit is what keeps a regression from shipping silently.

  * `check()` runs claim_check's selftest and passes every problem through VERBATIM, each labelled BROKEN
    INSTRUMENT, so a regression blocks the commit in the instrument's own words (a crash is reported the same way).
  * `selftest()` proves only this WRAPPER's mechanics, with injected fake instruments (broken / healthy / crashing).
    It never depends on the real claim_check's current health -- if it did, a real regression would surface as
    "this gate failed its own selftest", the registry would skip `check()`, and the message saying WHAT broke
    would never print (round 6's version did exactly that).
"""
from __future__ import annotations

import types

import tools.claim_check as claim_check

NAME = "claim-check-selftest"
CLASS_ID = "CCT"
BLOCKING = True
_LABEL = ("BROKEN INSTRUMENT: tools/claim_check.py's own selftest fails, so GATE 2's verdicts on findings cannot be "
          "trusted until it is fixed -- ")


def check(paths, _cc=None):
    """Ignores `paths`: a registry-level health check on claim_check ITSELF."""
    cc = _cc if _cc is not None else claim_check
    try:
        problems = list(cc.selftest())
    except Exception as e:                         # a crashing instrument is as broken as a failing one
        problems = ["claim_check.selftest() CRASHED: %s: %s" % (type(e).__name__, e)]
    return [_LABEL + p for p in problems]


def selftest():
    problems = []
    msg = "SELFTEST BROKEN: case demo expected FAIL, got PASS (demo)"
    broken = types.SimpleNamespace(selftest=lambda: [msg])
    healthy = types.SimpleNamespace(selftest=lambda: [])

    def _boom():
        raise RuntimeError("instrument exploded")
    crashing = types.SimpleNamespace(selftest=_boom)

    got = check(None, _cc=broken)
    if len(got) != 1 or not got[0].startswith("BROKEN INSTRUMENT") or not got[0].endswith(msg):
        problems.append("did not pass a broken instrument's problem through verbatim with the BROKEN INSTRUMENT "
                        "label: %r" % got)
    if check(None, _cc=healthy):
        problems.append("reported problems for a HEALTHY fake instrument")
    crashed = check(None, _cc=crashing)
    if len(crashed) != 1 or "CRASHED" not in crashed[0] or "instrument exploded" not in crashed[0]:
        problems.append("did not report a CRASHING instrument: %r" % crashed)
    if not callable(getattr(claim_check, "selftest", None)):
        problems.append("tools/claim_check.py exposes no selftest() to wrap")
    return problems
