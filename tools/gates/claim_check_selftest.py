"""CLASS CCT — claim_check's own selftest was never wired into the gate registry.

WHY (round 6, 2026-09-25 -- round 5's own SOUND-WITH-ISSUES review, issue 3). GATE 2 in
tools/githooks/pre-commit shells out to `tools/claim_check.py` directly against staged findings, but never runs
`claim_check.py --selftest`. This registry's own founding rule -- refuse to trust a gate whose selftest cannot
FAIL in the failing direction (failure class 3, `gates/__init__`) -- therefore never covered claim_check at
all: a broken claim_check (round 5's own `errors="replace"` regression, which silently turned a blocking crash
into a clean, wrong PASS) would have shipped with nothing in THIS registry noticing, even though
`claim_check.selftest()` itself is perfectly capable of catching that exact class of regression.

This is a THIN wrapper, deliberately: every actual check is `tools.claim_check.selftest()` (round 6 closes it
with one SELFTEST_CASES entry per fix, cross-checked against every historical revision in git). There is
exactly one place the logic lives; this module's only job is to make the registry's `run_all()` -- and
therefore `tools/githooks/pre-commit` GATE 5 -- actually run it on every commit, and BLOCK when it fails.

NOT the same thing as GATE 2 (which scans staged findings' actual claims): GATE 2 asks "are THESE documents'
numbers supported"; this gate asks "is the INSTRUMENT that answers that question still trustworthy right now".
"""
from __future__ import annotations

import tools.claim_check as claim_check

NAME = "claim-check-selftest"
CLASS_ID = "CCT"
BLOCKING = True


def check(paths):
    """Ignores `paths` -- this is a registry-level health check on claim_check ITSELF (GATE 2 already scans
    staged findings directly; this only asks whether that scan is trustworthy). Returns claim_check's own
    selftest problems, which BLOCKS the commit if the instrument itself is broken."""
    return list(claim_check.selftest())


def selftest():
    """The registry's contract: demonstrate FAILING in the failing direction. We cannot re-introduce round 5's
    actual regression here without editing tools/claim_check.py, so instead we corrupt one SELFTEST_CASES
    entry's own recorded expectation (the same mechanism a real regression would trip: `selftest()` compares
    the actual scan verdict against what a case DECLARES it should be) and confirm this wrapper surfaces it,
    then confirm the real, unmodified registry reports clean."""
    problems = []
    if not claim_check.SELFTEST_CASES:
        return ["no SELFTEST_CASES to verify against -- claim_check.selftest() would trivially pass"]

    case = claim_check.SELFTEST_CASES[0]
    original = case["expect"]
    case["expect"] = "FAIL" if original == "PASS" else "PASS"
    try:
        broken = check(None)
    finally:
        case["expect"] = original             # never leave shared module state mutated
    if not broken:
        problems.append("did NOT detect a deliberately corrupted SELFTEST_CASES expectation -- this wrapper "
                        "cannot be trusted to catch a real claim_check regression")

    clean = check(None)
    if clean:
        problems.append("reported problems against the UNMODIFIED selftest registry (false positive): %s"
                        % clean)
    return problems
