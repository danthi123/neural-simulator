#!/usr/bin/env python3
"""Retro-scan research/findings/*.md under the round-5 line-only claim_check AND under main's pre-round-5
block/section/standalone-marker checker (loaded read-only from git), for two purposes:

  1. CALIBRATION -- what should MIN_CHECK_FRACTION / LOW_COVERAGE_MIN_TOTAL (tools/claim_check.py) be set to?
     Reports the largest "all-derived" doc (checked=0, every numeric claim marked) under the NEW checker, which
     is the floor LOW_COVERAGE_MIN_TOTAL must sit above.
  2. INFORMATION -- how many EXISTING findings actually relied on block/section/standalone-marker scope: docs
     that use the pre-round-5 idiom at all (a standalone marker line / a '## Derived'-style heading / a
     `<!--/derived-->` close marker), and, more precisely, docs whose verdict FLIPS from PASS (under main's
     scope rules) to FAIL (under round 5's same-line-only rule) -- meaning main's block/section scope was
     actually hiding something the new rule now catches. `gates/*` and the pre-commit hook only ever check
     NEWLY ADDED findings, so this script never rewrites or re-gates anything already committed; it is purely
     informational.

    .venv/bin/python tools/claim_check_retro_scan.py [--since 2026-09-01] [--out path.tsv]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import importlib.util
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import tools.claim_check as cc                     # noqa: E402

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
# tools/claim_check.py immediately before round 5 (7e2edc08e = main's block/section/standalone-marker scope,
# the same SHA tests/test_claim_check_line_only.py calls "main").
_MAIN_SHA = "7e2edc08e"
_OLD_IDIOM_RE = re.compile(r"<!--/?derived-->|^\s*(?:>\s*)*#{1,6}\s*[*_`]*\s*derived\b", re.I | re.M)

_FIELDS = ["path", "date", "synthesis", "total_numeric", "checked", "suppressed_inline", "suppressed_synthesis",
           "low_coverage", "uses_old_idiom", "main_verdict", "new_verdict", "flips_pass_to_fail", "error"]


def _load_main_checker():
    """The pre-round-5 checker, loaded straight from git so this script cannot drift from what actually shipped."""
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/claim_check.py" % _MAIN_SHA],
                          capture_output=True, text=True, check=True).stdout
    tmp = os.path.join(ROOT, ".claim_check_retro_main_ref.py")
    open(tmp, "w", encoding="utf-8").write(src)
    try:
        spec = importlib.util.spec_from_file_location("claim_check_main_ref", tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(tmp)
    mod.ROOT = ROOT
    return mod


def _scan_one(path, main_mod):
    rel = os.path.relpath(path, ROOT)
    try:
        r = cc._scan(path)
    except Exception as e:                        # a doc this script cannot even read is reported, not fatal
        return dict(path=rel, error="%s: %s" % (type(e).__name__, e))
    text = open(path, encoding="utf-8", errors="replace").read()
    uses_idiom = bool(_OLD_IDIOM_RE.search(text))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc_main = main_mod.check(path, verbose=False)
    except Exception:
        rc_main = None
    rc_new = 1 if (r["missing"] or r["unsupported"] or r["low_coverage"]) else 0
    return dict(
        path=rel, synthesis=r["synthesis"], total_numeric=r["total_numeric"], checked=r["checked"],
        suppressed_inline=r["suppressed"]["inline"], suppressed_synthesis=r["suppressed"]["synthesis"],
        low_coverage=r["low_coverage"], uses_old_idiom=uses_idiom,
        main_verdict=("FAIL" if rc_main else "PASS") if rc_main is not None else "ERROR",
        new_verdict="FAIL" if rc_new else "PASS",
        flips_pass_to_fail=(rc_main == 0 and rc_new == 1),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD; only findings whose filename date is >= this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    main_mod = _load_main_checker()
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "research/findings/*.md"))):
        base = os.path.basename(path)
        m = _DATE_RE.match(base)
        date = m.group(1) if m else "0000-00-00"
        if args.since and date < args.since:
            continue
        row = _scan_one(path, main_mod)
        row["date"] = date
        rows.append(row)

    out = args.out or os.path.join(ROOT, "research/coordination/claimcheck_lineonly_retro_2026-09-25.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    n = len(rows)
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    idiom_n = sum(1 for r in ok if r.get("uses_old_idiom"))
    flip_n = sum(1 for r in ok if r.get("flips_pass_to_fail"))
    # LOW_COVERAGE never applies to a synthesis doc (`not synthesis` guards it in tools/claim_check.py), so a
    # synthesis doc with checked=0 is irrelevant to calibrating the floor -- exclude it.
    all_derived_totals = sorted(r["total_numeric"] for r in ok if not r.get("synthesis")
                                and r.get("checked") == 0 and (r.get("total_numeric") or 0) > 0)
    print("scanned %d finding(s)%s (%d unreadable)"
          % (n, (" since %s" % args.since) if args.since else " (whole corpus)", len(errors)))
    print("  use the pre-round-5 idiom (standalone marker / '## Derived' heading / close marker): %d / %d"
          % (idiom_n, len(ok)))
    print("  verdict FLIPS pass(main)->fail(new line-only): %d / %d" % (flip_n, len(ok)))
    if all_derived_totals:
        print("  all-derived NON-SYNTHESIS docs (checked=0, total_numeric>0): n=%d, max total_numeric=%d, "
              "top 5=%s" % (len(all_derived_totals), all_derived_totals[-1], all_derived_totals[-5:]))
    else:
        print("  all-derived NON-SYNTHESIS docs (checked=0, total_numeric>0): none")
    print("  wrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
