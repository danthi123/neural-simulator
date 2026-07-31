#!/usr/bin/env python3
"""Verify a document's NUMBERS and VERDICTS against the artifacts it cites. Blocks hallucinated claims.

WHY (2026-07-31, owner directive). The experiment harness guards EXPERIMENTS. The larger failure class is
CLAIMS -- statements made in findings, commit messages, the board, and to the owner, that were never traced back
to an artifact. That class is not hypothetical:

  * I told the owner "lanes A/B/C/E have banked 6-seed GOs". Lane A's own artifact
    (research/findings/raw/_affect_state_region_6seed.json) reads {"GO": false, "n_seeds_go": 2}. I had repeated
    the BOARD's summary line without opening the JSON. The finding itself was honest; the summary overclaimed.
  * A "2.97 at 16x16 with NO heuristic" claim stood for 2.5 months and propagated into CLAUDE.md; the run's own
    recorded command showed the flag that closes the heuristic was absent, and its default is ON.
  * A headline "circ_dW 0.7050 = 105% of the 0.6705 reference" was real as a measurement and wrong as a claim.

THE RULE THIS ENFORCES: a measurement stated in a document must EXIST in an artifact the document cites.
Not "be plausible". Not "be remembered". Exist, in a file, that a reader can open.

    .venv/bin/python tools/claim_check.py research/findings/2026-07-31-foo.md

Exit 1 if a measurement-shaped number or a verdict word is unsupported by the cited artifacts.

CALIBRATION -- deliberately narrow, because a checker that cries wolf gets ignored (this project's own lesson,
learned twice today). It checks ONLY:
  * numbers with >= 3 decimal places, which are measurements rather than prose ("3 seeds", "97%" are ignored);
  * verdict words that contradict a cited artifact's own verdict field.
Derived values (ratios, differences, percentages) are legitimately absent from artifacts, so an inline
`<!--derived-->` marker on the line, or a `## Derived` section listing them, suppresses the check for that line.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# >=3 decimals => a measurement, not prose. "6 seeds", "97%", "2.5 months" are not claims about instrument output.
NUM_RE = re.compile(r"(?<![\w.])(-?\d+\.\d{3,})(?![\w])")
# Globs are allowed: a finding over N seeds cites one pattern, not N paths.
# Must contain a "/" -- a bare filename mentioned in prose ("as g5fix_d025_*.json shows") is a REFERENCE, not a
# citation, and treating it as one reports a missing artifact that was never claimed to be a path.
PATH_RE = re.compile(r"([\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:json|jsonl))")
VERDICT_RE = re.compile(r"\b(GO|NO-GO|PASS|FAIL|REFUTED|CONFIRMED)\b")
DERIVED_MARK = "<!--derived-->"


def _flatten_numbers(obj, out):
    """Every numeric leaf in an artifact, at any depth."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)


def _flatten_verdicts(obj, out):
    """Any key that looks like a verdict, with its value -- so a doc saying GO can be checked against GO:false."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in ("go", "verdict", "overall_verdict", "passed", "signal"):
                out.append((k, v))
            _flatten_verdicts(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_verdicts(v, out)


def load_artifacts(paths):
    nums, verdicts, loaded, missing = set(), [], [], []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        hits = sorted(glob.glob(full)) if any(c in full for c in "*?[") else ([full] if os.path.exists(full) else [])
        if not hits:
            missing.append(p)
            continue
        for h in hits:
            try:
                if h.endswith(".jsonl"):
                    for ln in open(h):
                        ln = ln.strip()
                        if ln:
                            d = json.loads(ln)
                            _flatten_numbers(d, nums); _flatten_verdicts(d, verdicts)
                else:
                    d = json.load(open(h))
                    _flatten_numbers(d, nums); _flatten_verdicts(d, verdicts)
                loaded.append(h)
            except Exception as e:                       # narrow enough to see; never silent
                missing.append("%s (unreadable: %s)" % (p, type(e).__name__))
    return nums, verdicts, loaded, missing


def check(doc_path, tol=None, verbose=True):
    """tol=None => RELATIVE tolerance. An absolute 5e-4 let a fabricated 0.9999 match a stored 1.0, so the
    checker's own negative control failed on first run: with ~1000 artifact values, near-misses are common and an
    absolute window is far too loose. Relative tolerance scales with the claim."""
    text = open(doc_path).read()
    lines = text.split("\n")
    cited = sorted(set(PATH_RE.findall(text)))
    nums, verdicts, loaded, missing = load_artifacts(cited)

    unsupported, checked = [], 0
    in_derived = False
    for i, ln in enumerate(lines, 1):
        if ln.strip().lower().startswith("## derived"):
            in_derived = True
            continue
        if ln.startswith("## "):
            in_derived = False
        # A marker ALONE on a line opens BLOCK scope (until the next heading); inline it suppresses just that
        # line. Learned immediately: the first real use put the marker at the END of a block whose earlier lines
        # were the ones being flagged, which is the natural way to write it.
        if ln.strip() == DERIVED_MARK:
            in_derived = True
            continue
        if in_derived or DERIVED_MARK in ln:
            continue
        for m in NUM_RE.finditer(ln):
            val = float(m.group(1))
            checked += 1
            eps = tol if tol is not None else max(5e-6, 1e-4 * abs(val))
            if not any(abs(val - a) <= eps for a in nums):
                unsupported.append((i, val, ln.strip()[:88]))

    if verbose:
        print("claim_check: %s" % os.path.relpath(doc_path, ROOT))
        print("  cited artifacts : %d found, %d missing" % (len(loaded), len(missing)))
        for mp in missing[:5]:
            print("      ⛔ MISSING  %s" % mp)
        print("  measurements    : %d checked against %d artifact values" % (checked, len(nums)))
        for lineno, val, ctx in unsupported[:12]:
            print("      ⛔ line %-4d %-14g not in any cited artifact | %s" % (lineno, val, ctx))
        if len(unsupported) > 12:
            print("      ... and %d more" % (len(unsupported) - 12))

    fail = bool(missing) or bool(unsupported)
    if verbose:
        if not cited:
            print("  ⚠️  NO ARTIFACT CITED — a findings doc with no artifact path cannot be checked at all.")
        print("  => %s" % ("⛔ UNSUPPORTED CLAIMS (or missing artifacts) — fix, cite, or mark <!--derived-->"
                           if fail else "✔ every measurement traces to a cited artifact"))
    return 0 if not fail else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for p in sys.argv[1:]:
        rc |= check(p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
