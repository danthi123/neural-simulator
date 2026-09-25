#!/usr/bin/env python3
"""Retro-compare tools/claim_check.py (the working copy, round 8) against main / r5 / r6 / r7 over existing findings,
with a CAUSE for every number round 8 fails, and the CALIBRATION of CHANCE_MAX and LOW_COVERAGE_MIN_TOTAL.

For every findings doc (filename date >= --since) it records each revision's verdict (loaded read-only from git, so
this script cannot drift from what shipped), round 8's verdict and failing rules, and for every number round 8 does
not accept, WHY -- one cause per number:
  too-broad          matched (exact/rounding/legacy) but its own chance-match rate exceeds CHANCE_MAX -- a number
                     CORRECT at its written precision when the rule was exact/rounding
  identifier         an arXiv id, a DOI, a URL or a file path (checked like any number, by design)
  code-fence         inside a fenced/indented code block (a marker there is code, so it cannot be exempted)
  code-span          inside an inline code span
  html-comment       inside an HTML comment
  reader-split       only the reader's reading holds it (markup/invisible character inside the digits)
  sign               the artifact holds the opposite sign (a dash read as a minus)
  near-miss          a cited value lies within 5 units of its last decimal (a wrong rounding, a truncation, or a
                     wrong number)
  no-pool            nothing was loaded to match against (no citation, or every citation missing)
  prose              anything else: an unmarked derived/aggregated/quoted number, or a real error
The gate itself only ever checks NEWLY ADDED findings; this script re-gates nothing -- it is a measurement.

    .venv/bin/python tools/claim_check_retro_compare.py --since 2026-09-01 --out research/coordination/X.tsv
    .venv/bin/python tools/claim_check_retro_compare.py --since 2026-09-01 --calibrate
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import glob
import importlib.util
import io
import os
import re
import subprocess
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import tools.claim_check as cc                      # noqa: E402

REVS = {"main": "7e2edc08e", "r5": "4fda849d4", "r6": "f2b7db2b4", "r7": "4ff05b018"}
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
_ID_RE = re.compile(r"(?:\b(?:https?|ftp)://|\bwww\.)\S+|\barxiv[:\s]*\d{4}\.\d{4,5}|\bdoi[:\s]*10\.\d{4,9}/\S+|"
                    r"\b10\.\d{4,9}/\S+|[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.\w{1,5}\b", re.I)
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_MODS = {}
FIELDS = ["path", "main", "r5", "r6", "r7", "r8", "r8_fail_rules", "r8_failing_numbers", "r8_causes",
          "r8_correct_only", "total", "checked", "exempt", "chance_p50", "chance_max", "synthesis"]


def _load(tag):
    if tag in _MODS:
        return _MODS[tag]
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:tools/claim_check.py" % REVS[tag]],
                         capture_output=True, text=True, check=True).stdout
    tmp = os.path.join(ROOT, ".claim_check_retro_%s_%d.py" % (tag, os.getpid()))
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("claim_check_retro_%s" % tag, tmp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(tmp)
    mod.ROOT = ROOT
    _MODS[tag] = mod
    return mod


def _rc(mod, path):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return "FAIL" if mod.check(path, verbose=False) else "PASS"
    except Exception:
        return "FAIL"                              # a crash blocks a commit too


def _fence_lines(text):
    """0-based line numbers inside fenced or indented code blocks, from the SAME parser round 8 uses."""
    out = set()
    for tok in cc._parsers()[1].parse(text):
        if tok.type in ("fence", "code_block") and tok.map:
            out.update(range(tok.map[0], tok.map[1]))
    return out


def _cause(rec, line, pool, fence, comment_spans_by_line):
    if rec["status"] == "too_broad":
        return "too-broad"
    if not pool:
        return "no-pool"
    if rec["reading"] == "reader":
        return "reader-split"
    num = rec["text"].lstrip("-").rstrip("kKMBGT")
    if any(num in m.group(0) for m in _ID_RE.finditer(line)):
        return "identifier"
    if rec["line"] - 1 in fence:
        return "code-fence"
    if any(num in m.group(2) for m in _CODE_SPAN_RE.finditer(line)):
        return "code-span"
    if any(num in c for c in comment_spans_by_line.get(rec["line"] - 1, ())):
        return "html-comment"
    if rec.get("hint", "").startswith("the artifact holds +"):
        return "sign"
    if rec.get("hint", "").startswith("near miss"):
        return "near-miss"
    return "prose"


def scan_one(path):
    rel = os.path.relpath(path, ROOT)
    row = dict(path=rel)
    for tag in REVS:
        row[tag] = _rc(_load(tag), path)
    r = cc._scan(path)
    row["r8"] = cc._verdict(r)
    rules = []
    if r.get("unreadable"):
        rules.append("unreadable")
    if r["missing"]:
        rules.append("missing(%d)" % len(r["missing"]))
    if r["unsupported"]:
        rules.append("unsupported(%d)" % len(r["unsupported"]))
    if r["too_broad"]:
        rules.append("too_broad(%d)" % len(r["too_broad"]))
    if r["low_coverage"]:
        rules.append("low_coverage")
    row["r8_fail_rules"] = " ".join(rules)
    row["total"], row["checked"] = r["total_numeric"], r["checked"]
    row["exempt"] = r["suppressed"]["inline"]
    row["synthesis"] = r["synthesis"]
    ch = sorted(r["chance"])
    row["chance_p50"] = "%.3f" % ch[len(ch) // 2] if ch else ""
    row["chance_max"] = "%.3f" % ch[-1] if ch else ""
    causes = collections.Counter()
    correct_flags, n_fail = 0, 0
    detail = []
    if not r.get("unreadable"):
        text = open(path, encoding="utf-8").read()
        lines = text.split("\n")
        fence = _fence_lines(text)
        comments = {}
        for m in cc._COMMENT_RE.finditer(text):
            li = text.count("\n", 0, m.start())
            for k, seg in enumerate(m.group(0).split("\n")):
                comments.setdefault(li + k, []).append(seg)
        for rec in r["records"]:
            if rec["status"] == "too_broad" or (rec["status"] == "checked" and rec["rule"] is None):
                n_fail += 1
                line = lines[rec["line"] - 1] if rec["line"] - 1 < len(lines) else ""
                c = _cause(rec, line, r["nums"], fence, comments)
                causes[c] += 1
                precise = rec["status"] == "too_broad" and (rec["rule"] or "").split("+")[0] in ("exact", "rounding")
                correct_flags += precise
                detail.append("%d:%s:%s:%s" % (rec["line"], rec["text"], c,
                                               ("%.2f" % rec["chance"]) if rec["chance"] is not None else ""))
    row["r8_failing_numbers"] = n_fail
    row["r8_causes"] = " ".join("%s=%d" % kv for kv in sorted(causes.items()))
    # A doc fails ONLY on numbers correct at their written precision: every failing number is a too-broad match at
    # exact/rounding precision, and nothing else fails it.
    row["r8_correct_only"] = (row["r8"] == "FAIL" and n_fail > 0 and correct_flags == n_fail
                              and not r["missing"] and not r["low_coverage"] and not r.get("unreadable"))
    row["_detail"] = detail
    row["_chance_recs"] = [(rec["chance"], (rec["rule"] or "").split("+")[0], rec["decimals"])
                           for rec in r["records"] if rec["status"] in ("checked", "too_broad")
                           and rec["rule"] is not None]
    row["_coverage"] = (r["synthesis"], r["total_numeric"], r["checked_visible_distinct"])
    return row


def _paths(since):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "research/findings/*.md"))):
        m = _DATE_RE.match(os.path.basename(path))
        if since and (not m or m.group(1) < since):
            continue
        out.append(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate", action="store_true", help="print the CHANCE_MAX / coverage-floor tables")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    paths = _paths(args.since)
    with Pool(args.jobs) as pool:
        rows = pool.map(scan_one, paths, chunksize=4)

    n = len(rows)
    print("scanned %d finding(s)%s" % (n, (" since %s" % args.since) if args.since else ""))
    for tag in list(REVS) + ["r8"]:
        print("  FAIL %-4s %d" % (tag, sum(1 for r in rows if r[tag] == "FAIL")))
    r8_fail = [r for r in rows if r["r8"] == "FAIL"]
    rule_docs = collections.Counter()
    for r in r8_fail:
        for w in r["r8_fail_rules"].split():
            rule_docs[w.split("(")[0]] += 1
    print("  r8 failing docs by rule (a doc can fail on several): %s" % dict(rule_docs))
    alone = collections.Counter()
    for r in r8_fail:
        ws = [w.split("(")[0] for w in r["r8_fail_rules"].split()]
        if len(ws) == 1:
            alone[ws[0]] += 1
    print("  r8 failing docs on ONE rule only: %s" % dict(alone))
    print("  r8 docs failing ONLY on numbers correct at their written precision (too broad, exact/rounding match): "
          "%d" % sum(1 for r in rows if r["r8_correct_only"]))
    causes = collections.Counter()
    cause_docs = collections.Counter()
    only_cause_docs = collections.Counter()
    for r in rows:
        cs = dict(kv.split("=") for kv in r["r8_causes"].split()) if r["r8_causes"] else {}
        for k, v in cs.items():
            causes[k] += int(v)
            cause_docs[k] += 1
        if len(cs) == 1 and r["r8"] == "FAIL" and not r["r8_fail_rules"].count("missing") \
                and "low_coverage" not in r["r8_fail_rules"]:
            only_cause_docs[list(cs)[0]] += 1
    print("  r8 failing NUMBERS by cause: %s" % dict(causes.most_common()))
    print("  r8 docs with >=1 failing number of that cause: %s" % dict(cause_docs.most_common()))
    print("  r8 docs whose ONLY failure is numbers of that one cause: %s" % dict(only_cause_docs.most_common()))
    for tag in REVS:
        flips = collections.Counter((r[tag], r["r8"]) for r in rows)
        print("  %s -> r8 verdict pairs: %s" % (tag, dict(flips)))

    if args.calibrate:
        recs = [x for r in rows for x in r["_chance_recs"]]
        prec = [c for c, rule, d in recs if rule in ("exact", "rounding")]
        leg = [c for c, rule, d in recs if rule == "legacy"]
        for name, xs in (("precision-tier matches", prec), ("legacy-tier matches", leg)):
            xs = sorted(xs)
            if not xs:
                continue

            def q(p):
                return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
            print("  chance over %d %s: p50 %.3f p75 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f"
                  % (len(xs), name, q(.5), q(.75), q(.9), q(.95), q(.99), xs[-1]))
        print("  CHANCE_MAX sweep (docs with >=1 matched claim above T / docs failing on breadth ALONE, i.e. no "
              "unsupported, missing, low-coverage):")
        for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
            above, alone_n, claims = 0, 0, 0
            for r in rows:
                hi = [c for c, rule, d in r["_chance_recs"] if c > t]
                claims += len(hi)
                if hi:
                    above += 1
                    other = ("unsupported" in r["r8_fail_rules"] or "missing" in r["r8_fail_rules"]
                             or "low_coverage" in r["r8_fail_rules"] or "unreadable" in r["r8_fail_rules"])
                    alone_n += not other
            print("    T=%.2f: %d claims above, %d docs above, %d docs would fail on breadth alone"
                  % (t, claims, above, alone_n))
        cov = sorted((tot, vis) for syn, tot, vis in (r["_coverage"] for r in rows) if not syn and tot)
        low = [(tot, vis) for tot, vis in cov if vis / float(tot) < cc.MIN_CHECK_FRACTION]
        print("  coverage: non-synthesis docs below %.0f%% visible-distinct-checked: %d; their totals (largest 8): %s"
              % (100 * cc.MIN_CHECK_FRACTION, len(low), sorted(low)[-8:]))

    out = args.out
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS + ["detail"], delimiter="\t", lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                r = dict(r)
                r["detail"] = " ".join(r["_detail"][:60])
                w.writerow(r)
        print("  wrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
