#!/usr/bin/env python3
"""Retro-compare tools/claim_check.py (the working copy, round 8) against main / r5 / r6 / r7 / r8a over findings,
with a CAUSE for every number round 8 fails, and the CALIBRATION of the per-precision chance limits (against false
positives AND, with --replay, against wrong numbers let through) and LOW_COVERAGE_MIN_TOTAL.

For every findings doc (filename date >= --since) it records each revision's verdict (loaded read-only from git, so
this script cannot drift from what shipped), round 8's verdict and failing rules, and for every number round 8 does
not accept, WHY -- one cause per number:
  too-broad          matched (exact/rounding/legacy) but its own chance-match rate exceeds the limit for its
                     precision -- MATCHED at its written precision when the rule was exact/rounding, which by the
                     checker's own logic does not show it is correct (a random number of its shape matches too)
  identifier         an arXiv id, a DOI, a URL or a file path (checked like any number, by design)
  code-fence         inside a fenced/indented code block (a marker there is code, so it cannot be exempted)
  code-span          inside an inline code span
  html-comment       inside an HTML comment
  reader-split       only the reader's reading holds it (markup/invisible character inside the digits)
  sign               the artifact holds the opposite sign (a dash read as a minus)
  near-miss          a cited value lies within 1.5 units of its last decimal (a truncation or a wrong rounding)
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

# r8a = round 8 as reviewed (654d95664); the working copy is reported as "r8".
REVS = {"main": "7e2edc08e", "r5": "4fda849d4", "r6": "f2b7db2b4", "r7": "4ff05b018", "r8a": "654d95664"}
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
_ID_RE = re.compile(r"(?:\b(?:https?|ftp)://|\bwww\.)\S+|\barxiv[:\s]*\d{4}\.\d{4,5}|\bdoi[:\s]*10\.\d{4,9}/\S+|"
                    r"\b10\.\d{4,9}/\S+|[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.\w{1,5}\b", re.I)
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1")
_MODS = {}
FIELDS = ["path", "main", "r5", "r6", "r7", "r8a", "r8", "r8_fail_rules", "r8_failing_numbers", "r8_causes",
          "r8_matched_only", "total", "checked", "exempt", "chance_p50", "chance_max", "synthesis"]


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
    if rec.get("hint", "").startswith(("the artifact holds +", "the artifact holds -")):
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
    matched_flags, n_fail = 0, 0
    detail = []
    if not r.get("unreadable"):
        text = open(path, encoding="utf-8").read()
        lines = text.split("\n")
        fence = _fence_lines(text)
        comments = {}
        for a, b in cc._comment_spans(text):
            li = text.count("\n", 0, a)
            for k, seg in enumerate(text[a:b].split("\n")):
                comments.setdefault(li + k, []).append(seg)
        for rec in r["records"]:
            if rec["status"] == "too_broad" or (rec["status"] == "checked" and rec["rule"] is None):
                n_fail += 1
                line = lines[rec["line"] - 1] if rec["line"] - 1 < len(lines) else ""
                c = _cause(rec, line, r["nums"], fence, comments)
                causes[c] += 1
                precise = rec["status"] == "too_broad" and (rec["rule"] or "").split("+")[0] in ("exact", "rounding")
                matched_flags += precise
                detail.append("%d:%s:%s:%s" % (rec["line"], rec["text"], c,
                                               ("%.2f" % rec["chance"]) if rec["chance"] is not None else ""))
    row["r8_failing_numbers"] = n_fail
    row["r8_causes"] = " ".join("%s=%d" % kv for kv in sorted(causes.items()))
    # A doc fails ONLY on numbers MATCHED at their written precision: every failing number is a too-broad match at
    # exact/rounding precision, and nothing else fails it.
    row["r8_matched_only"] = (row["r8"] == "FAIL" and n_fail > 0 and matched_flags == n_fail
                              and not r["missing"] and not r["low_coverage"] and not r.get("unreadable"))
    row["_detail"] = detail
    row["_n_fail"], row["_n_matched"] = n_fail, matched_flags
    row["_chance_recs"] = [(rec["chance"], (rec["rule"] or "").split("+")[0], rec["decimals"])
                           for rec in r["records"] if rec["status"] in ("checked", "too_broad")
                           and rec["rule"] is not None]
    row["_coverage"] = (r["synthesis"], r["total_numeric"], r["checked_visible_distinct"])
    row["_replay"] = _replay(r) if _REPLAY else []
    # What round 8 does with every number an OLDER revision flagged: are the older revision's failures of this doc
    # made ONLY of numbers round 8 matches at their written precision (a rounding main's window rejects -- the r5/r6
    # positive), or at least one it also rejects?
    for tag in REVS:
        row["_old_%s" % tag] = _old_flag_fates(tag, path, r) if row[tag] == "FAIL" else None
    return row


REPLAY_K = 10
_REPLAY = False


def _replay(r):
    """WRONG numbers made from the claims round 8 MATCHES in this doc: each value shifted by k = +-1 .. +-REPLAY_K
    units of its last decimal (a typo, a wrong rounding, a neighbouring seed's value). For each: does main's rule
    accept it (relative window max(5e-6, 1e-4|x|), bare reading), does round 8's matching rule accept it, and at what
    chance rate (rated in the tier that matched)? Both rules see the SAME pool (round 8's), so the difference is the
    rule, not the citation parsing. Two samplings are read from the rows (see _replay_table):
      "distinct"  every distinct matched claim of the doc, both directions (the calibration's own);
      "review"    every matched claim record whose own chance is <= 0.20 (accepted by the flat limit round 8 was
                  reviewed with), repeats kept, +1 .. +REPLAY_K only -- the sampling the 2026-09-25 review described.
    -> [(decimals, main_ok, r8_matched, chance, in_distinct, in_review)]."""
    out = []
    pool = r["nums"]
    seen = set()
    cache = {}
    for rec in r["records"]:
        if rec["status"] not in ("checked", "too_broad") or rec["rule"] is None:
            continue
        key = (round(rec["value"], 12), rec["decimals"], rec["alts"])
        first = key not in seen
        seen.add(key)
        review = rec["chance"] <= 0.20
        if not (first or review):
            continue
        d, alts = rec["decimals"], rec["alts"]
        u = 10.0 ** (-d)
        for k in range(1, REPLAY_K + 1):
            for sgn in (1, -1):
                if not first and sgn < 0:
                    continue
                x = rec["value"] + sgn * k * u
                if (key, sgn, k) not in cache:
                    main_ok = cc._any_within(pool, x, max(5e-6, 1e-4 * abs(x)))
                    rule = cc._match_value(x, u, alts, pool)
                    ch = (cc._chance(cc.Claim(None, None, 0, x, d, u, alts, "", "raw"), pool, None, cc._tier(rule))
                          if rule else None)
                    cache[(key, sgn, k)] = (main_ok, rule is not None, ch)
                main_ok, matched, ch = cache[(key, sgn, k)]
                out.append((d, main_ok, matched, ch, first, review and sgn > 0))
    return out


def _set_replay(on):
    global _REPLAY
    _REPLAY = on


def _cap(caps, d):
    return caps.get(d, caps.get("default", cc.CHANCE_MAX))


def _replay_table(rows, caps, sampling="distinct"):
    """Acceptance of the replayed wrong numbers by stated precision: main vs round 8 under `caps`."""
    by_d = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for d, main_ok, matched, ch, in_distinct, in_review in r["_replay"]:
            if not (in_distinct if sampling == "distinct" else in_review):
                continue
            b = by_d[min(d, 6)]
            b[0] += 1
            b[1] += main_ok
            b[2] += matched and ch <= _cap(caps, d)
    return by_d


def _breadth_cost(rows, caps):
    """(docs failing, docs failing on breadth ALONE, flagged numbers matched at their written precision) if the
    per-precision limits were `caps`."""
    fail = alone = matched_flags = 0
    for r in rows:
        other = any(w in r["r8_fail_rules"] for w in ("unsupported", "missing", "low_coverage", "unreadable"))
        hi = [(c, rule, d) for c, rule, d in r["_chance_recs"] if c > _cap(caps, d)]
        matched_flags += sum(1 for _c, rule, _d in hi if rule in ("exact", "rounding"))
        fail += bool(other or hi)
        alone += bool(hi and not other)
    return fail, alone, matched_flags


_OUT_LINE_RE = re.compile(r"line\s+(\d+)\s+(-?[0-9.eE+-]+)\s+not in any cited artifact")


def _old_flags(tag, path):
    """[(line, value)] an older revision flags as unsupported, and whether it failed for any OTHER reason."""
    mod = _load(tag)
    if hasattr(mod, "_scan"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = mod._scan(path)
        except Exception:
            return None, True
        flags = [(ln, float(v)) for ln, v, _c in res.get("unsupported", ()) if ln]
        other = bool(res.get("missing") or res.get("low_coverage") or res.get("unreadable") or res.get("too_broad"))
        return flags, other
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.check(path, verbose=True)
    except Exception:
        return None, True
    out = buf.getvalue()
    other = "MISSING" in out or "LOW COVERAGE" in out
    return [(int(m.group(1)), float(m.group(2))) for m in _OUT_LINE_RE.finditer(out)], other


def _old_flag_fates(tag, path, r8):
    flags, other = _old_flags(tag, path)
    if flags is None:
        return {"error": 1}
    fates = collections.Counter()
    by_line = collections.defaultdict(list)
    for rec in r8["records"]:
        by_line[rec["line"]].append(rec)
    for ln, v in flags:
        cand = [x for x in by_line.get(ln, ()) if abs(x["value"] - v) <= 1e-6 * max(1.0, abs(v))]
        if not cand:
            fates["not-a-claim-in-r8"] += 1
            continue
        x = cand[0]
        rule = (x["rule"] or "").split("+")[0]
        if x["status"] == "exempt":
            fates["exempt-in-r8"] += 1
        elif rule in ("exact", "rounding"):
            fates["matched-at-precision"] += 1
        elif rule == "legacy":
            fates["legacy"] += 1
        else:
            fates["unsupported-in-r8"] += 1
    fates["_other_reason"] = int(other)
    fates["_n"] = len(flags)
    return fates


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
    ap.add_argument("--calibrate", action="store_true", help="print the chance-limit / coverage-floor tables")
    ap.add_argument("--replay", action="store_true",
                    help="also replay WRONG numbers (every matched claim shifted by +-1..+-%d units of its last "
                         "decimal) through main's rule and round 8's, by stated precision, with the breadth cost "
                         "of each per-precision limit" % REPLAY_K)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    paths = _paths(args.since)
    with Pool(args.jobs, initializer=_set_replay, initargs=(args.replay,)) as pool:
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
    print("  r8 docs failing ONLY on numbers matched at their written precision but too broad (exact/rounding "
          "match, chance over the limit -- not shown correct): %d" % sum(1 for r in rows if r["r8_matched_only"]))
    n_flag = sum(r["_n_fail"] for r in rows)
    n_ok = sum(r["_n_matched"] for r in rows)
    print("  r8 flagged numbers: %d, of which %d (%.0f%%) matched at their written precision but are too broad; "
          "failing docs: %d/%d (%.0f%%)" % (n_flag, n_ok, 100.0 * n_ok / max(1, n_flag), len(r8_fail), n,
                                            100.0 * len(r8_fail) / max(1, n)))
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
    for tag in REVS:
        fails = [r["_old_%s" % tag] for r in rows if r.get("_old_%s" % tag)]
        tot = collections.Counter()
        only_matched = 0
        for f in fails:
            for k, v in f.items():
                if not k.startswith("_"):
                    tot[k] += v
            if f.get("_n") and not f.get("_other_reason") and f.get("matched-at-precision", 0) == f["_n"]:
                only_matched += 1
        n = sum(v for k, v in tot.items() if k != "error")
        print("  %s: %d failing docs, %d flagged numbers; what round 8 makes of them: %s; %.0f%% matched at their "
              "written precision; docs failing ONLY on numbers matched at their written precision: %d"
              % (tag, len(fails), n, dict(tot.most_common()), 100.0 * tot["matched-at-precision"] / max(1, n),
                 only_matched))

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

    if args.replay:
        live = dict(cc.CHANCE_MAX_BY_DECIMALS, default=cc.CHANCE_MAX)
        for sampling, what in (("distinct", "every distinct matched claim, +-1..+-%d units of its last decimal"
                                % REPLAY_K),
                               ("review", "every matched claim record with chance <= 0.20, +1..+%d units (the "
                                "review's sampling)" % REPLAY_K)):
            tab = _replay_table(rows, live, sampling)
            print("  REPLAY (%s): wrong numbers from %s; accepted by main / by round 8 at its live limits %s:"
                  % (sampling, what, live))
            tot = [0, 0, 0]
            for d in sorted(tab):
                n, m, r8 = tab[d]
                tot = [a + b for a, b in zip(tot, tab[d])]
                print("    d=%s%s: %6d wrong, main accepts %5d (%5.2f%%), round 8 accepts %5d (%5.2f%%)%s"
                      % (d, "+" if d == 6 else " ", n, m, 100.0 * m / n, r8, 100.0 * r8 / n,
                         "" if r8 <= m else "   <-- MORE than main"))
            print("    all : %6d wrong, main accepts %5d (%5.2f%%), round 8 accepts %5d (%5.2f%%)"
                  % (tot[0], tot[1], 100.0 * tot[1] / max(1, tot[0]), tot[2], 100.0 * tot[2] / max(1, tot[0])))
        print("  per-precision limit sweep (d=3 limit / d=4 limit / d>=5 limit): wrong numbers accepted at d=3, d=4, "
              "all; docs failing; docs failing on breadth alone; flagged numbers matched at their written precision:")
        for c3, c4, c5 in ((0.20, 0.20, 0.20), (0.10, 0.20, 0.20), (0.08, 0.20, 0.20), (0.05, 0.20, 0.20),
                           (0.04, 0.20, 0.20), (0.03, 0.20, 0.20), (0.05, 0.15, 0.20), (0.04, 0.15, 0.20),
                           (0.03, 0.10, 0.20)):
            caps = {3: c3, 4: c4, "default": c5}
            accs = []
            for sampling in ("distinct", "review"):
                t = _replay_table(rows, caps, sampling)
                acc = {d: 100.0 * t[d][2] / t[d][0] for d in t if t[d][0]}
                allp = 100.0 * sum(t[d][2] for d in t) / max(1, sum(t[d][0] for d in t))
                accs.append("%5.1f%% %5.1f%% %5.1f%%" % (acc.get(3, 0.0), acc.get(4, 0.0), allp))
            fail, alone, mflags = _breadth_cost(rows, caps)
            print("    %.2f / %.2f / %.2f: distinct %s | review %s; %d failing, %d on breadth alone, %d flagged"
                  % (c3, c4, c5, accs[0], accs[1], fail, alone, mflags))

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
