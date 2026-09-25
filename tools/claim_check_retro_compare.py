#!/usr/bin/env python3
"""Retro-compare tools/claim_check.py (the working copy) against main / round 5 / round 6 over existing findings,
with a CAUSE column -- the measurement round 6 reported without committing its generator (round 6 review, issue 7).

For every findings doc (filename date >= --since) it records:
  * each revision's verdict (main 7e2edc08e, r5 4fda849d4, r6 f2b7db2b4 -- loaded read-only from git, so this
    script cannot drift from what shipped) and the working copy's (`cur`);
  * why `cur` fails, by RULE: unsupported(N) / missing(N) / low_coverage / too_broad(rate) / unreadable;
  * `cur`'s chance-match rate (rule B) and how its checked numbers matched (rule A: exact / rounding);
  * for every number round 6 flagged, what `cur` does with it now (the CAUSE column): `rounding` (a correct rounding
    of a cited value), `exact`, `identifier` (URL/DOI/arXiv/path, now skipped), `exempt` (marker in its own
    cell/segment, or a derived note), `still-unsupported`, or `not-a-number` (no longer read as a measurement);
  * the synthesis decision and, when the escape is refused, why.
The gate itself only ever checks NEWLY ADDED findings; this script re-gates nothing -- it is a measurement.

    .venv/bin/python tools/claim_check_retro_compare.py --since 2026-09-01 --out research/coordination/X.tsv
    .venv/bin/python tools/claim_check_retro_compare.py --legacy-tolerance ...   # rule A + the old relative window
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

REVS = {"main": "7e2edc08e", "r5": "4fda849d4", "r6": "f2b7db2b4"}
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")
_CORRECT = {"rounding", "exact", "truncation", "identifier", "sign-read"}
FIELDS = ["path", "main", "r5", "r6", "cur", "cur_fail_rules", "chance_rate", "checked", "exact", "rounding",
          "unsupported", "identifiers", "r6_flagged", "r6_flag_causes", "synthesis", "synthesis_note"]
_MODS = {}


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


def _r6_flags(path):
    return _flags("r6", path)[0]


def _flags(tag, path):
    """(flagged [(line, value)], missing_or_low_coverage) for one revision. r5/r6 expose `_scan`; main does not, so
    its per-line loop is replayed here from its own module's regexes and loader (main is frozen at 7e2edc08e)."""
    mod = _load(tag)
    try:
        if hasattr(mod, "_scan"):
            r = mod._scan(path)
            return ([(ln, v) for ln, v, _c in r["unsupported"] if ln > 0],
                    bool(r.get("missing") or r.get("low_coverage") or r.get("unreadable")))
        text = open(path).read()
        lines = text.split("\n")
        synthesis = text.startswith("---") and bool(mod.SYNTH_RE.search(text.split("\n---", 1)[0]))
        nums, _v, _l, missing = mod.load_artifacts(sorted(set(mod.PATH_RE.findall(text))))
        out, in_derived = [], False
        for i, ln in enumerate(lines, 1):
            if ln.strip().lower().startswith("## derived"):
                in_derived = True
                continue
            if ln.startswith("## "):
                in_derived = False
            if ln.strip() == mod.DERIVED_MARK:
                in_derived = True
                continue
            if in_derived or mod.DERIVED_MARK in ln or synthesis:
                continue
            for m in mod.NUM_RE.finditer(ln):
                val = float(m.group(1))
                if not any(abs(val - a) <= max(5e-6, 1e-4 * abs(val)) for a in nums):
                    out.append((i, val))
        return out, bool(missing)
    except Exception:
        return [], True


def _cause(flag_line, flag_val, recs, line_text):
    same = [x for x in recs if x["line"] == flag_line and abs(x["value"] - flag_val) <= 1e-9 * max(1, abs(flag_val))]
    flipped = [x for x in recs if x["line"] == flag_line and flag_val
               and abs(x["value"] + flag_val) <= 1e-9 * max(1, abs(flag_val))]
    if not same and flipped:
        # the older revision dropped a typographic minus (U+2212/U+2013); the current one reads the sign
        x = flipped[0]
        return "sign-read" if x["status"] == "checked" and x["rule"] else "sign-read-" + (x["rule"] or x["status"])
    if not same:
        if any(rx.search(line_text) for rx in cc._ID_RES):
            return "identifier"
        if "<!--" in line_text and "derived" in line_text.lower():
            return "exempt"
        return "not-a-number"
    x = same[0]
    if x["status"] == "exempt":
        return "exempt"
    if x["status"] == "synthesis":
        return "synthesis"
    return x["rule"] or "still-unsupported"


def one(path):
    rel = os.path.relpath(path, ROOT)
    row = dict(path=rel)
    for tag in REVS:
        row[tag] = _rc(_load(tag), path)
    r = cc._scan(path)
    row["cur"] = cc._verdict(r)
    rules = []
    if r.get("unreadable"):
        rules.append("unreadable")
    if r["missing"]:
        rules.append("missing(%d)" % len(r["missing"]))
    if r["unsupported"]:
        rules.append("unsupported(%d)" % len(r["unsupported"]))
    if r["low_coverage"]:
        rules.append("low_coverage")
    if r.get("too_broad"):
        rules.append("too_broad(%.3f)" % r["chance"])
    row["cur_fail_rules"] = ";".join(rules)
    row["chance_rate"] = "" if r["chance"] is None else "%.4f" % r["chance"]
    row["checked"] = r["checked"]
    row["exact"] = r["matched"].get("exact", 0)
    row["rounding"] = r["matched"].get("rounding", 0)
    row["unsupported"] = len(r["unsupported"])
    row["identifiers"] = r["identifiers"]
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    for tag in REVS:
        flags, other = _flags(tag, path)
        causes = collections.Counter(_cause(ln, v, r["records"], lines[ln - 1] if 0 < ln <= len(lines) else "")
                                     for ln, v in flags)
        correct = sum(n for k, n in causes.items() if k.split("+")[0] in _CORRECT)
        row["_%s_flags" % tag] = len(flags)
        row["_%s_correct" % tag] = correct
        # a RULE false positive: the revision failed the doc, and every number it flagged is one the current
        # checker shows is a correct rounding/truncation of a cited value or an identifier -- nothing else failed.
        row["_%s_rule_fp" % tag] = (row[tag] == "FAIL" and bool(flags) and correct == len(flags) and not other)
        if tag == "r6":
            row["r6_flagged"] = len(flags)
            row["r6_flag_causes"] = ";".join("%s:%d" % kv for kv in sorted(causes.items()))
    row["_cur_rule_fp"] = (row["cur"] == "FAIL" and not r["unsupported"] and not r["missing"]
                           and not r["low_coverage"] and bool(r.get("too_broad")))
    row["synthesis"] = r["synthesis"]
    row["synthesis_note"] = next((m for _l, k, m in r["warnings"] if k == "synthesis escape not applied"), "")
    text = open(path, encoding="utf-8", errors="replace").read()
    fm = cc._FRONTMATTER_RE.match(text)
    row["_declares_synthesis"] = bool(fm and cc.SYNTH_RE.search(fm.group(1)))
    if row["_declares_synthesis"] and not cc._fm_value(fm.group(1), "claim_check_reason"):
        # would the escape apply if the author ADDED a reason? (separates the reason rule from the verdict bar)
        patched = text.replace(fm.group(1), fm.group(1) + "\nclaim_check_reason: retro probe", 1)
        ok, _reason, why = cc._synthesis_status(patched, path)
        row["_with_reason"] = "applies" if ok else ("barred: " + (why or ""))
    else:
        row["_with_reason"] = ""
    row["_unsupported_detail"] = [(x["line"], x["text"], x.get("hint", "")) for x in r["records"]
                                  if x["status"] == "checked" and x["rule"] is None]
    row["_total_numeric"] = r["total_numeric"]
    row["_checked_visible_distinct"] = r["checked_visible_distinct"]
    return row


def _cur_only(path):
    """The current checker alone (for the --legacy-compare pass)."""
    r = cc._scan(path)
    return dict(path=os.path.relpath(path, ROOT), cur=cc._verdict(r), too_broad=bool(r.get("too_broad")),
                unsupported=len(r["unsupported"]), chance=r["chance"])


# The round-6 review's broad-citation scenarios (issue 1), re-measured under the current rules: 40 uniformly random
# WRONG numbers at 3 and 4 decimals against (a) the tracked 26 MB artifact, (b) the corpus's largest legitimate
# per-battery glob, (c) four whole raw directories cited at once.
ATTACKS = {
    "26 MB single artifact": ["research/findings/raw/v14_snr_stageB_fast_channel_clamp_cupy_v1.json"],
    "consol_opsweep_gpu battery glob": ["research/findings/raw/consol_opsweep_gpu/op*_seed42.json"],
    "4 raw directories": ["research/findings/raw/_lbf_borderline_isolated/*.json",
                          "research/findings/raw/_sleep_replay_capture_r2_smoke/*.json",
                          "research/findings/raw/metacog/*.json",
                          "research/findings/raw/vocal_action_selector_gate_v2/*.json"],
}


def _attacks():
    import random
    import tempfile
    print("  broad-citation scenarios (40 random WRONG numbers each; chance = rule B's decoy estimate):")
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".claim_check_retro_attack_") as d:
        for name, cites in ATTACKS.items():
            for dec in (3, 4):
                rng = random.Random(11)
                body = ("# Attack\n\n" + "".join("Artifact: `%s`\n" % c for c in cites) + "\n"
                        + "\n".join("The value was %.*f here." % (dec, rng.random()) for _ in range(40)) + "\n")
                p = os.path.join(d, "attack.md")
                with open(p, "w") as fh:
                    fh.write(body)
                s = cc._scan(p)
                passed = sum(1 for x in s["records"] if x["status"] == "checked" and x["rule"])
                print("      %-32s d=%d pool=%7d chance=%-6s wrong numbers accepted %2d/40 -> %s%s"
                      % (name, dec, len(s["nums"]), "n/a" if s["chance"] is None else "%.3f" % s["chance"],
                         passed, cc._verdict(s), " (TOO BROAD)" if s["too_broad"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--legacy-tolerance", action="store_true",
                    help="re-enable the pre-round-7 relative tolerance as a second rule-A window (measurement only)")
    ap.add_argument("--legacy-compare", action="store_true",
                    help="also re-run the current checker WITH the legacy tolerance and print the difference")
    ap.add_argument("--attacks", action="store_true", help="also measure the review's broad-citation scenarios")
    args = ap.parse_args(argv)
    if args.legacy_tolerance:
        cc.LEGACY_TOLERANCE = True
    docs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "research/findings/*.md"))):
        m = _DATE_RE.match(os.path.basename(p))
        if m and m.group(1) >= args.since:
            docs.append(p)
    for tag in REVS:                               # fail fast (and warm the parent) before forking
        _load(tag)
    with Pool(args.jobs, initializer=_init, initargs=(args.legacy_tolerance,)) as pool:
        rows = pool.map(one, docs, chunksize=2)

    out = args.out
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
    _summary(rows, args)
    if args.legacy_compare:
        with Pool(args.jobs, initializer=_init, initargs=(True,)) as pool:
            leg = pool.map(_cur_only, docs, chunksize=2)
        rates = sorted(x["chance"] for x in leg if x["chance"] is not None)
        print("  WITH the legacy relative tolerance as a second rule-A window: FAIL %d / %d (too_broad %d, "
              "unsupported docs %d, unsupported numbers %d; chance p90 %.3f, docs >= 0.5: %d)"
              % (sum(x["cur"] == "FAIL" for x in leg), len(leg), sum(x["too_broad"] for x in leg),
                 sum(1 for x in leg if x["unsupported"]), sum(x["unsupported"] for x in leg),
                 rates[int(0.9 * len(rates))] if rates else 0.0, sum(x >= 0.5 for x in rates)))
    if args.attacks:
        _attacks()
    if out:
        print("wrote %s" % os.path.relpath(out, ROOT))
    return 0


def _init(legacy):
    cc.LEGACY_TOLERANCE = legacy


def _summary(rows, args):
    n = len(rows)
    print("scanned %d finding(s) dated >= %s%s" % (n, args.since, "  [LEGACY_TOLERANCE on]" if args.legacy_tolerance
                                                  else ""))
    for tag in ("main", "r5", "r6", "cur"):
        fails = sum(r[tag] == "FAIL" for r in rows)
        rule_fp = sum(bool(r["_%s_rule_fp" % tag]) for r in rows)
        if tag == "cur":
            print("  %-4s FAIL %3d / %d; failing on breadth alone (chance-match rule): %d" % (tag, fails, n, rule_fp))
        else:
            fl = sum(r["_%s_flags" % tag] for r in rows)
            ok = sum(r["_%s_correct" % tag] for r in rows)
            print("  %-4s FAIL %3d / %d; failing ONLY on numbers the current checker shows are correct: %d; of its "
                  "%d flagged numbers, %d (%.0f%%) are correct roundings/identifiers"
                  % (tag, fails, n, rule_fp, fl, ok, 100.0 * ok / fl if fl else 0))
    rule_docs = collections.Counter()
    only = collections.Counter()
    for r in rows:
        if r["cur"] != "FAIL":
            continue
        kinds = [x.split("(")[0] for x in r["cur_fail_rules"].split(";") if x]
        for k in kinds:
            rule_docs[k] += 1
        only[" + ".join(kinds)] += 1
    print("  cur failures by rule (a doc can fail several):", dict(rule_docs))
    print("  cur failures by rule combination:", dict(only.most_common()))
    causes = collections.Counter()
    for r in rows:
        for kv in filter(None, r["r6_flag_causes"].split(";")):
            k, v = kv.split(":")
            causes[k] += int(v)
    print("  what cur does with the %d numbers r6 flagged: %s" % (sum(causes.values()), dict(causes.most_common())))
    tot = collections.Counter()
    for r in rows:
        tot["checked"] += r["checked"]
        tot["exact"] += r["exact"]
        tot["rounding"] += r["rounding"]
        tot["unsupported"] += r["unsupported"]
        tot["identifiers"] += r["identifiers"]
    print("  cur rule A over all checked numbers:", dict(tot))
    rates = sorted(float(r["chance_rate"]) for r in rows if r["chance_rate"] != "")
    if rates:
        q = lambda f: rates[min(len(rates) - 1, int(f * len(rates)))]           # noqa: E731
        print("  chance-match rate over %d docs with >=1 checked number: min %.3f  p50 %.3f  p90 %.3f  p95 %.3f  "
              "p99 %.3f  max %.3f" % (len(rates), rates[0], q(.5), q(.9), q(.95), q(.99), rates[-1]))
        for lo, hi in ((0, .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1.01)):
            print("      [%.2f, %.2f): %d" % (lo, hi, sum(lo <= x < hi for x in rates)))
        print("  CHANCE_MAX sensitivity (docs above the threshold / of those, docs that would fail on breadth ALONE):")
        for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            above = [r for r in rows if r["chance_rate"] != "" and float(r["chance_rate"]) > t]
            alone = [r for r in above if not re.search(r"unsupported|missing|low_coverage", r["cur_fail_rules"])]
            print("      %.2f: %3d / %3d%s" % (t, len(above), len(alone), "   <- CHANCE_MAX" if
                                              abs(t - cc.CHANCE_MAX) < 1e-9 else ""))
    marked = sorted(((r["_total_numeric"], r["path"]) for r in rows if not r["synthesis"]
                     and r["_total_numeric"] and r["_checked_visible_distinct"] == 0), reverse=True)
    if marked:
        print("  coverage: largest non-synthesis doc with ZERO checked visible values: %d numeric claims (%s); "
              "LOW_COVERAGE_MIN_TOTAL = %d" % (marked[0][0], marked[0][1][18:], cc.LOW_COVERAGE_MIN_TOTAL))
    decl = [r for r in rows if r["_declares_synthesis"]]
    refused = [r for r in decl if r["synthesis_note"]]
    print("  synthesis: %d doc(s) declare it, %d refused the escape (%s); with a claim_check_reason ADDED, %d would "
          "still be barred by a verdict word; verdict flips among declarers: r6 %d FAIL, cur %d FAIL"
          % (len(decl), len(refused),
             dict(collections.Counter(("no reason" if "claim_check_reason" in r["synthesis_note"] else "verdict")
                                      for r in refused)),
             sum(1 for r in decl if r["_with_reason"].startswith("barred")),
             sum(r["r6"] == "FAIL" for r in decl), sum(r["cur"] == "FAIL" for r in decl)))
    for r in decl:
        if r["_with_reason"].startswith("barred"):
            print("      barred even with a reason: %s -- %s" % (r["path"][18:80], r["_with_reason"][8:130]))


if __name__ == "__main__":
    sys.exit(main())
