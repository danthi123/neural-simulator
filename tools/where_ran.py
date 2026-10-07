#!/usr/bin/env python3
"""where_ran.py -- WHERE did each result run? Host, start time, git sha and entry script per result file.

Why (Claude's 10-06 review of Hermes): the affect audit (docs/drafts/stage1-exit/AUDIT-affect-pool-results.md) never
checked which machine produced each result, and that decided validity: the affect organ reads
research/findings/raw/_affect_learned_valence_map.json, which the pool provisioner did not ship, so every pool/AWS
run silently fell back to the raw Warriner norm while the workstation runs had the map. A number cited without its
host is a number whose validity nobody has checked. Run this in ANY audit and quote the host summary.

    python tools/where_ran.py research/findings/raw/_affect_*.json 'research/findings/raw/_load_bearing/_shards/<tag>/s*/*'
    python tools/where_ran.py <paths/globs/dirs...> [--json] [--expect pool|workstation|aws|server]

For each result JSON it reads the sibling `<file>.prov.json` (a `.prov.json` or a directory is also accepted; a
directory expands to every result JSON beneath it). A sidecar that names no host falls back to
research/findings/raw/_provenance/runs.jsonl, by `run_id`, then by the record's `artifact` / `--out` argument.
Host classes: pool (node-*), workstation (dant123-wk, wspool), aws (ip-172-31-*, or a host string naming AWS),
server (a 12-hex container hostname: homelab-server's hermes container), other, unknown (no host recorded anywhere).
Prints one line per file (host, class, start, git sha[+dirty], argv[0]) and a host summary. `--expect <class>` exits 1
if any file ran elsewhere (or nowhere known). Exit 0 otherwise; 2 when nothing matched.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "research", "findings", "raw", "_provenance", "runs.jsonl")
EXPECT = ("pool", "workstation", "aws", "server")


def classify_host(host):
    """pool | workstation | aws | server | other | unknown for a recorded hostname string."""
    if not isinstance(host, str) or not host.strip():
        return "unknown"
    h = host.strip()
    low = h.lower()
    if low.startswith("node-"):
        return "pool"
    if low == "dant123-wk" or low.startswith("dant123-wk.") or "wspool" in low:
        return "workstation"
    if low.startswith("ip-172-31-") or "aws" in low or re.match(r"^i-[0-9a-f]{8,}", low):
        return "aws"
    if re.fullmatch(r"[0-9a-f]{12}", low):
        return "server"
    return "other"


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class RunsIndex:
    """runs.jsonl by run_id and by artifact path (built lazily; the file is ~16k lines)."""

    def __init__(self, path):
        self.path, self._by_id, self._by_art, self._loaded = path, {}, {}, False

    def _load(self):
        self._loaded = True
        try:
            f = open(self.path, encoding="utf-8")
        except OSError:
            return
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                if r.get("run_id"):
                    self._by_id[str(r["run_id"])] = r
                argv = r.get("argv") or []
                arts = [r.get("artifact")] if r.get("artifact") else []
                arts += [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--out"]
                for a in arts:
                    if isinstance(a, str):
                        self._by_art[os.path.normpath(a)] = r

    def find(self, run_id=None, artifact=None):
        if not self._loaded:
            self._load()
        if run_id and str(run_id) in self._by_id:
            return self._by_id[str(run_id)]
        if artifact:
            return self._by_art.get(os.path.normpath(artifact))
        return None


def expand(args):
    """Result JSON paths for the given paths/globs/dirs (a .prov.json maps to its result file)."""
    out = []
    for a in args:
        hits = sorted(glob.glob(a, recursive=True)) or ([a] if os.path.exists(a) else [])
        for h in hits:
            if os.path.isdir(h):
                for dp, _dn, fns in os.walk(h):
                    out += [os.path.join(dp, fn) for fn in sorted(fns)
                            if fn.endswith(".json") and not fn.endswith(".prov.json")]
            elif h.endswith(".prov.json"):
                out.append(h[: -len(".prov.json")])
            else:
                out.append(h)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def where(path, runs):
    """One result file's provenance row."""
    rel = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
    row = {"path": path, "host": None, "class": "unknown", "started": None, "git_sha": None, "git_dirty": None,
           "argv0": None, "source": "none"}
    rec, src = _load_json(path + ".prov.json"), "sidecar"
    if rec is None:
        rec, src = runs.find(artifact=rel), "runs.jsonl(artifact)"
    elif not rec.get("host"):
        alt = runs.find(run_id=rec.get("run_id"), artifact=rel)
        if alt and alt.get("host"):
            rec = dict(rec, host=alt["host"], started=rec.get("started") or alt.get("started"))
            src = "sidecar+runs.jsonl"
    if rec is None:
        return row
    argv = rec.get("argv") or []
    row.update(host=rec.get("host"), started=rec.get("started"), git_sha=(rec.get("git_sha") or None),
               git_dirty=rec.get("git_dirty"), argv0=(argv[0] if argv else None), source=src)
    row["class"] = classify_host(row["host"])
    return row


def summarize(rows):
    by_class = Counter(r["class"] for r in rows)
    by_host = Counter("%s (%s)" % (r["host"] or "-", r["class"]) for r in rows)
    return {"n_files": len(rows), "by_class": dict(sorted(by_class.items())), "by_host": dict(sorted(by_host.items()))}


def render(rows, summary):
    lines = []
    for r in rows:
        sha = (r["git_sha"] or "-")[:9] + ("+dirty" if r["git_dirty"] else "")
        lines.append("%-24s %-11s %-19s %-16s %s  <- %s" % ((r["host"] or "-")[:24], r["class"], r["started"] or "-", sha,
                                                          r["path"], r["argv0"] or "-"))
    lines.append("")
    lines.append("host summary: %d file(s): %s" % (summary["n_files"], ", ".join(
        "%s=%d" % kv for kv in summary["by_class"].items())))
    for h, n in summary["by_host"].items():
        lines.append("  %-44s %d" % (h, n))
    if summary["by_class"].get("unknown"):
        lines.append("  NOTE: %d file(s) record no host anywhere: where they ran is UNKNOWN, not 'fine'"
                     % summary["by_class"]["unknown"])
    if len([c for c in summary["by_class"] if c != "unknown"]) > 1:
        lines.append("  NOTE: results span more than one host class: say which class each cited number came from, "
                     "and whether validity depends on it (inputs the provisioner does not ship, e.g. raw/ files)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="result JSONs, globs or directories")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expect", choices=EXPECT, help="exit 1 if any file did not run on this host class")
    ap.add_argument("--runs", default=RUNS, help="runs.jsonl fallback (default: the repo's)")
    a = ap.parse_args(argv)
    files = expand(a.paths)
    if not files:
        print("where_ran: no result file matched", file=sys.stderr)
        return 2
    runs = RunsIndex(a.runs)
    rows = [where(p, runs) for p in files]
    summary = summarize(rows)
    if a.json:
        print(json.dumps({"files": rows, "summary": summary}, indent=1))
    else:
        print(render(rows, summary))
    if a.expect:
        bad = [r for r in rows if r["class"] != a.expect]
        if bad:
            print("where_ran: %d file(s) did not run on %s: %s" % (len(bad), a.expect, ", ".join(
                "%s=%s" % (b["path"], b["class"]) for b in bad[:5])), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
