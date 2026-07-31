#!/usr/bin/env python3
"""Bind CODE to the BIOLOGY it implements, and verify the binding still resolves.

WHY (2026-07-31, owner directive). The biology is STABLE. A pipeline's interactions with other regions grow and
change, but the neuroscience it was based on does not. Yet this project keeps re-researching what it already
established, and -- worse -- keeps running configurations that contradict the biology it claims to implement.

The case that earned this, today: a 21-agent research round established that Bittner et al. 2017 reports BTSP as
a ONE-SHOT mechanism -- a SINGLE plateau creates a place field. The gap#5 runner's `laps` default was 5. Five
traversals re-potentiate every position and destroy the field; measured, place-specificity decays 4.40x -> 2.57x
-> 1.11x as laps go 1 -> 2 -> 5. The wrong default survived every sweep of density, w_max, lr and eligibility tau,
because nothing connected the runner to the paper. The biology was knowable from the first day of the arc.

WHAT THIS ENFORCES, three things a human cannot be relied on to keep doing:

  1. SOURCE POINTERS MUST RESOLVE. Each claim cites a local corpus file plus a distinctive `anchor` quote, and
     the checker OPENS the file and confirms the anchor is still present. A rotted pointer fails loudly instead of
     silently becoming folklore. (The project's local corpus paths had already rotted to dead `E:` drive paths
     once, and the RAG silently fell back to online search for weeks.)
  2. CONFIG MUST NOT CONTRADICT THE BIOLOGY. `constraints_config` maps a config/CLI key to the value the biology
     REQUIRES, with the reason. `laps: 1` for a one-shot mechanism is checkable, and would have caught the defect
     above on day one.
  3. IMPLEMENTED-BY MUST EXIST. An entry claiming a runner implements it is checked against the filesystem, so
     the map cannot quietly describe code that was renamed or deleted.

    .venv/bin/python tools/biology_check.py                 # check every entry
    .venv/bin/python tools/biology_check.py --config research/runners/_gap5_btsp_place_field_derisk.py

Entries live in research/biology/<id>.md with YAML-ish frontmatter. Deliberately a flat directory of small files:
the point is that opening ONE file tells you what the code is supposed to be doing and where that came from.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIO_DIR = os.path.join(ROOT, "research", "biology")


def _parse_frontmatter(text):
    """Minimal YAML subset: scalars, lists of scalars, and lists of `- key: value` blocks. No dependency."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    body, out, cur_key, cur_list = text[3:end], {}, None, None
    for raw in body.split("\n"):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.startswith("  - ") or raw.startswith("- "):
            item = raw.split("- ", 1)[1].strip()
            if ":" in item and not item.startswith(("http", '"')):
                k, v = item.split(":", 1)
                cur_list.append({k.strip(): v.strip().strip('"')})
            else:
                cur_list.append(item.strip('"'))
            continue
        if raw.startswith("    ") and cur_list and isinstance(cur_list[-1], dict):
            if ":" in raw:
                k, v = raw.strip().split(":", 1)
                cur_list[-1][k.strip()] = v.strip().strip('"')
            continue
        if ":" in raw:
            k, v = raw.split(":", 1)
            cur_key, v = k.strip(), v.strip().strip('"')
            if v == "":
                cur_list = []
                out[cur_key] = cur_list
            else:
                out[cur_key] = v
                cur_list = None
    return out


def _expand(p):
    return os.path.expanduser(p) if p.startswith("~") else (p if os.path.isabs(p) else os.path.join(ROOT, p))


def check_entry(path, verbose=True):
    text = open(path).read()
    fm = _parse_frontmatter(text)
    name = os.path.basename(path)
    problems = []
    if not fm:
        return ["%s: no frontmatter — an entry with no machine-readable head cannot be checked or indexed" % name]

    for req in ("id", "mechanism"):
        if not fm.get(req):
            problems.append("%s: missing required field '%s'" % (name, req))

    srcs = [s for s in (fm.get("sources") or []) if isinstance(s, dict)]
    if not srcs:
        problems.append("%s: NO SOURCE — a biology entry without a citation is folklore" % name)
    for s in srcs:
        p, anchor = s.get("path", ""), s.get("anchor", "")
        if not p:
            continue
        if p.startswith(("http", "doi:", "DOI:", "PMC", "PMID")):
            if not anchor:
                problems.append("%s: external source %s has no anchor quote to verify against" % (name, p[:60]))
            continue                                     # cannot resolve an external source offline
        full = _expand(p)
        if not os.path.exists(full):
            problems.append("%s: SOURCE PATH DOES NOT RESOLVE: %s" % (name, p))
            continue
        if anchor:
            try:
                with open(full, errors="ignore") as fh:
                    if anchor.lower() not in fh.read().lower():
                        problems.append("%s: ANCHOR NOT FOUND in %s — the pointer has rotted: %r"
                                        % (name, os.path.basename(full), anchor[:60]))
            except Exception as e:
                problems.append("%s: source unreadable (%s): %s" % (name, type(e).__name__, p))

    for impl in (fm.get("implemented_by") or []):
        ip = impl if isinstance(impl, str) else str(impl)
        if not os.path.exists(_expand(ip)):
            problems.append("%s: implemented_by path missing: %s" % (name, ip))

    if verbose and not problems:
        n_c = len([c for c in (fm.get("constraints_config") or [])])
        print("  ✔ %-34s %d source(s) resolve, %d config constraint(s)" % (fm.get("id", name), len(srcs), n_c))
    return problems


def check_config(runner_path, verbose=True):
    """Compare a runner's argparse defaults against every biology entry that claims to be implemented by it."""
    text = open(runner_path).read()
    rel = os.path.relpath(os.path.abspath(runner_path), ROOT)
    problems, checked = [], 0
    for f in sorted(glob.glob(os.path.join(BIO_DIR, "*.md"))):
        fm = _parse_frontmatter(open(f).read()) or {}
        impls = [str(i) for i in (fm.get("implemented_by") or [])]
        if not any(rel in i or os.path.basename(runner_path) in i for i in impls):
            continue
        for c in (fm.get("constraints_config") or []):
            if not isinstance(c, dict):
                continue
            key, want, why = c.get("key"), c.get("value"), c.get("why", "")
            if not key or want is None:
                continue
            checked += 1
            m = re.search(r'add_argument\(\s*["\']--%s["\'][^)]*?default\s*=\s*([^,)\s]+)' % re.escape(key), text)
            if not m:
                m = re.search(r'\b%s\s*=\s*([0-9.]+)' % re.escape(key), text)
            if not m:
                problems.append("%s: biology '%s' constrains '%s' but it was not found in the runner"
                                % (rel, fm.get("id"), key))
                continue
            actual = m.group(1).strip()
            try:
                same = abs(float(actual) - float(want)) < 1e-9
            except ValueError:
                same = actual.strip("\"'") == str(want)
            if not same:
                problems.append(
                    "%s: ⛔ CONFIG CONTRADICTS BIOLOGY — '%s' default is %s but %s requires %s.\n        WHY: %s"
                    % (rel, key, actual, fm.get("id"), want, why))
    if verbose:
        print("  %s: %d biology constraint(s) checked" % (rel, checked))
    return problems


def main():
    args = sys.argv[1:]
    problems = []
    if "--config" in args:
        for p in args[args.index("--config") + 1:]:
            problems += check_config(p)
    else:
        entries = sorted(glob.glob(os.path.join(BIO_DIR, "*.md")))
        if not entries:
            print("biology_check: no entries in research/biology/ — nothing to check.")
            return 0
        print("biology_check: %d entr(ies)" % len(entries))
        for e in entries:
            problems += check_entry(e)
        for e in entries:
            fm = _parse_frontmatter(open(e).read()) or {}
            for impl in (fm.get("implemented_by") or []):
                ip = _expand(str(impl))
                if os.path.exists(ip) and ip.endswith(".py"):
                    problems += check_config(ip, verbose=False)
    for p in problems:
        print("  ⛔ %s" % p)
    print("  => %s" % ("⛔ %d problem(s)" % len(problems) if problems
                       else "✔ every source resolves and no config contradicts its biology"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
