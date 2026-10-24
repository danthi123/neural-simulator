#!/usr/bin/env python3
"""tools/gate_preflight.py -- no GATE-seed line is queued until the REGISTERED scorer has read a dev-seed run of the
same packet at the same revision and found its cells DEFINED (owner-approved acceleration step, Claude 2026-10-16).

WHY. On 2026-10-16 the word-role flip reached its gate TSV twice with a defect only a scorer read could show: (1) at
the flipped default the parse row's engagement rule read every cell UNDEFINED (route word_role_commit, NEEDS
wrc-f1a-engagement); (2) the flip revision still forced both packet flags through REQUIRED_ENV, which the at-default
scorer reads as UNDEFINED; then the TSV's F1 block ran the standalone organ batteries instead of the LBF rows, with
shard dirs the scorer never reads and without the control rows. Each was found by a Claude reading, hours or days
late. The pool canary (tools/pool_canary.sh) proves a gate line can START on a pool node; nothing proved its output
can be SCORED. This closes that gap mechanically: run the packet's lines on a dev seed at the pinned revision, score
them with the registered scorer restricted to that seed (e.g. `tools/s2p_score.py R ... --seeds 7`), and record it.

  record  --prereg P --revision R [--log PATH] -- <scorer command ...>
          runs the scorer command from the repo root, keeps its full output (stdout + stderr + any JSON the command
          names with --out), and appends a row to research/queue/preflight.tsv. Verdict CLEAN iff: no Python
          traceback; at least one DEFINED cell; and every UNDEFINED mention is a COVERAGE gap (a seed or shard not
          run: missing / INCOMPLETE / no-shard / no-arm-files / UNMEASURED), never an instrument one (engagement,
          lesion flag, sidecar env, route, error). Else FLAGGED, with each offending line listed.
  ack     --prereg P --revision R --ruling ID
          a Claude ruling that a FLAGGED read is a false positive. Refused unless "ID" appears in the RULINGS
          section of research/coordination/LOCAL_LLM_RUNBOOK.md.
  check   --prereg P --revision R
          exit 0 iff the latest row for (P, R) is CLEAN or ACK (R matches on its first 9 hex chars); used by
          tools/pool_queue.sh before a GATE-seed line is accepted. Exit 2 with the reason otherwise.
          FIRST it runs the smoke-first check (tools/smoke_first.py, RULING smoke-first-1018, 2026-10-18): when P is a
          prereg first added on or after 2026-10-18 it must declare `flip: yes|no <reason>`, and a FLIP prereg (or an
          old one edited into a flip) must cite two committed, valid dev-seed answer_rate records at R, else exit 2
          before the pre-flight rows are even read. Older preregs that were already flips, and non-flip preregs, pass
          it. Only a missing smoke_first module skips the check; any other failure of it refuses (the commit-time gate
          blocks on the same crash), because a check that fails open on a crash is a bypass.

It cannot judge the bar; it judges that the instrument can read the brain. A CLEAN read with a FAILING dev verdict
is fine (a dev seed is no verdict); an UNDEFINED one is a defect to fix before gate seeds are spent.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.environ.get("GATE_PREFLIGHT_PATH") or os.path.join(ROOT, "research", "queue", "preflight.tsv")
RUNBOOK = os.path.join(ROOT, "research", "coordination", "LOCAL_LLM_RUNBOOK.md")
COLS = ("date", "prereg", "revision", "verdict", "detail", "log", "command")
COVERAGE = re.compile(r"missing|incomplete|no-shard|no_shard|no-arm-files|unmeasured|not run|fewer than", re.I)
UNDEF = re.compile(r"UNDEFINED[^\n]*")
DEFINED = re.compile(r"(?<![A-Za-z_])DEFINED(?![A-Za-z_])")
# tools/b2d_score.py prints no DEFINED word, only a count: `"n_defined": 82` (RULING lbf-preflight-ack-1020: its CLEAN
# dev read was FLAGGED "no DEFINED cell" until this). A zero count is still no defined cell.
DEFINED_COUNT = re.compile(r'"n_defined"\s*:\s*[1-9][0-9]*')


def _rev9(r):
    r = (r or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", r):
        raise SystemExit("gate_preflight: revision must be a hex sha (7-40 chars), got %r" % r)
    return r[:9]


def classify(text):
    """(verdict, offending lines) for one scorer read's combined output text."""
    bad = []
    if "Traceback (most recent call last)" in text:
        bad.append("python traceback in the scorer output")
    for m in UNDEF.finditer(text):
        line = m.group(0)[:200]
        if not COVERAGE.search(line):
            bad.append(line)
    if not (DEFINED.search(text) or DEFINED_COUNT.search(text)):
        bad.append("no DEFINED cell anywhere in the read (did the scorer read the dev seed?)")
    return ("FLAGGED" if bad else "CLEAN"), bad


def _rows():
    if not os.path.exists(TSV):
        return []
    out = []
    with open(TSV) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            out.append(dict(zip(COLS, parts + [""] * (len(COLS) - len(parts)))))
    return out


def _append(row):
    new = not os.path.exists(TSV)
    os.makedirs(os.path.dirname(TSV), exist_ok=True)
    with open(TSV, "a") as f:
        if new:
            f.write("# " + "\t".join(COLS) + "\n")
        f.write("\t".join(str(row.get(c, "")).replace("\t", " ").replace("\n", " | ") for c in COLS) + "\n")


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def cmd_record(a, command):
    if not command:
        raise SystemExit("gate_preflight record: give the scorer command after --")
    rev = _rev9(a.revision)
    if not os.path.exists(os.path.join(ROOT, a.prereg)):
        raise SystemExit("gate_preflight record: no prereg %s in this checkout" % a.prereg)
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    text = p.stdout + "\n" + p.stderr
    if "--out" in command:
        i = command.index("--out")
        if i + 1 < len(command) and os.path.exists(os.path.join(ROOT, command[i + 1])):
            with open(os.path.join(ROOT, command[i + 1])) as f:
                text += "\n" + f.read()
    verdict, bad = classify(text)
    log = a.log or os.path.join("research", "queue", "preflight",
                                "%s_%s.log" % (os.path.basename(a.prereg).replace(".md", ""), rev))
    os.makedirs(os.path.dirname(os.path.join(ROOT, log)), exist_ok=True)
    with open(os.path.join(ROOT, log), "w") as f:
        f.write("$ %s\n(rc=%d)\n%s" % (" ".join(command), p.returncode, text))
    _append({"date": _now(), "prereg": a.prereg, "revision": rev, "verdict": verdict,
             "detail": "; ".join(bad)[:600] or "ok", "log": log, "command": " ".join(command)})
    print("gate_preflight: %s %s @ %s (log %s)" % (verdict, a.prereg, rev, log))
    for b in bad:
        print("  - " + b)
    return 0 if verdict == "CLEAN" else 1


def ruling_section(rb):
    """Section 6 of the runbook only: from the `## 6. RULINGS` (or bare `## RULINGS`) header to the next `## ` header, so a
    NEEDS line never counts as a ruling. (Before 2026-10-18 this split on the literal "## RULINGS", which the real header
    `## 6. RULINGS (...)` never contains, so every ack was refused.)"""
    m = re.search(r"^##\s+(?:6\.\s*)?RULINGS\b.*$", rb, re.M)
    if not m:
        return ""
    rest = rb[m.end():]
    n = re.search(r"^## ", rest, re.M)
    return rest[:n.start()] if n else rest


def ruling_present(rb, key):
    """True when KEY is a RULING entry of section 6: `- KEY:` or `- <date> RULING KEY` (a bare substring is not enough)."""
    if not key:
        return False
    k = re.escape(key)
    return bool(re.search(r"^-\s+(?:%s:|(?:\d{4}-\d{2}-\d{2}\s+)?RULING\s+%s(?![A-Za-z0-9_.\-]))" % (k, k), ruling_section(rb), re.M))


def cmd_ack(a):
    rev = _rev9(a.revision)
    with open(RUNBOOK) as f:
        rb = f.read()
    if not ruling_present(rb, a.ruling):
        raise SystemExit("gate_preflight ack: ruling %r is not a RULING entry in section 6 of the runbook" % a.ruling)
    _append({"date": _now(), "prereg": a.prereg, "revision": rev, "verdict": "ACK",
             "detail": "ruling " + a.ruling, "log": "", "command": ""})
    print("gate_preflight: ACK %s @ %s by ruling %s" % (a.prereg, rev, a.ruling))
    return 0


def check(prereg, revision):
    """(ok, reason) for the latest row of (prereg, revision)."""
    rev = _rev9(revision)
    rows = [r for r in _rows() if r["prereg"] == prereg and r["revision"][:9] == rev]
    if not rows:
        return False, ("no pre-flight read of %s at %s: run its lines on a dev seed at this revision, then "
                       "tools/gate_preflight.py record --prereg %s --revision %s -- <registered scorer on that seed>"
                       % (prereg, rev, prereg, rev))
    last = rows[-1]
    if last["verdict"] in ("CLEAN", "ACK"):
        return True, "%s %s" % (last["verdict"], last["date"])
    return False, "latest pre-flight read of %s at %s is %s: %s (log %s)" % (
        prereg, rev, last["verdict"], last["detail"], last["log"])


def smoke_first_check(prereg, revision):
    """(ok, reason) from the shared smoke-first validator (tools/smoke_first.py). Fail-OPEN only when the module itself
    is missing (an older checkout without it); a determinate violation AND a crash of the check are refusals, as they
    are at commit time (review 2026-10-18: a validator that raises on a malformed record must not open the queue)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import smoke_first
    except ImportError as e:
        print("gate_preflight: smoke-first check skipped (%s)" % e, file=sys.stderr)
        return True, "smoke-first check skipped"
    except Exception as e:                                    # noqa: BLE001 -- e.g. a SyntaxError under an old python
        return False, "the smoke-first check could not be loaded (%s: %s)" % (type(e).__name__, e)
    try:
        return smoke_first.check_queue(prereg, revision, ROOT)
    except Exception as e:                                    # noqa: BLE001 -- refuse, with the cause
        return False, "the smoke-first check crashed (%s: %s); fix tools/smoke_first.py or ASK" % (type(e).__name__, e)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="step", required=True)
    for name in ("record", "ack", "check"):
        s = sp.add_parser(name)
        s.add_argument("--prereg", required=True)
        s.add_argument("--revision", required=True)
        if name == "record":
            s.add_argument("--log", default=None)
        if name == "ack":
            s.add_argument("--ruling", required=True)
    a = ap.parse_args(argv)
    if a.step == "record":
        return cmd_record(a, command)
    if a.step == "ack":
        return cmd_ack(a)
    ok, why = smoke_first_check(a.prereg, a.revision)
    if not ok:
        print("⛔ " + why, file=sys.stderr)
        return 2
    ok, why = check(a.prereg, a.revision)
    print(("✓ " if ok else "⛔ ") + why, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
