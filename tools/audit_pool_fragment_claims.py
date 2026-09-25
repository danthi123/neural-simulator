#!/usr/bin/env python3
"""Audit pool.queue.claims for LINE-FRAGMENT claims (the 2026-09-25 dispatcher incident, fixed in 096dfdae0).

WHAT HAPPENED. Until 096dfdae0, tools/pool_autodispatch.sh's revision_available() ran `ssh` WITHOUT `-n` from
inside pop_job's `while IFS= read -r cand; do ...; done < <(awk ...)` candidate loop. The probe's ssh inherited
that loop's stdin and swallowed part of the candidate stream. When the probe answered "revision not provisioned"
the loop `continue`d, and the next `read` resumed MID-LINE: the tail of a queue line was claimed and dispatched as
if it were a whole job. The tail always carries the line's `#checked:` reason, so it passed the record-check gate;
it never carries the leading `cd ~/derisk-pool/revisions/<sha> &&`, so whatever it ran, it ran in the UNPINNED
~/derisk-pool/sim tree. The tail's own `grep -vF "<TAB>$job"` removal never matches a real queue line (a mid-line
fragment is never preceded by the queue's TAB), so the FULL line stayed queued and ran again later.

WHAT THIS DOES (read-only; it never writes a queue file, never contacts a node):
  1. Detect fragment claims: a claim whose job text is a PROPER SUFFIX of another known full line (any claim at any
     time, the live queue, .unchecked, .malformed). A secondary scan lists claims whose first word is not a normal
     command start, so complete-but-malformed lines are not confused with fragments.
  2. For each fragment: node + executed text (pool.running), the dispatch.log line and the "not provisioned" probe
     that preceded it, what bash would run (`bash -n` syntax check + first command word after assignments),
     whether it is revision-pinned, its declared output paths, its JOB_B64 (the key of the node's job_status.log
     v2 record), and every later claim of a parent full line (the duplicate/real run).
  3. With --node-status DIR (files <node>.job_status.log OR <node>.job_status.tsv, fetched read-only by the
     operator -- the `.tsv` name exists because `.gitignore`'s `*.log` rule silently drops a committed copy of the
     real name; fix round r3, review HIGH item, 2026-09-25: durable evidence must be committed under a name the
     repo does not ignore), resolve each fragment's exit status.
  4. With --parent-status-archive FILE (a JSON of previously-resolved PARENT-occurrence records, matched by
     claim_line/parent_outputs/occurrence identity rather than by JOB_B64), fill in a parent occurrence's exit
     status when the live --node-status fetch above did not cover it. A live record always wins; an archive-filled
     entry is tagged node_status_source="archive:<path>" so it is never mistaken for a fresh fetch. Fix round r3,
     review MEDIUM item: a prior regeneration fetched only the fragments' and A2 lines' own job_status records and
     silently lost every parent full-line rc; this restores them from a durable, already-committed prior report
     (git commit 273cfc1b4) rather than a new live fetch (out of scope for a read-only, worktree-isolated audit).
  5. For output paths under research/findings/raw/_load_bearing/_shards/<tag>/, report the LOCAL cell's lb.json
     sidecar (git_sha / source_kind / started) and run tools/lb_shard.py's own pin rule on the cell, so the audit
     says whether the LBP pin gate would exclude a fragment-produced cell.

WARNINGS (fix round r3, review LOW item): a fail-open path that reports "rc=?" or "cells=0" and looks clean is
exactly how the missing job_status evidence went unnoticed the first time. main() now collects an explicit
`warnings` list -- --node-status given but zero v2 records loaded from it, a fragment or nonstandard-start line
still unresolved after --node-status was given, or a --scan-shards tag with n_cells==0 -- prints each to stderr,
and returns a non-zero exit code when any fired.

    .venv/bin/python tools/audit_pool_fragment_claims.py --json <scratch>/fragments.json [--node-status <dir>]

Earned by research/findings/2026-09-25-dispatcher-fragment-jobs-audit.md.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import json
import os
import re
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_ROOT = "/home/dant123/Projects/sim"
# First dispatcher start after the revision probe reached main (merge 561efa586, 2026-09-23 22:44:43 -0400); the
# dispatch.log records `[pool-dispatch] started 22:53:21` right after it. The whole file is still scanned, so the
# pre-cutoff span is a negative control for the detector.
DEFAULT_SINCE = "2026-09-23T22:53:21"
FIX_TIME = "2026-09-25T11:04:36"  # 096dfdae0
NORMAL_START = re.compile(
    r"^(cd |SIM_BACKEND=|CUDA_VISIBLE_DEVICES=|: |mkdir |\.venv/bin/python |env |LANE=|bash |OMP_NUM_THREADS=|"
    r"mem_gb=\d+ )")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# shell operators that separate one command from the next: skipped as WORD BOUNDARIES in first_command_word, so a
# NAME=value assignment following one is still recognised as a fresh assignment run (fix round r2, review LOW item:
# `X=1 && cmd` used to read first_word="&&" -> misclassified "command-not-found" even though `cmd` runs). Bare `&`
# is deliberately EXCLUDED here -- classify() special-cases it itself (a backgrounded assignment), and returning it
# unskipped from first_command_word is what lets that special case fire; see classify()'s own comment.
SHELL_OPERATORS = ("&&", "||", ";", "|")
OUT_FLAGS = ("--out", "--json", "--out-dir", "--output")
SHARD_RE = re.compile(r"research/findings/raw/_load_bearing/_shards/([^/]+)/(s\d+)/([^/]+)/lb\.json")


def _epoch(s):
    if re.fullmatch(r"\d+", s):
        return int(s)
    return int(dt.datetime.fromisoformat(s).timestamp())


def _fmt(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def read_tsv_jobs(path, ts_field=True):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ts, _, job = line.partition("\t")
            rows.append((n, int(ts) if ts.isdigit() else None, job))
    return rows


def read_running(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                ts = int(dt.datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S").timestamp())
            except ValueError:
                continue
            rows.append((n, ts, parts[1], parts[2]))
    return rows


def strip_checked(job):
    """pop_job's own execution transform, minus the POOL_CHECKED_REASON prefix."""
    return job.split("#checked:", 1)[0].rstrip()


def first_command_word(executed):
    """First word bash would EXECUTE after leading NAME=value assignments and any `&&`/`||`/`;`/`|` operator (each
    treated as a WORD BOUNDARY: an assignment immediately after one still counts as a fresh assignment, not a stray
    token). A bare `&` separator is returned AS THE WORD, not skipped -- classify()'s own special case for a
    backgrounded assignment (`POOL_CHECKED_REASON=x & rest`) depends on seeing it."""
    try:
        toks = shlex.split(executed, posix=True)
    except ValueError as e:
        return None, "shlex:%s" % e
    for i, t in enumerate(toks):
        if t in SHELL_OPERATORS or ASSIGN.match(t):
            continue
        return t, None
    return None, "assignments-only"


def bash_syntax_ok(executed):
    r = subprocess.run(["bash", "-n", "-c", executed], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[:200]


def output_paths(text):
    try:
        toks = shlex.split(text, posix=True)
    except ValueError:
        toks = text.split()
    outs = []
    for i, t in enumerate(toks[:-1]):
        if t in OUT_FLAGS:
            outs.append(toks[i + 1])
    return outs


def classify(executed):
    """What bash does with the executed text, statically. The node's rc (when available) is the ground truth."""
    body = executed
    # the POOL_CHECKED_REASON=<%q> prefix is one word; drop it for readability of the first-word test
    word, err = first_command_word(body)
    if err:
        return {"first_word": None, "expect": "parse-error:" + err}
    ok, serr = bash_syntax_ok(body)
    if not ok:
        return {"first_word": word, "expect": "syntax-error (nothing runs)", "bash_n": serr}
    if word == "&":
        # `POOL_CHECKED_REASON=x & rest` -- the assignment is backgrounded, then `rest` runs
        rest = body.split(" & ", 1)[1] if " & " in body else body
        w2, _ = first_command_word(rest)
        return {"first_word": "& " + str(w2), "expect": "RUNS `%s ...` after a backgrounded assignment" % w2}
    if word in (".venv/bin/python", "bash", "cd", "env", "mkdir", "python", "python3"):
        return {"first_word": word, "expect": "RUNS"}
    return {"first_word": word, "expect": "command-not-found (rc 127 expected; nothing after `&&` runs)"}


def local_cell_report(out_path, pin_hint):
    m = SHARD_RE.search(out_path)
    if not m:
        return None
    tag, seed, fac = m.groups()
    cell = os.path.join(LIVE_ROOT, "research/findings/raw/_load_bearing/_shards", tag, seed, fac)
    rep = {"cell": os.path.relpath(cell, LIVE_ROOT), "exists": os.path.isdir(cell)}
    if not rep["exists"]:
        return rep
    prov = os.path.join(cell, "lb.json.prov.json")
    try:
        pj = json.load(open(prov))
        rep["lb_prov"] = {k: pj.get(k) for k in ("git_sha", "source_kind", "started", "run_id")}
        rep["lb_prov"]["argv0"] = (pj.get("argv") or [None])[0]
    except Exception as e:  # noqa: BLE001 -- report, never raise
        rep["lb_prov"] = "unreadable: %s" % e
    pin_file = os.path.join(LIVE_ROOT, "research/findings/raw/_load_bearing/_shards", tag, "PIN.txt")
    pin = open(pin_file).read().strip() if os.path.exists(pin_file) else pin_hint
    rep["pin"] = pin
    rep["pin_source"] = "PIN.txt" if os.path.exists(pin_file) else "parent-line revision"
    try:
        sys.path.insert(0, REPO)
        from tools import lb_shard  # noqa: E402
        per_fac = []
        try:
            per_fac = json.load(open(os.path.join(cell, "lb.json"))).get("per_faculty", [])
        except Exception:  # noqa: BLE001
            pass
        fails, covered = lb_shard.cell_prov_fails(cell, pin, per_fac) if pin else (["no pin"], [])
        rep["pin_rule_fails"] = fails
        rep["pin_rule_covered_by_parent"] = covered
    except Exception as e:  # noqa: BLE001
        rep["pin_rule_error"] = "%s: %s" % (type(e).__name__, e)
    files = []
    for f in sorted(glob.glob(os.path.join(cell, "*"))):
        files.append({"name": os.path.basename(f), "mtime": _fmt(int(os.path.getmtime(f)))})
    rep["files"] = files
    return rep


def load_node_status(node_status_dir):
    """Load every `<node>.job_status.log` OR `<node>.job_status.tsv` file under `node_status_dir` into
    {job_b64: [{"node", "ts", "rc"}, ...]}. Both extensions are read (fix round r3, review HIGH item): `.log` is
    what a live node produces, but `.gitignore`'s `*.log` rule silently drops that name from any commit, so durable
    evidence committed for a finding uses `.tsv` instead -- a prior fix round claimed these were committed while
    they were actually gitignored, and this glob only ever looked for `.log`. Returns (status, n_files, n_records)
    so a caller can warn when a given directory yielded nothing."""
    status = {}
    paths = sorted(glob.glob(os.path.join(node_status_dir, "*.job_status.log"))
                    + glob.glob(os.path.join(node_status_dir, "*.job_status.tsv")))
    n_records = 0
    for p in paths:
        node = os.path.basename(p).split(".job_status.")[0]
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4 and parts[0] == "v2":
                status.setdefault(parts[3], []).append({"node": node, "ts": int(parts[1]), "rc": int(parts[2])})
                n_records += 1
    return status, len(paths), n_records


def load_parent_status_archive(path):
    """Load a --parent-status-archive FILE (see load_node_status's docstring / fix round r3, review MEDIUM item)
    into {(claim_line, tuple(parent_outputs), occurrence_src, occurrence_line, occurrence_time): node_status}."""
    data = json.load(open(path))
    out = {}
    for e in data["entries"]:
        key = (e["claim_line"], tuple(e["parent_outputs"]), e["occurrence_src"], e["occurrence_line"],
               e.get("occurrence_time"))
        out[key] = e["node_status"]
    return out


def scan_shard_tag(tag, pin):
    """Run lb_shard.py's own pin rule over EVERY cell of a local shard tag, and separately list every sidecar whose
    producing script lived in an UNPINNED node tree (`.../derisk-pool/sim/...`) -- the fingerprint of a fragment
    run (a pinned run's argv[0] is `.../derisk-pool/revisions/<sha>/...`)."""
    sys.path.insert(0, REPO)
    from tools import lb_shard  # noqa: E402
    base = os.path.join(LIVE_ROOT, "research/findings/raw/_load_bearing/_shards", tag)
    out = {"tag": tag, "pin": pin, "n_cells": 0, "pin_rule_failing_cells": {}, "unpinned_tree_sidecars": []}
    for lb in sorted(glob.glob(os.path.join(base, "s*", "*", "lb.json"))):
        cell = os.path.dirname(lb)
        out["n_cells"] += 1
        try:
            per_fac = json.load(open(lb)).get("per_faculty", [])
        except Exception:  # noqa: BLE001
            per_fac = []
        fails, _covered = lb_shard.cell_prov_fails(cell, pin, per_fac)
        if fails:
            out["pin_rule_failing_cells"][os.path.relpath(cell, base)] = fails
    for prov in sorted(glob.glob(os.path.join(base, "s*", "*", "*.prov.json"))):
        try:
            argv0 = (json.load(open(prov)).get("argv") or [""])[0]
        except Exception:  # noqa: BLE001
            argv0 = "<unreadable>"
        if "/derisk-pool/sim/" in argv0 or "/derisk-pool/revisions/" not in argv0:
            out["unpinned_tree_sidecars"].append({"sidecar": os.path.relpath(prov, base), "argv0": argv0})
    return out


def main():
    global LIVE_ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue-dir", default=os.path.join(LIVE_ROOT, "research/queue"))
    ap.add_argument("--since", default=DEFAULT_SINCE, help="epoch or ISO local time (default %(default)s)")
    ap.add_argument("--node-status", default=None,
                    help="dir of <node>.job_status.log or <node>.job_status.tsv copies (read-only fetch)")
    ap.add_argument("--parent-status-archive", default=None,
                    help="JSON of previously-resolved PARENT-occurrence node_status records (see "
                         "load_parent_status_archive); fills only entries a live --node-status fetch left empty")
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--scan-shards", action="append", default=[], metavar="TAG=PIN",
                    help="also run the pin rule over every local cell of this _load_bearing shard tag (repeatable)")
    ap.add_argument("--node-outputs", default=None,
                    help="dir of <node>/research/findings/raw/_load_bearing/... cells copied read-only from the "
                         "nodes' UNPINNED trees; each is run through the pin rule of its tag (pins from --scan-shards)")
    ap.add_argument("--root", default=None,
                    help="override LIVE_ROOT (the local repo root that --scan-shards/local_cell_report read shard "
                         "cells from) with a different absolute path; TESTS ONLY -- production runs omit this and "
                         "get the real checkout, exactly as before this flag existed")
    a = ap.parse_args()
    if a.root:
        LIVE_ROOT = os.path.abspath(a.root)
    qd = a.queue_dir
    since = _epoch(a.since)

    claims = read_tsv_jobs(os.path.join(qd, "pool.queue.claims"))
    known = {}
    for n, ts, job in claims:
        known.setdefault(job, []).append(("claims", n, ts))
    for fn in ("pool.queue", "pool.queue.unchecked", "pool.queue.malformed"):
        for n, ts, job in read_tsv_jobs(os.path.join(qd, fn)):
            known.setdefault(job, []).append((fn, n, ts))
    running = read_running(os.path.join(qd, "pool.running"))
    with open(os.path.join(qd, "dispatch.log"), encoding="utf-8", errors="replace") as fh:
        dlog = fh.read().split("\n")

    warnings = []

    # KNOWN LIMIT (fix round r2, review LOW item, not fixed this round -- see that round's commit message): the
    # node's job_status.log is keyed by the EXECUTED TEXT's own base64, not a per-dispatch id, so two DIFFERENT
    # dispatches that happen to run byte-identical text collapse onto the same key here and `rc_of`/`node_status`
    # then reports BOTH records for either occurrence, with no timestamp-based disambiguation. None of the 20
    # records this audit committed collide (verified: 20 distinct keys, see `research/findings/raw/
    # _dispatcher_fragment_audit/node_evidence/`), so no fragment or SETTLE-A2 record in this finding is affected.
    status = {}
    if a.node_status:
        status, n_status_files, n_status_records = load_node_status(a.node_status)
        if n_status_files == 0 or n_status_records == 0:
            warnings.append("--node-status %r yielded %d files / %d v2 records -- every fragment/nonstandard-start "
                             "line below will read rc=? (fix round r3, review LOW item: this used to fail open and "
                             "print a clean-looking 'rc=?' with no warning)" % (a.node_status, n_status_files,
                                                                                n_status_records))

    parent_archive = load_parent_status_archive(a.parent_status_archive) if a.parent_status_archive else {}

    def rc_of(executed):
        if not executed:
            return None
        return status.get(base64.b64encode(executed.encode()).decode())

    def running_for(ts, job):
        st = strip_checked(job)
        return [r for r in running if abs(r[1] - ts) <= 5 and r[3].endswith(st)]

    # KNOWN LIMITS of the proper-suffix rule below (fix round r2, review LOW item -- documented rather than
    # silently assumed away; NOT implemented as code this round, see that round's commit message for why):
    #   (a) false-flag risk -- a genuinely complete, unpinned claim whose text happens to equal the tail of some
    #       OTHER known (pinned) line would be counted as a fragment of that line, even though bash executed it as
    #       its own whole command. Nothing here distinguishes "tail of a real line, dispatched whole" from "tail of
    #       a real line, dispatched as a mid-line fragment" by TEXT alone.
    #   (b) false-miss risk -- a fragment whose PARENT line was removed from every queue file (claims, live queue,
    #       .unchecked, .malformed) before this scan ever saw it leaves no known full line for the suffix test to
    #       match against, so it is invisible to this detector.
    #   A second, INDEPENDENT detector (dispatch-adjacency: flag every dispatch that immediately follows a
    #   same-node "revision ... not provisioned" probe line, the one moment the underlying bug can act) would
    #   cross-check both risks and was run BY HAND for this audit's 14 fragments (agreement recorded in the
    #   finding's Plain statement / Summary) but is not yet implemented as a second code path here.
    frag_jobs = {job for n, ts, job in claims if any(k != job and k.endswith(job) for k in known)}
    frags, odd = [], []
    pre_cutoff_frags = 0
    for n, ts, job in claims:
        # a fragment is never a parent: only a line that is not itself a suffix of another counts as a full line
        parents = [k for k in known if k != job and k.endswith(job) and k not in frag_jobs]
        if job not in frag_jobs:
            if ts and ts >= since and not NORMAL_START.match(job):
                rr = running_for(ts, job)
                ex = rr[0][3] if rr else None
                odd.append({"claim_line": n, "claim_time": _fmt(ts), "job_head": job[:160],
                            "node": rr[0][2] if rr else None, "classification": classify(ex) if ex else None,
                            "node_status": rc_of(ex)})
            continue
        if ts is None or ts < since:
            pre_cutoff_frags += 1
        stripped = strip_checked(job)
        # pool.running: same dispatch, within a few seconds, whose executed text ends with the stripped fragment
        run = [r for r in running if abs(r[1] - ts) <= 5 and r[3].endswith(stripped)]
        node = run[0][2] if run else None
        executed = run[0][3] if run else None
        dline, probe = None, None
        if executed is not None:
            needle = "%s <- %s" % (node, executed)
            for i, L in enumerate(dlog):
                if L.endswith(needle) and L.startswith("[pool-dispatch] "):
                    dline = i + 1
                    for j in range(i - 1, max(i - 40, -1), -1):
                        pm = re.search(r"revision ([0-9a-f]+) not provisioned on (\S+)", dlog[j])
                        if pm:
                            probe = {"line": j + 1, "sha": pm.group(1), "node": pm.group(2)}
                            break
                        if re.search(r"\] \d\d:\d\d:\d\d \S+ <- ", dlog[j]) or "started " in dlog[j]:
                            break
                    break
        par = []
        for p in parents:
            occ = sorted((o for o in known[p]), key=lambda o: (o[2] or 0))
            sha = re.search(r"derisk-pool/revisions/([0-9a-f]{7,40})", p)
            outputs = output_paths(strip_checked(p))
            later = []
            for src, ln, t in occ:
                entry = {"src": src, "line": ln, "time": _fmt(t) if t else None}
                if src == "claims" and t and t >= ts:
                    pr = running_for(t, p)
                    entry["node"] = pr[0][2] if pr else None
                    entry["node_status"] = rc_of(pr[0][3]) if pr else None
                    if not entry["node_status"] and parent_archive:
                        # fix round r3, review MEDIUM item: a live --node-status fetch that does not cover this
                        # occurrence (e.g. it ran before/outside the fetched node logs' window) falls back to a
                        # prior, already-committed resolution -- see load_parent_status_archive's docstring.
                        archived = parent_archive.get((n, tuple(outputs), "claims", ln, entry["time"]))
                        if archived:
                            entry["node_status"] = archived
                            entry["node_status_source"] = "archive:%s" % a.parent_status_archive
                    later.append(entry)
                elif src != "claims" and t and t <= ts:
                    # a still-queued line is a candidate only if it was ADDED before the fragment was claimed (the
                    # queue's own timestamp is its add time); a claimed line's add time is not recorded, so a later
                    # claim stays a candidate -- the parent set over-approximates, never under-approximates
                    later.append(entry)
            if not later:
                continue  # claimed in full BEFORE the fragment (and not re-queued): it was not in the queue then
            par.append({"offset": len(p) - len(job), "revision": sha.group(1) if sha else None,
                        "outputs": outputs, "occurrences_at_or_after_fragment": later,
                        "head_tail": p[:len(p) - len(job)][-100:]})
        cls = classify(executed) if executed else {"expect": "no pool.running record"}
        rec = {
            "claim_line": n, "claim_time": _fmt(ts), "fragment": job, "stripped": stripped,
            "node": node, "running_line": run[0][0] if run else None, "dispatch_log_line": dline,
            "swallowing_probe": probe, "executed": executed, "classification": cls,
            "pinned": "derisk-pool/revisions/" in (executed or stripped),
            "outputs": output_paths(stripped),
            "job_b64": base64.b64encode(executed.encode()).decode() if executed else None,
            "n_parents": len(parents), "parents": par,
        }
        if executed and rec["job_b64"] in status:
            rec["node_status"] = status[rec["job_b64"]]
        rec["local_cells"] = []
        for o in rec["outputs"]:
            pin_hint = next((p["revision"] for p in par if p["revision"] and len(p["revision"]) == 40), None)
            c = local_cell_report(o, pin_hint)
            if c:
                rec["local_cells"].append(c)
        frags.append(rec)

    # Fragments that cut INSIDE the `#checked:` reason carry no `#checked:` and were quarantined by pop_job's
    # record-check gate (never executed, never claimed). dispatch.log keeps only their first 96 characters.
    blocked = []
    last_clock = None
    since_line = 0
    for i, L in enumerate(dlog):
        if "[pool-dispatch] started " + _fmt(since)[11:] in L:
            since_line = i + 1
    for i, L in enumerate(dlog):
        mclock = re.search(r"\] (\d\d:\d\d:\d\d) \S+ <- ", L)
        if mclock:
            last_clock = mclock.group(1)
        if i + 1 <= since_line or "BLOCKED unchecked job" not in L or i + 1 >= len(dlog):
            continue
        head = dlog[i + 1].strip()
        prev = dlog[i - 1] if i else ""
        pm = re.search(r"revision ([0-9a-f]+) not provisioned on (\S+)", prev)
        hosts = [k for k in known if head and head in k and not k.startswith(head)]
        blocked.append({"dispatch_log_line": i + 1, "after_clock": last_clock, "head96": head,
                        "preceded_by_probe": {"sha": pm.group(1), "node": pm.group(2)} if pm else None,
                        "is_mid_line_fragment": bool(hosts), "n_host_lines": len(hosts)})

    report = {"since": _fmt(since), "dispatch_log_since_line": since_line, "blocked_unchecked_since": blocked, "fix_commit_time": FIX_TIME, "n_claims": len(claims),
              "n_fragments": len(frags), "n_fragments_before_since": pre_cutoff_frags,
              "fragments": frags, "nonstandard_start_complete_lines": odd}

    # fix round r3, review LOW item: a fragment or nonstandard-start line that stays unresolved (no node_status)
    # after --node-status WAS given used to print a clean-looking "rc=?" with nothing to flag it as a problem --
    # exactly how the missing job_status evidence went unnoticed. Warn explicitly instead of failing open.
    if a.node_status:
        unresolved_frags = [r["claim_line"] for r in frags if not r.get("node_status")]
        if unresolved_frags:
            warnings.append("%d fragment(s) still unresolved (rc=?) after --node-status was given: claim_line(s) "
                             "%s" % (len(unresolved_frags), unresolved_frags))
        unresolved_odd = [o["claim_line"] for o in odd if not o.get("node_status")]
        if unresolved_odd:
            warnings.append("%d nonstandard-start line(s) still unresolved (rc=?) after --node-status was given: "
                             "claim_line(s) %s" % (len(unresolved_odd), unresolved_odd))

    report["shard_scans"] = []
    pins = {}
    for spec in a.scan_shards:
        tag, _, pin = spec.partition("=")
        pins[tag] = pin
        scan = scan_shard_tag(tag, pin)
        if scan["n_cells"] == 0:
            # fix round r3, review LOW item: a mistyped or missing tag used to report "cells=0
            # pin-rule-failing=0", which reads as a CLEAN scan rather than a scan that found nothing to check.
            warnings.append("--scan-shards %s=%s matched 0 cells -- check the tag spelling / shard path" % (
                tag, pin))
        report["shard_scans"].append(scan)
    report["node_output_cells"] = []
    if a.node_outputs:
        sys.path.insert(0, REPO)
        from tools import lb_shard  # noqa: E402
        for lb in sorted(glob.glob(os.path.join(a.node_outputs, "*", "research/findings/raw/_load_bearing/_shards",
                                                "*", "s*", "*", "lb.json"))):
            cell = os.path.dirname(lb)
            node = os.path.relpath(cell, a.node_outputs).split(os.sep)[0]
            m = SHARD_RE.search(lb)
            tag = m.group(1) if m else None
            try:
                rep = json.load(open(lb))
                pj = json.load(open(os.path.join(cell, "lb.json.prov.json")))
            except Exception as e:  # noqa: BLE001
                report["node_output_cells"].append({"node": node, "cell": cell, "error": str(e)})
                continue
            fails, _cov = (lb_shard.cell_prov_fails(cell, pins[tag], rep.get("per_faculty", []))
                           if tag in pins else (["no pin given for tag %r" % tag], []))
            report["node_output_cells"].append({
                "node": node, "cell": "/".join(m.groups()) if m else cell, "run_id": pj.get("run_id"),
                "argv0": (pj.get("argv") or [None])[0], "git_sha": pj.get("git_sha"),
                "source_kind": pj.get("source_kind"), "started_node_clock": pj.get("started"),
                "env_SIM_BACKEND": (pj.get("env") or {}).get("SIM_BACKEND"),
                "per_faculty": [(x.get("faculty"), x.get("verdict"), x.get("load_bearing"))
                                for x in rep.get("per_faculty", [])],
                "pin": pins.get(tag), "pin_rule_fails": fails})
    report["provenance"] = {
        "script": "tools/audit_pool_fragment_claims.py", "argv": sys.argv,
        "git_sha": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip(),
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "backend": "none -- host-side audit of queue/log/sidecar files; no simulation is run",
        "inputs": {fn: {"lines": sum(1 for _ in open(os.path.join(qd, fn), errors="replace")),
                        "mtime": _fmt(int(os.path.getmtime(os.path.join(qd, fn))))}
                   for fn in ("pool.queue.claims", "pool.running", "pool.queue", "pool.queue.unchecked",
                              "dispatch.log") if os.path.exists(os.path.join(qd, fn))},
        "read_only": "no queue file, node, or shard cell is written; node files are copies fetched with ssh -n"}
    report["warnings"] = warnings
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1)
    for w in warnings:
        print("WARNING: %s" % w, file=sys.stderr)
    for sc in report["shard_scans"]:
        print("shard scan %s pin=%s: cells=%d pin-rule-failing=%d unpinned-tree sidecars=%d" % (
            sc["tag"], sc["pin"][:10], sc["n_cells"], len(sc["pin_rule_failing_cells"]),
            len(sc["unpinned_tree_sidecars"])))
        for c, f in sc["pin_rule_failing_cells"].items():
            print("    FAIL %s: %s" % (c, "; ".join(f)[:160]))
        for u in sc["unpinned_tree_sidecars"]:
            print("    UNPINNED-TREE %s argv0=%s" % (u["sidecar"], u["argv0"]))
    for c in report["node_output_cells"]:
        print("node-output %s %s sha=%s verdict=%s pin-rule-fails=%d" % (
            c["node"], c.get("cell"), (c.get("git_sha") or "")[:10], c.get("per_faculty"),
            len(c.get("pin_rule_fails") or [])))
    print("claims=%d fragments=%d (before --since: %d) nonstandard-start complete lines=%d"
          % (len(claims), len(frags), pre_cutoff_frags, len(odd)))
    for r in frags:
        rc = ",".join("%s:rc=%s" % (s["node"], s["rc"]) for s in r.get("node_status", [])) or "rc=?"
        print("  L%-5d %s %-7s %-6s %-8s first=%-22s %s | %s" % (
            r["claim_line"], r["claim_time"], r["node"], "PINNED" if r["pinned"] else "UNPIN",
            rc, str(r["classification"].get("first_word"))[:22], r["classification"]["expect"][:40],
            r["stripped"][:60]))
    for o in odd:
        rc = ",".join("%s:rc=%s" % (x["node"], x["rc"]) for x in (o["node_status"] or [])) or "rc=?"
        print("  nonstandard L%d %s %s %s %s" % (o["claim_line"], o["claim_time"], o["node"], rc, o["job_head"][:70]))
    for b in blocked:
        print("  quarantined dispatch.log:%d after %s probe-node=%s mid-line-fragment=%s | %s" % (
            b["dispatch_log_line"], b["after_clock"], (b["preceded_by_probe"] or {}).get("node"),
            b["is_mid_line_fragment"], b["head96"][:70]))
    # fix round r3, review LOW item: fail LOUD, not open -- a caller (or a human skimming stdout) must not be able
    # to mistake a silently-incomplete resolution for a clean one.
    return 2 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
