#!/usr/bin/env python3
"""pool_stall_check.py -- catches a LIVE-BUT-STALLED pool the same way `parallel_audit.py` was fooled by one.

WHY (2026-09-25). The session heartbeat (`tools/parallel_audit.py`, run every ~15 min) printed `✓ SATURATED`
while 7 of 17 D6 processes running on pool41/pool42 were DUPLICATES of cells whose output had already landed
locally at the SAME pinned revision -- each had been running 7-26h. This is the 2026-07-24 failure class
("a live-but-stalled run", not idleness) recurring one layer up: `pgrep -fc research.runners` on each node
counts a process as a LANE the moment it exists, with no check that the process is doing anything the record
does not already have, or that it is still within its own historical running time.

WHAT THIS CHECKS, read-only, for every job currently RUNNING on every pool node:
  (a) DUP-OF-LANDED -- the job is pinned to a revision (`cd ~/derisk-pool/revisions/<sha> && ...`,
      `tools/pool_provision.sh --isolated`) AND its own `--out`/`--json` artifact already exists in THIS
      checkout with a `.prov.json` sidecar (`research/runners/__init__.py`) whose `git_sha` matches that same
      pinned revision -- i.e. the exact result this job is computing has already landed at the exact code it is
      running. Re-running it burns a core for a result nobody will read twice.
  (b) OVERDUE -- the job's elapsed wall time exceeds 3x the HISTORICAL duration this repo has actually observed
      for a job with the same (runner module, key args) signature, learned by pairing each node's
      `job_status.log` v2 completion records (epoch, rc, decoded command) against this checkout's own
      `research/queue/pool.queue.claims` (epoch, command) dispatch timestamps. No history for a signature ->
      report UNKNOWN, never silently OK (a runner this repo has never seen complete before is not "fine by
      default" just because nothing says otherwise).
  (c) a one-line, heartbeat-ready summary, and the EXACT `ssh ... kill` command a human can run for each flagged
      job -- this tool NEVER kills anything itself.

Identifying a running job: `remote_launch_command()` (tools/pool_autodispatch.sh) exports POOL_JOB_ID and
JOB_B64 (the base64-encoded job command) into the environment of the whole process tree it launches (setsid ->
the wrapper bash -> the job's own `bash -c "$job"`, which usually execs straight into the runner process) --
`node_is_idle()` already scans `/proc/*/environ` for POOL_JOB_ID for exactly this reason; this module reuses the
same signal and also pulls JOB_B64 to recover the actual command.

Usage:
    python -m tools.pool_stall_check                 # human-readable report, all pool nodes
    python -m tools.pool_stall_check --json           # machine-readable, for parallel_audit.py

Read-only. Never raises past `check_all()` in normal operation -- every ssh/parse step is wrapped so an
unreachable node or a malformed log line degrades to UNKNOWN/UNREACHABLE, never a crash, matching the
heartbeat's "exit-0-always, never why the cycle dies" contract every sibling probe here follows.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NODES = tools/pool_autodispatch.sh's own default + research/queue/.pool_extra_nodes, re-read every call (an
# AWS pool node can be added/removed with no restart -- see pool_autodispatch.sh's own comment on this file).
_DEFAULT_NODES = "pool40 pool41 pool42"
_QUEUE_PATH_DEFAULT = os.path.join(_ROOT, "research", "queue", "pool.queue")
_SSH_CONFIG_DEFAULT = os.path.join(_ROOT, "research", "queue", ".pool_ssh_config")
_EXTRA_NODES_DEFAULT = os.path.join(_ROOT, "research", "queue", ".pool_extra_nodes")

# The remote probe: (1) every PID carrying a POOL_JOB_ID in its environ (a job is usually 2-3 such PIDs -- the
# wrapper bash plus the exec'd runner -- see module docstring), with its elapsed seconds (`ps -o etimes=`) and
# the base64 job payload; (2) the node's job_status.log tail (always at ~/derisk-pool/sim/job_status.log --
# remote_launch_command's OUTER `cd ~/derisk-pool/sim &&` sets the wrapper's cwd unconditionally, even for a
# job whose OWN command later `cd`s into an isolated revision dir in a child subshell). `cut -d= -f2-` keeps
# everything after the FIRST `=` so a base64 value's own `=` padding is never truncated.
REMOTE_PROBE_SCRIPT = r"""
for e in /proc/[0-9]*/environ; do
  pid=$(basename "$(dirname "$e")")
  vars=$(tr '\0' '\n' < "$e" 2>/dev/null)
  jid=$(printf '%s\n' "$vars" | grep -m1 '^POOL_JOB_ID=' | cut -d= -f2-)
  if [ -z "$jid" ]; then continue; fi
  jb64=$(printf '%s\n' "$vars" | grep -m1 '^JOB_B64=' | cut -d= -f2-)
  etimes=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -z "$etimes" ]; then continue; fi
  printf 'RUN\t%s\t%s\t%s\t%s\n' "$pid" "$jid" "$etimes" "$jb64"
done
echo '===STATUS==='
tail -n 4000 ~/derisk-pool/sim/job_status.log 2>/dev/null
"""

_MODULE_RE = re.compile(r"research\.runners\.([A-Za-z0-9_.]+)")
_REV_RE = re.compile(r"derisk-pool/revisions/([0-9a-f]{7,40})")
_OUT_RE = re.compile(r"--(?:out|output|json)[= ]+(\S+)")
# Key args that materially change a runner's RUNTIME (not its identity/output path) -- the task's own examples
# (--n-facts/--family) plus the other size-shaped flags this repo's D6/battery runners commonly take. Extend
# this list rather than inventing a second one; an unmatched flag is silently absent from the signature (a
# coarser signature under-splits history, which is the SAFE direction -- more samples, not a missed pairing).
_KEY_ARG_FLAGS = ("n-facts", "family", "n-steps", "n-episodes", "n-trials", "epochs", "n-cells", "n-seeds", "seeds")


def get_pool_nodes():
    """tools/pool_autodispatch.sh's NODES default (POOL_NODES env, else pool40/41/42) plus every non-comment,
    non-blank line of research/queue/.pool_extra_nodes (POOL_EXTRA_NODES_FILE), re-read every call -- same two
    knobs pool_autodispatch.sh itself reads, so an AWS node added via `aws_pool_node.sh up` is picked up here
    with no restart, exactly as it is for the dispatcher."""
    nodes = os.environ.get("POOL_NODES", _DEFAULT_NODES).split()
    extra_path = os.environ.get("POOL_EXTRA_NODES_FILE", _EXTRA_NODES_DEFAULT)
    if os.path.isfile(extra_path):
        try:
            with open(extra_path, errors="ignore") as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        nodes.extend(line.split())
        except OSError:
            pass
    seen, out = set(), []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def ssh_config_args():
    cfg = os.environ.get("POOL_SSH_CONFIG", _SSH_CONFIG_DEFAULT)
    return ["-F", cfg] if os.path.isfile(cfg) else []


def queue_claims_path():
    queue = os.environ.get("POOL_QUEUE_PATH", _QUEUE_PATH_DEFAULT)
    return queue + ".claims"


def _decode_b64(s):
    if not s:
        return None
    s = s.strip()
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", errors="replace")
    except Exception:
        return None


def probe_node(node, timeout=12, connect_timeout=6):
    """SSH once to `node`, read-only, never raises. Returns the probe's raw stdout, or None if the node is
    unreachable/times out (the caller reports that distinctly -- never conflated with 'nothing running')."""
    cmd = (["ssh", "-n"] + ssh_config_args()
           + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % connect_timeout, node, REMOTE_PROBE_SCRIPT])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0 and "===STATUS===" not in (p.stdout or ""):
        return None
    return p.stdout or ""


def parse_probe_output(node, text):
    """-> (running: {job_id: {"node", "pids": set(), "max_etimes": int, "job_text": str|None}},
           completions: [(epoch:int, rc:int, job_text:str), ...])
    Malformed/partial lines are skipped, never raise -- a node mid-write or truncated by the timeout still
    yields whatever complete lines it produced."""
    running = {}
    completions = []
    section = "run"
    for line in (text or "").splitlines():
        if line.strip() == "===STATUS===":
            section = "status"
            continue
        if section == "run":
            if not line.startswith("RUN\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            _tag, pid, jid, etimes = parts[0], parts[1], parts[2], parts[3]
            jb64 = parts[4] if len(parts) > 4 else ""
            if not jid or not etimes.isdigit():
                continue
            rec = running.setdefault(jid, {"node": node, "pids": set(), "max_etimes": 0, "job_text": None})
            if pid:
                rec["pids"].add(pid)
            et = int(etimes)
            if et > rec["max_etimes"]:
                rec["max_etimes"] = et
            if rec["job_text"] is None:
                txt = _decode_b64(jb64)
                if txt:
                    rec["job_text"] = txt
        else:
            if not line.startswith("v2\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            _tag, epoch_s, rc_s, jb64 = parts
            if not epoch_s.isdigit() or not re.match(r"^-?\d+$", rc_s):
                continue
            txt = _decode_b64(jb64)
            if not txt:
                continue
            completions.append((int(epoch_s), int(rc_s), txt))
    return running, completions


def job_signature(text):
    """(runner module, sorted key-arg tuple), or None if no `research.runners.X` module is found in TEXT.
    Deliberately a REGEX search over the whole decoded command, not a positional parse -- the command is
    wrapped in varying amounts of `cd ... &&` / `POOL_CHECKED_REASON=...` prefix depending on whether it came
    from a live job's JOB_B64 or a raw pool.queue.claims line, and a search is robust to both."""
    if not text:
        return None
    m = _MODULE_RE.search(text)
    if not m:
        return None
    module = m.group(1)
    args = []
    for flag in _KEY_ARG_FLAGS:
        am = re.search(r"--%s[= ]+(\S+)" % re.escape(flag), text)
        if am:
            args.append((flag, am.group(1).rstrip(",;")))
    return (module, tuple(sorted(args)))


def pinned_revision(text):
    if not text:
        return None
    m = _REV_RE.search(text)
    return m.group(1) if m else None


def declared_out_path(text):
    if not text:
        return None
    m = _OUT_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def load_claims(claims_path=None, now=None, max_age_s=None):
    """[(epoch:int, job_text:str), ...] from research/queue/pool.queue.claims (the dispatcher's own append-only
    pop-time record -- see pool_autodispatch.sh's pop_job). job_text still carries any trailing '#checked:...'
    annotation; job_signature() is robust to that (see its own docstring)."""
    path = claims_path or queue_claims_path()
    out = []
    if not os.path.isfile(path):
        return out
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2 or not parts[0].isdigit():
                    continue
                epoch = int(parts[0])
                if max_age_s is not None and now is not None and now - epoch > max_age_s:
                    continue
                out.append((epoch, parts[1]))
    except OSError:
        return out
    return out


def historical_durations(completions, claims, max_pair_gap_s=172800):
    """{signature: [duration_seconds, ...]} -- for each COMPLETED job (from any node's job_status.log), pair it
    with the temporally-nearest LOCAL claim of the same signature that started at or before it (within
    max_pair_gap_s, default 48h). This is a DISTRIBUTION of typical durations for a signature, not an exact
    per-job join: two jobs with the identical signature dispatched close together (duplicates -- precisely the
    2026-09-25 incident) cannot be told apart from these two logs alone, and are not meant to be -- the
    resulting spread still bounds "how long does this kind of job normally take", which is all OVERDUE needs."""
    by_sig_claims = {}
    for epoch, text in claims:
        sig = job_signature(text)
        if sig is None:
            continue
        by_sig_claims.setdefault(sig, []).append(epoch)
    out = {}
    for epoch, _rc, text in completions:
        sig = job_signature(text)
        if sig is None:
            continue
        starts = by_sig_claims.get(sig)
        if not starts:
            continue
        candidates = [s for s in starts if s <= epoch and epoch - s <= max_pair_gap_s]
        if not candidates:
            continue
        best = max(candidates)
        out.setdefault(sig, []).append(epoch - best)
    return out


def overdue_verdict(elapsed_s, sig, durations, factor=3.0):
    """"OVERDUE" / "OK" / "UNKNOWN" -- UNKNOWN (never a silent OK) when this signature has no learned history."""
    hist = durations.get(sig) if sig else None
    if not hist:
        return "UNKNOWN"
    ref = statistics.median(hist)
    if ref <= 0:
        return "UNKNOWN"
    return "OVERDUE" if elapsed_s > factor * ref else "OK"


def check_dup_of_landed(root, out_path, pinned_sha):
    """True/False/None. None = not applicable (unpinned job, or no --out/--json on its command line) -- this is
    a DIFFERENT state from False (checked, and it is not a duplicate): callers must not conflate the two."""
    if not out_path or not pinned_sha:
        return None
    local_path = os.path.normpath(os.path.join(root, out_path))
    root_norm = os.path.normpath(root)
    if os.path.commonpath([local_path, root_norm]) != root_norm:
        return None  # a --out escaping the repo root is not something this check follows
    prov_path = local_path + ".prov.json"
    if not os.path.isfile(local_path) or not os.path.isfile(prov_path):
        return False
    try:
        with open(prov_path, errors="ignore") as f:
            prov = json.load(f)
    except (OSError, ValueError):
        return None
    local_sha = str(prov.get("git_sha") or "").strip().lower()
    if not local_sha or local_sha == "unknown":
        return False
    pinned = pinned_sha.strip().lower()
    return local_sha == pinned or local_sha.startswith(pinned) or pinned.startswith(local_sha)


def kill_command(node, pids):
    """The EXACT command a human can paste to end this job. Sorted, deduped PIDs; SIGTERM first (a runner that
    traps it can still flush a partial artifact) -- this tool never runs it."""
    ssh_args = " ".join(ssh_config_args())
    prefix = "ssh -n %s" % ssh_args if ssh_args else "ssh -n"
    pid_list = " ".join(sorted(pids, key=lambda p: int(p) if p.isdigit() else 0))
    return '%s %s "kill -TERM %s"' % (prefix, node, pid_list)


def check_all(nodes=None, root=None, timeout=12, connect_timeout=6, claims_path=None, now=None):
    """The whole read-only check. Never raises: every per-node probe / parse / provenance read is guarded, so an
    unreachable node or a corrupt log degrades that one signal to UNKNOWN/UNREACHABLE rather than aborting the
    others. Returns a JSON-serializable dict:
        {"nodes": [...], "unreachable": [...], "running": [<row>, ...], "flagged": [<row subset>, ...],
         "n_running": int, "n_dup": int, "n_overdue": int, "n_unknown_overdue": int, "summary_line": str}
    where each <row> carries node/job_id/pids/elapsed_s/module/signature/out_path/pinned_sha/dup_of_landed/
    overdue/kill_cmd.
    """
    import time as _time
    root = root or _ROOT
    now = now if now is not None else int(_time.time())
    node_list = nodes if nodes is not None else get_pool_nodes()

    all_running = {}   # job_id -> record (job_id is unique per dispatch, so no cross-node collision expected)
    all_completions = []
    unreachable = []
    for node in node_list:
        try:
            out = probe_node(node, timeout=timeout, connect_timeout=connect_timeout)
        except Exception:
            out = None
        if out is None:
            unreachable.append(node)
            continue
        try:
            running, completions = parse_probe_output(node, out)
        except Exception:
            running, completions = {}, []
        for jid, rec in running.items():
            all_running.setdefault(jid, rec)
        all_completions.extend(completions)

    try:
        claims = load_claims(claims_path, now=now)
    except Exception:
        claims = []
    try:
        durations = historical_durations(all_completions, claims)
    except Exception:
        durations = {}

    rows = []
    for jid, rec in all_running.items():
        text = rec.get("job_text")
        sig = job_signature(text)
        module = sig[0] if sig else None
        out_path = declared_out_path(text)
        sha = pinned_revision(text)
        try:
            dup = check_dup_of_landed(root, out_path, sha)
        except Exception:
            dup = None
        try:
            overdue = overdue_verdict(rec["max_etimes"], sig, durations)
        except Exception:
            overdue = "UNKNOWN"
        row = {
            "node": rec["node"],
            "job_id": jid,
            "pids": sorted(rec["pids"], key=lambda p: int(p) if p.isdigit() else 0),
            "elapsed_s": rec["max_etimes"],
            "module": module,
            "out_path": out_path,
            "pinned_sha": sha,
            "dup_of_landed": dup,
            "overdue": overdue,
            "kill_cmd": kill_command(rec["node"], rec["pids"]),
        }
        rows.append(row)
    rows.sort(key=lambda r: (-r["elapsed_s"]))

    flagged = [r for r in rows if r["dup_of_landed"] is True or r["overdue"] == "OVERDUE"]
    n_dup = sum(1 for r in rows if r["dup_of_landed"] is True)
    n_overdue = sum(1 for r in rows if r["overdue"] == "OVERDUE")
    n_unknown = sum(1 for r in rows if r["overdue"] == "UNKNOWN")

    bits = []
    if n_dup:
        bits.append("%d DUP-OF-LANDED" % n_dup)
    if n_overdue:
        bits.append("%d OVERDUE" % n_overdue)
    if not bits:
        bits.append("clean")
    summary_line = ("POOL STALL CHECK: %s (of %d running across %d node(s), %d unknown-history, %d unreachable)"
                     % (", ".join(bits), len(rows), len(node_list), n_unknown, len(unreachable)))

    return {
        "nodes": node_list,
        "unreachable": unreachable,
        "running": rows,
        "flagged": flagged,
        "n_running": len(rows),
        "n_dup": n_dup,
        "n_overdue": n_overdue,
        "n_unknown_overdue": n_unknown,
        "summary_line": summary_line,
    }


def format_row(row):
    tag = []
    if row["dup_of_landed"] is True:
        tag.append("DUP-OF-LANDED")
    if row["overdue"] == "OVERDUE":
        tag.append("OVERDUE")
    if not tag:
        tag.append(row["overdue"])
    return ("%s job=%s pid(s)=%s elapsed=%ds module=%s rev=%s out=%s [%s] -- kill: %s"
            % (row["node"], row["job_id"], ",".join(row["pids"]) or "?", row["elapsed_s"],
               row["module"] or "?", row["pinned_sha"] or "-", row["out_path"] or "-",
               "/".join(tag), row["kill_cmd"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only pool duplicate/stall check (never kills anything).")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=int, default=12)
    args = ap.parse_args(argv)
    report = check_all(timeout=args.timeout)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(report["summary_line"])
    if report["unreachable"]:
        print("  unreachable: %s" % ", ".join(report["unreachable"]))
    for row in report["running"]:
        marker = "⚠ " if row in report["flagged"] else "  "
        print("%s%s" % (marker, format_row(row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
