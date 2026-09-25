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

ALSO CHECKS (2026-09-25 addition), read-only, every QUEUED (not-yet-dispatched) line in research/queue/pool.queue
-- `check_queue()`, see its own module-level comment for the incident (six revision-pinned lines sat queued 7.5h
because the revision was never provisioned where it could fit, and nothing outside the dispatcher's own
per-cycle log line ever said so):
  (d) UNRUNNABLE -- a revision-pinned line whose revision is missing (no `.provisioned_ok`) on every node whose
      raw MemTotal could ever fit its declared `mem_gb`. Reports the exact `pool_provision.sh --isolated`
      command to fix it.
  (e) memory_budget_stalled -- a line whose declared `mem_gb` exceeds every KNOWN node's raw ceiling outright --
      no amount of provisioning helps; it needs a smaller size or a bigger node.

Usage:
    python -m tools.pool_stall_check                 # human-readable report, all pool nodes (running + queue)
    python -m tools.pool_stall_check --json           # machine-readable, for parallel_audit.py
    python -m tools.pool_stall_check --skip-queue     # running-jobs check only (no pool.queue scan)

Read-only. Never raises past `check_all()`/`check_queue()` in normal operation -- every ssh/parse step is wrapped
so an unreachable node or a malformed log line degrades to UNKNOWN/UNREACHABLE, never a crash, matching the
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

# job_est_gb()'s size hints, mirroring pool_autodispatch.sh's job_est_gb() bash function EXACTLY (same three
# fallbacks, same order) so a queued line's declared size here agrees with what the dispatcher itself would
# reserve for it -- a drift between the two would make the UNRUNNABLE/memory-budget check below disagree with
# the dispatcher about whether a node could ever take the job.
_MEM_GB_RE = re.compile(r"mem_gb=(\d+)")
_MEMCAP_RE = re.compile(r"memcap\.sh (\d+)")
_RUNNER_MOD_FLAG_RE = re.compile(r"-m research\.runners\.([A-Za-z0-9_]+)")
_POOL_RUNNER_MEM_DEFAULT = os.path.join(_ROOT, "tools", "pool_runner_mem.tsv")

# The ONE marker filename `.provisioned_ok` means "this revision dir completed provisioning" (see
# tools/pool_revision_marker.sh's own docstring for the fix history behind requiring it, not a bare directory
# check). That file defines POOL_REVISION_MARKER_FILE as the single source of truth for the bash dispatcher/
# queue scripts; this module cannot `source` bash, so the same literal is duplicated here -- kept from drifting
# apart by test_provisioned_marker_filename_matches_bash_source (tests/test_pool_stall_check.py), which reads
# pool_revision_marker.sh's own line and asserts this constant still matches it.
POOL_REVISION_MARKER_FILE = ".provisioned_ok"


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


def queue_path():
    return os.environ.get("POOL_QUEUE_PATH", _QUEUE_PATH_DEFAULT)


def queue_claims_path():
    return queue_path() + ".claims"


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


def _load_epoch_text_lines(path, now=None, max_age_s=None):
    """[(epoch:int, text:str), ...] from a `epoch\\ttext` file -- the format shared by BOTH
    research/queue/pool.queue (staged, not-yet-dispatched lines, timestamped at `add` time -- see
    pool_queue.sh's `add`) and research/queue/pool.queue.claims (the dispatcher's own append-only pop-time
    record -- see pool_autodispatch.sh's pop_job). `text` still carries any trailing '#checked:...' annotation;
    job_signature()/pinned_revision()/job_est_gb() are all robust to that (regex search over the whole string,
    not a positional parse -- see job_signature's own docstring)."""
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


def load_claims(claims_path=None, now=None, max_age_s=None):
    """[(epoch:int, job_text:str), ...] from research/queue/pool.queue.claims -- see _load_epoch_text_lines."""
    return _load_epoch_text_lines(claims_path or queue_claims_path(), now=now, max_age_s=max_age_s)


def load_queue(path=None, now=None, max_age_s=None):
    """[(epoch:int, job_text:str), ...] from research/queue/pool.queue -- the STAGED, not-yet-dispatched lines
    (a line is removed from this file the moment pop_job hands it to a node; the DISPATCH timestamp then lives
    in pool.queue.claims instead). See _load_epoch_text_lines for the shared format."""
    return _load_epoch_text_lines(path or queue_path(), now=now, max_age_s=max_age_s)


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


# ============================================================================================ QUEUE-LINE CHECKS
# WHY (2026-09-25, one addition found while reviewing the check above). The RUNNING-job checks above cannot see a
# job that never got to START: six SETTLE A2 lines, pinned to one revision, sat in research/queue/pool.queue for
# 7.5h because that revision was never provisioned on the AWS nodes (pool1/pool2) and the mini-PC nodes
# (pool41/pool42) never had room for the line's declared mem_gb. pool_autodispatch.sh's own per-cycle log
# ("revision ... not provisioned on pool2") is the ONLY place this was ever said -- a line in a log nobody was
# tailing, not a heartbeat line, not a report. These checks read research/queue/pool.queue directly (the STAGED,
# not-yet-dispatched lines -- see load_queue) and flag two DIFFERENT ways a line can never run:
#   UNRUNNABLE           -- its pinned revision is missing (no .provisioned_ok) on every node that could ever
#                            physically fit its declared size. Fixable: provision that revision on those nodes.
#   memory_budget_stalled -- its declared size exceeds EVERY known node's raw capacity, regardless of revision.
#                            Not fixable by provisioning -- the line needs a smaller size or a bigger node.
# Both are read-only probes (MemTotal, the .provisioned_ok marker) -- this never provisions, dispatches, or
# removes anything, matching the running-job checks' own read-only contract.


def job_est_gb(text, runner_mem_path=None):
    """The GB a queued line declares for itself, mirroring pool_autodispatch.sh's job_est_gb() bash function
    exactly (same three fallbacks in the same order), so this module's notion of a job's size never disagrees
    with what the dispatcher itself would reserve for it:
      1. an explicit `mem_gb=N` anywhere in the line (e.g. in its --checked reason);
      2. else a `tools/memcap.sh N` wrapper's own declared cap;
      3. else this checkout's measured-peak table (tools/pool_runner_mem.tsv), keyed by the `-m
         research.runners.<module>` it runs;
      4. else POOL_JOB_EST_GB (env, default 1) -- the dispatcher's own final fallback.
    """
    if text:
        m = _MEM_GB_RE.search(text)
        if m:
            return int(m.group(1))
        m = _MEMCAP_RE.search(text)
        if m:
            return int(m.group(1))
        m = _RUNNER_MOD_FLAG_RE.search(text)
        if m:
            mod = m.group(1)
            path = runner_mem_path or os.environ.get("POOL_RUNNER_MEM_PATH", _POOL_RUNNER_MEM_DEFAULT)
            if os.path.isfile(path):
                try:
                    with open(path, errors="ignore") as f:
                        for line in f:
                            line = line.rstrip("\n")
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 2 and parts[0] == mod:
                                try:
                                    return int(parts[1])
                                except ValueError:
                                    break
                except OSError:
                    pass
    try:
        return int(os.environ.get("POOL_JOB_EST_GB", "1"))
    except ValueError:
        return 1


def probe_mem_total_gb(node, timeout=10, connect_timeout=6):
    """This node's raw /proc/meminfo MemTotal in whole GB, read-only, or None if unreachable/unparseable. This
    is a CEILING (what the machine physically has), not current availability -- deliberately: current free RAM
    changes every second and is not what decides whether a job could EVER run here."""
    cmd = (["ssh", "-n"] + ssh_config_args()
           + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % connect_timeout, node,
              "awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo"])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0:
        return None
    try:
        return int((p.stdout or "").strip())
    except ValueError:
        return None


def all_mem_totals(nodes, timeout=10, connect_timeout=6):
    """-> ({node: mem_total_gb}, [unreachable_node, ...]). Never raises: a per-node probe failure degrades that
    node to 'unreachable', never aborts the others."""
    totals, unreachable = {}, []
    for node in nodes:
        try:
            gb = probe_mem_total_gb(node, timeout=timeout, connect_timeout=connect_timeout)
        except Exception:
            gb = None
        if gb is None:
            unreachable.append(node)
        else:
            totals[node] = gb
    return totals, unreachable


def check_provisioned(node, sha, timeout=10, connect_timeout=6):
    """True/False/None -- read-only probe of `~/derisk-pool/revisions/<sha>/POOL_REVISION_MARKER_FILE` on
    `node`, the exact predicate tools/pool_revision_marker.sh:revision_marker_probe_cmd defines for the bash
    dispatcher/queue scripts (see POOL_REVISION_MARKER_FILE's own comment for why the literal is duplicated
    here). None means the probe itself could not be completed (unreachable, timeout, non-standard ssh failure)
    -- a DIFFERENT state from False (reached the node, the marker is genuinely absent): callers must not treat
    an unreachable node as a confident 'not provisioned'."""
    remote_dir = "derisk-pool/revisions/%s" % sha
    cmd = (["ssh", "-n"] + ssh_config_args()
           + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % connect_timeout, node,
              "[ -f ~/%s/%s ]" % (remote_dir, POOL_REVISION_MARKER_FILE)])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode == 0:
        return True
    if p.returncode == 1:
        return False
    return None  # ssh itself failed (unreachable, auth, ...) -- not a confident answer either way


def fix_provision_command(sha, nodes):
    """The EXACT command a human can paste to provision `sha` (isolated) on the given nodes -- this tool never
    runs it. Empty `nodes` still returns a syntactically valid (if useless) command rather than raising; callers
    only reach here with a non-empty missing-node list in practice (see check_queue)."""
    return "bash tools/pool_provision.sh --revision %s --isolated %s" % (sha, " ".join(nodes))


DEFAULT_UNRUNNABLE_MIN_AGE_MIN = 30   # a normal provision (rsync + venv + sanity build) finishes in minutes, not this
DEFAULT_MEMBUDGET_MIN_AGE_H = 1       # a structural ceiling doesn't change with time, but age-gate anyway (less noise)


def check_queue(nodes=None, root=None, queue_path_=None, now=None, timeout=10, connect_timeout=6,
                 unrunnable_min_age_min=None, membudget_min_age_h=None, os_reserve_gb=None):
    """Read-only scan of research/queue/pool.queue (STAGED, not-yet-dispatched lines) for two ways a line can
    never run that pool_autodispatch.sh's own per-cycle log never surfaces anywhere else (see the module-level
    comment above this section for the 2026-09-25 incident this closes):
      UNRUNNABLE            -- pinned to a revision missing (no .provisioned_ok) on every node whose raw
                                MemTotal could ever fit the line's declared mem_gb + POOL_OS_RESERVE_GB.
      memory_budget_stalled -- declared mem_gb exceeds every KNOWN node's raw ceiling, full stop (no revision
                                would help; the line needs a smaller size or a bigger node).
    Both are age-gated (a line staged moments ago is not flagged mid-provisioning). Never raises past this
    function in normal operation -- every ssh probe and file read is guarded, degrading to unreachable/UNKNOWN
    rather than aborting the scan, matching check_all()'s own contract."""
    import time as _time
    root = root or _ROOT
    now = now if now is not None else int(_time.time())
    node_list = nodes if nodes is not None else get_pool_nodes()
    unrunnable_min_age_s = 60 * (unrunnable_min_age_min if unrunnable_min_age_min is not None
                                  else int(os.environ.get("POOL_UNRUNNABLE_MIN_AGE_MIN", DEFAULT_UNRUNNABLE_MIN_AGE_MIN)))
    membudget_min_age_s = 3600 * (membudget_min_age_h if membudget_min_age_h is not None
                                   else float(os.environ.get("POOL_MEMBUDGET_MIN_AGE_H", DEFAULT_MEMBUDGET_MIN_AGE_H)))
    reserve = os_reserve_gb if os_reserve_gb is not None else int(os.environ.get("POOL_OS_RESERVE_GB", "2"))

    try:
        mem_totals, mem_unreachable = all_mem_totals(node_list, timeout=timeout, connect_timeout=connect_timeout)
    except Exception:
        mem_totals, mem_unreachable = {}, list(node_list)

    try:
        entries = load_queue(queue_path_, now=now)
    except Exception:
        entries = []

    marker_cache = {}   # (node, sha) -> True/False/None -- one probe per pair even if several lines share a sha

    def _provisioned(node, sha):
        key = (node, sha)
        if key not in marker_cache:
            try:
                marker_cache[key] = check_provisioned(node, sha, timeout=timeout, connect_timeout=connect_timeout)
            except Exception:
                marker_cache[key] = None
        return marker_cache[key]

    unrunnable = []
    mem_stalled = []
    for epoch, text in entries:
        age_s = now - epoch
        mem_gb = job_est_gb(text)
        sha = pinned_revision(text)
        sig = job_signature(text)
        mod = sig[0] if sig else None
        capable = [n for n, gb in mem_totals.items() if (gb - reserve) >= mem_gb]

        if not capable:
            if age_s >= membudget_min_age_s:
                ceilings = [gb - reserve for gb in mem_totals.values()]
                mem_stalled.append({
                    "epoch": epoch, "age_s": age_s, "module": mod, "mem_gb": mem_gb,
                    "max_known_ceiling_gb": max(ceilings) if ceilings else None,
                    "nodes_checked": sorted(mem_totals.keys()),
                    "command_snippet": text[:160],
                })
            continue

        if sha is None:
            continue   # unpinned (the ~/derisk-pool/sim compatibility path) -- always "provisioned"; not this check's job
        if age_s < unrunnable_min_age_s:
            continue

        status = {n: _provisioned(n, sha) for n in capable}
        if any(v is True for v in status.values()):
            continue   # at least one node that could run this already has the code -- not a provisioning stall

        missing_nodes = sorted(n for n, v in status.items() if v is not True)
        unrunnable.append({
            "epoch": epoch, "age_s": age_s, "module": mod, "pinned_sha": sha, "mem_gb": mem_gb,
            "capable_nodes": sorted(capable), "node_status": status,
            "fix_cmd": fix_provision_command(sha, missing_nodes),
            "command_snippet": text[:160],
        })

    unrunnable.sort(key=lambda r: -r["age_s"])
    mem_stalled.sort(key=lambda r: -r["age_s"])

    bits = []
    if unrunnable:
        bits.append("%d UNRUNNABLE (revision missing on every capable node)" % len(unrunnable))
    if mem_stalled:
        bits.append("%d over every known node's memory ceiling" % len(mem_stalled))
    if not bits:
        bits.append("clean")
    summary_line = ("POOL QUEUE CHECK: %s (of %d queued line(s), %d node(s) mem-unreachable)"
                     % (", ".join(bits), len(entries), len(mem_unreachable)))

    return {
        "nodes": node_list, "mem_totals": mem_totals, "mem_unreachable": mem_unreachable,
        "n_queued": len(entries), "unrunnable": unrunnable, "memory_budget_stalled": mem_stalled,
        "summary_line": summary_line,
    }


def format_unrunnable_row(row):
    missing = ",".join(n for n, v in row["node_status"].items() if v is not True) or "-"
    return ("queued %.1fh module=%s rev=%s mem_gb=%s capable=%s missing-on=%s -- fix: %s"
            % (row["age_s"] / 3600.0, row["module"] or "?", row["pinned_sha"], row["mem_gb"],
               ",".join(row["capable_nodes"]) or "-", missing, row["fix_cmd"]))


def format_membudget_row(row):
    ceiling = row["max_known_ceiling_gb"]
    return ("queued %.1fh module=%s mem_gb=%s exceeds every known node's ceiling (best=%s) -- needs a smaller "
            "mem_gb or a bigger node, re-provisioning will not help"
            % (row["age_s"] / 3600.0, row["module"] or "?", row["mem_gb"],
               ceiling if ceiling is not None else "unknown"))


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
    ap = argparse.ArgumentParser(
        description="Read-only pool duplicate/stall/unrunnable-queue check (never kills or provisions anything).")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--skip-queue", action="store_true",
                     help="skip the queued-line UNRUNNABLE/memory-budget check (running-jobs check only)")
    args = ap.parse_args(argv)
    report = check_all(timeout=args.timeout)
    queue_report = None if args.skip_queue else check_queue(timeout=args.timeout)
    if args.json:
        out = dict(report)
        if queue_report is not None:
            out["queue"] = queue_report
        print(json.dumps(out, indent=2))
        return 0
    print(report["summary_line"])
    if report["unreachable"]:
        print("  unreachable: %s" % ", ".join(report["unreachable"]))
    for row in report["running"]:
        marker = "⚠ " if row in report["flagged"] else "  "
        print("%s%s" % (marker, format_row(row)))
    if queue_report is not None:
        print(queue_report["summary_line"])
        for row in queue_report["unrunnable"]:
            print("⚠ %s" % format_unrunnable_row(row))
        for row in queue_report["memory_budget_stalled"]:
            print("⚠ %s" % format_membudget_row(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
