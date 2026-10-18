#!/usr/bin/env python3
"""tools/stall_watch.py -- READ-ONLY stall alarm for pool and GPU lines (RULING stall-alarm-1018, owner-approved
2026-10-18; research/coordination/LOCAL_LLM_RUNBOOK.md section 6). It never stops, releases or edits anything: it prints,
Hermes acts on the output per the ruling (check the pin's age against main's fixes, read the line's log, stop it, file
`[stall] INFO`, re-queue once and only after a fix).

WHY. The B2d Amendment-2 probe ran for days (11, by the ruling) on a stale pin and nobody compared a running line's age
with what its runner needs (FAILURE_LOG 2026-10-17 and 2026-10-18). A line whose output is still absent after TWICE its
expected wall is a STALL.

THE RULE. For every pool claim and for the workstation GPU queue's current job:
  expected wall = the runner's CANARY wall (latest PASS record of its module in research/queue/pool_canaries.tsv, the
                  `wall=<N>s` in its evidence column),
              else the duration of the LAST COMPLETED RUN of that runner at any pin (provenance sidecars, below),
              else 6 h.
  STALL         = elapsed since the claim/START > 2 x expected, AND the line's `--out` file(s) still absent, AND the line is
                  still running (pool: its JOB_B64 is in a live process environ on its node, the same liveness scan
                  tools/pool_queue.sh uses; GPU: gpu_queue.sh's status says it is the current job).
A claim that ended (no live process) without writing its output is a FAILURE, which status.sh's "Recent failures" shows;
it is not a stall and is not printed. A claim on a node that cannot be reached is UNKNOWN, never a guess.

WHERE THE DATA IS (read on the server, which has no GPU).
  pool claims   research/queue/pool.running (POOL_RUNNING_PATH, else beside POOL_QUEUE_PATH): `YYYY-MM-DD HH:MM:SS<TAB>node
                <TAB>job`, appended by tools/pool_autodispatch.sh at dispatch. The job text carries the pin
                (`cd ~/derisk-pool/revisions/<sha>`), the runner (`-m research.runners.x`) and `--out <path>`. The file only
                grows, so claims older than --max-age-days (30) are ignored, and of identical (node, job) repeats only the
                latest counts. Timestamps are read in THIS process's timezone: run it with the dispatcher's TZ.
  liveness      one ssh per node that has an overdue claim (`/proc/*/environ` JOB_B64 scan, 20 s timeout, ssh BatchMode).
  GPU job       tools/gpu_remote.sh status + log from the server (the 3090's queue lives on the workstation); on the
                workstation itself the queue files directly (gpu.running, gpu_queue.log). `current:` says whether a job
                runs; the log's START line without a later DONE gives its start time and full text.
  canaries      research/queue/pool_canaries.tsv (POOL_CANARIES_PATH): recorded_at, module, sha, verdict, node, seed_kind,
                via, evidence. The wall is in the evidence text (`rc=0 wall=1234s json=...`).
  provenance    research/findings/raw/**/*.prov.json, written at the END of a run by research/runners/__init__.py with
                `runner` and `started`. Duration = ended_utc_ns - started_utc_ns when the sidecar has them (v2); else the
                artifact's own top-level wall_s / wall_seconds / elapsed_seconds; else the sidecar's mtime minus `started`
                (refused when 25 or more sidecars share that mtime second: a git checkout or rsync stamp, not a write time).
                research/findings/raw/_provenance/runs.jsonl records starts only, so it cannot give a duration.

OUTPUT. One line per STALL:
  STALL <id> node=<node> runner=<runner> elapsed=<..> expected=<..> (<source>) pin=<sha9|none> pin_age_commits=<N> out=<path>
pin_age_commits = commits on main since the pin (`git rev-list --count <pin>..main`; `unknown` when the pin is not in this
repository, `n/a` for an unpinned line). Then `stall_watch: CLEAN` when there is no STALL and every source was readable;
`stall_watch: UNKNOWN <source>: <why>` for each source that was not (with no STALL, the run is not claimed CLEAN).
Exit 0 with no STALL (UNKNOWN included), 1 with any STALL. It never raises: a failing source becomes an UNKNOWN line.

CAVEATS. "Last completed run" may be a short smoke of a runner whose gate lines are long; expected has a floor
(--min-wall-s, 300 s) and the output names the source so the reader can discount it. A job that writes its `--out`
incrementally reads as not stalled once the file exists. Stdlib only, Python 3.8+.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.dirname(HERE)
DEFAULT_WALL_S = 6 * 3600
FACTOR = 2.0
MIN_WALL_S = 300
MAX_AGE_DAYS = 30
SENTINEL = "POOL_LIVENESS_SCAN_OK"
LIVE_SCAN = ("for e in /proc/[0-9]*/environ; do tr '\\0' '\\n' < $e 2>/dev/null; done | grep '^JOB_B64=' | sort -u; "
             "echo " + SENTINEL)
WALL_KEYS = ("wall_s", "wall_seconds", "wall_time_s", "elapsed_s", "elapsed_seconds", "runtime_s", "duration_s",
             "total_wall_s")
BULK_MTIME = 25
TS_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
START_RE = re.compile(r"^(%s) START: (.*)$" % TS_RE)
DONE_RE = re.compile(r"^(%s) DONE\(rc=-?\d+\): (.*)$" % TS_RE)
CURRENT_RE = re.compile(r"^current: (.*?)(?: \(pid (\d+)\))?\s*$")


# ---------------------------------------------------------------- small helpers
def fmt_dur(s):
    s = int(max(0, s))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d:
        return "%dd%02dh" % (d, h)
    if h:
        return "%dh%02dm" % (h, m)
    return "%dm" % m


def parse_ts(text):
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return None


def run_cmd(argv, timeout, cwd=None):
    """(rc, stdout, why). Never raises: a missing binary, a timeout or an OS error is (None, '', reason)."""
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None, "", "timed out after %ds" % timeout
    except (OSError, ValueError) as e:
        return None, "", "%s: %s" % (type(e).__name__, e)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace").strip()[:160]


def git_out(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") or k in ("GIT_SSH", "GIT_SSH_COMMAND")}
    try:
        r = subprocess.run(["git", "-C", root] + list(args), capture_output=True, timeout=20, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.decode("utf-8", "replace").strip() if r.returncode == 0 else None


def data_root(opts):
    if opts.root:
        return os.path.abspath(opts.root)
    if os.environ.get("STALL_WATCH_ROOT"):                  # tests: keep the read away from the shared checkout
        return os.path.abspath(os.environ["STALL_WATCH_ROOT"])
    if os.environ.get("SIM_WORKSPACE"):
        return os.environ["SIM_WORKSPACE"]
    common = git_out(SCRIPT_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return os.path.dirname(common) if common else SCRIPT_ROOT


# ---------------------------------------------------------------- a job line -> its features
def strip_reason_prefix(job):
    """tools/pool_queue.sh strip_checked_reason_prefix in Python: pop_job prepends `POOL_CHECKED_REASON=<%q token> `."""
    if not job.startswith("POOL_CHECKED_REASON="):
        return job
    s = job[len("POOL_CHECKED_REASON="):]
    i = 0
    if s.startswith("$'"):
        i = 2
        while i < len(s):
            if s[i] == "\\":
                i += 2
            elif s[i] == "'":
                i += 1
                break
            else:
                i += 1
    else:
        while i < len(s):
            if s[i] == "\\":
                i += 2
            elif s[i] == " ":
                break
            else:
                i += 1
    return s[i:].lstrip(" ")


def features(job):
    """{module, runner, runner_path, outs, pin, cd} of one job text."""
    cmd = strip_reason_prefix(job)
    mods = re.findall(r"(?:^|\s)-m\s+([A-Za-z_][\w.]*)", cmd)
    module = next((m for m in mods if m.startswith("research.runners")), mods[0] if mods else None)
    path = None
    if not module:
        m = re.search(r"((?:research/runners|tools)/[\w./-]+\.py)", cmd)
        if m:
            path = m.group(1)
            module = path[:-3].replace("/", ".")
    if module and not path:
        path = module.replace(".", "/") + ".py"
    if module:
        runner = module
    else:
        first = cmd.split()
        runner = os.path.basename(first[0]) if first else "?"
    outs = []
    for m in re.finditer(r"--out(?:put)?(?:=|\s+)([^\s;&|)<>\"']+)", cmd):
        v = m.group(1)
        if not v.startswith("-") and "$" not in v and v not in outs:
            outs.append(v)
    pin = re.search(r"derisk-pool/revisions/([0-9a-f]{7,40})(?:-\d+)?(?![0-9a-zA-Z])", cmd)
    cd = re.match(r"\s*cd\s+(\S+)\s*&&", cmd)
    return {"module": module, "runner": runner, "runner_path": path, "outs": outs,
            "pin": pin.group(1) if pin else None, "cd": cd.group(1) if cd else None}


def outs_present(root, feat):
    """True iff the line declares output file(s) and ALL exist (as given, under the data root, or under its cd dir)."""
    if not feat["outs"]:
        return False
    for o in feat["outs"]:
        cands = [os.path.expanduser(o)]
        if not os.path.isabs(o):
            cands.append(os.path.join(root, o))
            if feat["cd"]:
                cands.append(os.path.join(os.path.expanduser(feat["cd"]), o))
        if not any(os.path.exists(c) for c in cands):
            return False
    return True


def line_id(kind, node, ts, job):
    h = hashlib.sha1(job.encode("utf-8", "surrogateescape")).hexdigest()[:6]
    return "%s-%s-%s" % (node if kind == "pool" else "gpu", time.strftime("%Y%m%dT%H%M%S", time.localtime(ts)), h)


# ---------------------------------------------------------------- expected wall
def load_canaries(path):
    """{module: (wall_s, recorded_at)} from the latest PASS record that carries a wall; {} when unreadable."""
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if not ln.strip() or ln.lstrip().startswith("#"):
                    continue
                f = ln.rstrip("\n").split("\t")
                if len(f) < 8 or f[3] != "PASS":
                    continue
                m = re.search(r"\bwall=(\d+)s\b", f[7])
                if m and int(m.group(1)) > 0:
                    out[f[1]] = (int(m.group(1)), f[0][:10])
    except OSError:
        return {}
    return out


class Provenance:
    """Completed-run durations per runner path, read lazily from the .prov.json sidecars under the data root. The index
    (runner, started, mtime per sidecar) is one cheap pass over ~15k small files; a duration is worked out only for
    the newest sidecars of the runner being asked about, and the artifact is opened only when the sidecar has no end."""

    _RUNNER = re.compile(r'"runner":\s*"([^"]*)"')
    _STARTED = re.compile(r'"started":\s*"([^"]*)"')

    def __init__(self, root):
        self.root = root
        self._by_runner = None
        self._bulk = {}

    def _index(self):
        self._by_runner = {}
        pat = os.path.join(self.root, "research", "findings", "raw", "**", "*.prov.json")
        for f in glob.glob(pat, recursive=True):
            try:
                mtime = int(os.stat(f).st_mtime)
                with open(f, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            self._bulk[mtime] = self._bulk.get(mtime, 0) + 1
            r, st = self._RUNNER.search(text), self._STARTED.search(text)
            if r:
                self._by_runner.setdefault(r.group(1), []).append(
                    (self._started(st.group(1)) if st else None, f, mtime))
        for v in self._by_runner.values():
            v.sort(key=lambda t: t[0] or 0, reverse=True)

    @staticmethod
    def _started(text):
        try:
            return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return None

    def _duration(self, f, started, mtime):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return None, None
        if not isinstance(d, dict):
            return None, None
        s, e = d.get("started_utc_ns"), d.get("ended_utc_ns")
        if isinstance(s, int) and isinstance(e, int) and e > s:
            return (e - s) / 1e9, "sidecar"
        art = d.get("artifact")
        if isinstance(art, str):
            p = art if os.path.isabs(art) else os.path.join(self.root, art)
            try:
                if os.path.getsize(p) <= 64 * 1024 * 1024:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        doc = json.load(fh)
                    if isinstance(doc, dict):
                        for k in WALL_KEYS:
                            v = doc.get(k)
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                                return float(v), "artifact " + k
            except (OSError, ValueError):
                pass
        if started and self._bulk.get(mtime, 0) < BULK_MTIME and 0 < mtime - started < 14 * 86400:
            return float(mtime - started), "sidecar mtime"
        return None, None

    def last_runs(self, runner_path, n=1):
        """[(dur_s, how, started_text)] for the n most recent completed runs of `runner_path` ([] when none)."""
        if self._by_runner is None:
            self._index()
        out = []
        for started, f, mtime in self._by_runner.get(runner_path, [])[:200]:
            dur, how = self._duration(f, started, mtime)
            if dur and dur > 0:
                out.append((dur, how, time.strftime("%Y-%m-%d", time.localtime(started)) if started else "?"))
                if len(out) >= n:
                    break
        return out


def expected_wall(feat, canaries, prov, opts):
    """(seconds, source text)."""
    mod = feat["module"]
    if mod and mod in canaries:
        w, day = canaries[mod]
        return max(w, opts.min_wall_s), "canary %ds %s" % (w, day) + (" +floor" if w < opts.min_wall_s else "")
    if feat["runner_path"]:
        runs = prov.last_runs(feat["runner_path"], opts.last_n)
        if runs:
            dur, how, started = max(runs)
            return max(dur, opts.min_wall_s), "last run %s %s (%s)" % (fmt_dur(dur), started, how) + (
                " +floor" if dur < opts.min_wall_s else "")
    return float(DEFAULT_WALL_S), "default 6h"


# ---------------------------------------------------------------- sources
def running_path(opts, root):
    if opts.running:
        return opts.running
    if os.environ.get("POOL_RUNNING_PATH"):
        return os.environ["POOL_RUNNING_PATH"]
    q = os.environ.get("POOL_QUEUE_PATH")
    if q:
        return (q[:-len(".queue")] if q.endswith(".queue") else q) + ".running"
    return os.path.join(root, "research", "queue", "pool.running")


def read_pool_claims(path, now, max_age_days):
    """(claims, unknown_reason). claims = [{ts, node, job}] the latest per (node, job), within the age window."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as fh:
            rows = fh.read().splitlines()
    except OSError as e:
        return [], "cannot read %s (%s)" % (path, e.strerror or e)
    latest = {}
    for ln in rows:
        f = ln.split("\t", 2)
        if len(f) < 3 or not f[1] or not f[2]:
            continue
        ts = parse_ts(f[0])
        if ts is None or now - ts > max_age_days * 86400:
            continue
        latest[(f[1], f[2])] = max(ts, latest.get((f[1], f[2]), 0))
    return [{"ts": ts, "node": n, "job": j} for (n, j), ts in latest.items()], None


def node_live_set(node, opts):
    """(set of live JOB_B64 values | None, why). The scan tools/pool_queue.sh node_live_b64_set runs, with its sentinel."""
    argv = [opts.ssh]
    cfg = opts.ssh_config or os.path.join(SCRIPT_ROOT, "research", "queue", ".pool_ssh_config")
    if cfg and os.path.isfile(cfg):
        argv += ["-F", cfg]
    argv += ["-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, LIVE_SCAN]
    rc, out, why = run_cmd(argv, 20)
    if rc is None or SENTINEL not in out:
        return None, why or "no liveness answer (rc %s)" % rc
    return {ln[len("JOB_B64="):].strip() for ln in out.splitlines() if ln.startswith("JOB_B64=")}, ""


def gpu_texts(opts, root):
    """(status_text, log_text, reason). reason is non-empty when the GPU queue could not be read."""
    if opts.gpu == "none":
        return None, None, "off"
    mode = opts.gpu
    qdir = os.environ.get("GPU_QUEUE_DIR") or os.path.join(root, "research", "queue")
    if opts.gpu_status_cmd:
        mode = "cmd"
    elif mode == "auto":
        if os.environ.get("GPU_QUEUE_DIR") or any(os.path.exists(os.path.join(qdir, f)) for f in
                                                 ("gpu_queue.dpid", "gpu.running", "gpu_queue.log")):
            mode = "local"
        elif os.path.exists(os.path.join(SCRIPT_ROOT, "tools", "gpu_remote.sh")):
            mode = "remote"
        else:
            return None, None, "off"
    n = str(opts.gpu_log_lines)
    if mode == "local":
        if not os.path.isdir(qdir):
            return None, None, "no GPU queue dir %s" % qdir
        try:
            running = os.path.join(qdir, "gpu.running")
            if os.path.exists(running):
                with open(running, encoding="utf-8", errors="replace") as fh:
                    pid, _, job = fh.read().rstrip("\n").partition("\t")
                status = "current: %s (pid %s)" % (job[:100], pid)
            else:
                status = "current: (idle)"
            log = ""
            lp = os.path.join(qdir, "gpu_queue.log")
            if os.path.exists(lp):
                with open(lp, encoding="utf-8", errors="replace") as fh:
                    log = "".join(fh.readlines()[-opts.gpu_log_lines:])
        except OSError as e:
            return None, None, "cannot read the GPU queue files (%s)" % (e.strerror or e)
        return status, log, ""
    if mode == "remote":
        sh = os.path.join(SCRIPT_ROOT, "tools", "gpu_remote.sh")
        s_argv, l_argv = ["bash", sh, "status"], ["bash", sh, "log", n]
    else:                                                    # explicit commands (tests, odd setups)
        s_argv = shlex.split(opts.gpu_status_cmd)
        l_argv = shlex.split(opts.gpu_log_cmd) if opts.gpu_log_cmd else None
    rc, status, why = run_cmd(s_argv, opts.gpu_timeout)
    if rc != 0:
        return None, None, "GPU status command failed (%s)" % (why or "rc %s" % rc)
    log = ""
    if l_argv:
        rc, log, why = run_cmd(l_argv, opts.gpu_timeout)
        if rc != 0:
            return None, None, "GPU log command failed (%s)" % (why or "rc %s" % rc)
    return status, log, ""


def gpu_current(status, log):
    """(None, None) idle | (job, start_ts) running with a known START | ('?', reason) running but unknown."""
    cur = next((CURRENT_RE.match(ln) for ln in status.splitlines() if ln.startswith("current:")), None)
    if not cur:
        return "?", "the GPU status has no `current:` line"
    text = cur.group(1).strip()
    if text == "(idle)":
        return None, None
    open_jobs = []
    for ln in log.splitlines():
        m = START_RE.match(ln)
        if m:
            open_jobs.append((m.group(1), m.group(2)))
            continue
        m = DONE_RE.match(ln)
        if m:
            for i in range(len(open_jobs) - 1, -1, -1):
                if open_jobs[i][1] == m.group(2):
                    del open_jobs[i]
                    break
    for ts, job in reversed(open_jobs):
        if job.startswith(text) or text.startswith(job[:100]):
            t = parse_ts(ts)
            return (job, t) if t is not None else ("?", "unreadable START time %r" % ts)
    return "?", "the current job %r has no START line in the last log lines" % text[:60]


# ---------------------------------------------------------------- the scan
def pin_age(root, pin, main_ref):
    if not pin:
        return "n/a"
    n = git_out(root, "rev-list", "--count", "%s..%s" % (pin, main_ref))
    return n if n is not None and n.isdigit() else "unknown"


def main_ref_of(root):
    forced = os.environ.get("STALL_WATCH_MAIN_REF")
    for ref in ([forced] if forced else []) + ["main", "origin/main"]:
        if git_out(root, "rev-parse", "--verify", "-q", ref + "^{commit}"):
            return ref
    return "main"


def scan(opts, now=None):
    """-> (stalls, unknowns). stalls = dicts ready to print; unknowns = reason strings. Never raises on a source."""
    now = time.time() if now is None else now
    root = data_root(opts)
    canaries_path = opts.canaries or os.environ.get("POOL_CANARIES_PATH") or os.path.join(
        root, "research", "queue", "pool_canaries.tsv")
    canaries = load_canaries(canaries_path)
    prov = Provenance(root)
    stalls, unknowns, cand, future = [], [], [], []
    main_ref = None

    def consider(kind, node, ts, job):
        elapsed = now - ts
        if elapsed < 0:
            future.append(line_id(kind, node, ts, job))
            return
        if elapsed <= opts.factor * opts.min_wall_s:
            return                                                         # younger than any possible stall
        feat = features(job)
        if outs_present(root, feat):
            return
        exp, src = expected_wall(feat, canaries, prov, opts)
        if elapsed <= opts.factor * exp:
            return
        cand.append({"kind": kind, "node": node, "ts": ts, "job": job, "feat": feat, "elapsed": elapsed,
                     "expected": exp, "source": src})

    # --- pool
    if opts.pool:
        claims, why = read_pool_claims(running_path(opts, root), now, opts.max_age_days)
        if why:
            unknowns.append("pool: " + why)
        for c in claims:
            consider("pool", c["node"], c["ts"], c["job"])
    # --- GPU
    status, log, why = gpu_texts(opts, root)
    if why and why != "off":
        unknowns.append("gpu: " + why)
    elif status is not None:
        job, t = gpu_current(status, log)
        if job == "?":
            unknowns.append("gpu: " + t)
        elif job is not None:
            consider("gpu", "gpu", t, job)
    if future:
        unknowns.append("%d claim(s) are dated in the future, e.g. %s (timezone mismatch? run with the dispatcher's TZ)"
                        % (len(future), future[0]))
    # --- liveness for the overdue pool claims (one ssh per node, concurrently)
    live_nodes = sorted({c["node"] for c in cand if c["kind"] == "pool"})
    live = {}
    if live_nodes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(live_nodes))) as ex:
            futs = {n: ex.submit(node_live_set, n, opts) for n in live_nodes}
            for n, f in futs.items():
                try:
                    live[n] = f.result(timeout=60)
                except Exception as e:                                     # noqa: BLE001 -- UNKNOWN, never a crash
                    live[n] = (None, "%s: %s" % (type(e).__name__, e))
    unreachable = {}
    for c in sorted(cand, key=lambda c: c["ts"]):
        if c["kind"] == "pool":
            alive, w = live.get(c["node"], (None, "no answer"))
            if alive is None:
                unreachable.setdefault(c["node"], [w, []])[1].append(line_id("pool", c["node"], c["ts"], c["job"]))
                continue
            b64 = base64.b64encode(c["job"].encode("utf-8", "surrogateescape")).decode()
            if b64 not in alive:
                continue                                                   # ended: a failure, not a stall
        f = c["feat"]
        if f["pin"] and main_ref is None:
            main_ref = main_ref_of(root)
        stalls.append({"id": line_id(c["kind"], c["node"], c["ts"], c["job"]), "node": c["node"], "runner": f["runner"],
                       "elapsed": c["elapsed"], "expected": c["expected"], "source": c["source"],
                       "pin": f["pin"][:9] if f["pin"] else "none", "pin_age": pin_age(root, f["pin"], main_ref),
                       "out": f["outs"][0] if f["outs"] else "none"})
    for n, (w, ids) in sorted(unreachable.items()):
        unknowns.append("node %s unreachable (%s): %d claim(s) past %gx their expected wall cannot be checked, e.g. %s"
                        % (n, w, len(ids), opts.factor, ids[0]))
    return stalls, unknowns


def render(stalls, unknowns):
    out = []
    for s in stalls:
        out.append("STALL %s node=%s runner=%s elapsed=%s expected=%s (%s) pin=%s pin_age_commits=%s out=%s"
                   % (s["id"], s["node"], s["runner"], fmt_dur(s["elapsed"]), fmt_dur(s["expected"]), s["source"],
                      s["pin"], s["pin_age"], s["out"]))
    for u in unknowns:
        out.append("stall_watch: UNKNOWN " + u)
    if not stalls and not unknowns:
        out.append("stall_watch: CLEAN")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="data root (default: STALL_WATCH_ROOT, SIM_WORKSPACE, else the parent of the git "
                                   "common dir)")
    ap.add_argument("--running", help="pool.running path (default: POOL_RUNNING_PATH / beside POOL_QUEUE_PATH / root)")
    ap.add_argument("--canaries", help="pool_canaries.tsv path (default: POOL_CANARIES_PATH / root)")
    ap.add_argument("--ssh", default=os.environ.get("STALL_WATCH_SSH") or "ssh", help="ssh binary (tests use a stub)")
    ap.add_argument("--ssh-config", default=os.environ.get("POOL_SSH_CONFIG"))
    ap.add_argument("--gpu", choices=("auto", "local", "remote", "none"), default=os.environ.get("STALL_WATCH_GPU") or "auto")
    ap.add_argument("--gpu-status-cmd", help="command printing `gpu_queue.sh status` (tests)")
    ap.add_argument("--gpu-log-cmd", help="command printing the tail of gpu_queue.log (tests)")
    ap.add_argument("--gpu-log-lines", type=int, default=3000)
    ap.add_argument("--gpu-timeout", type=int, default=25)
    ap.add_argument("--no-pool", dest="pool", action="store_false")
    ap.add_argument("--factor", type=float, default=FACTOR)
    ap.add_argument("--min-wall-s", type=float, default=MIN_WALL_S)
    ap.add_argument("--max-age-days", type=float, default=MAX_AGE_DAYS)
    ap.add_argument("--last-n", type=int, default=1, help="use the longest of the n most recent completed runs")
    ap.add_argument("--now", type=float, help="epoch seconds to use as now (tests)")
    return ap.parse_args(argv)


def main(argv=None):
    try:
        opts = parse_args(argv)
        stalls, unknowns = scan(opts, opts.now)
    except SystemExit:
        raise
    except Exception as e:                                   # noqa: BLE001 -- a watcher that crashes is no watcher
        print("stall_watch: UNKNOWN internal error (%s: %s)" % (type(e).__name__, e))
        return 0
    print("\n".join(render(stalls, unknowns)))
    return 1 if stalls else 0


if __name__ == "__main__":
    sys.exit(main())
