#!/usr/bin/env python3
"""attention_watch: wake Claude ONLY when something needs action (owner 2026-10-28).

Why: the owner kept finding problems before Claude did (a ruled driver task stalled two days with no NEEDS, the
workstation GPU idle with an empty queue, a GPU job launched from a checkout missing its untracked corpus, which only
surfaced when memcap's oom-kill notice reached the owner's desktop). The older heartbeat (tools/heartbeat_cmd.sh)
is workstation-only, prints every cycle whether or not anything is wrong, and was not re-armed after a compaction.

This checks the live sources on both hosts and prints one line per NEW actionable event, then exits (so a
background `--loop` run notifies once per event; re-arm after handling). Silence means nothing needs action.
Events already reported are suppressed for REPEAT_H hours (state file outside the repo).

  python3 tools/attention_watch.py --once          # print current events (does not update the state file)
  python3 tools/attention_watch.py --loop          # poll every --interval s; print new events and exit
  python3 tools/attention_watch.py --ack KEY       # silence one event key for REPEAT_H hours
  tools/status.sh runs `--once` (on the server: the server checks, run locally), so the driver sees every event each wake;
  .claude/hooks/watcher_guard.py tells Claude to re-arm `--loop` whenever it is not running (reboot, new session, compaction).

Checks: W GPU idle with an empty queue; a GPU job exiting non-zero; a degraded brain build in the GPU log
(missing untracked inputs); an oom-kill; S NEEDS CLAUDE lines; the S driver silent; the S pool idle or not claiming;
a recent RULING whose requested [key] report no later driver line mentions after RULE_STALE_H hours; a driver STOP
written only in the WORKLOG that no RULING answers.
"""
import argparse, json, os, re, subprocess, sys, time

def _shared_root():
    """The shared checkout root (parent of the git common dir): the gpu_queue and its log live there, not per worktree."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        cd = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, timeout=10).stdout.strip()
        return os.path.dirname(cd) if cd else os.path.dirname(here)
    except Exception:
        return os.path.dirname(here)


REPO = _shared_root()
STATE = os.path.expanduser("~/Projects/sim-rev/watch/attention_state.json")
S_REPO = "/home/dant123/Projects/neural-simulator"
S_EXEC = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "homelab-server",
          "docker", "exec", "-u", "99", "-e", "HOME=/opt/data/home", "hermes", "bash", "-c"]
REPEAT_H = 6.0          # re-report a still-true event after this long
GPU_IDLE_MIN = 30       # W GPU queue empty and idle this long
GPU_HELD_MIN = 30       # W gpu_queue has jobs but none running this long (VRAM held outside the queue)
DRIVER_SILENT_H = 3.0   # S runbook not written this long
POOL_STALL_MIN = 60     # S pool has queued lines but no claim this long
PI_STATUS = "/var/lib/host-status/pi-driver/status.json"   # tools/pi_driver supervisor.py writes it
PI_SILENT_MIN = 20      # Pi driver status not refreshed this long (the supervisor writes every 5 s)
PI_RESTARTS = 5         # Pi driver restarts worth a look (one alert per 5)
BIG_STARVE_MIN = 90     # a >= 8 GB pool line queued this long while small lines are claimed = starvation
RULE_STALE_H = 24.0     # a requested [key] report not mentioned by the driver this long after its RULING
RULE_LOOKBACK_D = 4
STOP_LOOKBACK_D = 2     # driver STOP lines this recent need a RULING naming their [key]


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception as e:  # a failed probe is itself reported, never silent
        return 99, f"{type(e).__name__}: {e}"


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"seen": {}, "gpu_idle_since": None, "gpu_log_off": 0, "journal_since": None}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def w_checks(st, now):
    ev = []
    rc, out = sh(["bash", os.path.join(REPO, "tools/gpu_queue.sh"), "status"])
    paused = "PAUSED" in out
    idle = "current: (idle)" in out and re.search(r"queued: 0\b", out) is not None
    # an empty queue is not an idle GPU when something outside the queue holds it (the local model, a trial):
    # >4 GB in use means occupied
    _, mem = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=15)
    if mem.strip().split("\n")[0].strip().isdigit() and int(mem.strip().split("\n")[0]) > 4096:
        idle = False
    if idle and not paused:
        st["gpu_idle_since"] = st.get("gpu_idle_since") or now
        mins = (now - st["gpu_idle_since"]) / 60
        if mins >= GPU_IDLE_MIN:
            ev.append(("w-gpu-idle", f"W GPU idle {mins:.0f} min with an empty gpu_queue (not paused): stock GPU work"))
    else:
        st["gpu_idle_since"] = None
    # the opposite stall: jobs queued, none running, not paused. On 2026-10-31 an orphaned local-model engine held
    # 21 GB of VRAM for 5 h; the dispatcher held every job and w-gpu-idle stayed silent because the queue was not empty.
    m_q = re.search(r"queued: (\d+)", out)
    held = "current: (idle)" in out and m_q is not None and int(m_q.group(1)) > 0
    if held and not paused:
        st["gpu_held_since"] = st.get("gpu_held_since") or now
        mins = (now - st["gpu_held_since"]) / 60
        if mins >= GPU_HELD_MIN:
            _, apps = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                          "--format=csv,noheader"], timeout=15)
            big = "; ".join(l.strip() for l in apps.splitlines() if "MiB" in l and int(l.split(",")[1].split()[0]) > 1024)
            ev.append(("w-gpu-held", f"W gpu_queue has {m_q.group(1)} job(s) held {mins:.0f} min with nothing running "
                                     f"(not paused); VRAM holders: {big or 'none >1 GB'}"))
    else:
        st["gpu_held_since"] = None
    log = os.path.join(REPO, "research/queue/gpu_queue.log")
    try:
        size = os.path.getsize(log)
        off = st.get("gpu_log_off", 0)
        if off > size:
            off = 0
        with open(log, errors="replace") as f:
            f.seek(off)
            new = f.read()
        st["gpu_log_off"] = size
        for m in re.finditer(r"DONE\(rc=(\d+)\): (.{0,140})", new):
            if m.group(1) != "0":
                ev.append((f"w-gpu-rc-{abs(hash(m.group(2))) % 10**8}", f"W GPU job exited rc={m.group(1)}: {m.group(2)}"))
        for pat in ("build FAILED", "FileNotFoundError", "degrading to standalone"):
            for m in re.finditer(re.escape(pat) + r".{0,160}", new):
                ev.append((f"w-gpu-degraded-{abs(hash(m.group(0))) % 10**8}",
                           f"W GPU job log shows a degraded build (inputs missing?): {m.group(0)[:200]}"))
    except OSError:
        pass
    since = st.get("journal_since") or "-30min"
    rc, out = sh(["journalctl", "--user", "--since", since, "--no-pager", "-q", "-g", "oom-kill"])
    st["journal_since"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    for line in out.splitlines():
        if "oom-kill" in line:
            ev.append((f"w-oom-{abs(hash(line)) % 10**8}", f"W oom-kill: {line[-160:]}"))
    return ev


S_PROBE = r"""
cd %(repo)s || exit 3
rb=research/coordination/LOCAL_LLM_RUNBOOK.md
echo "@@NOW $(date +%%s)"
echo "@@RB_MTIME $(stat -c %%Y $rb)"
echo "@@QDEPTH $(grep -c . research/queue/pool.queue 2>/dev/null)"
echo "@@CLAIMS_MTIME $(stat -c %%Y research/queue/pool.queue.claims 2>/dev/null || echo 0)"
echo "@@BIGOLD $(awk -F'\t' 'match($2,/mem_ok\.sh [0-9]+/){n=substr($2,RSTART+10,RLENGTH-10)+0; if(n>=8 && (m==""||$1<m)) m=$1} END{print m+0}' research/queue/pool.queue)"
echo "@@BIGN $(grep -cE 'mem_ok\.sh ([89]|[1-9][0-9]) ' research/queue/pool.queue)"
echo "@@SMALLN $(grep -cvE 'mem_ok\.sh ([89]|[1-9][0-9]) ' research/queue/pool.queue)"
echo "@@NEEDS"
awk '/^## NEEDS CLAUDE/{f=1;next} f&&/^Format/{exit} f&&/^- 20/' $rb
echo "@@RULINGS"
awk -v c="$(date -d '-%(days)d days' +%%Y-%%m-%%d)" '/^## 6\. RULINGS/{f=1;next} /^## NEEDS CLAUDE/{f=0} f{ if ($0 ~ /^- 20[0-9][0-9]-/) keep=(substr($2,1,10)>=c); if (keep) print }' $rb
echo "@@PI"
cat %(pi_status)s 2>/dev/null | tr -d '\n'; echo
echo "@@TAIL"
sed -n '/^## NEEDS CLAUDE/,$p' $rb | tail -n 800
""" % {"pi_status": PI_STATUS, "repo": S_REPO, "days": RULE_LOOKBACK_D}


def s_checks(now, local=False):
    ev = []
    rc, out = sh(["bash", "-c", S_PROBE] if local else S_EXEC + [S_PROBE], timeout=90)
    if rc != 0 or "@@NOW" not in out:
        return [("s-unreachable", f"S probe failed rc={rc}: {out[-200:]}")]
    sec, cur = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            parts = line[2:].split(" ", 1)
            cur = parts[0]
            sec[cur] = parts[1] if len(parts) > 1 else []
            if len(parts) == 1:
                sec[cur] = []
        elif isinstance(sec.get(cur), list):
            sec[cur].append(line)
    s_now = int(sec["NOW"])
    rb_age_h = (s_now - int(sec["RB_MTIME"])) / 3600
    if rb_age_h >= DRIVER_SILENT_H:
        ev.append(("s-driver-silent", f"S driver silent: runbook not written for {rb_age_h:.1f} h"))
    depth = int(sec.get("QDEPTH") or 0)
    claim_age_min = (s_now - int(sec.get("CLAIMS_MTIME") or 0)) / 60
    if depth == 0:
        ev.append(("s-pool-empty", "S pool.queue is empty: pool compute idle, stock lines"))
    elif claim_age_min >= POOL_STALL_MIN:
        ev.append(("s-pool-stall", f"S pool has {depth} queued lines but no claim for {claim_age_min:.0f} min"))
    big_old = int(sec.get("BIGOLD") or 0)
    if big_old and int(sec.get("SMALLN") or 0) > 0 and claim_age_min < POOL_STALL_MIN and (s_now - big_old) / 60 >= BIG_STARVE_MIN:
        ev.append(("s-pool-starve", f"S pool: {sec.get('BIGN')} big-memory lines (>= 8 GB) queued, the oldest for "
                   f"{(s_now - big_old) / 60:.0f} min, while small lines keep being claimed (backfill starvation): hold the small lines"))
    for line in sec.get("NEEDS", []):
        ev.append((f"s-needs-{abs(hash(line)) % 10**8}", f"S NEEDS CLAUDE: {line[:260]}"))
    ev += pi_checks(" ".join(sec.get("PI") or []), s_now)
    # ruled-task staleness: a [key] the ruling asks the driver to post, mentioned by no later non-RULING line
    tail = "\n".join(sec.get("TAIL", []))
    tail_l = tail.lower()
    block, blocks = None, []
    for line in sec.get("RULINGS", []):
        m = re.match(r"^- (20\d\d-\d\d-\d\d)(?: (\d\d:\d\d))? RULING (\S+)", line)
        if m:
            block = {"date": m.group(1), "name": m.group(3), "text": line}
            blocks.append(block)
        elif block is not None:
            block["text"] += "\n" + line
    for n, b in enumerate(blocks):
        later_l = "\n".join(x["text"] for x in blocks[n + 1:]).lower()  # a later RULING that already handled it
        age_h = (now - time.mktime(time.strptime(b["date"], "%Y-%m-%d"))) / 3600
        if age_h < RULE_STALE_H:
            continue
        if _mentioned(tail_l, b["name"]):
            continue  # the driver cites the ruling by name: the task has started
        for key in sorted(set(re.findall(r"[Pp]ost (?:NEEDS )?\[([a-z0-9][a-z0-9-]*)\]", b["text"]))):
            if not _mentioned(tail_l, key) and f"[{key}]" not in later_l:
                ev.append((f"s-rule-{b['name']}-{key}",
                           f"S ruled task not started: RULING {b['name']} ({b['date']}) asks for [{key}]; no driver line mentions it"))
    # a driver STOP written only in the WORKLOG (not on the NEEDS list) that no RULING has answered (2026-10-28:
    # [bg-tonic-sync] STOP sat six hours unseen because the NEEDS list was empty)
    rulings_l = "\n".join(sec.get("RULINGS", [])).lower()
    cutoff = time.strftime("%Y-%m-%d", time.localtime(now - STOP_LOOKBACK_D * 86400))
    for m in re.finditer(r"^- (20\d\d-\d\d-\d\d)(?: \d\d:\d\d)? \[([a-z0-9][a-z0-9-]*)\] STOP\b(.{0,160})", tail, re.M):
        if m.group(1) >= cutoff and f"[{m.group(2)}]" not in rulings_l:
            ev.append((f"s-stop-{m.group(2)}", f"S driver STOP [{m.group(2)}] ({m.group(1)}) has no RULING: {m.group(3).strip()}"))
    return ev


def pi_checks(raw, s_now):
    """The Pi sim driver (tools/pi_driver/README.md): silent when it is not deployed (no status file)."""
    try:
        st = json.loads(raw)
    except ValueError:
        return []
    ev = []
    updated = time.mktime(time.strptime(st["updated"][:19], "%Y-%m-%dT%H:%M:%S"))
    if s_now - updated >= PI_SILENT_MIN * 60:
        ev.append(("s-pi-silent", f"S Pi driver: status.json not updated for {(s_now - updated) / 60:.0f} min (supervisor down?)"))
    for need in st.get("needs", []):
        ev.append((f"s-pi-need-{abs(hash(need)) % 10**8}", f"S Pi driver NEEDS: {need[:260]}"))
    if st.get("restarts", 0) >= PI_RESTARTS:
        ev.append((f"s-pi-restarts-{st['restarts'] // PI_RESTARTS}", f"S Pi driver restarted {st['restarts']} times: {st.get('status', '')[:160]}"))
    return ev


def _mentioned(tail_l, key):
    """A key counts as mentioned in its bracket form or as words (driver prose writes [bg-t3] as "BG T3")."""
    key = key.lower()
    return f"[{key}]" in tail_l or key in tail_l or key.replace("-", " ") in tail_l


def run_once(st, now, record, host="auto"):
    if host == "auto":
        host = "s" if os.path.realpath(REPO) == os.path.realpath(S_REPO) else "both"
    events = []
    if host in ("both", "w"):
        events += w_checks(st, now)
    if host in ("both", "s"):
        events += s_checks(now, local=(host == "s"))
    fresh = []
    for key, msg in events:
        last = st["seen"].get(key)
        if last is None or (now - last) / 3600 >= REPEAT_H:
            fresh.append((key, msg))
            if record:
                st["seen"][key] = now
    return fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--ack", default=None)
    ap.add_argument("--host", choices=("auto", "both", "w", "s"), default="auto",
                    help="auto: on the server checkout run the server checks locally; elsewhere both hosts (server via ssh)")
    a = ap.parse_args()
    st = load_state()
    if a.ack:
        st["seen"][a.ack] = time.time()
        save_state(st)
        print("acked", a.ack)
        return 0
    if a.once:
        for key, msg in run_once(st, time.time(), record=False, host=a.host):
            print(f"[{key}] {msg}")
        return 0
    while True:
        fresh = run_once(st, time.time(), record=True, host=a.host)
        save_state(st)
        if fresh:
            for key, msg in fresh:
                print(f"ATTENTION [{key}] {msg}", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
