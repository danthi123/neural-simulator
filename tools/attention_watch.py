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

Checks: W GPU idle with an empty queue; a GPU job exiting non-zero; a degraded brain build in the GPU log
(missing untracked inputs); an oom-kill; S NEEDS CLAUDE lines; the S driver silent; the S pool idle or not claiming;
a recent RULING whose requested [key] report no later driver line mentions after RULE_STALE_H hours.
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
DRIVER_SILENT_H = 3.0   # S runbook not written this long
POOL_STALL_MIN = 60     # S pool has queued lines but no claim this long
RULE_STALE_H = 24.0     # a requested [key] report not mentioned by the driver this long after its RULING
RULE_LOOKBACK_D = 4


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
    if idle and not paused:
        st["gpu_idle_since"] = st.get("gpu_idle_since") or now
        mins = (now - st["gpu_idle_since"]) / 60
        if mins >= GPU_IDLE_MIN:
            ev.append(("w-gpu-idle", f"W GPU idle {mins:.0f} min with an empty gpu_queue (not paused): stock GPU work"))
    else:
        st["gpu_idle_since"] = None
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
echo "@@NEEDS"
awk '/^## NEEDS CLAUDE/{f=1;next} f&&/^Format/{exit} f&&/^- 20/' $rb
echo "@@RULINGS"
awk -v c="$(date -d '-%(days)d days' +%%Y-%%m-%%d)" '/^## 6\. RULINGS/{f=1;next} /^## NEEDS CLAUDE/{f=0} f{ if ($0 ~ /^- 20[0-9][0-9]-/) keep=(substr($2,1,10)>=c); if (keep) print }' $rb
echo "@@TAIL"
sed -n '/^## NEEDS CLAUDE/,$p' $rb | tail -n 800
""" % {"repo": S_REPO, "days": RULE_LOOKBACK_D}


def s_checks(now):
    ev = []
    rc, out = sh(S_EXEC + [S_PROBE], timeout=90)
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
    for line in sec.get("NEEDS", []):
        ev.append((f"s-needs-{abs(hash(line)) % 10**8}", f"S NEEDS CLAUDE: {line[:260]}"))
    # ruled-task staleness: a [key] the ruling asks the driver to post, mentioned by no later non-RULING line
    tail = "\n".join(sec.get("TAIL", []))
    block, blocks = None, []
    for line in sec.get("RULINGS", []):
        m = re.match(r"^- (20\d\d-\d\d-\d\d)(?: (\d\d:\d\d))? RULING (\S+)", line)
        if m:
            block = {"date": m.group(1), "name": m.group(3), "text": line}
            blocks.append(block)
        elif block is not None:
            block["text"] += "\n" + line
    for b in blocks:
        age_h = (now - time.mktime(time.strptime(b["date"], "%Y-%m-%d"))) / 3600
        if age_h < RULE_STALE_H:
            continue
        for key in sorted(set(re.findall(r"[Pp]ost (?:NEEDS )?\[([a-z0-9][a-z0-9-]*)\]", b["text"]))):
            if f"[{key}]" not in tail:
                ev.append((f"s-rule-{b['name']}-{key}",
                           f"S ruled task not started: RULING {b['name']} ({b['date']}) asks for [{key}]; no driver line mentions it"))
    return ev


def run_once(st, now, record):
    events = w_checks(st, now) + s_checks(now)
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
    a = ap.parse_args()
    st = load_state()
    if a.ack:
        st["seen"][a.ack] = time.time()
        save_state(st)
        print("acked", a.ack)
        return 0
    if a.once:
        for key, msg in run_once(st, time.time(), record=False):
            print(f"[{key}] {msg}")
        return 0
    while True:
        fresh = run_once(st, time.time(), record=True)
        save_state(st)
        if fresh:
            for key, msg in fresh:
                print(f"ATTENTION [{key}] {msg}", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
