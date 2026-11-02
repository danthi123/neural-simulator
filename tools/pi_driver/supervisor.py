#!/usr/bin/env python3
"""supervisor.py -- keeps the Pi sim driver running, fed, steerable and visible (design section 5).

    PI_DRIVER_CONFIG=/var/lib/pi-driver/config.json python3 tools/pi_driver/supervisor.py

Runs `pi --mode rpc` in the workspace with the sim-driver extension and the router model (sim/auto), and:
  * feeds the next lane item (cfg lane_file, `- [ ]` lines) whenever Pi is idle, nothing is parked and no wait is open;
  * turns files dropped in <state_dir>/inbox/ (tools/pi_driver/pi-steer) into a steer (while busy) or a prompt (idle);
  * re-prompts a PARKED run (the router found no model: W down) every park_retry_min;
  * aborts and re-kicks a run with no event for stall_min;
  * restarts Pi with --continue when it exits, with backoff, and tells it it was restarted;
  * writes <status_dir>/status.json: its own state merged with the extension's agent_status.json.
It never touches git or the queues itself; everything the driver does goes through Pi's tools and their guard.
"""
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.environ.get("PI_DRIVER_CONFIG") or sys.exit("PI_DRIVER_CONFIG is not set")
CFG = json.load(open(CFG_PATH))
STATE = CFG["state_dir"]
INBOX = os.path.join(STATE, "inbox")
LOG = os.path.join(STATE, "supervisor.log")
PI_BIN = os.environ.get("PI_BIN", "pi")
EXT = os.path.join(HERE, "extensions", "sim-driver", "index.ts")
LANE_PROMPT = open(os.path.join(HERE, "LANE_PROMPT.md")).read()
FEED_GAP_S = 120          # never re-feed the lane faster than this (a run that settles at once must not spin)
MAX_FEEDS = 3             # an item fed this often without its lane line changing is stuck: skip it, report it


def log(msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def open_lane_items():
    try:
        lines = open(os.path.join(CFG["workspace"], CFG["lane_file"])).read().splitlines()
    except OSError:
        return []
    return [l.strip()[6:] for l in lines if l.strip().startswith("- [ ] ")]


class Pi:
    """One `pi --mode rpc` child: JSONL in, events out (split on LF only, as docs/rpc.md requires)."""

    def __init__(self):
        # Pi's agent dir (models.json, auth) lives in state_dir/agent, never the account's own ~/.pi
        env = dict(os.environ, PI_DRIVER_CONFIG=CFG_PATH, PI_CODING_AGENT_DIR=os.path.join(STATE, "agent"))
        cmd = [PI_BIN, "--mode", "rpc", "--session-dir", os.path.join(STATE, "sessions"), "--continue",
               "--extension", EXT, "--model", "sim/auto"]
        self.proc = subprocess.Popen(cmd, cwd=CFG["workspace"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(os.path.join(STATE, "pi.stderr.log"), "ab"), start_new_session=True)
        self.busy = False
        self.last_event = time.time()
        self.last_settle = 0.0
        self.last_error = ""
        self.unsettled_steers = []   # owner steers whose run has not settled yet: re-delivered if Pi dies first
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                rec = json.loads(raw.rstrip(b"\r\n"))
            except ValueError:
                continue
            t = rec.get("type")
            with self.lock:
                self.last_event = time.time()
                if t == "agent_start":
                    self.busy = True
                    self.last_error = ""
                elif t == "agent_settled":
                    self.busy = False
                    self.last_settle = time.time()
                    self.unsettled_steers = []
                elif t == "message_end":
                    m = rec.get("message") or {}
                    if m.get("role") == "assistant" and m.get("stopReason") == "error":
                        self.last_error = m.get("errorMessage") or "error"
                elif t == "response" and not rec.get("success", True):
                    log("rpc %s failed: %s" % (rec.get("command"), rec.get("error")))

    def send(self, rec):
        self.proc.stdin.write((json.dumps(rec) + "\n").encode())
        self.proc.stdin.flush()

    def prompt(self, text, steer=False):
        rec = {"type": "prompt", "message": text}
        if steer:
            rec["streamingBehavior"] = "steer"
        self.send(rec)
        with self.lock:
            self.busy = True        # until agent_settled says otherwise (no double-feed in the gap)
            self.last_event = time.time()

    def alive(self):
        return self.proc.poll() is None

    def reap(self):
        """Kill what a dead Pi left running. Its tool commands are session leaders of their own (they outlive a
        kill -9 of Pi), so the supervisor is a child subreaper (main()) and they re-parent to it: kill each one's group."""
        for pid in children_of(os.getpid()):
            if pid != self.proc.pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass


def collect_zombies(keep):
    """As a subreaper this process inherits orphans; wait on the ones that exited (never on Pi itself: Popen owns it)."""
    for pid in children_of(os.getpid()):
        if pid != keep:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def children_of(ppid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if int(open("/proc/%s/stat" % d).read().rsplit(")", 1)[1].split()[1]) == ppid:
                    out.append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    return out

def main():
    # PR_SET_CHILD_SUBREAPER: orphaned tool commands of a dead Pi re-parent here, so reap() can find them
    ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
    os.makedirs(os.path.join(INBOX, "done"), exist_ok=True)
    backoff, restarts, note = 30, 0, None
    state = {"last_feed": 0.0, "status": "starting", "feeds": {}}   # survives Pi restarts (feed counts too)
    while True:
        log("starting pi (restart #%d)" % restarts)
        pi = Pi()
        started = time.time()
        if note:
            pi.prompt(note)
        while pi.alive():
            time.sleep(5)
            collect_zombies(pi.proc.pid)
            agent = read_json(os.path.join(STATE, "agent_status.json"), {})
            items = open_lane_items()
            state["status"] = tick(pi, agent, items, state)
            write_status(state["status"], restarts, agent, items, state.get("stuck", []))
            if time.time() - started > 600:
                backoff = 30
        pi.reap()
        write_status("restarting", restarts, read_json(os.path.join(STATE, "agent_status.json"), {}), open_lane_items(),
                     state.get("stuck", []))
        log("pi exited (code %s); restarting in %ds" % (pi.proc.returncode, backoff))
        time.sleep(backoff)
        backoff = min(backoff * 2, 600)
        restarts += 1
        note = ("You were restarted after the driver process exited (code %s). Your session was resumed with --continue: "
                "an interrupted tool call may not have finished. Check git status and the last command's effect, then "
                "continue." % pi.proc.returncode)
        if pi.unsettled_steers:
            note += "\n\nThe owner's steering sent just before the restart (apply it now):\n" + "\n".join(
                "- [owner steer] " + m for m in pi.unsettled_steers)


def tick(pi, agent, items, state):
    """One supervision step; returns the status line for status.json."""
    with pi.lock:
        busy, quiet, err, settled = pi.busy, time.time() - pi.last_event, pi.last_error, pi.last_settle
    # 1. the owner's steering always goes first
    for name in sorted(os.listdir(INBOX)):
        p = os.path.join(INBOX, name)
        if not os.path.isfile(p):
            continue
        msg = open(p).read().strip()
        shutil.move(p, os.path.join(INBOX, "done", "%d-%s" % (time.time(), name)))
        if msg:
            log("steer (%s): %s" % ("busy" if busy else "idle", msg[:200]))
            pi.prompt("[owner steer] " + msg, steer=busy)
            with pi.lock:
                pi.unsettled_steers.append(msg)
            busy = True
    if busy:
        if quiet > CFG.get("stall_min", 30) * 60:
            log("stalled %d min: abort + re-kick" % (quiet // 60))
            pi.send({"type": "abort"})
            time.sleep(10)
            pi.prompt("You produced no events for %d minutes and were aborted by the supervisor. Check what the "
                      "last command left behind (git status, running jobs), then continue the lane item." % (quiet // 60))
        return "working"
    if err:
        # PARKED (the router found no model) or a provider error ended the run: retry it after park_retry_min
        if time.time() - settled > CFG.get("park_retry_min", 10) * 60:
            log("retrying after: %s" % err[:200])
            pi.prompt("continue (the last run ended with: %s)" % err[:300])
        return ("parked: " + err[err.find("PARKED") + 8:][:200]) if "PARKED" in err else ("error: " + err[:200])
    if agent.get("waiters"):
        return "waiting: " + ", ".join(w["label"] for w in agent["waiters"])[:300]
    feeds = state.setdefault("feeds", {})
    todo = [i for i in items if feeds.get(i, 0) < MAX_FEEDS]
    state["stuck"] = [i for i in items if feeds.get(i, 0) >= MAX_FEEDS]
    if not todo:
        return "idle: lane empty" if not items else "stuck: every open lane item was fed %d times without progress" % MAX_FEEDS
    if time.time() - state["last_feed"] > FEED_GAP_S:
        state["last_feed"] = time.time()
        feeds[todo[0]] = feeds.get(todo[0], 0) + 1   # keyed by the line's text: editing the line resets its count
        log("feeding lane item (attempt %d): %s" % (feeds[todo[0]], todo[0][:200]))
        pi.prompt(LANE_PROMPT.replace("{lane_file}", CFG["lane_file"]).replace("{item}", todo[0]))
        return "working"
    return state["status"]


def write_status(status, restarts, agent, items, stuck):
    # needs: what only the owner or Claude can unblock (read by tools/attention_watch.py)
    needs = ["stuck lane item: " + i[:200] for i in stuck]
    needs += ["open checkpoint (Claude cap reached or failed): " + c for c in agent.get("open_checkpoints", [])]
    write_json(os.path.join(CFG["status_dir"], "status.json"), {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": status, "restarts": restarts,
        "lane_open": items[:10], "needs": needs, "agent": agent})


if __name__ == "__main__":
    main()
