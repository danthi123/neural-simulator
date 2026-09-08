#!/usr/bin/env python3
"""openhands_loop.py — continuous-session driver for OpenHands, run by tools/openhands_takeover.sh.

PRODUCTIONIZED 2026-09-08 (was a prototype-only design artifact as of 2026-09-06 — see
docs/2026-09-06-openhands-harness-prototype.md for the evaluation that validated this live). Started/
stopped by `tools/openhands_takeover.sh {on,off}`, which also sets the shared driver sentinel this
loop reads (`research/queue/OPENHANDS_ACTIVE`) and starts `tools/qwen_supervisor.sh __daemon` — see
below for why VRAM load/unload itself is NOT done here.

⛔ SINGLE ACTIVE DRIVER. Do not run this at the same time as `hermes-loop` (or any other local
driver) — `tools/openhands_takeover.sh on` and `tools/hermes_takeover.sh on` must never both be
active; hand back the current driver first (`... off`) before switching.

VRAM LIFECYCLE IS OWNED BY tools/qwen_supervisor.sh, NOT THIS LOOP (2026-09-08 change from the
2026-09-06 prototype, which called `tools/qwen_serve.sh up/down` itself — that duplicated the exact
unload/reload decision `qwen_supervisor.sh` already makes for Hermes, which is the double-load race
its own header warns about if two independent deciders ever ran together). By default this loop only
READS state (`qwen_up()`, `gpu_busy()`) and never calls `qwen_serve.sh` itself — it waits for the
shared supervisor daemon to bring Qwen up/down, exactly mirroring how a Hermes turn never calls
`qwen_serve.sh` directly either. Set `OPENHANDS_LOOP_MANAGE_QWEN=1` to restore the old
self-contained 2026-09-06 behavior (calls `qwen_serve.sh up/down` itself) for standalone testing
without the supervisor running — do not set it while `qwen_supervisor.sh __daemon` is also active.

DESIGN — the read-only GPU-queue introspection (gpu_busy/_running_job/dispatcher_alive) is ported
from tools/hermes/loop.py's proven functions, because that logic is already hardened against real
incidents (stale gpu.running, dispatcher death mid-job, a hung nvidia-smi during a GPU-crash — see
gpu_queue.sh's and loop.py's own comments) and there is no reason to re-derive it. The structural
difference from Hermes' loop remains: instead of firing a brand-new webui/gateway session per turn
(loop.py's fire_turn()/poll_run()), each iteration calls `conversation.send_message() +
conversation.run()` on the ONE persisted OpenHands Conversation (agent_config.build_conversation(),
fixed conversation_id) — so the session survives every offload cycle instead of restarting cold.
That persistence is the entire point of this harness (see agent_config.py's docstring and the
write-up doc).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import agent_config as cfg

REPO = cfg.DEFAULT_WORKSPACE
MANAGE_QWEN = os.environ.get("OPENHANDS_LOOP_MANAGE_QWEN", "0") == "1"


def _shared_queue_root(repo):
    """Resolve the SAME shared root tools/gpu_queue.sh uses (its own header: "SINGLETON across
    worktrees ... resolved at the git-common-dir root, not per-checkout"). `git rev-parse
    --show-toplevel` from inside a worktree returns the WORKTREE's own root, whose tracked
    research/queue/* is a git blob snapshot, NOT the live runtime queue the shared dispatcher reads
    and writes — reading gpu.queue/gpu.running/gpu_queue.dpid via the worktree path silently reads
    stale committed content instead of ground truth. Mirrors gpu_queue.sh's own
    `git rev-parse --path-format=absolute --git-common-dir` + dirname."""
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            common = out.stdout.strip()
            parent = os.path.dirname(common)
            if os.path.isdir(parent):
                return parent
    except Exception:
        pass
    return repo


SHARED_ROOT = _shared_queue_root(REPO)
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
os.makedirs(STATE, exist_ok=True)
GPU_QUEUE_STATE = os.path.join(SHARED_ROOT, "research", "queue")  # shared ground truth
# The driver sentinel is the SAME file tools/qwen_supervisor.sh and tools/openhands_takeover.sh use
# (research/queue/OPENHANDS_ACTIVE) — a single source of truth alongside Hermes' HERMES_ACTIVE, not a
# private tools/openhands_proto/state/ file (that was the 2026-09-06 prototype's own sentinel, kept
# private on purpose while unvetted; now that openhands_takeover.sh drives this loop, sharing the
# sentinel with the supervisor is what lets `openhands_takeover.sh off` stop BOTH cleanly).
ACTIVE_SENTINEL = os.path.join(GPU_QUEUE_STATE, "OPENHANDS_ACTIVE")
LOG = os.path.join(STATE, "openhands_loop.log")
GPU_QUEUE = os.path.join(SHARED_ROOT, "tools", "gpu_queue.sh")
# NOTE: unlike gpu_queue.sh, tools/qwen_serve.sh has NO singleton-across-worktrees resolution of its
# own (it derives its PID-file/log directory from ITS OWN script path's parent, not a git-common-dir
# lookup) — invoking a worktree's copy would track a DIFFERENT qwen_server.pid than the canonical
# checkout's Hermes loop uses, a real double-launch risk. So always invoke the SHARED_ROOT's copy,
# even if this loop script itself is run from a worktree (see the write-up doc's "gap found" note).
SERVE = os.path.join(SHARED_ROOT, "tools", "qwen_serve.sh")

IDLE_SLEEP = int(os.environ.get("OPENHANDS_LOOP_IDLE_SLEEP", "8"))
QWEN_UP_TIMEOUT = int(os.environ.get("OPENHANDS_LOOP_QWEN_UP_TIMEOUT", "1800"))


def log(msg):
    line = "%s %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def loop_active():
    return os.path.exists(ACTIVE_SENTINEL)


# --- GPU-queue introspection, ported verbatim (read-only) from tools/hermes/loop.py --------------
def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _queue_depth():
    q = os.path.join(GPU_QUEUE_STATE, "gpu.queue")
    try:
        return sum(1 for ln in open(q) if ln.strip())
    except Exception:
        return 0


def _running_job():
    r = os.path.join(GPU_QUEUE_STATE, "gpu.running")
    try:
        line = open(r).read().strip()
    except Exception:
        return None
    if not line:
        return None
    pid = line.split("\t")[0]
    return line if pid and _pid_alive(pid) else None


def gpu_busy():
    return _queue_depth() > 0 or _running_job() is not None


def dispatcher_alive():
    try:
        pid = open(os.path.join(GPU_QUEUE_STATE, "gpu_queue.dpid")).read().strip()
        return bool(pid) and _pid_alive(pid)
    except Exception:
        return False


# --- qwen lifecycle, identical shape to tools/hermes/loop.py (down/up wrap tools/qwen_serve.sh) --
def qwen_down_cmd():
    try:
        subprocess.run(["bash", SERVE, "down"], cwd=REPO, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log("qwen down FAILED (non-fatal): %s" % e)


def qwen_up_cmd():
    try:
        r = subprocess.run(["bash", SERVE, "up"], cwd=REPO, timeout=QWEN_UP_TIMEOUT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = (r.stdout or b"")[-300:].decode("utf-8", "replace").replace("\n", " ")
            log("qwen up rc=%d: %s" % (r.returncode, tail))
    except Exception as e:
        log("qwen up FAILED (non-fatal): %s" % e)


def vram_handoff():
    """A GPU job is queued/running. Default (MANAGE_QWEN=False): tools/qwen_supervisor.sh already owns
    the unload/reload decision (it polls the same gpu.queue/gpu.running this loop reads) — just wait
    for the queue to drain. MANAGE_QWEN=True (opt-in standalone mode, no supervisor running): unload
    qwen ourselves, the 2026-09-06 prototype's original self-contained behavior. Either way qwen is
    NEVER reloaded BY THIS LOOP while a GPU job still holds the card — identical invariant to
    tools/hermes/loop.py and to qwen_supervisor.sh itself."""
    log("GPU job present -> %s" % ("unloading qwen for the run (MANAGE_QWEN=1)" if MANAGE_QWEN else
                                    "waiting for it to drain (tools/qwen_supervisor.sh owns qwen up/down)"))
    if MANAGE_QWEN:
        qwen_down_cmd()
    t0 = last_hb = time.time()
    while gpu_busy():
        if not loop_active():
            return "aborted"
        now = time.time()
        if now - last_hb > 900:
            log("GPU job still running (%dm) -> qwen stays DOWN" % ((now - t0) // 60))
            last_hb = now
        time.sleep(IDLE_SLEEP)
    if MANAGE_QWEN:
        time.sleep(3)
        log("GPU queue drained -> reloading qwen")
        qwen_up_cmd()
    return "gpu_ok"


# --- one turn on the ONE persisted conversation ---------------------------------------------------
def run_one_turn(conversation):
    def _cb(event):
        pass  # run_turn.py prints a transcript when run standalone; the loop stays quiet in its log
    conversation.send_message(cfg.TURN_PROMPT)
    conversation.run()
    return getattr(conversation.state, "execution_status", "?")


def main():
    mode = "self-managed VRAM (OPENHANDS_LOOP_MANAGE_QWEN=1)" if MANAGE_QWEN else \
        "supervisor-managed VRAM (default — tools/qwen_supervisor.sh owns qwen up/down)"
    log("openhands_loop up [%s] (sentinel: %s)" % (mode, ACTIVE_SENTINEL))
    conversation = None
    while True:
        try:
            if not loop_active():
                time.sleep(IDLE_SLEEP)
                continue
            if gpu_busy():
                vram_handoff()
                continue
            if not cfg.qwen_up():
                if not MANAGE_QWEN:
                    # tools/qwen_supervisor.sh's daemon owns bringing qwen up; just wait for it.
                    time.sleep(IDLE_SLEEP)
                    continue
                log("qwen down + no GPU job -> loading qwen (MANAGE_QWEN=1)")
                qwen_up_cmd()
                if not cfg.qwen_up():
                    log("qwen failed to come up -> backing off 60s")
                    time.sleep(60)
                    continue
            if gpu_busy():
                continue
            if conversation is None:
                conversation = cfg.build_conversation()
                log("conversation ready: id=%s persistence_dir=%s" % (cfg.CONVERSATION_ID, cfg.PERSISTENCE_DIR))
            log("firing turn on the persisted conversation")
            status = run_one_turn(conversation)
            log("turn done: status=%s" % status)
        except Exception as e:
            log("loop iteration error (non-fatal): %r" % e)
            time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
