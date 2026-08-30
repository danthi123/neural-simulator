#!/usr/bin/env python3
"""Hermes visible autonomous research loop — v2 orchestrator (single owner of the loop).

Design: docs/plans/2026-08-30-hermes-visible-loop-v2-design.md. Replaces the v1 supervisor firing.

Each iteration = ONE fresh, short webui-visible session re-anchored from durable state (live_state.md),
so nothing ever accumulates enough to balloon the context or leave a stale-stream jam. Turns are
tracked by the gateway runs API (run_id -> poll to terminal) — the reliable "is the turn done?" signal.
VRAM is handed off cleanly: after a turn, if Hermes launched a GPU research job, Qwen is unloaded so
the job gets the card, then reloaded before the next turn.

Runs as the `hermes-loop` systemd service. Active only while HERMES_ACTIVE and not GAME_MODE.

Auth: gateway runs API (127.0.0.1:8642, Bearer API_SERVER_KEY from ~/.hermes/.env). Session titling
is best-effort via the webui API (127.0.0.1:8787, password from the webui unit) — purely cosmetic.

Robustness: a failed/errored turn is logged and the loop simply moves to the NEXT fresh session — a
single turn can never permanently stall the loop (the v1 failure mode).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(REPO, "research", "queue")
ENVFILE = os.path.expanduser("~/.hermes/.env")
WEBUI_UNIT = os.path.expanduser("~/.config/systemd/user/hermes-webui.service")
LOG = os.path.join(STATE, "hermes_loop.log")
TURNFILE = os.path.join(STATE, ".hermes_loop_turn")
SERVE = os.path.join(REPO, "tools", "qwen_serve.sh")

GATEWAY = "http://127.0.0.1:8642"
QWEN = "http://127.0.0.1:8033"
WEBUI = "http://127.0.0.1:8787"
REASONING = os.environ.get("HERMES_LOOP_REASONING", "medium")
POLL = int(os.environ.get("HERMES_LOOP_POLL", "10"))          # run-status poll interval (s)
TURN_MAX = int(os.environ.get("HERMES_LOOP_TURN_MAX", "1800"))  # kill a turn stuck > this (s)
IDLE_SLEEP = int(os.environ.get("HERMES_LOOP_IDLE_SLEEP", "8"))

TURN_PROMPT = (
    "You are Hermes, autonomously driving the neural-simulator research project (Claude usage is out). "
    "This is one turn of a continuous loop. Work IN /home/dant123/Projects/sim; use ABSOLUTE paths for "
    "file reads (the read_file tool's cwd may differ) or the terminal tool. STEPS: (1) read "
    "/home/dant123/Projects/sim/research/coordination/live_state.md and the CURRENT STATE atop "
    "/home/dant123/Projects/sim/GAP_CLOSURE_MISSION.md; (2) harvest any just-completed run(s); (3) take "
    "the next concrete action from the ordered NEXT ACTIONS (edit files, commit via "
    "tools/push_both.sh — NEVER --no-verify); (4) if a GPU experiment is the next step, launch exactly "
    "ONE via `bash tools/hermes_gpu_run.sh '<cmd>'` (do NOT run GPU python directly — it fights the "
    "model for VRAM). Then END your turn; the loop re-invokes you when the run completes. Obey CLAUDE.md "
    "(brain-based-only, one-brain, no-defer, 6-seed, gates authoritative). Keep this turn focused — one "
    "concrete step, then stop."
)


def log(msg):
    line = "%s %s" % (time.strftime("%F %T"), msg)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _env_val(path, name):
    try:
        with open(path) as f:
            for ln in f:
                m = re.match(r"\s*(?:Environment=)?%s=(.*)$" % re.escape(name), ln.strip())
                if m:
                    return m.group(1).strip().strip('"')
    except Exception:
        pass
    return ""


API_KEY = _env_val(ENVFILE, "API_SERVER_KEY")
WEBUI_PW = _env_val(WEBUI_UNIT, "HERMES_WEBUI_PASSWORD")


def _req(url, data=None, headers=None, method=None, timeout=20):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=h, method=method or ("POST" if data is not None else "GET"))
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        raw = r.read()
        return r.getcode(), (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"_err": str(e)}


# --- state -----------------------------------------------------------------------------------------
def hermes_active():
    return os.path.exists(os.path.join(STATE, "HERMES_ACTIVE"))


def game_mode():
    return os.path.exists(os.path.join(STATE, "GAME_MODE"))


def gpu_busy():
    q = os.path.join(STATE, "gpu.queue")
    return (os.path.exists(q) and os.path.getsize(q) > 0) or os.path.exists(os.path.join(STATE, "gpu.running"))


def qwen_up():
    code, _ = _req(QWEN + "/health", timeout=4)
    return code == 200


def qwen_down_cmd():
    subprocess.run(["bash", SERVE, "down"], cwd=REPO, timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def qwen_up_cmd():
    subprocess.run(["bash", SERVE, "up"], cwd=REPO, timeout=600,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --- turn lifecycle --------------------------------------------------------------------------------
def _next_turn_no():
    try:
        n = int(open(TURNFILE).read().strip()) + 1
    except Exception:
        n = 1
    try:
        open(TURNFILE, "w").write(str(n))
    except Exception:
        pass
    return n


def _webui_cookie():
    if not WEBUI_PW:
        return None
    code, _ = 0, None
    try:
        req = urllib.request.Request(WEBUI + "/api/auth/login",
                                     data=json.dumps({"password": WEBUI_PW}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        r = urllib.request.urlopen(req, timeout=8)
        c = r.headers.get("Set-Cookie", "")
        return c.split(";")[0] if c else None
    except Exception:
        return None


def _title_session(sid, n):
    """Best-effort: give the per-turn session a distinguishable title in the webui sidebar."""
    ck = _webui_cookie()
    if not ck:
        return
    title = "\U0001F916 loop · turn %d · %s" % (n, time.strftime("%H:%M"))
    _req(WEBUI + "/api/session/rename", data={"session_id": sid, "title": title},
         headers={"Cookie": ck}, timeout=8)


def fire_turn():
    """Create a fresh session + fire one turn via the runs API. Returns (run_id, session_id) or (None,None)."""
    n = _next_turn_no()
    sid = "loop-%d-%d" % (int(time.time()), n)
    body = {"input": TURN_PROMPT, "session_id": sid, "model_options": {"reasoning_effort": REASONING}}
    code, out = _req(GATEWAY + "/v1/runs", data={**body},
                     headers={"Authorization": "Bearer %s" % API_KEY}, timeout=25)
    if code not in (200, 201, 202) or not out.get("run_id"):   # runs API returns 202 Accepted on success
        log("fire FAILED (HTTP %s: %s)" % (code, str(out)[:160]))
        return None, None
    rid = out["run_id"]
    log("turn %d fired: run_id=%s session=%s reasoning=%s" % (n, rid, sid, REASONING))
    _title_session(sid, n)
    return rid, sid


def poll_run(rid):
    """Poll a run to terminal status. Returns the final status string."""
    t0 = time.time()
    while time.time() - t0 < TURN_MAX:
        code, out = _req(GATEWAY + "/v1/runs/%s" % rid,
                         headers={"Authorization": "Bearer %s" % API_KEY}, timeout=8)
        st = out.get("status")   # extract status whenever present, regardless of 200/202
        if st in ("completed", "failed", "cancelled", "error"):
            return st
        # if a GPU job appeared, the turn has launched its run near its end — stop waiting on the turn
        if gpu_busy():
            return "launched_gpu"
        if not hermes_active() or game_mode():
            return "aborted"
        time.sleep(POLL)
    log("turn %s exceeded TURN_MAX=%ss — stopping it" % (rid, TURN_MAX))
    _req(GATEWAY + "/v1/runs/%s/stop" % rid, data={},
         headers={"Authorization": "Bearer %s" % API_KEY}, timeout=8)
    return "timeout"


def vram_handoff():
    """A GPU research job is queued/running -> give it the card: unload Qwen, wait for the queue to
    drain, reload Qwen. Bounded so a hung job can't wedge the loop forever."""
    log("GPU job present -> unloading Qwen for the run")
    qwen_down_cmd()
    t0 = time.time()
    while gpu_busy():
        if not hermes_active() or game_mode():
            return
        if time.time() - t0 > 6 * 3600:  # 6h safety: don't wait forever on a stuck job
            log("GPU job still running after 6h — reloading Qwen anyway")
            break
        time.sleep(IDLE_SLEEP)
    log("GPU queue drained -> reloading Qwen")
    qwen_up_cmd()


def main():
    log("hermes-loop v2 up (reasoning=%s, per-turn sessions, run-status tracked)" % REASONING)
    consec_fail = 0
    while True:
        if not hermes_active() or game_mode():
            if qwen_up():
                log("HERMES_ACTIVE off / GAME_MODE -> unloading Qwen, loop idle")
                qwen_down_cmd()
            time.sleep(IDLE_SLEEP)
            continue
        if gpu_busy():                 # a research run is queued/running -> hand off the GPU
            vram_handoff()
            continue
        if not qwen_up():              # need the brain to fire a turn
            log("Qwen down + no GPU job -> loading Qwen")
            qwen_up_cmd()
            if not qwen_up():
                time.sleep(IDLE_SLEEP)
                continue
        rid, sid = fire_turn()
        if not rid:
            consec_fail += 1
            time.sleep(min(30 * consec_fail, 300))
            continue
        status = poll_run(rid)
        log("turn done: run=%s status=%s" % (rid, status))
        # Backoff on a genuinely failed/errored/timed-out turn so a persistent failure mode (e.g. a
        # bad model_options value 500ing every turn, as reasoning=high did) can never rapid-fire a
        # cascade of dead turns. Escalates 30s->5min; resets the moment a turn is productive.
        if status in ("failed", "error", "timeout"):
            consec_fail += 1
            back = min(30 * consec_fail, 300)
            log("turn %s (consec_fail=%d) -> backing off %ss before next" % (status, consec_fail, back))
            time.sleep(back)
        else:
            consec_fail = 0
        # if status==launched_gpu, next iter's gpu_busy() triggers the handoff; otherwise
        # (completed) we just fire the next fresh turn immediately.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
