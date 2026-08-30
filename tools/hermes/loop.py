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
QWEN_UP_TIMEOUT = int(os.environ.get("HERMES_LOOP_QWEN_UP_TIMEOUT", "1800"))  # cold load + drafter-fallback headroom
MIN_TURN_FLOOR = int(os.environ.get("HERMES_LOOP_MIN_TURN_FLOOR", "20"))      # floor a degenerate instant completion (s)
GPU_QUEUE = os.path.join(REPO, "tools", "gpu_queue.sh")

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


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _queue_depth():
    q = os.path.join(STATE, "gpu.queue")
    try:
        return sum(1 for ln in open(q) if ln.strip())
    except Exception:
        return 0


def _running_job():
    """gpu.running content IF its recorded pid is alive; else the file is STALE (the dispatcher died
    mid-job leaving it behind — gpu_queue.sh's EXIT trap removes only its dpid, never gpu.running) so we
    clear it and report no job. Without this, a stale gpu.running wedges the loop into a permanent Qwen
    flap (M1a)."""
    r = os.path.join(STATE, "gpu.running")
    try:
        line = open(r).read().strip()
    except Exception:
        return None
    if not line:
        return None
    pid = line.split("\t")[0]
    if pid and _pid_alive(pid):
        return line
    try:
        os.remove(r)
        log("cleared STALE gpu.running (pid %s dead) — dispatcher died mid-job" % pid)
    except Exception:
        pass
    return None


def gpu_busy():
    return _queue_depth() > 0 or _running_job() is not None


def dispatcher_alive():
    """Mirror gpu_queue.sh daemon_alive: the dpid file names a live process."""
    try:
        pid = open(os.path.join(STATE, "gpu_queue.dpid")).read().strip()
        return bool(pid) and _pid_alive(pid)
    except Exception:
        return False


def ensure_dispatcher():
    """A queued/running job is only real if the gpu_queue dispatcher is alive to run it. `add` only
    appends; nothing else starts the daemon (M1b). Start it if absent so unloading Qwen can't strand the
    loop on a job that never runs."""
    if dispatcher_alive():
        return True
    log("gpu_queue dispatcher not alive -> starting it")
    try:
        subprocess.run(["bash", GPU_QUEUE, "start"], cwd=REPO, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log("dispatcher start err: %s" % e)
    return dispatcher_alive()


def _gpu_log_offset():
    try:
        return sum(1 for _ in open(os.path.join(STATE, "gpu_queue.log")))
    except Exception:
        return 0


def _gpu_rcs_since(offset):
    """Return the rc of every DONE(rc=N) the dispatcher logged after `offset` lines (M4)."""
    rcs = []
    try:
        lines = open(os.path.join(STATE, "gpu_queue.log")).read().splitlines()
        for ln in lines[offset:]:
            m = re.search(r"DONE\(rc=(-?\d+)\)", ln)
            if m:
                rcs.append(int(m.group(1)))
    except Exception:
        pass
    return rcs


def qwen_up():
    code, _ = _req(QWEN + "/health", timeout=4)
    return code == 200


def qwen_down_cmd():
    # NEVER let a hung serve script (e.g. nvidia-smi wedged during a GPU crash) crash the loop (M5): a
    # TimeoutExpired/OSError here is logged and swallowed, not propagated.
    try:
        subprocess.run(["bash", SERVE, "down"], cwd=REPO, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log("qwen down FAILED (non-fatal): %s" % e)


def qwen_up_cmd():
    """Load Qwen. Bounded by QWEN_UP_TIMEOUT (cold load + drafter fallback can exceed the old 600s -> the
    loop must not TimeoutExpire-crash, M6). Capture rc + a stderr/stdout tail so a load FAILURE is visible
    in the log instead of silently spinning."""
    try:
        r = subprocess.run(["bash", SERVE, "up"], cwd=REPO, timeout=QWEN_UP_TIMEOUT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = (r.stdout or b"")[-300:].decode("utf-8", "replace").replace("\n", " ")
            log("qwen up rc=%d: %s" % (r.returncode, tail))
    except Exception as e:
        log("qwen up FAILED (non-fatal): %s" % e)


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
    """Poll a run to terminal status. Returns the final status string.

    Does NOT early-return on gpu_busy: a turn that enqueues a GPU job must still be polled to terminal
    BEFORE the handoff unloads Qwen (M3 — the old early-return cut turns mid-generation and logged them as
    success). The dispatcher peeks-don't-pops and won't start the queued job while Qwen holds VRAM, so
    waiting for the turn to end costs nothing; the handoff then runs in main's next iteration, between
    turns, exactly as the design promised."""
    t0 = time.time()
    while time.time() - t0 < TURN_MAX:
        code, out = _req(GATEWAY + "/v1/runs/%s" % rid,
                         headers={"Authorization": "Bearer %s" % API_KEY}, timeout=8)
        st = out.get("status")   # extract status whenever present, regardless of 200/202
        if st in ("completed", "failed", "cancelled", "error"):
            return st
        if not hermes_active() or game_mode():
            return "aborted"
        time.sleep(POLL)
    log("turn %s exceeded TURN_MAX=%ss — stopping it" % (rid, TURN_MAX))
    _req(GATEWAY + "/v1/runs/%s/stop" % rid, data={},
         headers={"Authorization": "Bearer %s" % API_KEY}, timeout=8)
    # verify the stop took — the single --parallel 1 slot stays held until it does (S6)
    for _ in range(6):
        time.sleep(POLL)
        _, out = _req(GATEWAY + "/v1/runs/%s" % rid,
                      headers={"Authorization": "Bearer %s" % API_KEY}, timeout=8)
        if out.get("status") in ("completed", "failed", "cancelled", "error"):
            return "timeout"
    log("turn %s did not terminate after stop -> restarting Qwen to clear the slot" % rid)
    qwen_down_cmd()   # main reloads Qwen at the top of the next iteration
    return "timeout"


def vram_handoff():
    """A GPU research job is queued/running -> give it the card: unload Qwen, wait for the queue to
    GENUINELY drain, then reload Qwen. Returns 'gpu_ok' / 'gpu_fail' / 'no_dispatcher' / 'aborted'.

    Invariant (M1/M2/M7): Qwen is NEVER (re)loaded while any GPU job still holds the card. The old 6h cap
    VIOLATED this — on expiry it reloaded Qwen ON TOP of a still-running job = two 27B brains on one 24GB
    card -> CUDA OOM / the 3090 off the bus (reboot-only). We instead keep Qwen DOWN for as long as the job
    actually runs (one brain at a time; multi-hour jobs are in scope) and rely on _running_job()'s dead-pid
    detection to unblock automatically if the job crashes."""
    # A queued/running marker is only real if a dispatcher exists to run it (M1b). If none can be started,
    # do NOT unload Qwen into a permanent wedge — leave Qwen up and re-evaluate next cycle.
    if not ensure_dispatcher():
        log("⛔ GPU job present but NO dispatcher could be started -> NOT unloading Qwen (would wedge); investigate gpu_queue")
        time.sleep(IDLE_SLEEP)
        return "no_dispatcher"
    off = _gpu_log_offset()
    log("GPU job present -> unloading Qwen for the run")
    qwen_down_cmd()
    t0 = last_hb = time.time()
    while gpu_busy():
        if not hermes_active() or game_mode():
            return "aborted"   # leave Qwen down; the top of main handles game/inactive
        now = time.time()
        if now - last_hb > 900:   # visible heartbeat so a legit multi-hour job isn't silent
            log("GPU job still running (%dm) -> Qwen stays DOWN (one brain at a time)" % ((now - t0) // 60))
            last_hb = now
        time.sleep(IDLE_SLEEP)
    time.sleep(3)   # let the dispatcher write its DONE(rc=) line after it removes gpu.running
    rcs = _gpu_rcs_since(off)
    if any(rc != 0 for rc in rcs):
        log("⚠ GPU job(s) FAILED rc=%s -> reloading Qwen (loop will back off)" % rcs)
        status = "gpu_fail"
    else:
        log("GPU queue drained (rc=%s) -> reloading Qwen" % (rcs if rcs else "n/a"))
        status = "gpu_ok"
    qwen_up_cmd()
    return status


def main():
    log("hermes-loop v2 up (reasoning=%s, per-turn sessions, run-status tracked)" % REASONING)
    consec_fail = 0
    while True:
        try:
            if not hermes_active() or game_mode():
                if qwen_up():
                    log("HERMES_ACTIVE off / GAME_MODE -> unloading Qwen, loop idle")
                    qwen_down_cmd()
                time.sleep(IDLE_SLEEP)
                continue
            if gpu_busy():                 # a research run is queued/running -> hand off the GPU
                st = vram_handoff()
                if st == "gpu_fail":       # the job crashed -> back off so a broken command can't fast-cycle
                    consec_fail += 1
                    time.sleep(min(30 * consec_fail, 300))
                elif st == "gpu_ok":       # a real GPU run completed cleanly
                    consec_fail = 0
                continue
            if not qwen_up():              # need the brain to fire a turn
                log("Qwen down + no GPU job -> loading Qwen")
                qwen_up_cmd()
                if not qwen_up():          # load failed -> escalate, don't silently spin every 8s (M6/S1)
                    consec_fail += 1
                    back = min(30 * consec_fail, 300)
                    log("Qwen failed to come up (consec_fail=%d) -> backing off %ss" % (consec_fail, back))
                    time.sleep(back)
                    continue
            if gpu_busy():                 # a job may have been enqueued during the load (S3)
                continue
            t_fire = time.time()
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
            else:                          # completed / aborted
                consec_fail = 0
                # Floor a degenerate instant completion so it can't rapid-fire fresh webui sessions and
                # hammer the gateway (S5). A real turn takes minutes, so this never bites a healthy loop.
                el = time.time() - t_fire
                if 0 <= el < MIN_TURN_FLOOR:
                    time.sleep(MIN_TURN_FLOOR - el)
        except Exception as e:
            # One bad iteration (a hung serve script, a transient API error) must NEVER crash the loop
            # into a systemd restart-loop (M5). Log, back off, keep going.
            log("loop iteration error (non-fatal): %r" % e)
            time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
