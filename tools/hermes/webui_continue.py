#!/usr/bin/env python3
"""Fire ONE visible Hermes turn into the single persistent webui conversation.

The supervisor calls this after each GPU run completes + Qwen reloads. It posts the continuation
prompt to ONE fixed webui session (id persisted in research/queue/.hermes_webui_session_id, titled
"🤖 Autonomous research loop"), so the owner watches the whole cycle stream in one place:
  Hermes works (harvest → decide → launch one run) → run executes → Hermes checks results → repeat.

The turn runs through webui -> gateway -> local Qwen and appears in that conversation. /api/chat/start
is ASYNC (returns as soon as the run is accepted), so this exits in ~1-2s and never blocks the
supervisor's VRAM loop (the supervisor also detaches this call as belt-and-suspenders).

Continuity is via DURABLE STATE (live_state.md + repo), not the transcript, so each fired turn
re-reads live_state and picks up where the loop left off.

Auth: logs in with the webui password read from the webui systemd unit (one source of truth).

Usage: webui_continue.py ["<message>"]   (message defaults to the harvest+continue prompt)
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIT = os.path.expanduser("~/.config/systemd/user/hermes-webui.service")
SID_FILE = os.path.join(ROOT, "research", "queue", ".hermes_webui_session_id")
TITLE = "\U0001F916 Autonomous research loop"
DEFAULT_MSG = (
    "[autonomous cycle] A GPU experiment just finished and your model reloaded. Read "
    "research/coordination/live_state.md and the CURRENT STATE of GAP_CLOSURE_MISSION.md, harvest "
    "the completed run's result, take the next concrete action (edits/commits via tools/push_both.sh), "
    "then launch the NEXT single experiment via tools/hermes_gpu_run.sh and end your turn — you will "
    "be re-invoked automatically when it completes. Obey CLAUDE.md. Never end on a status report alone."
)


def _unit_env(name, default=""):
    try:
        with open(UNIT, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*Environment=%s=(.*)$" % re.escape(name), line)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return os.environ.get(name, default)


def _base():
    return "http://127.0.0.1:%s" % (_unit_env("HERMES_WEBUI_PORT", "8787") or "8787")


def _read_sid():
    try:
        with open(SID_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def _write_sid(sid):
    try:
        os.makedirs(os.path.dirname(SID_FILE), exist_ok=True)
        with open(SID_FILE, "w", encoding="utf-8") as f:
            f.write(sid)
    except Exception:
        pass


def _post(base, path, headers, body, timeout=25):
    try:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.getcode(), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        print("webui_continue: POST %s failed: %s" % (path, e), file=sys.stderr)
        return 0, {}


def _login(base, pw):
    if not pw:
        return None
    try:
        req = urllib.request.Request(base + "/api/auth/login",
                                     data=json.dumps({"password": pw}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        r = urllib.request.urlopen(req, timeout=8)
        c = r.headers.get("Set-Cookie", "")
        return c.split(";")[0] if c else None
    except Exception as e:
        print("webui_continue: login failed: %s" % e, file=sys.stderr)
        return None


def _create(base, headers):
    code, out = _post(base, "/api/session/new", headers, {})
    if code == 200:
        sid = ((out or {}).get("session") or {}).get("session_id")
        if sid:
            _write_sid(sid)
            _post(base, "/api/session/rename", headers, {"session_id": sid, "title": TITLE})
            return sid
    print("webui_continue: /api/session/new -> %s %s" % (code, str(out)[:160]), file=sys.stderr)
    return None


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MSG
    base = _base()
    pw = _unit_env("HERMES_WEBUI_PASSWORD")
    headers = {"Content-Type": "application/json"}   # no Origin -> non-browser client, no CSRF gate
    cookie = _login(base, pw)
    if cookie:
        headers["Cookie"] = cookie
    elif pw:
        return 2
    sid = _read_sid() or _create(base, headers)
    if not sid:
        return 5
    for attempt in (1, 2, 3):
        code, out = _post(base, "/api/chat/start", headers, {"session_id": sid, "message": msg})
        if code == 200:
            if out.get("status") == "suppressed":
                print("webui_continue: suppressed (%s)" % out.get("reason"), file=sys.stderr)
                return 4
            print("webui_continue: fired a visible turn into %s (%s)" % (sid, out.get("status", "ok")))
            return 0
        # 409 = the session is jammed by a stale active-stream lock (left by an interrupted turn).
        # Self-heal: delete the jammed session and start a fresh one so the loop never permanently
        # stalls (the overnight-jam failure mode). 404 = session gone -> just recreate.
        if code in (404, 409) and attempt < 3:
            print("webui_continue: session %s %s -> deleting + recreating (self-heal)"
                  % (sid, "jammed(409)" if code == 409 else "gone(404)"), file=sys.stderr)
            if code == 409:
                _post(base, "/api/session/delete", headers, {"session_id": sid})
            try:
                os.remove(SID_FILE)
            except Exception:
                pass
            sid = _create(base, headers)
            if not sid:
                return 5
            continue
        print("webui_continue: chat/start -> %s %s" % (code, str(out)[:160]), file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    sys.exit(main())
