#!/usr/bin/env python3
"""Drive ONE persistent, webui-visible Hermes conversation.

Used by tools/qwen_supervisor.sh to fire the between-runs continuation turn INTO the
webui (so the owner watches it stream + types into the same conversation), instead of a
headless `hermes -z` turn. The turn runs through the webui -> gateway -> local Qwen, and
appears as the single conversation whose session_id is HERMES_AUTONOMOUS_SESSION_ID.

Continuity is via DURABLE STATE (live_state.md + repo), not the transcript, so each fired
turn re-reads live_state and picks up where the loop left off — exactly like the old
`hermes -z` path, but visible + engageable.

Auth: logs in with the webui password (login -> cookie), read from the webui systemd unit
(the single source of truth the webui itself uses). One source, no duplicated secret.

Exit 0 = the turn was accepted by the webui. Non-zero = could not (webui down / auth /
suppressed) -> the caller falls back to the headless path so autonomy never stalls.

Usage: webui_continue.py "<message>"   (message defaults to a live-state re-anchor prompt)
"""
import json
import os
import re
import sys
import urllib.request
import urllib.error

# The webui assigns session ids; we create ONE session, persist its id here, and reuse it
# so every fired turn appends to the SAME conversation the owner watches.
UNIT = os.path.expanduser("~/.config/systemd/user/hermes-webui.service")
SID_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "research", "queue", ".hermes_webui_session_id",
)
DEFAULT_MSG = (
    "[autonomous continuation] A local GPU run just finished and your model was reloaded. "
    "Read research/coordination/live_state.md and the CURRENT STATE at the top of "
    "GAP_CLOSURE_MISSION.md, harvest the just-completed run(s), then take the next concrete "
    "action (edit files / launch the next run via tools/hermes_gpu_run.sh / commit via "
    "tools/push_both.sh). Obey CLAUDE.md constraints. Never end on a status report alone."
)


def _unit_env(name: str, default: str = "") -> str:
    """Read an Environment=NAME=VALUE line from the webui systemd unit (falls back to os.environ)."""
    try:
        with open(UNIT, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*Environment=%s=(.*)$" % re.escape(name), line)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return os.environ.get(name, default)


def _base_url() -> str:
    port = _unit_env("HERMES_WEBUI_PORT", "8787") or "8787"
    return "http://127.0.0.1:%s" % port


def _read_sid() -> str | None:
    try:
        with open(SID_FILE, encoding="utf-8") as f:
            sid = f.read().strip()
            return sid or None
    except Exception:
        return None


def _write_sid(sid: str) -> None:
    try:
        os.makedirs(os.path.dirname(SID_FILE), exist_ok=True)
        with open(SID_FILE, "w", encoding="utf-8") as f:
            f.write(sid)
    except Exception:
        pass


def _post(base: str, endpoint: str, headers: dict, body: dict, timeout: int = 20):
    """Return (status_code, parsed_json) — status_code 0 on a transport error."""
    try:
        req = urllib.request.Request(
            base + endpoint, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        print("webui_continue: POST %s failed: %s" % (endpoint, e), file=sys.stderr)
        return 0, {}


def _create_session(base: str, headers: dict) -> str | None:
    """Create a fresh webui session (defaults to the repo workspace) and return its id."""
    code, out = _post(base, "/api/session/new", headers, {})
    if code == 200:
        sid = ((out or {}).get("session") or {}).get("session_id")
        if sid:
            _write_sid(sid)
            # Give it a findable title so the owner can spot THE autonomous loop in the sidebar.
            _post(base, "/api/session/rename", headers,
                  {"session_id": sid, "title": "\U0001F916 Autonomous research loop"})
            return sid
    print("webui_continue: /api/session/new -> %s %s" % (code, str(out)[:160]), file=sys.stderr)
    return None


def _login(base: str, pw: str) -> str | None:
    if not pw:
        return None
    try:
        req = urllib.request.Request(
            base + "/api/auth/login",
            data=json.dumps({"password": pw}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=8)
        cookie = resp.headers.get("Set-Cookie", "")
        return cookie.split(";")[0] if cookie else None
    except Exception as e:
        print("webui_continue: login failed: %s" % e, file=sys.stderr)
        return None


def main() -> int:
    msg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MSG
    base = _base_url()
    pw = _unit_env("HERMES_WEBUI_PASSWORD")
    # No Origin header -> treated as a non-browser API client (no CSRF gate), same as mcp_server.
    headers = {"Content-Type": "application/json"}
    cookie = _login(base, pw)
    if cookie:
        headers["Cookie"] = cookie
    elif pw:
        return 2  # a password is set but login failed -> do not proceed unauth

    sid = _read_sid() or _create_session(base, headers)
    if not sid:
        return 5  # could not obtain a session id

    # PRIMARY re-engagement: if the owner set a standing goal (/goal ...), the loop drives itself
    # turn-after-turn — but a GPU run unloads Qwen and stalls its auto-continue, so after the run we
    # RESUME the goal (it then harvests + takes the next step). Falls through to a one-shot chat/start
    # continuation only when there is no active goal to resume.
    if os.environ.get("HERMES_CONTINUE_RESUME_GOAL", "1") == "1":
        code, out = _post(base, "/api/goal", headers, {"session_id": sid, "args": "resume"})
        blob = json.dumps(out).lower()
        no_goal = ("no active goal" in blob or "no goal" in blob or "not found" in blob
                   or (isinstance(out, dict) and out.get("error")))
        if code == 200 and not no_goal:
            print("webui_continue: resumed the standing goal in session %s" % sid)
            return 0
        # else: no goal set (or resume unavailable) -> one-shot continuation below.

    # Send; if the persisted session was deleted (404), recreate once and retry.
    for attempt in (1, 2):
        code, out = _post(base, "/api/chat/start", headers, {"session_id": sid, "message": msg})
        if code == 200:
            if out.get("status") == "suppressed":
                print("webui_continue: suppressed (%s)" % out.get("reason"), file=sys.stderr)
                return 4
            print("webui_continue: fired into webui session %s (%s)" % (sid, out.get("status", "ok")))
            return 0
        if code == 404 and attempt == 1:
            print("webui_continue: session %s gone -> recreating" % sid, file=sys.stderr)
            sid = _create_session(base, headers)
            if not sid:
                return 5
            continue
        print("webui_continue: chat/start -> %s %s" % (code, str(out)[:160]), file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    sys.exit(main())
