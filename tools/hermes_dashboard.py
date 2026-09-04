#!/usr/bin/env python3
"""Hermes control + progress dashboard — self-contained (Python stdlib only), runs FULLY INDEPENDENTLY of Claude.

Launch (one short command):
    python3 tools/hermes_dashboard.py            # serves on http://0.0.0.0:8765
    python3 tools/hermes_dashboard.py --port 9000 --host 127.0.0.1

Then open http://localhost:8765 (or http://<this-box-LAN-ip>:8765 from another device on your LAN).

What it does, live (auto-refreshes every 10s), reading real state from research/queue/ + git and driving the
EXISTING control scripts — no Claude, no pip deps:
  * CONTROL: Start / Stop the Hermes takeover, Pause-for-gaming / Resume (buttons POST to
    tools/hermes_takeover.sh + tools/game.sh).
  * COMPUTE: GPU util+VRAM, the gpu_queue (running + depth), the mini-PC pool depth.
  * PROGRESS: the live FRONTIER + NEXT ACTIONS (from research/coordination/live_state.md), recent commits,
    recent findings.
  * HERMES LOOP: a live tail of research/queue/hermes_loop.log (what Hermes is doing each turn) + the gpu_queue log.
"""
import argparse
import html
import os
import subprocess
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(REPO, "research", "queue")


def sh(cmd, timeout=25):
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(err: {e})"


def exists(name):
    return os.path.exists(os.path.join(Q, name))


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip() or "(empty)"
    except Exception:
        return "(no file)"


def count_lines(path):
    try:
        with open(path) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def gpu_line():
    out = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
              "--format=csv,noheader,nounits"], timeout=10)
    if "," in out:
        p = [x.strip() for x in out.split(",")]
        try:
            return f"{p[0]}% util · {int(p[1])/1024:.1f}/{int(p[2])/1024:.1f} GB · {p[3]}°C"
        except Exception:
            return out
    return out or "n/a"


def gpu_queue():
    running = tail(os.path.join(Q, "gpu.running"), 1)
    cur = "(idle)"
    if running and running not in ("(empty)", "(no file)"):
        cur = running.split("\t")[-1][:120] if "\t" in running else running[:120]
    depth = count_lines(os.path.join(Q, "gpu.queue"))
    return cur, depth


def gpu_research_count():
    """Count brain-loading research GPU procs (python with >1.5GB VRAM) — excludes desktop apps (kwin/Discord/etc.)."""
    out = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
              "--format=csv,noheader,nounits"], timeout=10)
    n = 0
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 3 and "python" in p[2].lower():
            try:
                if int(p[1]) > 1500:
                    n += 1
            except Exception:
                pass
    return n


def read_state():
    active = exists("HERMES_ACTIVE")
    game = exists("GAME_MODE")
    gpaused = exists("GPU_PAUSE")
    cur, gdepth = gpu_queue()
    return {
        "driver": "HERMES (autonomous)" if active else "idle / Claude-driven",
        "active": active,
        "game": game,
        "gpaused": gpaused,
        "gpu": gpu_line(),
        "gpu_cur": cur,
        "gpu_depth": gdepth,
        "research_gpu": gpu_research_count(),
        "last_action": tail(os.path.join(Q, "dashboard_actions.log"), 6),
        "pool_depth": count_lines(os.path.join(Q, "pool.queue")),
        "live_state": tail(os.path.join(REPO, "research", "coordination", "live_state.md"), 40),
        "commits": sh(["git", "log", "--oneline", "-14"], timeout=10),
        "findings": sh(["bash", "-lc", "ls -t research/findings/*.md 2>/dev/null | head -6"], timeout=10),
        "loop_log": tail(os.path.join(Q, "hermes_loop.log"), 30),
        "gpu_log": tail(os.path.join(Q, "gpu_queue.log"), 10),
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def badge(label, on, on_txt="ON", off_txt="off", on_color="#e0533d", off_color="#3a7a4a"):
    c = on_color if on else off_color
    return f'<span class="badge" style="background:{c}">{html.escape(label)}: {on_txt if on else off_txt}</span>'


def btn(do, label, cls="b"):
    return (f'<form method="POST" action="/action" style="display:inline">'
            f'<input type="hidden" name="do" value="{do}">'
            f'<button class="{cls}" type="submit">{html.escape(label)}</button></form>')


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Dashboard</title><style>
:root{{color-scheme:dark}}
body{{background:#12141a;color:#d8dde6;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:18px}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#8b94a3;margin:20px 0 8px;border-bottom:1px solid #262a33;padding-bottom:4px}}
.sub{{color:#6b7482;font-size:12px;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.card{{background:#1a1d25;border:1px solid #262a33;border-radius:8px;padding:12px}}
.k{{color:#8b94a3}} .v{{color:#e8edf4;font-weight:600}}
.badge{{display:inline-block;padding:3px 9px;border-radius:5px;color:#fff;font-size:12px;font-weight:600;margin:2px 4px 2px 0}}
button.b{{background:#2a3040;color:#dfe6f0;border:1px solid #3a4252;border-radius:6px;padding:8px 14px;margin:3px 4px 3px 0;cursor:pointer;font:inherit}}
button.b:hover{{background:#343c4e}} button.go{{background:#2f5d3a;border-color:#3f7a4d}} button.stop{{background:#5d2f2f;border-color:#7a3f3f}}
pre{{background:#0e1015;border:1px solid #22262e;border-radius:6px;padding:10px;overflow:auto;max-height:340px;font-size:12px;white-space:pre-wrap;word-break:break-word}}
.row{{margin:3px 0}}
</style></head><body>
<h1>🧠 Hermes Dashboard</h1>
<div class="sub">refreshes every 10s · {now} · repo: {repo}</div>

<h2>Control</h2>
<div class="card">
  <div class="row">Driver: <span class="v">{driver}</span> &nbsp; {b_active} {b_game} {b_gpause}</div>
  <div class="row" style="margin-top:8px;font-size:16px;font-weight:700">{game_status}</div>
  <div style="margin-top:10px">
    {btn_on} {btn_off} &nbsp;|&nbsp; {btn_game_on} {btn_game_off}
  </div>
  <div class="sub" style="margin-top:8px">Start/Stop = tools/hermes_takeover.sh · Pause = tools/game.sh on --force (kills any resident GPU job + frees VRAM for gaming) · Resume = game.sh off. Buttons act on THIS box.</div>
  <div class="k" style="margin-top:8px">last action (tools/game.sh / takeover output):</div><pre style="max-height:130px">{last_action}</pre>
</div>

<h2>Compute</h2>
<div class="grid">
  <div class="card"><div class="k">GPU</div><div class="v">{gpu}</div></div>
  <div class="card"><div class="k">GPU queue</div><div class="v">depth {gpu_depth}</div><div class="sub" style="margin-top:4px">current: {gpu_cur}</div></div>
  <div class="card"><div class="k">Mini-PC pool</div><div class="v">depth {pool_depth}</div></div>
</div>

<h2>Frontier &amp; Next Actions (live_state)</h2>
<pre>{live_state}</pre>

<h2>Recent commits</h2>
<pre>{commits}</pre>

<h2>Recent findings</h2>
<pre>{findings}</pre>

<h2>Hermes loop log (last 30)</h2>
<pre>{loop_log}</pre>

<h2>GPU queue log (last 10)</h2>
<pre>{gpu_log}</pre>
</body></html>"""


def render():
    s = read_state()
    rg = s["research_gpu"]
    if s["game"] and rg == 0:
        gs = '<span style="color:#5fd97a">🎮 SAFE TO GAME — GPU free (no research job resident)</span>'
    elif s["game"] and rg >= 1:
        gs = f'<span style="color:#e0533d">⚠️ PAUSED but {rg} GPU job still resident — click Pause again (it force-kills)</span>'
    elif rg > 1:
        gs = f'<span style="color:#e0533d">⚠️ {rg} GPU jobs resident — possible contention; click Pause to clear</span>'
    elif rg == 1:
        gs = '<span style="color:#8b94a3">⚙️ 1 research job running on the GPU (normal)</span>'
    else:
        gs = '<span style="color:#8b94a3">GPU idle (no research job)</span>'
    return PAGE.format(
        now=html.escape(s["now"]), repo=html.escape(REPO), driver=html.escape(s["driver"]),
        game_status=gs, last_action=html.escape(s["last_action"]),
        b_active=badge("Hermes", s["active"], "ACTIVE", "idle"),
        b_game=badge("Game-pause", s["game"]),
        b_gpause=badge("GPU-pause", s["gpaused"]),
        btn_on=btn("hermes_on", "▶ Start Hermes", "b go"),
        btn_off=btn("hermes_off", "■ Stop Hermes", "b stop"),
        btn_game_on=btn("game_on", "⏸ Pause (gaming)", "b"),
        btn_game_off=btn("game_off", "▶ Resume", "b go"),
        gpu=html.escape(s["gpu"]), gpu_depth=s["gpu_depth"], gpu_cur=html.escape(s["gpu_cur"]),
        pool_depth=s["pool_depth"],
        live_state=html.escape(s["live_state"]), commits=html.escape(s["commits"]),
        findings=html.escape(s["findings"]), loop_log=html.escape(s["loop_log"]), gpu_log=html.escape(s["gpu_log"]),
    )


ACTIONS = {
    "hermes_on": ["bash", "tools/hermes_takeover.sh", "on"],
    "hermes_off": ["bash", "tools/hermes_takeover.sh", "off"],
    "game_on": ["bash", "tools/game.sh", "on", "--force"],
    "game_off": ["bash", "tools/game.sh", "off"],
}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode("utf-8", "replace")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/action"):
            self.send_response(303); self.send_header("Location", "/"); self.end_headers(); return
        try:
            self._send(200, render())
        except Exception as e:
            self._send(500, f"<pre>dashboard error: {html.escape(repr(e))}</pre>")

    def do_POST(self):
        if self.path.split("?")[0] == "/action":
            n = int(self.headers.get("Content-Length", 0) or 0)
            data = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace")) if n else {}
            do = (data.get("do") or [""])[0]
            if do in ACTIONS:
                # fire-and-report; takeover on/off can take a while (Qwen load), so cap it and let the page refresh
                subprocess.Popen(ACTIONS[do], cwd=REPO,
                                 stdout=open(os.path.join(Q, "dashboard_actions.log"), "a"),
                                 stderr=subprocess.STDOUT)
        self.send_response(303); self.send_header("Location", "/"); self.end_headers()

    def log_message(self, *a):
        pass  # quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"Hermes dashboard → http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print(f"  from another LAN device: http://<this-box-ip>:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
