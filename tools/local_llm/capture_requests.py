#!/usr/bin/env python3
"""capture_requests.py -- a tiny logging reverse proxy for the prompt-cache/template-divergence investigation
(research/local-llm-prompt-cache branch, 2026-09-25 round 2, coordinator directive).

Sits in front of a live llama-server and writes every request body it forwards to a numbered file, verbatim,
before relaying it unchanged to the real server and returning the real response to the client. Point
ANTHROPIC_BASE_URL at this proxy's port (instead of directly at llama-server) for a `claude -p` session, and the
exact wire-format request bodies Claude Code sends -- the real Anthropic Messages API JSON, "system"/"messages"/
"tools" and all -- land on disk for offline inspection (e.g. to check exactly how a mid-conversation "system
reminder" is actually shaped: a top-level `system` block, or a `role: "system"` entry inside `messages`).

Usage: capture_requests.py <listen_port> <upstream_port> <capture_dir>
Runs until killed (SIGTERM/SIGINT); the caller starts it as a background thread or subprocess and stops it once
the Claude Code task under test has finished.
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.error
import urllib.request

_counter_lock = threading.Lock()
_counter = [0]


def _next_index():
    with _counter_lock:
        _counter[0] += 1
        return _counter[0]


def make_handler(upstream_port, capture_dir):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass   # keep stdout clean; the capture files ARE the log

        def _forward(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            idx = _next_index()
            safe_path = self.path.strip("/").replace("/", "_") or "root"
            fname = "%04d_%s_%s.json" % (idx, self.command, safe_path)
            try:
                # Pretty-print if it parses as JSON (much easier to read later); fall back to raw bytes.
                parsed = json.loads(body) if body else None
                with open(os.path.join(capture_dir, fname), "w") as f:
                    json.dump(parsed, f, indent=1)
            except Exception:
                with open(os.path.join(capture_dir, fname), "wb") as f:
                    f.write(body)
            url = "http://127.0.0.1:%d%s" % (upstream_port, self.path)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "connection")}
            req = urllib.request.Request(url, data=body if length else None, method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=1200) as resp:
                    self.send_response(resp.status)
                    resp_body = resp.read()
                    for k, v in resp.getheaders():
                        if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    if self.command == "POST":
                        # Raw response bytes (often an SSE stream) saved verbatim beside the request capture,
                        # so hypothesis 2 (assistant turns re-rendered differently from what was generated) can
                        # be checked directly: compare what the model actually streamed back this turn against
                        # how that same turn renders when Claude Code resends it as history next turn.
                        with open(os.path.join(capture_dir, fname + ".response"), "wb") as f:
                            f.write(resp_body)
            except urllib.error.HTTPError as e:
                body_err = e.read()
                self.send_response(e.code)
                self.send_header("Content-Length", str(len(body_err)))
                self.end_headers()
                self.wfile.write(body_err)
            except Exception as e:
                self.send_response(502)
                msg = ("proxy forwarding error: %r" % e).encode()
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def do_GET(self):
            self._forward()

        def do_POST(self):
            self._forward()

    return Handler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run(listen_port, upstream_port, capture_dir):
    os.makedirs(capture_dir, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", listen_port), make_handler(upstream_port, capture_dir))
    server.serve_forever()


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
