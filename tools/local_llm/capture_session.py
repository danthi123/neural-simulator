#!/usr/bin/env python3
"""capture_session.py -- capture the REAL request bodies Claude Code sends across a short multi-turn session,
via capture_requests.py's logging proxy sitting in front of a real llama-server.

research/local-llm-prompt-cache branch, 2026-09-25 round 2 (coordinator directive): test directly, rather than
infer, whether the chat template's system-message hoisting (merging every system/developer message into the ONE
leading block) is what breaks prompt-cache reuse -- capture two consecutive real request bodies, then a separate
script (render_and_diff.py) renders each through the template and diffs.

Layout: llama-server on LLAMA_PORT (real inference), capture_requests.py proxy on PROXY_PORT (ANTHROPIC_BASE_URL
points HERE), a throwaway git worktree running the short multi-turn `claude -p` task. Captured request bodies
land in --capture-dir, one JSON file per request, numbered in arrival order.

Run through the GPU queue (needs the whole card, like cache_probe.py):
    bash tools/gpu_queue.sh add 'cd <worktree> && python3 tools/local_llm/capture_session.py --capture-dir ...'
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_probe as cp   # reuse build_cmd/load_default_profile/start_server/stop_server/worktree/claude_bin

LLAMA_PORT = 8093
PROXY_PORT = 8094


def run_claude_task_via(base_url, cwd, prompt, ctx_tokens, max_turns, timeout):
    """Same shape as cache_probe.run_claude_task, but pointed at an arbitrary base_url (the capture proxy)."""
    env = dict(os.environ, ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN="local", ANTHROPIC_API_KEY="",
               ANTHROPIC_MODEL="local", ANTHROPIC_SMALL_FAST_MODEL="local", ANTHROPIC_DEFAULT_HAIKU_MODEL="local",
               ANTHROPIC_DEFAULT_SONNET_MODEL="local", ANTHROPIC_DEFAULT_OPUS_MODEL="local",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1", API_TIMEOUT_MS="1200000",
               SIM_NO_PROVENANCE="1", CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(ctx_tokens))
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(k, None)
    cmd = [cp.claude_bin(), "-p", prompt, "--output-format", "json", "--max-turns", str(max_turns),
           "--dangerously-skip-permissions", "--strict-mcp-config",
           "--settings", os.path.join(HERE, "claude_local_settings.json"),
           "--disallowedTools", "WebSearch", "ReportFindings", "Bash(git push:*)", "Bash(git commit:*)",
           "Bash(git merge:*)", "Bash(git checkout:*)", "Bash(git reset:*)", "Bash(bash tools/push_both.sh:*)"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        raw = p.stdout.strip()
        try:
            j = json.loads(raw.splitlines()[-1]) if raw else {}
        except Exception:
            j = {"unparsed": raw[-2000:]}
        return {"wall_s": round(time.time() - t0, 1), "exit": p.returncode, "num_turns": j.get("num_turns"),
                "is_error": j.get("is_error"), "result": str(j.get("result", j.get("unparsed", "")))[-1500:],
                "stderr_tail": p.stderr[-1500:]}
    except subprocess.TimeoutExpired:
        return {"wall_s": round(time.time() - t0, 1), "exit": "timeout", "result": "", "is_error": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--template", default=None, help="override chat_template_file path (default: profile's own)")
    ap.add_argument("--task", default=cp.DEFAULT_TASK)
    ap.add_argument("--max-turns", type=int, default=cp.DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=cp.DEFAULT_TIMEOUT_S)
    a = ap.parse_args()

    global LLAMA_PORT
    cp.PORT = LLAMA_PORT   # cache_probe.build_cmd() reads the module-level PORT
    profile = dict(cp.load_default_profile())
    if a.template:
        profile["chat_template_file"] = os.path.relpath(a.template, cp.ROOT)
    cmd = cp.build_cmd(profile, 1, [])   # config A shape: -np 1, no extra flags -- the config under investigation
    os.makedirs(os.path.join(cp.HERE, "results", "template_divergence"), exist_ok=True)
    log_path = os.path.join(cp.HERE, "results", "template_divergence", "capture_session.server.log")
    proc = None
    proxy = None
    wt = None
    try:
        proc, load_s = cp.start_server(cmd, log_path)
        print("server up after %.1fs" % load_s, flush=True)
        os.makedirs(a.capture_dir, exist_ok=True)
        proxy = subprocess.Popen([sys.executable, os.path.join(HERE, "capture_requests.py"),
                                   str(PROXY_PORT), str(LLAMA_PORT), a.capture_dir])
        time.sleep(1)
        wt = cp.worktree("capture-session")
        result = run_claude_task_via("http://127.0.0.1:%d" % PROXY_PORT, wt, a.task, profile["ctx"],
                                      a.max_turns, a.timeout)
        print("claude task result: %s" % json.dumps(result, indent=1), flush=True)
    finally:
        if proxy:
            proxy.terminate()
            try:
                proxy.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proxy.kill()
        if wt:
            cp.remove_worktree(wt)
        cp.stop_server(proc)
    files = sorted(os.listdir(a.capture_dir))
    print("captured %d files: %s" % (len(files), files), flush=True)


if __name__ == "__main__":
    sys.exit(main())
