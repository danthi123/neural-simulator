#!/usr/bin/env python3
"""One-off verification for the 2026-09-29 serving simplification (llm.sh's "local-llm" unit now runs llama-swap
in front of llama-server instead of launching llama-server directly, tools/local_llm/llama-swap.yaml): confirms
throughput through llama-swap is within ~5% of the SAME profile served directly, so fronting it with a proxy
cost nothing worth caring about. Compares against the bake-off's own direct-llama-server number for the default
profile (research/serving-bakeoff, tools/local_llm/results/summary.md: 74 tok/s, qwen38-27b-iq4nl-mtp-128k-q4).

Runs its OWN llama-server (port 18300) and its OWN llama-swap (port 18301) -- neither touches the real
"local-llm" unit or its port 8080, so this is safe to run even while that unit is doing something else; it only
needs the GPU itself to be free (one model loads at a time either way). This needs the whole GPU, so it must run
through the GPU queue, never directly:
    bash tools/gpu_queue.sh add 'cd /home/dant123/Projects/sim/.claude/worktrees/serving-simplify && \\
        .venv/bin/python tools/local_llm/swap_speed_check.py' (or the primary checkout's own venv/copy)
Result: tools/local_llm/results/swap_speed_check.json (also printed to stdout, which lands in gpu_queue's log).
"""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LLM_SH = os.path.join(HERE, "llm.sh")
YAML_PATH = os.path.join(HERE, "llama-swap.yaml")
RESULTS = os.path.join(HERE, "results")
DIRECT_PORT = 18300
SWAP_PORT = 18301
PROFILE = "qwen38-27b-iq4nl-mtp-128k-q4"   # tools/local_llm/default_profile -- the bake-off winner
PROMPT = ("Write a 200-word explanation of how flash attention reduces memory bandwidth during transformer "
          "inference, aimed at someone who already knows what attention is.")


def http(url, payload, timeout=1800, method=None):
    # method=None lets urllib pick its default (POST, since `data` is given). llama-swap's own API is not
    # uniformly POST -- PUT /api/profiles/active specifically (its server.go registers exactly that method) --
    # a 2026-09-29 bug found live: a plain POST there 405's instantly, which is why the first queued run of
    # this script failed in 14s (an HTTPError) instead of actually measuring anything.
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(url, proc, deadline_s=600):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError("process exited during load (exit %s)" % proc.returncode)
        try:
            urllib.request.urlopen(url, timeout=5).read()
            return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("did not become healthy within %ss (%s)" % (deadline_s, url))


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def measure_completion(base_url, model=None, warmup=True):
    payload = {"prompt": PROMPT, "n_predict": 300, "temperature": 0, "cache_prompt": False}
    if model:
        payload["model"] = model
    if warmup:
        http(base_url + "/completion", {**payload, "n_predict": 16})   # first request pays one-time warmup cost
    r = http(base_url + "/completion", payload, timeout=600)
    t = r.get("timings", {})
    return {"gen_tok_s": round(t.get("predicted_per_second", 0), 2),
            "pp_tok_s": round(t.get("prompt_per_second", 0), 2),
            "predicted_n": t.get("predicted_n")}


def measure_direct():
    cmd_line = subprocess.run(["bash", LLM_SH, "__profile_cmd", PROFILE],
                              env={**os.environ, "LLM_PORT": str(DIRECT_PORT)},
                              capture_output=True, text=True, check=True).stdout.strip()
    log = open(os.path.join(RESULTS, "swap_speed_check.direct.log"), "w")
    proc = subprocess.Popen(shlex.split(cmd_line), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        wait_health("http://127.0.0.1:%d/health" % DIRECT_PORT, proc)
        load_s = round(time.time() - t0, 1)
        result = measure_completion("http://127.0.0.1:%d" % DIRECT_PORT)
        result["load_s"] = load_s
        return result
    finally:
        stop(proc)


def measure_through_swap():
    swap_cmd = subprocess.run(
        ["bash", "-c",
         'PORT=%d; printf "%%s -config %%s -listen 127.0.0.1:%%s -watch-config" '
         '"$(command -v llama-swap || echo "$HOME/.local/bin/llama-swap")" %s "$PORT"'
         % (SWAP_PORT, shlex.quote(YAML_PATH))],
        capture_output=True, text=True, check=True).stdout
    log = open(os.path.join(RESULTS, "swap_speed_check.swap.log"), "w")
    env = {**os.environ, "LLAMA_SWAP_REPO_ROOT": ROOT}
    proc = subprocess.Popen(shlex.split(swap_cmd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=env)
    try:
        t0 = time.time()
        wait_health("http://127.0.0.1:%d/health" % SWAP_PORT, proc)   # llama-swap's own health (fast: no model yet)
        # activate_profile() equivalent: pin "local" then force-load, same as llm.sh's cmd_on would.
        http("http://127.0.0.1:%d/api/profiles/active" % SWAP_PORT, {"name": PROFILE}, method="PUT")
        urllib.request.urlopen("http://127.0.0.1:%d/upstream/%s/health" % (SWAP_PORT, PROFILE), timeout=600).read()
        load_s = round(time.time() - t0, 1)
        result = measure_completion("http://127.0.0.1:%d" % SWAP_PORT, model=PROFILE)
        result["load_s"] = load_s
        return result
    finally:
        stop(proc)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rec = {"profile": PROFILE, "device": "cuda (RTX 3090, llama-server; run via tools/gpu_queue.sh)", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec["direct_llama_server"] = measure_direct()
    rec["through_llama_swap"] = measure_through_swap()
    d, s = rec["direct_llama_server"]["gen_tok_s"], rec["through_llama_swap"]["gen_tok_s"]
    rec["gen_tok_s_delta_pct"] = round(100.0 * (s - d) / d, 2) if d else None
    rec["within_5pct"] = abs(rec["gen_tok_s_delta_pct"]) <= 5.0 if rec["gen_tok_s_delta_pct"] is not None else None
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(RESULTS, "swap_speed_check.json")
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print("saved: %s" % out)
    return 0 if rec["within_5pct"] else 1


if __name__ == "__main__":
    sys.exit(main())
