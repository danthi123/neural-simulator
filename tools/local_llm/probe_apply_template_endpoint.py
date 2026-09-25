import json, os, subprocess, sys, time, urllib.request, urllib.error, signal

ROOT = "/home/dant123/Projects/sim/.claude/worktrees/agent-a50e75206aabe3466"
PORT = 8093
MODEL = os.path.expanduser("~/models/qwen3.8-27b-gguf/unsloth-iq4nl/Qwen3.8-27B-IQ4_NL.gguf")
TEMPLATE = os.path.join(ROOT, "tools/local_llm/templates/qwen38-27b-iq4nl-mtp.jinja")
LOG = os.path.join(ROOT, "tools/local_llm/results/template_divergence/endpoint_probe.server.log")

cmd = ["llama-server", "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--alias", "local",
       "-ngl", "99", "-np", "1", "-fa", "on", "-c", "131072", "-ctk", "q4_0", "-ctv", "q4_0", "--jinja",
       "--chat-template-file", TEMPLATE]
log = open(LOG, "w")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def http(path, payload=None, method="GET"):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return None, repr(e)


t0 = time.time()
up = False
while time.time() - t0 < 300:
    if proc.poll() is not None:
        print("SERVER EXITED EARLY, see log")
        sys.exit(1)
    st, body = http("/health")
    if st == 200:
        up = True
        break
    time.sleep(2)
if not up:
    print("SERVER DID NOT COME UP")
    sys.exit(1)
print("server up after %.1fs" % (time.time() - t0))

candidates = [
    ("apply-template_oai_simple", "/apply-template", {"messages": [{"role": "user", "content": "hi"}]}),
    ("apply-template_oai_system_mid", "/apply-template", {"messages": [
        {"role": "system", "content": "leading sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "mid sys reminder"},
        {"role": "user", "content": "q2"},
    ]}),
    ("apply-template_anthropic_shape", "/apply-template", {"system": "leading sys",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "q1"}]}]}),
    ("count_tokens_oai", "/v1/messages/count_tokens", {"model": "local", "system": "leading sys",
        "messages": [{"role": "user", "content": "q1"}]}),
    ("props", "/props", None),
]
out = {
    # Not a sim/research run -- an HTTP endpoint-shape discovery against llama-server itself. See
    # render_and_diff.py's identical note for why this is recorded explicitly.
    "backend": "llama.cpp/HTTP endpoint probe (not a SIM_BACKEND numpy/cupy run)",
}
for name, path, payload in candidates:
    method = "GET" if payload is None else "POST"
    st, body = http(path, payload, method)
    out[name] = {"status": st, "body": body[:4000]}
    print("=== %s (%s %s) -> %s ===" % (name, method, path, st))
    print(body[:2000])
    print()

json.dump(out, open(os.path.join(ROOT, "tools/local_llm/results/template_divergence/endpoint_probe.json"), "w"), indent=1)

os.killpg(proc.pid, signal.SIGTERM)
try:
    proc.wait(timeout=60)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)
print("done")
