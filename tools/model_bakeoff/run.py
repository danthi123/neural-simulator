#!/usr/bin/env python3
"""Model bake-off runner (docs: tools/model_bakeoff/README.md).

Runs INSIDE the throwaway bake-off container started by launch.sh (no workspace, no ssh keys, no RAG index):
for each (model, task) it clones the task's prepared base (/bk/bases/<task>, history only up to the base commit),
runs the real Hermes CLI on it (`hermes chat --oneshot`, the production approvals with HERMES_CRON_SESSION=1, the
production deny list, provider_routing.data_collection=deny), then grades:
  command / exact  -> the suite's command, run in the clone after restoring protected setup files;
  rubric           -> left for judge.py (the answer, final text and diff are saved).
Every run is audited from its own transcript: a command that touched the network or a path outside its clone
disqualifies the run (reported, never silently dropped). Results: /bk/runs/<batch>/<model-slug>/<task>/result.json.
Re-running skips any pair that already has a result.json.

  python3 run.py --batch screen1 [--models a,b] [--tasks x,y] [--parallel 3]
  python3 run.py --preflight            # which OpenRouter models have a no-data-collection endpoint
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

BK = os.environ.get("BAKEOFF_ROOT", "/bk")
TOOLS = os.path.join(BK, "tools")
BASES = os.path.join(BK, "bases")
RUNS = os.path.join(BK, "runs")
HOME = os.environ.get("HERMES_HOME", os.path.join(BK, "home"))
HERMES = os.environ.get("BAKEOFF_HERMES", "/opt/hermes/.venv/bin/hermes")
WS_VENV = "/home/dant123/Projects/neural-simulator/.venv"      # mounted read-only at its real path
CATALOG = "/home/dant123/Projects/sim-catalog"                  # mounted read-only at its real path
OR_API = "https://openrouter.ai/api/v1"
SUITE = os.environ.get("BAKEOFF_SUITE", "suite.json")                 # a smoke suite can stand in
TOOLSETS = "terminal,file,todo,code_execution"                   # no memory/session_search (cross-run leakage), no web

PREAMBLE = """BAKE-OFF RULES. You are being evaluated as an autonomous agent on the neural-simulator repository.
Your working directory is a disposable clone of the repository at an earlier commit: work only inside it.
Do not use the network for anything (no curl, wget, pip, git fetch/pull/clone, ssh), and do not read files outside
this directory, except the Python environment through ./.venv. No human will answer questions: finish the task on
your own. If something blocks you, write what blocked you in BAKEOFF_ANSWER.md and stop. Commit only if the task
asks for a commit, and only through the normal gates (never --no-verify).

TASK:
"""

# Disqualifying: a command that reaches the network, or a path that could hold an answer (other runs, the bases, the
# Hermes data dir, the workspace). A bare URL inside text is only a warning (it can be quoted documentation).
NET_RE = re.compile(r"(?:^|[\s;&|(`$])(curl|wget|pip3?\s+install|uv\s+(?:pip|add|sync)|npm\s+(?:i|install)|apt(?:-get)?\s+install"
                    r"|ssh|scp|sftp|ncat|telnet)\b|\bgit\s+(?:fetch|pull|clone|ls-remote|push|remote\s+add)\b", re.I)
URL_RE = re.compile(r"https?://(?!localhost|127\.0\.0\.1)", re.I)
LEAKY_ROOTS = ("/bk/", "/opt/data", "/home/", "/root", "/srv", "/mnt")
ALLOWED_OUTSIDE = (WS_VENV, "/home/dant123/Projects/neural-simulator/.python", CATALOG)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def or_key():
    """The OpenRouter key from the bake-off HERMES_HOME's .env (bind-mounted read-only); never printed."""
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    try:
        for line in open(os.path.join(HOME, ".env")):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def or_request(path, body=None, timeout=120):
    key = or_key()
    if not key:
        raise RuntimeError("no OPENROUTER_API_KEY in the bake-off .env")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OR_API + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def key_usage():
    try:
        return float(or_request("/key")["data"]["usage"])
    except Exception:
        return None


def preflight(models):
    """One tiny request per OpenRouter model with data_collection=deny: eligible only if OpenRouter can route it."""
    out = {}
    for m in models:
        if m["provider"] != "openrouter":
            out[m["id"]] = {"eligible": True, "why": "local"}
            continue
        body = {"model": m["id"], "max_tokens": 8, "messages": [{"role": "user", "content": "Reply: ok"}],
                "provider": {"data_collection": "deny"}}
        try:
            or_request("/chat/completions", body, timeout=90)
            out[m["id"]] = {"eligible": True, "why": "routed with data_collection=deny"}
        except urllib.error.HTTPError as e:
            out[m["id"]] = {"eligible": False, "why": "HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:300])}
        except Exception as e:
            out[m["id"]] = {"eligible": False, "why": str(e)[:300]}
        print("%-40s %s" % (m["id"], "ELIGIBLE" if out[m["id"]]["eligible"] else "EXCLUDED: " + out[m["id"]]["why"]))
    return out


def slug(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def sh(cmd, cwd, timeout, env=None):
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + p.stderr)


def audit(transcript_path, run_dir):
    """Flag network use and reads/writes outside the run's own directory, from the run's tool calls."""
    flags, warns = [], []
    try:
        lines = open(transcript_path).read().splitlines()
    except OSError:
        return ["no transcript"], []
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") != "tool_use":
            continue
        inp = ev.get("input") or {}
        text = " ".join(str(v) for v in inp.values()) if isinstance(inp, dict) else str(inp)
        if ev.get("name") in ("terminal", "code_execution") and NET_RE.search(text):
            flags.append("network: " + text[:160])
        elif URL_RE.search(text):
            warns.append("url: " + text[:120])
        for path in re.findall(r"(/[A-Za-z0-9_.-]+(?:/[^\s'\"`;|&)<>]*)?)", text):
            if not path.startswith(LEAKY_ROOTS) or path.startswith(run_dir) or path.startswith(ALLOWED_OUTSIDE):
                continue
            flags.append("outside: " + path[:160])
    return sorted(set(flags)), sorted(set(warns))[:20]


def is_protected(task, rel):
    """Files the model must not be able to change to pass: the task's explicit `protect` list, else (implement
    tasks) the test files setup brought in. Setup's other files (e.g. the code to fix) are the model's to edit."""
    if task.get("protect") is not None:
        return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in task["protect"])
    base = os.path.basename(rel)
    return task.get("category") == "implement" and (rel.startswith("tests/") or "/tests/" in rel
                                                     or base.startswith("test_") or base == "conftest.py")


def restore_protected(task_id, repo, only=None):
    """Recreate setup's state in a fresh clone (only=None), or, before grading, put back just the files for which
    only(rel) is true."""
    src = os.path.join(BASES, task_id + ".protected")
    n = 0
    try:
        deleted = open(os.path.join(src, ".deleted")).read().splitlines()
    except OSError:
        deleted = []
    for rel in deleted:   # setup removed these from the base clone; mirror that inside this run's clone only
        if only is not None and not only(rel):
            continue
        target = os.path.realpath(os.path.join(repo, rel))
        if rel and target.startswith(os.path.realpath(repo) + os.sep) and os.path.isfile(target):
            os.unlink(target)
            n += 1
    for dp, _, fns in os.walk(src):
        for fn in fns:
            if fn in (".base_sha", ".deleted"):
                continue
            s = os.path.join(dp, fn)
            rel = os.path.relpath(s, src)
            if only is not None and not only(rel):
                continue
            d = os.path.join(repo, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    return n


def grade(task, repo, run_dir):
    g = task["grading"]
    out = {"type": g["type"]}
    if g["type"] == "rubric":
        out["pending"] = "judge.py"
        return out
    out["restored_protected"] = restore_protected(task["id"], repo, only=lambda rel: is_protected(task, rel))
    try:
        rc, text = sh(g["command"], repo, timeout=int(task.get("grade_timeout_s", 1500)))
    except subprocess.TimeoutExpired:
        rc, text = 124, "grader timed out"
    out["rc"] = rc
    out["output_tail"] = text[-3000:]
    if g["type"] == "command":
        out["pass"] = rc == 0
    else:
        got = (text.strip().splitlines() or [""])[-1].strip()
        out["got"] = got[:300]
        out["pass"] = rc == 0 and got.lower() == str(g.get("expected", "")).strip().lower()
    return out


def run_one(batch, model, task, budget_s, max_turns, lock):
    run_dir = os.path.join(RUNS, batch, slug(model["id"]), task["id"])
    res_path = os.path.join(run_dir, "result.json")
    if os.path.exists(res_path):
        return load_json(res_path)
    os.makedirs(run_dir, exist_ok=True)
    repo = os.path.join(run_dir, "repo")
    shutil.rmtree(repo, ignore_errors=True)
    # --shared: objects come from the read-only base through alternates (fast, nothing copied, base untouched).
    subprocess.run(["git", "clone", "-q", "--shared", os.path.join(BASES, task["id"]), repo], check=True)
    git = ["git", "-C", repo]
    subprocess.run(git + ["config", "core.hooksPath", "tools/githooks"], check=True)   # the production gates
    subprocess.run(git + ["config", "user.name", "Bake-off Agent"], check=True)
    subprocess.run(git + ["config", "user.email", "bakeoff@localhost"], check=True)
    restore_protected(task["id"], repo)                       # setup's own files (uncommitted in the base)
    # The post-setup state as a tree object (no commit, no hooks), so the saved diff shows only the model's work.
    subprocess.run(git + ["add", "-A"], check=True)
    setup_tree = subprocess.run(git + ["write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(git + ["reset", "-q"], check=True)
    os.symlink(WS_VENV, os.path.join(repo, ".venv"))
    os.symlink(CATALOG, os.path.join(run_dir, "sim-catalog"))  # biology gate: ~/Projects/sim-catalog = a sibling
    for h in (os.path.join(run_dir, "home"), os.path.join(HOME, "home")):
        # bases older than da0e80669 expand ~/Projects/sim-catalog through HOME (Hermes may set either HOME)
        os.makedirs(os.path.join(h, "Projects"), exist_ok=True)
        link = os.path.join(h, "Projects", "sim-catalog")
        if not os.path.lexists(link):
            os.symlink(CATALOG, link)
    base_sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prompt_path = os.path.join(run_dir, "prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(PREAMBLE + task["prompt"].strip() + "\n")

    cmd = [HERMES, "chat", "--query-file", prompt_path, "--oneshot", "-Q", "--format", "stream-json",
           "--max-turns", str(max_turns), "--run-budget", str(budget_s), "-t", TOOLSETS]
    if model["provider"] == "openrouter":
        cmd += ["-m", model["id"], "--provider", "openrouter"]
    env = dict(os.environ, HERMES_HOME=HOME, HERMES_CRON_SESSION="1", HOME=os.path.join(run_dir, "home"))
    t0 = time.time()
    transcript = os.path.join(run_dir, "transcript.jsonl")
    with open(transcript, "w") as tf:
        try:
            p = subprocess.run(cmd, cwd=repo, stdout=tf, stderr=subprocess.STDOUT, env=env, timeout=budget_s + 600)
            exit_code = p.returncode
        except subprocess.TimeoutExpired:
            exit_code = "hard-timeout"
    wall = time.time() - t0

    final = {}
    for ln in open(transcript):
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") == "result":
            final = ev
    flags, warns = audit(transcript, run_dir)
    answer = os.path.join(repo, "BAKEOFF_ANSWER.md")
    if os.path.exists(answer):
        shutil.copy2(answer, os.path.join(run_dir, "answer.md"))
    subprocess.run(git + ["add", "-A", "--", ".", ":!.venv"], capture_output=True)
    diff = subprocess.run(git + ["diff", "--cached", setup_tree, "--", ".", ":!.venv"], capture_output=True, text=True).stdout
    with open(os.path.join(run_dir, "diff.patch"), "w") as f:
        f.write(diff)
    commits = subprocess.run(git + ["log", "--oneline", base_sha + "..HEAD"], capture_output=True, text=True).stdout
    subprocess.run(git + ["reset", "-q"], capture_output=True)   # unstage only; the working tree stays as the model left it
    g = grade(task, repo, run_dir)
    result = {
        "batch": batch, "model": model["id"], "label": model.get("label"), "tier": model.get("tier"),
        "task": task["id"], "category": task.get("category"), "role": task.get("role"),
        "base_sha": base_sha, "exit_code": exit_code, "wall_s": round(wall, 1),
        "tokens": final.get("tokens"), "duration_ms": final.get("duration_ms"), "final_text": (final.get("text") or "")[-4000:],
        "commits": commits.strip().splitlines(), "diff_lines": diff.count("\n"),
        "audit_flags": flags, "audit_warnings": warns, "disqualified": bool(flags), "grade": g,
    }
    with open(res_path, "w") as f:
        json.dump(result, f, indent=1)
    shutil.rmtree(repo, ignore_errors=True)
    with lock:
        print("%s %-34s %-44s %s %s" % (time.strftime("%H:%M:%S"), model["id"][:34], task["id"][:44],
              ("PASS" if g.get("pass") else "fail") if "pass" in g else "rubric", "DQ" if flags else ""), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="screen1")
    ap.add_argument("--models", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--max-spend", type=float, default=3.0,
                    help="USD: no new paid run starts once this batch's real OpenRouter spend passes it (owner: as cheap as possible)")
    a = ap.parse_args()
    cfg = load_json(os.path.join(TOOLS, "models.json"))
    models = cfg["models"]
    if a.models:
        want = a.models.split(",")
        models = [m for m in models if m["id"] in want]
    if a.preflight:
        res = preflight(models)
        with open(os.path.join(RUNS, "preflight.json"), "w") as f:
            json.dump(res, f, indent=1)
        return
    pf_path = os.path.join(RUNS, "preflight.json")
    if os.path.exists(pf_path):
        pf = load_json(pf_path)
        dropped = [m["id"] for m in models if not pf.get(m["id"], {}).get("eligible", m["provider"] != "openrouter")]
        if dropped:
            print("skipping models without a no-data-collection endpoint (or not preflighted):", ", ".join(dropped))
        models = [m for m in models if m["id"] not in dropped]
    elif any(m["provider"] == "openrouter" for m in models):
        raise SystemExit("run --preflight first: no OpenRouter model runs without a data_collection=deny check")
    tasks = load_json(os.path.join(TOOLS, SUITE))["tasks"]
    if a.tasks:
        want = a.tasks.split(",")
        tasks = [t for t in tasks if t["id"] in want]
    os.makedirs(os.path.join(RUNS, a.batch), exist_ok=True)
    usage0 = key_usage()
    lock = threading.Lock()
    local = [(m, t) for m in models if m["provider"] != "openrouter" for t in tasks]
    remote = [(m, t) for t in tasks for m in models if m["provider"] == "openrouter"]

    def budget(t):
        return int(min(45, max(15, 3 * float(t.get("est_minutes") or 10))) * 60)

    def guarded(m, t):
        # Spend cap, checked against OpenRouter's own counter before every paid run (unknown usage = stop).
        now = key_usage()
        if usage0 is None or now is None or now - usage0 >= a.max_spend:
            with lock:
                print("SKIPPED (spend cap $%.2f reached or usage unknown): %s %s" % (a.max_spend, m["id"], t["id"]), flush=True)
            return None
        return run_one(a.batch, m, t, budget(t), a.max_turns, lock)

    with cf.ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        futs = [ex.submit(guarded, m, t) for m, t in remote]
        # the local model has one slot: its runs go one at a time, beside the remote ones
        def local_chain():
            return [run_one(a.batch, m, t, budget(t), a.max_turns, lock) for m, t in local]
        futs.append(ex.submit(local_chain))
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                with lock:
                    print("RUN ERROR:", repr(e)[:400], flush=True)
    usage1 = key_usage()
    with open(os.path.join(RUNS, a.batch, "key_usage.json"), "w") as f:
        json.dump({"before": usage0, "after": usage1,
                   "spent": (usage1 - usage0) if usage0 is not None and usage1 is not None else None}, f)
    print("batch done; OpenRouter spend this batch:", (usage1 - usage0) if usage0 is not None and usage1 is not None else "unknown")


if __name__ == "__main__":
    main()
