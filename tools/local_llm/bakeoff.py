#!/usr/bin/env python3
"""Local-LLM bake-off: which local model can actually work on THIS repo, on THIS machine, through Claude Code?

For each candidate profile it starts llama-server (tools/local_llm/profiles.json), then measures, with the desktop
running as it normally does:
  * VRAM: peak total GPU memory in use (desktop included) and what is left of the 24 GB card;
  * speed: generation tok/s on a short prompt, and prompt-processing + generation tok/s on a ~60K-token prompt;
  * long-context recall: a passphrase hidden half-way into that prompt must come back (catches the open llama.cpp
    long-context bugs for the Qwen3.8 architecture);
  * agentic work, through the real Claude Code CLI pointed at the local server (no Anthropic account involved), in a
    fresh detached worktree per task, each scored by an objective check, never by the model's own claim:
      T1 locate  - name the file that implements the delete guard and the test that pins it;
      T2 debug   - a planted bug makes tests/test_require_memcap_hook.py fail; fix it without editing the test;
      T3 extend  - add an ALLOW case to that test file and run the suite green.

It needs the whole GPU, so run it through the GPU queue:
    bash tools/gpu_queue.sh add 'cd /home/dant123/Projects/sim && python3 tools/local_llm/bakeoff.py'
Results: tools/local_llm/results/<profile>.json and tools/local_llm/results/summary.md
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools", "local_llm")
RESULTS = os.path.join(HERE, "results")
PORT = 8091
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(VENV_PY):
    # A linked worktree has no .venv of its own (a 2026-09-25 re-run from one failed every task on it): use the
    # primary checkout's, found through git's common dir.
    _common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    if _common:
        VENV_PY = os.path.join(os.path.dirname(_common), ".venv", "bin", "python")
PASSPHRASE = "ORCHID-7431-TANGERINE"


def claude_bin():
    found = shutil.which("claude")
    if found:
        return found
    base = os.path.expanduser("~/.config/Claude/claude-code")
    vers = sorted((v for v in os.listdir(base) if os.path.exists(os.path.join(base, v, "claude"))),
                  key=lambda s: [int(x) for x in re.findall(r"\d+", s)])
    return os.path.join(base, vers[-1], "claude")


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


class PeakVram:
    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib()[0])
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def http(path, payload=None, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(profile):
    cmd = ["llama-server", "-m", os.path.expanduser(profile["model"]), "--port", str(PORT), "--alias", "local",
           "-ngl", "99", "-np", "1", "-fa", "on", "-c", str(profile["ctx"]), "-ctk", profile["kv"], "-ctv", profile["kv"],
           "--jinja"]
    if profile.get("chat_template_file"):
        # A model's stock embedded template only merges a LEADING run of system messages and raises on
        # any other one; Claude Code sends leading system blocks AND later mid-conversation "system
        # reminders", which is exactly what broke every T1/T2/T3 agentic task in the bake-off (see
        # tools/local_llm/templates/). This profile-specific patched copy fixes that; see that
        # directory's *.orig.jinja for the unmodified original and test_templates_offline.py for the
        # offline render/diff check that must pass before this ever reaches the GPU.
        cmd += ["--chat-template-file", os.path.join(ROOT, profile["chat_template_file"])]
    cmd += profile.get("extra", [])
    log = open(os.path.join(RESULTS, profile["name"] + ".server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during load (see %s.server.log)" % profile["name"])
        try:
            if http("/health", timeout=5).get("status") == "ok":
                return proc, time.time() - t0, cmd
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready in 600 s")


def stop_server(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def long_prompt(target_tokens):
    """~target_tokens of real repo prose with a passphrase planted at 50% depth."""
    sources = ["GAP_CLOSURE_MISSION.md", "docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md", "docs/ENGINE_REFERENCE.md",
               "docs/project-history-archive.md", "research/FAILURE_LOG.md", "ROADMAP.md"]
    text = ""
    for s in sources:
        p = os.path.join(ROOT, s)
        if os.path.exists(p):
            text += "\n\n# FILE: %s\n\n" % s + open(p, errors="replace").read()
    text = text[: target_tokens * 6]              # ~4 chars/token; a margin, and never megabytes of text
    toks = http("/tokenize", {"content": text})["tokens"]
    toks = toks[:target_tokens]
    body = http("/detokenize", {"tokens": toks})["content"]
    mid = len(body) // 2
    cut = body.rfind("\n", 0, mid)
    cut = mid if cut < 0 else cut
    return (body[:cut] + "\n\nIMPORTANT NOTE FOR THE READER: the bake-off passphrase is %s.\n\n" % PASSPHRASE + body[cut:]
            + "\n\nQuestion: what is the bake-off passphrase stated in the note above? Answer with the passphrase only.")


def speed_and_context(profile):
    out = {}
    r = http("/completion", {"prompt": "Write a Python function that parses an ISO-8601 date string without using "
                                       "datetime, with a docstring and three doctests.", "n_predict": 400,
                             "temperature": 0, "cache_prompt": False})
    out["short_gen_tok_s"] = round(r["timings"]["predicted_per_second"], 1)
    prompt = long_prompt(profile.get("long_tokens", 60000))
    with PeakVram() as pv:
        r = http("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                                          "temperature": 0, "cache_prompt": False,
                                          "chat_template_kwargs": {"enable_thinking": False}}, timeout=3600)
    t = r.get("timings", {})
    msg = r["choices"][0]["message"]
    answer = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    out.update(long_prompt_tokens=t.get("prompt_n"), long_pp_tok_s=round(t.get("prompt_per_second", 0), 1),
               long_gen_tok_s=round(t.get("predicted_per_second", 0), 1), long_answer=answer.strip()[:200],
               long_recall_ok=PASSPHRASE in answer, vram_peak_mib_long=pv.peak)
    return out


def worktree(tag):
    path = os.path.join(ROOT, ".claude", "worktrees", "llm-bakeoff-" + tag)
    if os.path.exists(path):
        subprocess.run(["git", "worktree", "remove", "--force", path], cwd=ROOT)
    subprocess.run(["git", "worktree", "add", "-q", "--detach", path, "HEAD"], cwd=ROOT, check=True)
    return path


def run_claude(cwd, prompt, max_turns=30, timeout=1500):
    env = dict(os.environ, ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % PORT, ANTHROPIC_AUTH_TOKEN="local",
               ANTHROPIC_MODEL="local", ANTHROPIC_SMALL_FAST_MODEL="local", ANTHROPIC_DEFAULT_HAIKU_MODEL="local",
               ANTHROPIC_DEFAULT_SONNET_MODEL="local", ANTHROPIC_DEFAULT_OPUS_MODEL="local",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1", API_TIMEOUT_MS="1200000",
               SIM_NO_PROVENANCE="1")
    for k in ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(k, None)
    t0 = time.time()
    try:
        p = subprocess.run([claude_bin(), "-p", prompt, "--output-format", "json", "--max-turns", str(max_turns),
                            "--dangerously-skip-permissions", "--disallowedTools",
                            "Bash(git push:*)", "Bash(git commit:*)", "Bash(git merge:*)", "Bash(git checkout:*)",
                            "Bash(git reset:*)", "Bash(bash tools/push_both.sh:*)"],
                           cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=timeout)
        raw = p.stdout.strip()
        try:
            j = json.loads(raw.splitlines()[-1]) if raw else {}
        except Exception:
            j = {"unparsed": raw[-2000:]}
        return {"wall_s": round(time.time() - t0), "exit": p.returncode, "num_turns": j.get("num_turns"),
                "is_error": j.get("is_error"), "result": str(j.get("result", j.get("unparsed", "")))[-1500:],
                "stderr_tail": p.stderr[-800:]}
    except subprocess.TimeoutExpired:
        return {"wall_s": round(time.time() - t0), "exit": "timeout", "result": "", "is_error": True}


def pytest_ok(cwd, rel):
    p = subprocess.run([VENV_PY, "-m", "pytest", "-q", rel], cwd=cwd, capture_output=True, text=True,
                       env=dict(os.environ, SIM_NO_PROVENANCE="1"), timeout=600)
    return p.returncode == 0, p.stdout.strip().splitlines()[-1:] if p.stdout else []


def git_diff_names(cwd):
    return subprocess.run(["git", "diff", "--name-only"], cwd=cwd, capture_output=True, text=True).stdout.split()


def tasks(profile):
    res = {}
    # T1 locate
    wt = worktree(profile["name"] + "-t1")
    r = run_claude(wt, "Which file in this repository implements the PreToolUse guard that blocks deletes aimed at "
                       "~/.claude, and which test file pins it? Reply with just the two repository-relative paths, "
                       "one per line, and nothing else.", max_turns=15)
    r["pass"] = (".claude/hooks/guard_protected_delete.py" in r["result"]
                 and "tests/test_guard_protected_delete.py" in r["result"])
    res["T1_locate"] = r
    subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT)
    # T2 debug a planted bug
    wt = worktree(profile["name"] + "-t2")
    hook = os.path.join(wt, ".claude", "hooks", "require_memcap_for_brain_builds.py")
    src = open(hook).read()
    planted = src.replace("--collect-only|--selftest|", "--collect-only|", 1)
    assert planted != src, "the planted bug did not apply"
    open(hook, "w").write(planted)
    test_path = os.path.join(wt, "tests", "test_require_memcap_hook.py")
    test_before = open(test_path).read()
    assert not pytest_ok(wt, "tests/test_require_memcap_hook.py")[0], "the planted bug did not make the suite fail"
    r = run_claude(wt, "The test suite tests/test_require_memcap_hook.py is failing. Find the cause in the code under "
                       "test and fix it. Do NOT edit the test file. Run the suite with `SIM_NO_PROVENANCE=1 %s -m "
                       "pytest -q tests/test_require_memcap_hook.py` to confirm it passes, then reply with the one-line "
                       "pytest summary." % VENV_PY)
    ok, tail = pytest_ok(wt, "tests/test_require_memcap_hook.py")
    test_untouched = open(test_path).read() == test_before
    r.update({"pass": ok and test_untouched, "pytest": tail, "test_untouched": test_untouched,
              "changed": git_diff_names(wt)})
    res["T2_debug"] = r
    subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT)
    # T3 extend a test file
    wt = worktree(profile["name"] + "-t3")
    case = ".venv/bin/python -m research.runners.load_bearing_fraction --score results.json"
    r = run_claude(wt, "In tests/test_require_memcap_hook.py add one new entry to the ALLOW list for the command "
                       "`%s` (a scoring run that builds no brain), then run `SIM_NO_PROVENANCE=1 %s -m pytest -q "
                       "tests/test_require_memcap_hook.py` and reply with the one-line pytest summary." % (case, VENV_PY))
    ok, tail = pytest_ok(wt, "tests/test_require_memcap_hook.py")
    added = case in open(os.path.join(wt, "tests", "test_require_memcap_hook.py")).read()
    r.update({"pass": ok and added, "pytest": tail, "case_added": added, "changed": git_diff_names(wt)})
    res["T3_extend"] = r
    subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT)
    return res


def run_profile(profile):
    rec = {"profile": profile["name"], "model": profile["model"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    used0, total = gpu_used_mib()
    rec["vram_before_mib"], rec["vram_total_mib"] = used0, total
    proc = None
    try:
        with PeakVram() as pv:
            proc, load_s, cmd = start_server(profile)
            rec.update(load_s=round(load_s, 1), server_cmd=" ".join(cmd), vram_after_load_mib=gpu_used_mib()[0])
            rec.update(speed_and_context(profile))
            rec["tasks"] = tasks(profile)
        rec["vram_peak_mib"] = pv.peak
        rec["vram_headroom_mib"] = total - pv.peak
    except Exception as exc:
        rec["error"] = repr(exc)
    finally:
        stop_server(proc)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(rec, open(os.path.join(RESULTS, profile["name"] + ".json"), "w"), indent=1)
    return rec


def summarize(recs):
    ctx_of = {p["name"]: p for p in json.load(open(os.path.join(HERE, "profiles.json")))}
    lines = ["| profile | ctx | load s | peak VRAM (MiB, desktop incl.) | headroom | short gen tok/s | long prompt tokens | "
             "long prompt tok/s | long gen tok/s | recall @long | T1 locate | T2 debug | T3 extend |", "|" + "---|" * 13]
    for r in recs:
        prof = ctx_of.get(r["profile"], {})
        t = r.get("tasks", {})
        cell = lambda k: ("PASS" if t.get(k, {}).get("pass") else "fail") + " (%ss)" % t.get(k, {}).get("wall_s", "?")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["profile"], prof.get("ctx"), r.get("load_s"), r.get("vram_peak_mib"), r.get("vram_headroom_mib"), r.get("short_gen_tok_s"),
            prof.get("long_tokens", 60000), r.get("long_pp_tok_s"), r.get("long_gen_tok_s"), r.get("long_recall_ok"), cell("T1_locate"),
            cell("T2_debug"), cell("T3_extend")) + ("  ERROR: %s" % r["error"] if r.get("error") else ""))
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", help="profile names from profiles.json (default: all)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    profiles = json.load(open(os.path.join(HERE, "profiles.json")))
    if a.profiles:
        profiles = [p for p in profiles if p["name"] in a.profiles]
    for p in profiles:
        run_profile(p)
    # summarize EVERY profile that has a result on disk, so a partial re-run (--profiles X) keeps the earlier rows
    # beside it instead of overwriting summary.md with only this run's profiles
    recs = []
    for p in json.load(open(os.path.join(HERE, "profiles.json"))):
        f = os.path.join(RESULTS, p["name"] + ".json")
        if os.path.exists(f):
            recs.append(json.load(open(f)))
    summarize(recs)


if __name__ == "__main__":
    sys.exit(main())
