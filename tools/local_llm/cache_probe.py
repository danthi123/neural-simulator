#!/usr/bin/env python3
"""cache_probe.py -- measure llama-server PROMPT-CACHE reuse across a real multi-turn Claude Code session, for
several server CONFIGS, to find why `llm.sh claude` reprocesses the whole prompt every turn.

BACKGROUND (research/local-llm-prompt-cache branch, 2026-09-25). A live `llm claude -p` session against the
default profile (qwen38-27b-iq4nl-mtp-128k-q4: Qwen3.8-27B IQ4_NL, a HYBRID architecture -- 48 of 64 layers Gated
DeltaNet linear-attention with a fixed-size recurrent state, 16 full-attention layers -- plus MTP speculative
decoding) showed every request re-processing the ENTIRE prompt (`prompt processing, n_tokens = N, progress =
1.00` at t ~= N / 950 tok/s), with "selected slot by LCP similarity, sim_best" FALLING each turn (0.62 -> 0.23,
f_keep 0.71 -> 0.40). Three hypotheses, NOT assumed true:
  (a) Claude Code's own side requests (small-fast-model calls sharing -np 1's one slot) evict the main
      conversation's cached prefix between turns;
  (b) the hybrid recurrent layers cannot roll back to an arbitrary prefix -- only to a context CHECKPOINT
      (`-ctxcp`/`-cms`), which may be too sparse;
  (c) something early in the prompt changes every turn, so even an exact-prefix cache would not help.

METHOD. For each CONFIG this starts its OWN llama-server (same model/template/ctx as the default profile, one
knob changed at a time), runs the SAME short real Claude Code task through it (a throwaway git worktree, the
same ANTHROPIC_* env + context-trim flags `llm.sh claude` uses), then parses the server's own per-request log
lines -- llama-server logs `slot <name>: id <N> | task <T> | ...` lines whose exact formats are read verbatim
from the shipped binary (see `_LINE_RE` / `_EVENT_RES` below; verified 2026-09-25 against
/usr/lib/libllama-server-impl.so version 10042) -- for tokens ACTUALLY reprocessed vs the prompt size submitted,
peak VRAM, and total wall time. `--selftest-parse` proves the parser against embedded synthetic log lines,
including a NEGATIVE case, without touching a GPU.

Needs the whole card -- run through the GPU queue, never standalone:
    bash tools/gpu_queue.sh add 'cd <this worktree, absolute> && python3 tools/local_llm/cache_probe.py'

Results: tools/local_llm/results/cache_probe.json (raw) + tools/local_llm/results/cache_probe.md (table + verdict).
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
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "tools", "local_llm")
RESULTS = os.path.join(HERE, "results")
LOGDIR = os.path.join(RESULTS, "cache_probe_logs")
PORT = int(os.environ.get("CACHE_PROBE_PORT", "8093"))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(VENV_PY):
    # A linked worktree has no .venv of its own -- fall back to the primary checkout's (see bakeoff.py, same fix).
    _common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    if _common:
        VENV_PY = os.path.join(os.path.dirname(_common), ".venv", "bin", "python")

# The task run through each config's server. Deliberately mirrors the bakeoff.py example in spirit (a short,
# read-only, multi-tool-call task) but touches enough real files across several turns to build up a non-trivial,
# GROWING shared prefix -- exactly the shape a prompt cache needs to pay off on.
DEFAULT_TASK = (
    "Read tools/local_llm/llm.sh and tools/gpu_queue.sh, then in 5 bullet points explain how the LOCAL-LLM "
    "AUTOSWAP between them works. Then run `bash tools/status.sh` and summarize its output in 2 lines. Do not "
    "edit any files."
)
DEFAULT_MAX_TURNS = 8
DEFAULT_TIMEOUT_S = 900

# ---------------------------------------------------------------------------------------------------------------
# CONFIGS. Each is (name, np, extra_flags) layered on top of the default profile's own base command (model, ctx,
# kv quant, chat template, sampling/speculative-decoding flags) -- so the ONLY thing that differs between configs
# is the cache/slot knobs under test, never the model or generation settings.
#
# B/C/D explicitly pass -kvu: --kv-unified's OWN default is "enabled if number of slots is auto" (llama-server
# --help, verbatim) -- since every config here passes -np explicitly (never -1/auto), kv-unified would NOT turn
# itself on by default for -np 2, and WITHOUT it llama-server splits the total -c budget evenly across slots
# (halving the main conversation's usable context to ~65536 for a 2-slot server) -- exactly the "same total ctx"
# constraint the task calls for. -kvu keeps a single shared buffer so either slot can use close to the full
# ctx_size on demand.
CONFIGS = [
    {"name": "A_np1_baseline", "np": 1, "extra": [],
     "note": "today's production flags, unchanged (profile_cmd()/bakeoff.py start_server())"},
    {"name": "B_np2_kvu", "np": 2, "extra": ["-kvu"],
     "note": "a second slot for Claude Code's small-fast-model side calls, so they never evict the main "
             "conversation's slot (tests hypothesis a)"},
    {"name": "C_np2_kvu_denser_checkpoints", "np": 2, "extra": ["-kvu", "-ctxcp", "32", "-cms", "1024"],
     "note": "B + checkpoints every >=1024 tokens instead of the 8192 default, so a hybrid-model rollback lands "
             "closer to the actual divergence point (tests hypothesis b)"},
    {"name": "D_np2_kvu_checkpoints_cram16g", "np": 2, "extra": ["-kvu", "-ctxcp", "32", "-cms", "1024", "-cram", "16384"],
     "note": "C + doubled cache-ram (8192 -> 16384 MiB) so idle-slot caching/checkpoint storage is not the "
             "limiting factor"},
    {"name": "E_np1_cache_reuse", "np": 1, "extra": ["--cache-reuse", "256"],
     "note": "A + --cache-reuse (min chunk 256 tokens; default 0 = OFF in every other config here). Isolates "
             "whether the plain prefix-match path even ATTEMPTS a KV trim/checkpoint-restore for this hybrid "
             "model at all when explicitly told to, independent of -np/-kvu."},
]


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
            try:
                self.peak = max(self.peak, gpu_used_mib()[0])
            except Exception:
                pass
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def load_default_profile():
    profiles = json.load(open(os.path.join(HERE, "profiles.json")))
    name = open(os.path.join(HERE, "default_profile")).read().strip()
    for p in profiles:
        if p["name"] == name:
            return p
    sys.exit("default_profile %r not found in profiles.json" % name)


def build_cmd(profile, np, extra_flags):
    """The base command is byte-for-byte the same shape as llm.sh's profile_cmd() / bakeoff.py's start_server():
    same model, ctx, kv quant, chat template, and the profile's own sampling/speculative-decoding flags. Only
    `-np` and the config's extra cache/slot flags differ."""
    cmd = ["llama-server", "-m", os.path.expanduser(profile["model"]), "--host", "127.0.0.1", "--port", str(PORT),
           "--alias", "local", "-ngl", "99", "-np", str(np), "-fa", "on", "-c", str(profile["ctx"]),
           "-ctk", profile["kv"], "-ctv", profile["kv"], "--jinja"]
    if profile.get("chat_template_file"):
        cmd += ["--chat-template-file", os.path.join(ROOT, profile["chat_template_file"])]
    cmd += profile.get("extra", [])
    cmd += extra_flags
    return cmd


def http_health(timeout=5):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def start_server(cmd, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during load (see %s)" % log_path)
        if http_health():
            return proc, time.time() - t0
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready in 600 s (see %s)" % log_path)


def stop_server(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def worktree(tag):
    path = os.path.join(ROOT, ".claude", "worktrees", "cache-probe-" + tag)
    if os.path.exists(path):
        subprocess.run(["git", "worktree", "remove", "--force", path], cwd=ROOT)
    subprocess.run(["git", "worktree", "add", "-q", "--detach", path, "HEAD"], cwd=ROOT, check=True)
    return path


def remove_worktree(path):
    subprocess.run(["git", "worktree", "remove", "--force", path], cwd=ROOT, capture_output=True)


def run_claude_task(cwd, prompt, ctx_tokens, max_turns=DEFAULT_MAX_TURNS, timeout=DEFAULT_TIMEOUT_S):
    """Same ANTHROPIC_* env + the `llm.sh claude` / `LLM_CLAUDE_FULL=0` context-trim flags (--strict-mcp-config,
    the bio-research-plugin-off --settings, --disallowedTools WebSearch ReportFindings), plus
    --dangerously-skip-permissions and a small git-safety disallow list so an unattended run in a throwaway
    worktree can finish without a human approving each tool call (the same non-interactive pattern
    tools/local_llm/bakeoff.py already uses against this exact server)."""
    env = dict(os.environ, ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % PORT, ANTHROPIC_AUTH_TOKEN="local",
               ANTHROPIC_API_KEY="", ANTHROPIC_MODEL="local", ANTHROPIC_SMALL_FAST_MODEL="local",
               ANTHROPIC_DEFAULT_HAIKU_MODEL="local", ANTHROPIC_DEFAULT_SONNET_MODEL="local",
               ANTHROPIC_DEFAULT_OPUS_MODEL="local", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_AUTOUPDATER="1", API_TIMEOUT_MS="1200000", SIM_NO_PROVENANCE="1",
               CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(ctx_tokens))
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_OAUTH_TOKEN"):
        env.pop(k, None)
    cmd = [claude_bin(), "-p", prompt, "--output-format", "json", "--max-turns", str(max_turns),
           "--dangerously-skip-permissions", "--strict-mcp-config",
           "--settings", '{"enabledPlugins":{"bio-research@inline":false}}',
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


# ---------------------------------------------------------------------------------------------------------------
# Server-log parsing. Format strings VERIFIED against this machine's actual llama-server output (llama.cpp-cuda
# b10042/50b29f6, verbosity=3 default -- NOT the "new prompt, ..., task.n_tokens = %d" string `strings` found in
# libllama-server-impl.so, which turned out to be dead in this build/verbosity for our workload: it never once
# appeared across any of the real probe logs). What actually appears, per real request, in order:
#   "slot get_availabl: id <slot> | task -1 | selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f"
#     (or "... by LRU, t_last = %d" for the very first request, when there is nothing yet to compare against)
#   "slot launch_slot_: id <slot> | task <N> | processing task, is_child = 0"          <- marks a real request START
#   "slot print_timing: id <slot> | task <N> | prompt processing, n_tokens = %d, progress = %.2f, t = %.2f s / %.2f tokens per second"  (repeated, growing)
#   "slot print_timing: id <slot> | task <N> | prompt eval time = %.2f ms / %d tokens (...)"   <- AUTHORITATIVE reprocessed-token count for this request
#   "slot print_timing: id <slot> | task <N> |        eval time = %.2f ms / %d tokens (...)"   <- generated-token count (note: no "prompt " prefix)
#   "slot      release: id <slot> | task <N> | stop processing: n_tokens = %d, truncated = %d" <- total context length (prompt + generated) once done
# requested_tokens (the FULL prompt length submitted this turn, cached or not) = release_n_tokens - generated_tokens;
# reprocessed_tokens = the "prompt eval time" count (llama.cpp's own prompt_eval counter already EXCLUDES whatever
# was actually skipped via cache reuse -- confirmed by construction: for the first-ever request on a fresh slot,
# with nothing to reuse, requested_tokens == reprocessed_tokens exactly). reused_frac = 1 - reprocessed/requested.
# A `get_availabl` selection line always immediately PRECEDES the `launch_slot_` line for the request it decided,
# and both carry task=-1 (not yet assigned) -- correlated here purely by ORDER (buffer the most recent selection,
# attach it to the next `launch_slot_`), since the log gives no other shared key between them.
# ---------------------------------------------------------------------------------------------------------------
_LINE_RE = re.compile(r"slot\s+(\S+)\s*:\s*id\s*(\d+)\s*\|\s*task\s*(-?\d+)\s*\|\s*(.*)$")
_EVENT_RES = {
    "sim_selected": re.compile(
        r"selected slot by LCP similarity, sim_best\s*=\s*([\d.]+) \(>\s*([\d.]+) thold\), f_keep\s*=\s*([\d.]+)"),
    "lru_selected": re.compile(r"selected slot by LRU, t_last\s*=\s*(-?\d+)"),
    "launch": re.compile(r"processing task, is_child\s*=\s*(\d+)"),
    "prompt_eval": re.compile(r"^prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"),
    "gen_eval": re.compile(r"^eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"),
    "release": re.compile(r"stop processing: n_tokens\s*=\s*(\d+), truncated\s*=\s*(\d+)"),
    "checkpoint_created": re.compile(r"created context checkpoint"),
    "checkpoint_restored": re.compile(r"restored context checkpoint .*n_past\s*=\s*(\d+)"),
    "context_reuse": re.compile(r"after context reuse, new n_past\s*=\s*(\d+)"),
}


def parse_server_log(text):
    """Returns (requests, n_checkpoints_created). `requests` is a list of per-request dicts, one per real
    generation request (a `launch_slot_`/"processing task" event), in arrival order. Each dict: task_id, slot_id,
    requested_tokens, reprocessed_tokens, generated_tokens, reused_frac (None if requested_tokens could not be
    determined), selection ("lcp"/"lru"/None), sim_best, f_keep, n_past_after_reuse (from context-reuse/
    checkpoint-restore, if any -- None when neither ever fired). `n_checkpoints_created` counts how many times the
    server actually created a context checkpoint during the run -- if it never did, `-ctxcp`/`-cms` had nothing to
    act on regardless of their values."""
    pending_selection = None   # (kind, sim_best, f_keep) from the most recent task=-1 "get_availabl" line
    by_task = {}
    order = []
    cur = None   # the record currently accumulating prompt_eval/gen_eval/release lines
    n_checkpoints_created = 0
    for line in text.splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        tag, slot_id, task_id, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if _EVENT_RES["checkpoint_created"].search(rest):
            n_checkpoints_created += 1
            continue
        if task_id == -1:
            m2 = _EVENT_RES["sim_selected"].search(rest)
            if m2:
                pending_selection = ("lcp", float(m2.group(1)), float(m2.group(3)))
                continue
            m2 = _EVENT_RES["lru_selected"].search(rest)
            if m2:
                pending_selection = ("lru", None, None)
                continue
            continue
        m2 = _EVENT_RES["launch"].search(rest)
        if m2:
            sel = pending_selection or (None, None, None)
            pending_selection = None
            cur = {"task_id": task_id, "slot_id": slot_id, "selection": sel[0], "sim_best": sel[1], "f_keep": sel[2],
                   "requested_tokens": None, "reprocessed_tokens": None, "generated_tokens": None,
                   "pp_time_s": None, "gen_time_s": None, "n_past_after_reuse": None}
            by_task[task_id] = cur
            order.append(task_id)
            continue
        rec = by_task.get(task_id)
        if rec is None:
            continue
        m2 = _EVENT_RES["prompt_eval"].search(rest)
        if m2:
            rec["pp_time_s"] = round(float(m2.group(1)) / 1000.0, 2)
            rec["reprocessed_tokens"] = int(m2.group(2))
            continue
        m2 = _EVENT_RES["gen_eval"].search(rest)
        if m2:
            rec["gen_time_s"] = round(float(m2.group(1)) / 1000.0, 2)
            rec["generated_tokens"] = int(m2.group(2))
            continue
        m2 = _EVENT_RES["release"].search(rest)
        if m2:
            total = int(m2.group(1))
            if rec["generated_tokens"] is not None:
                rec["requested_tokens"] = total - rec["generated_tokens"]
            continue
        m2 = _EVENT_RES["context_reuse"].search(rest)
        if m2:
            rec["n_past_after_reuse"] = int(m2.group(1))
            continue
        m2 = _EVENT_RES["checkpoint_restored"].search(rest)
        if m2:
            rec["n_past_after_reuse"] = int(m2.group(1))
            continue
    out = []
    for tid in order:
        rec = by_task[tid]
        req, reproc = rec["requested_tokens"], rec["reprocessed_tokens"]
        if req and req > 0 and reproc is not None:
            rec["reused_frac"] = round(max(0.0, min(1.0, 1 - reproc / req)), 3)
        else:
            rec["reused_frac"] = None
        out.append(rec)
    return out, n_checkpoints_created


def summarize_requests(reqs, n_checkpoints_created=0):
    if not reqs:
        return {"n_requests": 0, "checkpoints_created": n_checkpoints_created}
    scored = [r for r in reqs if r["requested_tokens"] and r["reprocessed_tokens"] is not None]
    total_req = sum(r["requested_tokens"] for r in scored)
    total_reproc = sum(r["reprocessed_tokens"] for r in scored)
    total_pp_s = sum(r["pp_time_s"] or 0.0 for r in reqs)
    sizes = [r["requested_tokens"] for r in reqs]
    monotonic = all((b or 0) >= (a or 0) for a, b in zip(sizes, sizes[1:]))
    return {
        "n_requests": len(reqs), "n_scored_requests": len(scored),
        "total_requested_tokens": total_req, "total_reprocessed_tokens": total_reproc,
        "overall_reused_frac": round(max(0.0, min(1.0, 1 - total_reproc / total_req)), 3) if total_req else None,
        "total_pp_time_s": round(total_pp_s, 1),
        "last_request_reused_frac": reqs[-1]["reused_frac"], "last_request_sim_best": reqs[-1]["sim_best"],
        "last_request_f_keep": reqs[-1]["f_keep"], "prompt_sizes_monotonic_nondecreasing": monotonic,
        "prompt_sizes": sizes, "checkpoints_created": n_checkpoints_created,
    }


# ---------------------------------------------------------------------------------------------------------------
def run_config(cfg, profile, task, max_turns, timeout, keep_worktree=False):
    print("=== config %s: np=%d extra=%s ===" % (cfg["name"], cfg["np"], cfg["extra"]), flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    log_path = os.path.join(LOGDIR, cfg["name"] + ".server.log")
    cmd = build_cmd(profile, cfg["np"], cfg["extra"])
    rec = {"config": cfg["name"], "np": cfg["np"], "extra": cfg["extra"], "note": cfg["note"],
           "server_cmd": " ".join(cmd), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    used0, total_vram = gpu_used_mib()
    rec["vram_before_mib"] = used0
    proc = None
    wt = None
    try:
        with PeakVram() as pv:
            proc, load_s = start_server(cmd, log_path)
            rec["load_s"] = round(load_s, 1)
            rec["vram_after_load_mib"] = gpu_used_mib()[0]
            wt = worktree(cfg["name"])
            rec["claude"] = run_claude_task(wt, task, ctx_tokens=profile["ctx"], max_turns=max_turns, timeout=timeout)
        rec["vram_peak_mib"] = pv.peak
        rec["vram_headroom_mib"] = total_vram - pv.peak
        server_log_text = open(log_path, errors="replace").read()
        reqs, n_ckpt = parse_server_log(server_log_text)
        rec["requests"] = reqs
        rec["summary"] = summarize_requests(reqs, n_ckpt)
    except Exception as exc:
        rec["error"] = repr(exc)
    finally:
        stop_server(proc)
        if wt and not keep_worktree:
            remove_worktree(wt)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return rec


# Static write-up appended after the (reproducible, script-generated) table by write_markdown() -- kept here so
# `--reparse` regenerates the file WITHOUT losing the analysis. Update this alongside the CONFIGS list if a new
# config changes the verdict.
VERDICT_MD = """
## ROUND 1 (server-flag sweep): 0% reuse on all five configs -- but the wrong layer was blamed

Every one of the five configs above (`-np 1` baseline, `-np 2` with `-kvu`, `-kvu` plus denser context
checkpoints, + doubled `-cram`, and `-np 1` with `--cache-reuse` explicitly enabled) produced **0% prompt-cache
reuse across turns** and **created zero context checkpoints**, for the exact same real, growing, single-thread
Claude Code conversation. This is not "no common prefix to reuse": in every config, the request-2 (or -3)
`selected slot by LCP similarity` line reports `f_keep` of 0.39-0.73 -- llama-server's own slot-selection
heuristic correctly DETECTS that a large fraction of the previously-cached prompt is still a valid prefix of the
new, longer one. Despite that, the subsequent `prompt eval time = ... / N tokens` line for the same request
shows N within a few tokens of the FULL new prompt length every single time.

**ROUND 1's conclusion -- "this is an upstream llama-server hybrid-model limitation, nothing to fix" -- was
WRONG, and is corrected by round 2 below on a coordinator challenge that it rested on an untested, inferred
cause rather than a real captured request.** Round 1 never captured or rendered a real request; it inferred the
mechanism from server-side log symptoms alone. It was right that llama-server's checkpoint/`-ctxcp`/`-cms`/
`-cram`/`--cache-reuse` machinery never engaged (still true, see round 2) -- but wrong about why the prefix was
unusable in the first place. The actual cause was in OUR OWN chat template, entirely fixable, and fixing it took
prompt-cache reuse on this exact model from 0% to 60.6% overall (98.6% on the largest turn) with NO server-flag
changes at all.

## ROUND 2 (coordinator directive, real captured requests): the chat template was hoisting mid-conversation system messages

**Method.** Captured the raw Anthropic `/v1/messages` request bodies of a real 3-turn `claude -p` session via a
logging reverse proxy (`tools/local_llm/capture_requests.py` + `tools/local_llm/capture_session.py`), then
rendered pairs of them through the live chat template via llama-server's own `/apply-template` endpoint
(`tools/local_llm/render_and_diff.py`) to find the exact first character where two consecutive requests'
rendered prompts diverge. Full captures and diffs: `tools/local_llm/results/template_divergence/`.

**Finding.** Claude Code sends its per-turn "system reminders" (a live `<total_tokens>N tokens left</total_tokens>`
line, refreshed every turn; a stable session-start reminder) as literal `role: "system"` entries embedded
directly in the `messages` array, not just in the leading `system` field -- confirmed directly from the captured
JSON (`capture_old_template/0003_POST_v1_messages?beta=true.json`, message indices 1 and 4). The
round-1 chat-template fix (`tools/local_llm/templates/qwen38-27b-iq4nl-mtp.jinja`, "LOCAL FIX 2026-09-25") merged
EVERY such message, wherever it appeared, into the ONE leading system block to stop a real
"System message must be at the beginning" failure. That fix worked for correctness but broke caching: each new
turn's fresh reminder text changes the CONTENT of that leading block, so the leading-block/conversation-turns
BOUNDARY shifts by a few bytes every turn, and everything after it -- the entire rest of the conversation, even
though byte-identical -- counts as changed. Measured directly: rendering turn 1 and turn 2 of the SAME real
session through the round-1 template diverges at char 62857 of turn 1's 82848-char prompt (75.9% in), right at
that exact boundary (`diff_old_template_turn1_vs_turn2.json`).

**Fix.** `tools/local_llm/templates/qwen38-27b-iq4nl-mtp.jinja` ("LOCAL FIX 2026-09-25 ROUND 2"): only the
LEADING contiguous run of system/developer messages is merged into the one stable leading block now (as the
pre-round-1 template did for a single message); every LATER system/developer message renders IN PLACE, as its
own `<|im_start|>system ... <|im_end|>` turn at its own position, and the main loop skips only the leading run
by index rather than every system-role message by role. Never raises (the original bug stays fixed). Verified
offline (no GPU) in `tools/local_llm/templates/test_templates_offline.py::check_prefix_stability_qwen`, which
would fail against the round-1 template (confirmed: divergence at char 366/849 in the synthetic fixture) and
passes against the fix.

**Re-measured on the SAME real captured requests, through the FIXED template:** turn 1's entire 82877-char
rendered prompt is now a byte-for-byte PREFIX of turn 2's 144447-char prompt -- divergence at char 82877, i.e.
100.0% of turn 1 (`diff_new_template_turn1_vs_turn2.json`). **Re-measured end-to-end with a real llama-server
+ a real 3-turn Claude Code session** (`A_np1_baseline_ROUND1TEMPLATE` vs `A_np1_baseline_ROUND2TEMPLATE` in
the table above, same `-np 1` config, template swapped): overall reused fraction 0.0 -> **0.606**, last-turn
reused fraction 0.0 -> **0.986**, total prompt-processing time 107.9s -> **46.2s** for a slightly LARGER
conversation. Context checkpoints created: 0 in both -- expected and fine, not a regression: with a byte-exact
prefix the server needs a plain forward CONTINUATION (pick up decoding where the previous turn's cache already
ends), which every architecture supports natively; a checkpoint-based REWIND (round 1's target) is only needed
when the divergence point is somewhere back in the MIDDLE of the cache, which no longer happens here.

**Hypothesis 2 (assistant turns re-rendered differently from what was generated) was checked directly and
ruled out as a contributing factor**, not merely assumed away: the model's actual streamed "thinking" content
(from the captured SSE response) is byte-identical to what Claude Code resends as history in the next request
(verified on `capture_old_template/0001_...json.response` vs `0003_...json` message index 2), and the template
renders historical assistant turns through the exact same formatting code path used for live generation, so
there is no additional divergence source here to fix.

**Hypothesis (a)** (interleaved small-fast-model side calls) remains unreproduced under
`--dangerously-skip-permissions` (needed for unattended runs) -- every capture shows exactly as many
`/v1/messages` requests as conversation turns, no interleaved extra calls. Not needed to explain the bug either
way: the fix above fully explains and resolves the measured symptom without it.

## Chosen config: `A_np1_baseline` + the FIXED chat template (no llama-server flag changes)

The server-flag sweep (round 1: `-np`/`-kvu`/`-ctxcp`/`-cms`/`-cram`/`--cache-reuse`) is still valid as a
NEGATIVE result on its own terms -- none of those flags move the needle, and `B`/`D` (the `-kvu` multi-slot
variants) cost more peak VRAM for it. The actual fix was the chat template, requires no `-np`/`-kvu` change, and
is already the profile's own template file, so `tools/local_llm/llm.sh`'s `profile_cmd()` and
`tools/local_llm/bakeoff.py`'s `start_server()` keep `-np 1` with no extra flags -- comments at each now point
to this corrected history instead of the retracted round-1 conclusion.

**Agentic-task regression check** (`python3 tools/local_llm/bakeoff.py --profiles qwen38-27b-iq4nl-mtp-128k-q4`,
run against the FIXED template as its own GPU-queue job): all three tasks still PASS -- T1 locate 44s, T2 debug
64s, T3 extend 42s -- noticeably faster than this same profile's original (pre-branch) bake-off with the
round-1 template, 90s/213s/139s; long-context passphrase recall at 120K tokens still `True`; peak VRAM 22831
MiB, in line with every other measurement in this file. The fix does not just avoid regressing these tasks, it
makes the multi-turn ones (T2/T3, which are themselves several turns within one Claude Code session) noticeably
faster, for the same reason the cache-probe numbers above improved.
"""


def write_markdown(recs, path):
    lines = [
        "# Local-LLM prompt-cache probe results",
        "",
        "One llama-server per config (qwen38-27b-iq4nl-mtp-128k-q4, same model/ctx/template for every row), the "
        "SAME short multi-turn Claude Code task through each, server log parsed for tokens actually reprocessed "
        "per request vs the prompt size submitted. See tools/local_llm/cache_probe.py for the exact method and "
        "tools/local_llm/results/cache_probe.json for the full per-request data.",
        "",
        "| config | np | extra flags | load s | peak VRAM MiB | headroom MiB | task wall s | # requests | "
        "prompt sizes (tokens) | overall reused frac | last-req reused frac | last sim_best / f_keep | "
        "total pp time s | checkpoints created |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        s = r.get("summary", {})
        cl = r.get("claude", {})
        sizes = s.get("prompt_sizes", [])
        sizes_str = ",".join(str(x) for x in sizes) if sizes else "?"
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s | %s |" % (
            r["config"], r["np"], " ".join(r["extra"]) or "(none)", r.get("load_s"), r.get("vram_peak_mib"),
            r.get("vram_headroom_mib"), cl.get("wall_s"), s.get("n_requests"), sizes_str,
            s.get("overall_reused_frac"), s.get("last_request_reused_frac"), s.get("last_request_sim_best"),
            s.get("last_request_f_keep"), s.get("total_pp_time_s"), s.get("checkpoints_created"),
        ) + ("  ERROR: %s" % r["error"] if r.get("error") else ""))
    text = "\n".join(lines) + "\n" + VERDICT_MD
    open(path, "w").write(text)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", help="config names to run (default: all)")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--out", default=os.path.join(RESULTS, "cache_probe.json"))
    ap.add_argument("--md-out", default=os.path.join(RESULTS, "cache_probe.md"))
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--selftest-parse", action="store_true", help="prove the log parser against synthetic lines and exit (no GPU)")
    ap.add_argument("--reparse", action="store_true",
                     help="re-parse the ALREADY-SAVED server logs in tools/local_llm/results/cache_probe_logs/ "
                          "for --out's existing configs and rewrite --out/--md-out, WITHOUT touching the GPU or "
                          "re-running any Claude task -- for when the parser itself changes after a run.")
    ap.add_argument("--template", default=None,
                     help="override the default profile's chat_template_file for this run only (profiles.json "
                          "is not touched) -- e.g. to A/B an in-progress template fix against the shipped one.")
    ap.add_argument("--config-suffix", default="",
                     help="appended to every config's name/log/worktree for this run, so e.g. --template X "
                          "--config-suffix _newtpl does not collide with a prior run's files for the same config.")
    a = ap.parse_args()

    if a.selftest_parse:
        return selftest_parse()

    if a.reparse:
        if not os.path.exists(a.out):
            sys.exit("%s does not exist -- nothing to reparse" % a.out)
        recs = json.load(open(a.out))
        for rec in recs:
            log_path = os.path.join(LOGDIR, rec["config"] + ".server.log")
            if not os.path.exists(log_path):
                print("skip %s: %s not found" % (rec["config"], log_path))
                continue
            reqs, n_ckpt = parse_server_log(open(log_path, errors="replace").read())
            rec["requests"] = reqs
            rec["summary"] = summarize_requests(reqs, n_ckpt)
        json.dump(recs, open(a.out, "w"), indent=1)
        write_markdown(recs, a.md_out)
        print("reparsed -- results in %s and %s" % (a.out, a.md_out))
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    profile = dict(load_default_profile())
    if a.template:
        profile["chat_template_file"] = os.path.relpath(a.template, ROOT)
    configs = CONFIGS if not a.configs else [c for c in CONFIGS if c["name"] in a.configs]
    if not configs:
        sys.exit("no matching configs (known: %s)" % ", ".join(c["name"] for c in CONFIGS))
    if a.config_suffix:
        configs = [dict(c, name=c["name"] + a.config_suffix) for c in configs]
    # Merge onto any EXISTING results at --out (keyed by config name) rather than overwrite, so running a subset
    # via --configs (e.g. to add one new config after the fact) does not lose earlier configs' data.
    by_name = {}
    if os.path.exists(a.out):
        for rec in json.load(open(a.out)):
            by_name[rec["config"]] = rec
    order = list(by_name.keys())
    for cfg in configs:
        rec = run_config(cfg, profile, a.task, a.max_turns, a.timeout, keep_worktree=a.keep_worktrees)
        if rec["config"] not in by_name:
            order.append(rec["config"])
        by_name[rec["config"]] = rec
        recs = [by_name[n] for n in order]
        json.dump(recs, open(a.out, "w"), indent=1)   # write after EVERY config, so a crash mid-run loses nothing
        write_markdown(recs, a.md_out)
    print("done -- results in %s and %s" % (a.out, a.md_out))


# ---------------------------------------------------------------------------------------------------------------
def selftest_parse():
    """No GPU needed: proves parse_server_log() against synthetic log text shaped EXACTLY like the real
    llama-server output captured during this branch's probe runs (see the module docstring / _EVENT_RES comment
    -- verified against real logs, not the dead "new prompt, ..., task.n_tokens=" string `strings` found in the
    .so but which never actually appears at verbosity=3 for this workload)."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and cond

    # Case 1: two back-to-back requests on one slot -- the REAL, observed shape (config A/B/C's actual logs):
    # request 2 gets a non-trivial LCP similarity estimate (f_keep=0.73-ish) yet its "prompt eval" count is
    # STILL essentially the full requested size (the bug this whole probe exists to characterize).
    log1 = """
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
slot launch_slot_: id  0 | task 1 | processing task, is_child = 0
slot print_timing: id  0 | task 1 | prompt eval time =   10.00 ms / 5000 tokens (0.00 ms per token, 500.00 tokens per second)
slot print_timing: id  0 | task 1 |        eval time =    2.00 ms /  200 tokens (0.00 ms per token, 100.00 tokens per second)
slot      release: id  0 | task 1 | stop processing: n_tokens = 5200, truncated = 0
slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.230 (> 0.100 thold), f_keep = 0.730
slot launch_slot_: id  0 | task 2 | processing task, is_child = 0
slot print_timing: id  0 | task 2 | prompt eval time =   16.00 ms / 7999 tokens (0.00 ms per token, 500.00 tokens per second)
slot print_timing: id  0 | task 2 |        eval time =    3.00 ms /  300 tokens (0.00 ms per token, 100.00 tokens per second)
slot      release: id  0 | task 2 | stop processing: n_tokens = 8299, truncated = 0
"""
    reqs1, ckpt1 = parse_server_log(log1)
    check("case1: two requests parsed", len(reqs1) == 2)
    check("case1: request 1 requested=5000 (5200 total - 200 generated)", reqs1[0]["requested_tokens"] == 5000)
    check("case1: request 1 fully reprocessed (no earlier cache)", reqs1[0]["reused_frac"] == 0.0)
    check("case1: request 2 requested=7999 (8299 total - 300 generated)", reqs1[1]["requested_tokens"] == 7999)
    check("case1: request 2 essentially fully reprocessed despite non-trivial f_keep (the bug)",
          reqs1[1]["reused_frac"] == round(1 - 7999 / 7999, 3))
    check("case1: request 2 sim_best/f_keep captured", reqs1[1]["sim_best"] == 0.230 and reqs1[1]["f_keep"] == 0.730)
    check("case1: no checkpoints created", ckpt1 == 0)

    # Case 2: a request whose prefix WAS genuinely reused -- "prompt eval" only covers the NEW 2000 tokens of an
    # 8000-token requested prompt (6000 came from cache), and a context checkpoint was created along the way.
    log2 = """
slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.900 (> 0.100 thold), f_keep = 0.900
slot launch_slot_: id  0 | task 3 | processing task, is_child = 0
slot ctx_shift...: id  0 | task 3 | after context reuse, new n_past = 6000
slot ctx_shift...: id  0 | task 3 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 6000, n_tokens = 6000, size = 12.000 MiB)
slot print_timing: id  0 | task 3 | prompt eval time =    4.00 ms / 2000 tokens (0.00 ms per token, 500.00 tokens per second)
slot print_timing: id  0 | task 3 |        eval time =    1.00 ms /  100 tokens (0.00 ms per token, 100.00 tokens per second)
slot      release: id  0 | task 3 | stop processing: n_tokens = 8100, truncated = 0
"""
    reqs2, ckpt2 = parse_server_log(log2)
    check("case2: one request parsed", len(reqs2) == 1)
    check("case2: n_past_after_reuse captured", reqs2[0]["n_past_after_reuse"] == 6000)
    check("case2: requested=8000 (8100 total - 100 generated)", reqs2[0]["requested_tokens"] == 8000)
    check("case2: reused_frac = 1 - 2000/8000 = 0.75", reqs2[0]["reused_frac"] == 0.75)
    check("case2: one checkpoint creation counted", ckpt2 == 1)

    s1 = summarize_requests(reqs1, ckpt1)
    check("summary: prompt sizes are [5000, 7999] and monotonic",
          s1["prompt_sizes"] == [5000, 7999] and s1["prompt_sizes_monotonic_nondecreasing"])
    check("summary: overall_reused_frac is 0.0 for case1 (nothing reused)", s1["overall_reused_frac"] == 0.0)
    check("summary: checkpoints_created propagated", s1["checkpoints_created"] == 0)

    # NEGATIVE control: garbage text must yield zero requests, never a crash or a fabricated record.
    reqs_neg, ckpt_neg = parse_server_log("hello\nworld\nno slot lines here\n")
    check("negative: unrelated text yields no requests", reqs_neg == [] and ckpt_neg == 0)

    print("selftest_parse: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
