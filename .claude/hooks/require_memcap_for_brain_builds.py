#!/usr/bin/env python3
"""PreToolUse hook: a command that builds the FULL brain must run under tools/memcap.sh (a kernel-enforced cgroup cap).

WHY THIS IS A HOOK AND NOT A NOTE. The memcap rule was written down after 2026-09-18, when an uncapped full-brain
battery ballooned to ~28 GB on this 46 GB box and the global OOM-killer took the Claude session with it. On
2026-09-24 it happened again in slow motion: three uncapped `pytest tests/test_webapp_server.py` runs from review
agents (9.8 + 5.3 + 2.3 GB) sat beside two capped battery workers and left 2 GB available; the main session killed the
largest by hand. A rule that twelve agents each have to remember is not a mechanism.

WHAT IT BLOCKS, at a command position only (never inside quotes, heredoc bodies or comments, so a job string handed
to tools/pool_queue.sh or tools/gpu_queue.sh, a commit message, or an echo is never judged):
  * pytest on tests/test_webapp_server.py (the brain_chat server tests build the full brain per test);
  * python -m research.runners.onebrain_regression_battery / research.runners.load_bearing_fraction.
unless the same command line runs it through tools/memcap.sh (or systemd-run with MemoryMax). --collect-only,
--selftest, --score, --dry-run and --help do not build a brain and pass.

Exit 2 blocks the call and shows the message to the model.

    .venv/bin/python -m pytest -q tests/test_require_memcap_hook.py
"""
import json
import re
import sys

HEAVY = [
    (re.compile(r"(?:^|\s)(?:\S*/)?(?:python[\d.]*\s+(?:-\S+\s+)*-m\s+)?pytest\b[^\n;&|]*?tests/test_webapp_server\.py"),
     "the brain_chat server tests (tests/test_webapp_server.py build the full brain)"),
    (re.compile(r"(?:^|\s)(?:\S*/)?python[\d.]*\s+(?:-\S+\s+)*-m\s+research\.runners\."
                r"(?:onebrain_regression_battery|load_bearing_fraction)\b"),
     "a full-brain battery runner"),
]
CAPPED = re.compile(r"memcap\.sh|systemd-run\b[^\n;&|]*MemoryMax")
HARMLESS = re.compile(r"--collect-only|--selftest|--score|--dry-run|(?:^|\s)(?:-h|--help)(?:\s|$)")


def _strip_noncode(text):
    """Drop heredoc bodies, quoted strings and comments: text that is handed on or discussed, not run here."""
    out, in_heredoc, term = [], False, None
    for ln in text.split("\n"):
        if in_heredoc:
            if ln.strip() == term:
                in_heredoc = False
            continue
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", ln)
        if m:
            in_heredoc, term = True, m.group(1)
            ln = ln[:m.start()]
        ln = re.sub(r"'[^']*'", "''", ln)
        ln = re.sub(r'"(?:\\.|[^"\\])*"', '""', ln)
        out.append(re.sub(r"(?:^|\s)#.*$", "", ln))
    return "\n".join(out)


def problems(command):
    code = _strip_noncode(command or "")
    found = []
    for segment in re.split(r"&&|\|\||[;\n|&]", code):
        for rx, label in HEAVY:
            m = rx.search(segment)
            if not m or HARMLESS.search(segment):
                continue
            if CAPPED.search(segment[:m.start()]):
                continue
            found.append(label)
    return found


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    found = problems((payload.get("tool_input") or {}).get("command", ""))
    if not found:
        return 0
    sys.stderr.write(
        "BLOCKED: this runs %s without a memory cap.\n"
        "Full-brain builds must run under the kernel-enforced cgroup cap, or one ballooning process can take down the\n"
        "box and the Claude session (2026-09-18: a 28 GB battery did exactly that; 2026-09-24: three uncapped test runs\n"
        "left 2 GB free beside the capped battery workers).\n\n"
        "Wrap it, with a RAM check first:\n"
        "  bash tools/mem_ok.sh 12 3 && bash tools/memcap.sh 12 -- <your command>\n"
        "If mem_ok refuses, wait for RAM or run it on the pool (tools/pool_queue.sh add --checked ...).\n"
        % found[0])
    return 2


if __name__ == "__main__":
    sys.exit(main())
