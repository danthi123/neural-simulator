#!/usr/bin/env python3
"""Tests for the kill-safety PreToolUse hook.

Rule 8 of verify-go: a monitor must be tested against a run you KNOW is broken, not only a healthy one — and a
FALSE ALARM is as corrosive as a missed one. This hook has already produced one of each within a minute of being
written (it blocked its own commit), so both directions are pinned here.

    python3 .claude/hooks/test_block_self_matching_kill.py
"""
import json
import os
import subprocess
import sys

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_self_matching_kill.py")

BLOCK, ALLOW = 2, 0
DANGER = "p" + "kill -f"          # assembled so this test file is not itself an example to copy

CASES = [
    # (expected_exit, description, command)
    (BLOCK, "bare dangerous form",        DANGER + " gap4_resumable"),
    (BLOCK, "mid-pipeline after &&",      "echo hi && " + DANGER + " myrunner"),
    (BLOCK, "after a semicolon",          "cd /tmp; " + DANGER + " myrunner"),
    (BLOCK, "flags combined (-9f)",       "p" + "kill -9f myrunner"),
    (BLOCK, "under sudo",                 "sudo " + DANGER + " myrunner"),
    (BLOCK, "killall",                    "killall python"),
    # The bracket trick is NOT an exemption: it protects the pattern but not the rest of the command line.
    (BLOCK, "bracket trick still blocked", DANGER + " '[g]ap4'"),

    (ALLOW, "safe PID idiom",
     "PIDS=$(ps -eo pid,args | grep '[g]ap4' | awk '{print $1}'); for P in $PIDS; do kill $P; done"),
    (ALLOW, "plain name kill (no -f)",    "p" + "kill firefox"),
    # The false positive that this hook produced on its very first run.
    (ALLOW, "commit message describing it",
     "git commit -F - << 'MSG'\nenforce kill-by-PID\nthe " + DANGER + " form matches the caller\nMSG"),
    (ALLOW, "mentioned in a comment",     "echo ok   # never use " + DANGER + " here"),
    (ALLOW, "grep for it in the record",  "grep -rn '" + DANGER + "' research/findings/ | head"),
    (ALLOW, "not a Bash tool call", None),   # handled below
]


def run(command, tool_name="Bash"):
    payload = {"tool_name": tool_name, "tool_input": {"command": command}}
    p = subprocess.run([sys.executable, HOOK], input=json.dumps(payload),
                       capture_output=True, text=True)
    return p.returncode


def main():
    failures = 0
    for expected, desc, command in CASES:
        got = run("rm -rf /", tool_name="Write") if command is None else run(command)
        ok = got == expected
        failures += 0 if ok else 1
        print("%-4s %-30s expected=%d got=%d" % ("PASS" if ok else "FAIL", desc, expected, got))

    # A malformed payload must never block — a broken hook that blocks everything is worse than no hook.
    p = subprocess.run([sys.executable, HOOK], input="not json", capture_output=True, text=True)
    ok = p.returncode == 0
    failures += 0 if ok else 1
    print("%-4s %-30s expected=0 got=%d" % ("PASS" if ok else "FAIL", "malformed payload", p.returncode))

    print("\n%d/%d passed" % (len(CASES) + 1 - failures, len(CASES) + 1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
