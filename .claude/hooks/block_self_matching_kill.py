#!/usr/bin/env python3
"""PreToolUse hook: BLOCK the self-matching kill forms — kill by PID instead.

WHY THIS IS A HOOK AND NOT A NOTE (2026-07-30). `pkill -f <pattern>` matches the *full command line of every
process*, INCLUDING the shell that is running it — because that shell's argv contains the pattern. It therefore
kills the caller. This happened **SEVEN times in one session**, costing a lost commit, a lost launch, and several
truncated tool calls (exit 144). The rule "kill by PID" was written down, banked in a skill, and re-derived in a
commit message — and then violated again twice under load. A rule that has failed seven times after being written
down three times is not a knowledge problem; it needs a mechanism that can say NO.

The `[p]attern` bracket trick is NOT a reliable escape: it protects the pattern itself, but the same command
usually mentions the plain string elsewhere (a launch line, an echo, an --out path), and the kill matches that.

SAFE REPLACEMENT (what to do instead):
    PIDS=$(ps -eo pid,args | grep '[m]y_runner' | awk '{print $1}')   # bracket = grep won't match itself
    for P in $PIDS; do kill "$P"; done                                 # kill by PID only
Then VERIFY: `ps -eo args | grep -c '[m]y_runner'`.

SCOPE IS DELIBERATELY NARROW, and that narrowness was earned within a minute of writing this file: the FIRST
version matched the raw string anywhere in the command, and it blocked the very commit that ADDED it, because the
commit message *describes* the dangerous form. It would equally have blocked every findings doc, comment and echo
that discusses the rule. An over-broad blocker is precisely how a check earns a reputation for crying wolf and
gets switched off — the failure mode this repo has already banked twice (verify-go rule 8: a false alarm is as
corrosive as a missed one). So: heredoc bodies and comments are stripped, and the match must be at a COMMAND
POSITION. Talking about it is always fine; running it is not.

Exit 2 blocks the call and shows this message to the model.
"""
import json
import re
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)                      # never block on a malformed payload

if payload.get("tool_name") != "Bash":
    sys.exit(0)

cmd = (payload.get("tool_input") or {}).get("command", "") or ""


def _strip_noncode(text):
    """Drop heredoc BODIES and comments — text that is discussed, not executed."""
    out, in_heredoc, term = [], False, None
    for ln in text.split("\n"):
        if in_heredoc:
            if ln.strip() == term:
                in_heredoc = False
            continue                                   # drop the heredoc body entirely
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", ln)
        if m:
            in_heredoc, term = True, m.group(1)
            out.append(ln[:m.start()])                 # keep what precedes the heredoc opener
            continue
        out.append(re.sub(r"#.*$", "", ln))            # drop trailing comments
    return "\n".join(out)


code = _strip_noncode(cmd)

# Require a COMMAND POSITION: start of line, or after ; | & or a newline. Plain `pkill <name>` (no -f) matches
# only the process NAME, which cannot match the invoking shell, so it is allowed.
patterns = [
    (r"(?:^|[;|&\n])\s*(?:sudo\s+)?pkill\b[^|;&\n]*\s-\w*f", "pkill -f"),
    (r"(?:^|[;|&\n])\s*(?:sudo\s+)?killall\b", "killall"),
]

for rx, label in patterns:
    if re.search(rx, code):
        sys.stderr.write(
            "BLOCKED: `%s` matches the FULL COMMAND LINE of every process, including the shell running it,\n"
            "so it kills the caller. This self-kill happened SEVEN times in one session (exit 144), losing a\n"
            "commit and a launch. The bracket trick does NOT reliably help: the same command usually mentions\n"
            "the plain string elsewhere (a launch line, an echo, an --out path) and the kill matches that.\n\n"
            "USE PID INSTEAD:\n"
            "  PIDS=$(ps -eo pid,args | grep '[m]y_runner' | awk '{print $1}')\n"
            "  for P in $PIDS; do kill \"$P\"; done\n"
            "  ps -eo args | grep -c '[m]y_runner'   # verify it is gone\n\n"
            "Name-based killing without -f (matches only the process name, cannot match the invoking shell)\n"
            "is allowed, as is any mention of these commands inside a heredoc, comment or quoted message.\n"
            % label
        )
        sys.exit(2)

sys.exit(0)
