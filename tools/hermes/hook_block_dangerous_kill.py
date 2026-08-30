#!/usr/bin/env python3
"""Hermes `pre_tool_call` shell hook (matcher: "terminal") — ports
.claude/hooks/block_self_matching_kill.py to Hermes's wire protocol.

WHY A PORT INSTEAD OF POINTING AT THE ORIGINAL FILE VERBATIM. The logic (strip heredocs/comments,
require a command-position match, block `pkill -f` / `killall`) is unchanged -- copied intact from
the Claude Code version. Only the wire adapter differs: Hermes's tool name for a shell/terminal
call is "terminal", not "Bash" (agent/display.py:458 maps `"terminal": "command"` for its primary
arg), and Hermes's block directive is `{"decision": "block", "reason": "..."}` OR plain exit code 2
(both accepted per agent/shell_hooks.py::_parse_response) -- the original script already does
exactly that (writes to stderr, `sys.exit(2)`), so the only line that had to change is the
`tool_name` check.

WHY THIS STILL MATTERS ON HERMES. `pkill -f <pattern>` matches the full command line of every
process INCLUDING the shell invoking it -- it kills the caller regardless of which agent runs it.
This cost the Claude Code session seven self-kills in one day (CLAUDE.md skill note); the failure
mode is a shell/process-model fact, not a Claude-Code-specific one, so it applies identically to
any agent (Hermes included) driving a `terminal` tool over this repo.

Registration (~/.hermes/config.yaml, see hermes-parity/config.hooks.snippet.yaml):

    hooks:
      pre_tool_call:
        - matcher: "terminal"
          command: "python3 /home/dant123/Projects/sim/tools/hermes/hook_block_dangerous_kill.py"
          timeout: 5
          fail_closed: false

Exit code 2 blocks the call (Claude-Code/Cursor-compatible, honored by Hermes's pre_tool_call
dispatcher). Any malformed payload fails OPEN (exit 0) -- a hook that can hang or crash the agent
on garbage input is worse than one that no-ops, matching both the Claude Code original and Hermes's
own default (non-fail_closed) shell-hook semantics.
"""
import json
import re
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # never block on a malformed payload

if payload.get("tool_name") != "terminal":  # Hermes tool name (Claude Code: "Bash")
    sys.exit(0)

cmd = (payload.get("tool_input") or {}).get("command", "") or ""


def _strip_noncode(text):
    """Drop heredoc BODIES and comments — text that is discussed, not executed."""
    out, in_heredoc, term = [], False, None
    for ln in text.split("\n"):
        if in_heredoc:
            if ln.strip() == term:
                in_heredoc = False
            continue  # drop the heredoc body entirely
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", ln)
        if m:
            in_heredoc, term = True, m.group(1)
            out.append(ln[:m.start()])  # keep what precedes the heredoc opener
            continue
        out.append(re.sub(r"#.*$", "", ln))  # drop trailing comments
    return "\n".join(out)


code = _strip_noncode(cmd)

# Require a COMMAND POSITION: start of line, or after ; | & or a newline. Plain `pkill <name>`
# (no -f) matches only the process NAME, which cannot match the invoking shell, so it is allowed.
patterns = [
    (r"(?:^|[;|&\n])\s*(?:sudo\s+)?pkill\b[^|;&\n]*\s-\w*f", "pkill -f"),
    (r"(?:^|[;|&\n])\s*(?:sudo\s+)?killall\b", "killall"),
]

for rx, label in patterns:
    if re.search(rx, code):
        sys.stderr.write(
            "BLOCKED: `%s` matches the FULL COMMAND LINE of every process, including the shell\n"
            "running it, so it kills the caller. USE PID INSTEAD:\n"
            "  PIDS=$(ps -eo pid,args | grep '[m]y_runner' | awk '{print $1}')\n"
            "  for P in $PIDS; do kill \"$P\"; done\n"
            "  ps -eo args | grep -c '[m]y_runner'   # verify it is gone\n\n"
            "Name-based killing without -f (matches only the process name, cannot match the\n"
            "invoking shell) is allowed, as is any mention of these commands inside a heredoc,\n"
            "comment, or quoted message.\n"
            % label
        )
        sys.exit(2)

sys.exit(0)
