#!/usr/bin/env python3
"""tools/agent_gateway_parse.py -- H1 fix (2026-09-29 security review). tools/agent_gateway.sh used to run
`eval "set -- $REST"` on text taken directly from SSH_ORIGINAL_COMMAND -- attacker-controlled, since the
sim-agent CONTAINER's own ssh client chooses this string. `pool-submit x $(cmd) --checked y` ran `cmd`: `eval`
performs a full extra round of shell parsing (command substitution, variable expansion, globbing, redirection,
...), not merely "split into words".

This splits the SAME text with Python's `shlex` (POSIX shell WORD-SPLITTING/QUOTING rules only -- no command
substitution, no variable expansion, no globbing, no redirection: shlex has no concept of any of those, so there
is nothing for an attacker's `$(...)` or `` `...` `` to DO here beyond being literal, inert characters in a
token) and rejects any resulting token containing a control character (a bare newline/CR could otherwise be used
to smuggle an extra line into a downstream queue file, exactly like the pool_queue.sh --checked fix). Tokens are
printed one per line on stdout -- safe as a delimiter precisely because a token containing a newline is already
refused, so no valid token can ever be mistaken for two.

Usage: printf '%s' "$REST" | agent_gateway_parse.py
Exit 0 + one token per line on stdout, or exit 1 + a one-line reason on stderr and no stdout.
"""
import shlex
import sys


def parse(text):
    """Returns (tokens, error) -- error is None on success."""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as e:
        return None, f"could not parse arguments: {e}"
    for tok in tokens:
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in tok):
            return None, f"argument contains a control character: {tok!r}"
    return tokens, None


def main():
    text = sys.stdin.read()
    tokens, err = parse(text)
    if err is not None:
        print(err, file=sys.stderr)
        return 1
    for tok in tokens:
        print(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
