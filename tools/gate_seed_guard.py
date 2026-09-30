#!/usr/bin/env python3
"""tools/gate_seed_guard.py -- shared GATE-seed / --prereg check (CLAUDE.md non-negotiable: a GATE seed result
is a VERDICT, so it requires a committed preregistration; running a GATE seed with no --prereg is refused,
guardrail table row "ran GATE seeds without a committed prereg", 2026-09-28).

Extracted 2026-09-29 (security review, H2) from tools/local_llm/gpu_add.sh's own embedded python heredoc, so
tools/agent_gateway_submit.py's new structured gpu-submit/pool-submit path enforces the EXACT SAME rule instead
of re-deriving a second copy that could silently drift from it. gpu_add.sh now calls this same module too (see
its own comment) -- there is exactly ONE place this check is written.

CLI:      gate_seed_guard.py '<full command string>'
          Prints "<0|1> <prereg-path-or-'-'>" on stdout (gpu_add.sh's historical PY_OUT shape, unchanged).
Library:  find_gate_hit_and_prereg(cmd: str) -> tuple[bool, str | None]
"""
import re
import sys

GATE_SEEDS = {42, 43, 44, 100, 101, 102}


def _values_after(flag, cmd):
    return re.findall(flag + r'''(?:=|\s+)("[^"]*"|'[^']*'|\S+)''', cmd)


def find_gate_hit_and_prereg(cmd):
    """cmd: a full command-line STRING (a job line, or a space-joined list of already-validated argument
    tokens -- either is fine, since this only ever does regex text-matching, never shell interpretation).
    Returns (gate_hit, prereg_path_or_None)."""
    hit = False
    for val in _values_after(r'--seeds?', cmd):
        nums = re.findall(r'\d+', val)
        if any(int(n) in GATE_SEEDS for n in nums):
            hit = True
            break

    prereg = None
    m = (re.findall(r'--prereg(?:=|\s+)("[^"]*"|\'[^\']*\'|\S+)', cmd)
         + re.findall(r'\bPREREG=("[^"]*"|\'[^\']*\'|\S+)', cmd))
    if m:
        prereg = m[0].strip('"\'')
    return hit, prereg


def main(argv):
    if len(argv) != 2:
        print("usage: gate_seed_guard.py '<full command string>'", file=sys.stderr)
        return 2
    hit, prereg = find_gate_hit_and_prereg(argv[1])
    print(f"{'1' if hit else '0'} {prereg or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
