#!/usr/bin/env python3
"""cost_audit.py — ENFORCE cost-routing (owner-flagged 2026-08-19: ~50% of the weekly limit in 1.5 days).

Cost-overburn is the SAME failure class as under-parallelization: a failure of OMISSION with no bad commit to
gate at the moment it happens. So it gets the SAME enforcement `parallel_audit` uses — a measured check that runs
INSIDE the heartbeat every cycle, fires regardless of choices, NAMES the cheaper routing, and RECURS until fixed.

THE RULE it enforces (see `.claude/skills/cost-routing/SKILL.md`): every `agent()` in a workflow MUST declare its
`model` tier explicitly (haiku=mechanical / sonnet=moderate / opus=hard-judgment). An `agent()` with NO `model:`
inherited Opus by default — the exact leak the owner flagged. This scans the session's persisted workflow scripts
(committed AND ad-hoc) and flags every un-tiered agent, so a defaulted-to-Opus fan-out cannot pass unnoticed.

Also usable as a library: `find_untiered_agents(script_text)` is imported by `tools/gates/workflow_cost_tiering.py`
so the SAME detector blocks an un-tiered workflow at commit time.

Exit 0 always (advisory-to-the-shell, blocking-to-me — like parallel_audit).
"""
import glob
import os
import re
import sys

ROOT = "/home/dant123/Projects/sim"
# ad-hoc workflow scripts persist here; committed reusable ones live in .claude/workflows/. The heartbeat cares
# about LIVE burn (scripts run this session) + the durable committed workflows — NOT ancient history from other
# projects. So the ad-hoc glob is filtered to recently-touched scripts; committed workflows are always scanned.
RECENT_MIN = 180  # a script touched within this many minutes counts as "live" for the heartbeat
_AD_HOC_GLOB = "/home/dant123/.claude/projects/*/*/workflows/scripts/*.js"
_COMMITTED_GLOB = os.path.join(ROOT, ".claude", "workflows", "*.js")


def _now_via_mtime():
    # Date.now() is unavailable in workflow scripts but this is a plain tool — use the newest file mtime as "now"
    # so "recent" is relative to actual activity, robust to clock questions.
    files = glob.glob(_AD_HOC_GLOB) + glob.glob(_COMMITTED_GLOB)
    return max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=0.0)

# an agent() call and the opts object that (may) follow the prompt. We look for `model:` anywhere in the call's
# opts. Non-greedy across the call; JS `agent(` may span lines, so scan a window after each `agent(`.
_AGENT_RE = re.compile(r"\bagent\s*\(", re.M)
_MODEL_RE = re.compile(r"\bmodel\s*:")


def find_untiered_agents(text):
    """Return a list of (line_no, snippet) for agent() calls with NO explicit `model:` in their opts.

    Heuristic but conservative: for each `agent(` we scan forward to the matching close paren (balanced) and
    check whether `model:` appears inside. `agent()` is a workflow-only symbol, so false positives are rare.
    """
    out = []
    for m in _AGENT_RE.finditer(text):
        i = m.end()  # just after 'agent('
        depth = 1
        j = i
        n = len(text)
        while j < n and depth > 0:
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
            if j - i > 6000:  # safety bound on one call
                break
        call = text[m.start():j]
        if not _MODEL_RE.search(call):
            line_no = text.count("\n", 0, m.start()) + 1
            snippet = " ".join(call[:110].split())
            out.append((line_no, snippet))
    return out


def _scan_files(paths):
    problems = []  # (path, line_no, snippet)
    for p in paths:
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        if "export const meta" not in txt and "agent(" not in txt:
            continue
        for ln, snip in find_untiered_agents(txt):
            problems.append((os.path.relpath(p, ROOT) if p.startswith(ROOT) else p, ln, snip))
    return problems


def main():
    now = _now_via_mtime()
    cutoff = now - RECENT_MIN * 60
    # ad-hoc scripts: only those touched within RECENT_MIN (this session's LIVE burn); committed workflows: always.
    ad_hoc = [f for f in glob.glob(_AD_HOC_GLOB) if os.path.exists(f) and os.path.getmtime(f) >= cutoff]
    committed = [f for f in glob.glob(_COMMITTED_GLOB) if os.path.exists(f)]
    recent = sorted(set(ad_hoc) | set(committed))
    problems = _scan_files(recent)
    print("─ COST AUDIT ─ workflow scripts scanned=%d | un-tiered agent() calls=%d"
          % (len(recent), len(problems)))
    if problems:
        print("⛔ COST-ROUTING VIOLATION (a burn leak, not a note) — %d agent() call(s) declared NO model tier, so"
              " they inherited OPUS by default:" % len(problems))
        for path, ln, snip in problems[:8]:
            print("     • %s:%d  %s…" % (path, ln, snip))
        print("   FIX: give each agent() an explicit model — haiku (mechanical) / sonnet (moderate) / opus (hard"
              " judgment). See .claude/skills/cost-routing/SKILL.md. Reserve opus for the ONE stage that needs it.")
    else:
        print("✓ COST-CLEAN (every scanned workflow agent declares its model tier).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
