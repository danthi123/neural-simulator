"""workflow_cost_tiering (class CT, BLOCK) — a committed workflow script must TIER its agent models.

WHY (owner-flagged 2026-08-19, reiterated: "enforce these changes, not just rely on memory"). Cost burn hit
~50% of the weekly limit in 1.5 days, and the biggest leak was agent/workflow fan-outs inheriting OPUS by
default (a 6-agent Opus doc workflow burned ~500k tokens). The cost-routing discipline (haiku=mechanical /
sonnet=moderate / opus=hard-judgment) was a skill — i.e. remembered, not enforced. This makes it BLOCK: a
workflow script committed under `.claude/workflows/` (a reusable, re-run-forever workflow) may not contain an
`agent()` call that declares no `model` — an un-declared model inherits Opus, the exact leak. Every stage must
CHOOSE its tier explicitly (including the opus judgment stage — declared, not defaulted).

The SAME detector runs in `tools/cost_audit.py` inside the heartbeat over ad-hoc (uncommitted) workflow scripts,
so the live fan-outs are flagged too; this gate is the commit-time half for the durable reusable workflows.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
from cost_audit import find_untiered_agents  # noqa: E402  (shared detector)

NAME = "workflow-cost-tiering"
CLASS_ID = "CT"
BLOCKING = True


def _is_workflow_script(path):
    return path.replace("\\", "/").startswith(".claude/workflows/") and path.endswith(".js")


def check(paths):
    if paths is None:
        # full-tree fallback: scan everything under .claude/workflows/
        import glob
        cand = glob.glob(os.path.join(_ROOT, ".claude", "workflows", "*.js"))
    else:
        cand = [os.path.join(_ROOT, p) for p in paths if _is_workflow_script(p)]
    problems = []
    for p in cand:
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        for ln, snip in find_untiered_agents(txt):
            rel = os.path.relpath(p, _ROOT)
            problems.append("%s:%d — agent() declares NO `model` tier (inherits Opus). Give it an explicit "
                            "model: haiku (mechanical) / sonnet (moderate) / opus (hard judgment). %s…"
                            % (rel, ln, snip))
    return problems


def selftest():
    """FAILING DIRECTION FIRST: an un-tiered agent() MUST be caught; a fully-tiered script must PASS."""
    bad = []
    untiered = "export const meta={name:'x',description:'y'}\nconst a = await agent('do a thing', {label:'a'})\n"
    if not find_untiered_agents(untiered):
        bad.append("did NOT catch an agent() with no model: (the defaulted-to-Opus leak)")
    tiered = ("export const meta={name:'x',description:'y'}\n"
              "const a = await agent('mech', {label:'a', model:'haiku', effort:'low'})\n"
              "const b = await agent('judge', {label:'b', model:'opus'})\n")
    if find_untiered_agents(tiered):
        bad.append("FALSE POSITIVE: flagged a fully-tiered script")
    # multiline agent() with model on a later line must still count as tiered
    multiline = "const c = await agent(\n  'moderate task',\n  {label:'c',\n   model: 'sonnet'}\n)\n"
    if find_untiered_agents(multiline):
        bad.append("FALSE POSITIVE: flagged a multi-line agent() that DOES declare model")
    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
