"""CLASS AP — PENDING WORK SERIALISED while dispatchable agents sit unused. BLOCKING.

WHY (owner, 2026-07-31): "parallel work isn't strictly a compute thing, especially when there's an AI with
dispatchable subagents involved. The only real limit is per-agent context, and even that is loose with
compaction."

Every parallelism check in this repo measured COMPUTE — GPU lanes, pool queue depth, idle CPU cores. NONE
measured whether AGENT work was being fanned out. So on 2026-07-31 I worked an 8-item backlog strictly one item
at a time — build, test, commit, next — for hours, while six of those items were entirely file-disjoint and could
have run at once. Every compute check was GREEN throughout, because the crux and the pool were busy. The axis
with the loosest limit was the one nothing watched.

THE RULE: if the board lists N pending items and fewer than MIN_AGENTS agents are working, that is serialisation,
not prioritisation. Dispatch, or say why not.

The original implementation parsed an old Markdown list and counted processes named ``claude``. That became
silently useless under Codex: the JSON workboard could carry a ready lane, or a completed agent could leave its
lane marked running, while this gate stayed green. The tracked workboard and its explicit lane-to-agent mapping
are now the only signal.

THE REAL CONSTRAINT, and why the threshold is not higher: agents must own DISJOINT FILES. Two agents editing one
file corrupt each other. That is a scheduling problem, not a reason to serialise — the fan-out that prompted this
gate split "stale citations" from "plans frontmatter" solely because both would have touched docs/plans/.

WHAT IT CANNOT CATCH: whether the parallel work is the RIGHT work. Six agents on trivia passes. That is
judgement and stays with the human and with me.
"""
from __future__ import annotations

NAME = "agent-parallelism"
CLASS_ID = "AP"
BLOCKING = True

from tools.autonomous_coordinator import agent_lanes_without_live_owner, load_board


def check(paths=None):
    del paths
    try:
        missing = agent_lanes_without_live_owner(load_board())
    except ValueError as exc:
        return [f"cannot validate agent parallelism because the workboard is invalid: {exc}"]
    if not missing:
        return []
    return [
        "delegated workboard lane(s) have no live assigned agent: " + ", ".join(missing) + "\n"
        "        This includes a running lane whose agent already completed; that state loses the controller's\n"
        "        pending review step and makes serial work look active.\n"
        "        FIX: dispatch and register an agent, or explicitly move the lane to controller ownership,\n"
        "        blocked with a recovery action, or completed with its next roadmap action recorded."
    ]


def selftest():
    """FAILING DIRECTION FIRST: actionable delegated work without a live agent must fire."""
    bad = []
    board = {
        "lanes": {"lane-a": {"status": "running", "delegation": "agent", "agent_id": "agent-a"}},
        "agents": {"agent-a": {"status": "completed"}},
    }
    if agent_lanes_without_live_owner(board) != ["lane-a"]:
        bad.append("does not catch a running delegated lane whose assigned agent completed")
    board["agents"]["agent-a"]["status"] = "running"
    if agent_lanes_without_live_owner(board):
        bad.append("false positive on a delegated lane with a live assigned agent")
    board["lanes"]["lane-a"]["delegation"] = "controller"
    board["agents"]["agent-a"]["status"] = "completed"
    if agent_lanes_without_live_owner(board):
        bad.append("false positive on controller-owned work")
    return bad
