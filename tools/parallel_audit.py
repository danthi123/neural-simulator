#!/usr/bin/env python3
"""parallel_audit.py — the parallelization ENFORCEMENT check.

Under-parallelization is a failure of OMISSION: there is no bad commit to gate, so the
commit-gates cannot catch it. Past fixes failed because they were MANUAL (lane_check I had to
remember), ADVISORY (a heartbeat line read past), or PASSIVE (a memory). This runs INSIDE the
heartbeat every cycle, so it fires regardless of choices; it NAMES the idle capacity + the exact
launchable work; it RECURS until resolved; and it prints a STALL verdict, not a note.

THE RULE it enforces: "holding" is only earned when this prints SATURATED. If it prints
UNDER-PARALLELIZED, holding IS a stall — launch the listed work (independent lanes: agents for
build/research, pool for CPU de-risks, GPU for the big run) BEFORE holding.

Output is one heartbeat-friendly block. Exit 0 always (advisory-to-the-shell, blocking-to-me).
"""
import json, os, subprocess, sys

ROOT = "/home/dant123/Projects/sim"
POOL = ["pool40", "pool41", "pool42"]
VIK = os.path.join(ROOT, "tools", "vikunja.sh")


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def local_idle():
    nproc = int(sh("nproc") or 20)
    load1 = float((sh("cut -d' ' -f1 /proc/loadavg") or "0"))
    lanes = int(sh("ps -eo args | grep -c '[r]esearch.runners'") or 0)
    return nproc, load1, max(0, int(nproc - load1)), lanes


def gpu_state():
    u = sh("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")
    try:
        return int(u)
    except Exception:
        return -1  # no GPU / unknown


def pool_idle():
    idle, lanes, up = 0, 0, 0
    for h in POOL:
        out = sh("timeout 8 ssh -o BatchMode=yes %s \"nproc; cut -d' ' -f1 /proc/loadavg; pgrep -fc research.runners || echo 0\" 2>/dev/null" % h, timeout=12)
        parts = out.split()
        if len(parts) >= 3:
            up += 1
            n, ld, ln = int(parts[0]), float(parts[1]), int(parts[2])
            idle += max(0, int(n - ld))
            lanes += ln
    return idle, lanes, up


def active_agents():
    # Count in-flight Claude subagents: task transcripts touched in the last ~12 min. A building
    # agent has idle cores now but WILL consume them — counting it prevents a build-phase false alarm.
    out = sh("find /tmp/claude-1000/-home-dant123-Projects-sim/*/tasks -maxdepth 1 -name '*.output' -mmin -12 2>/dev/null | wc -l")
    try:
        return int(out)
    except Exception:
        return 0


def open_tasks():
    raw = sh("%s --json list-tasks 2 2>/dev/null" % VIK, timeout=15)
    try:
        ts = json.loads(raw)
    except Exception:
        return None, []
    # actionable = open + not done + priority>=1. NOTE (2026-08-26 fix): the old cap `<= 4` excluded the
    # priority-5 CRUX tasks (e.g. #150 knowledge-scale) and undercounted the frontier to ~1, which made the
    # SATURATED bar (n_open > lanes) trivially met and the whole check UNABLE TO FIRE. Include p5; the raw
    # count still under-represents the true parallelizable backlog (dozens of buildable de-risks never land on
    # the board), which is WHY the agent-floor below (compute-independent) is the real enforcement, not this count.
    act = [t for t in ts if not t.get("done") and (t.get("priority") or 0) >= 1]
    act.sort(key=lambda t: -(t.get("priority") or 0))
    return len(act), [(t.get("priority") or 0, t.get("title", "")) for t in act]


def main():
    nproc, load1, idle_local, lanes_local = local_idle()
    gpu = gpu_state()
    idle_pool, lanes_pool, pool_up = pool_idle()
    n_open, top = open_tasks()
    agents = active_agents()

    total_lanes = lanes_local + lanes_pool + agents
    gpu_free = (0 <= gpu < 30)
    # idle CAPACITY worth filling, for the informative message: >6 local cores, idle pool cores (>10), or a free GPU.
    cap = []
    if idle_local > 6: cap.append("%d local cores" % idle_local)
    if idle_pool > 10: cap.append("%d pool cores (%d/3 nodes up)" % (idle_pool, pool_up))
    if gpu_free: cap.append("GPU idle(%d%%)" % gpu)

    # DEDICATED idle = a whole pool node or the GPU sitting idle. THESE are pure waste when idle (they exist only to
    # run our jobs), so an idle one with ready work IS under-parallelization — this is the same signal workflow_check.sh
    # fires on, so the two checks now AGREE instead of contradicting. Idle LOCAL cores are a weaker signal (the box runs
    # other things), so they inform the message but do NOT by themselves trip the compute branch (avoids crying wolf).
    dedicated_idle = (idle_pool > 10) or gpu_free
    have_ready = (n_open is not None and n_open > 0)
    # ROOT-CAUSE FIX (2026-08-26, owner-flagged 3x). Two INDEPENDENT triggers; EITHER fires:
    #  (1) under_agents — agent-bound BUILD/RESEARCH/VERIFY/WIRING work is NOT compute-limited, so a live frontier with
    #      fewer than FLOOR concurrent agents is under-parallelized regardless of cores. Compute-independent = the real fix.
    #  (2) under_compute — a DEDICATED lane (pool node / GPU) idle while a frontier exists. NOTE the removed clause: the
    #      old bar also required `n_open > total_lanes`, which made it UNABLE TO FIRE (the board count reads ~1-3 while the
    #      TRUE parallelizable backlog — roadmap de-risks, pending 6-seed validations, wirings — is dozens). Gating idle
    #      dedicated compute on the board count was the bug; the backlog is ALWAYS bigger than the board, so idle pool/GPU
    #      + ready work fires on its own. (This is what let SATURATED print while 3 pool nodes sat idle with crashed jobs.)
    AGENT_FLOOR = int(os.environ.get("PARALLEL_AGENT_FLOOR", "3"))
    under_agents = have_ready and (agents < AGENT_FLOOR)      # compute-INDEPENDENT
    under_compute = have_ready and dedicated_idle            # a dedicated lane idle with ready work
    under = under_agents or under_compute

    print("─ PARALLEL AUDIT ─ lanes=%d (local %d + pool %d + agents %d) | GPU=%s | open-tasks=%s"
          % (total_lanes, lanes_local, lanes_pool, agents, ("%d%%" % gpu if gpu >= 0 else "n/a"),
             (str(n_open) if n_open is not None else "?")))
    if under:
        why = []
        if under_agents:
            why.append("only %d build/research agent(s) running (floor %d) — agent work is NOT compute-limited, FAN OUT MORE"
                       % (agents, AGENT_FLOOR))
        if under_compute:
            why.append("idle %s ; %d ready tasks vs %d lanes" % (", ".join(cap), n_open, total_lanes))
        print("⛔ UNDER-PARALLELIZED (a STALL, not a hold) — %s." % " ; ".join(why))
        print("   The parallelizable backlog is ALWAYS bigger than the board — roadmap de-risks, pending 6-seed")
        print("   validations, faculty wirings, consolidations. LAUNCH concurrent agents/workflows now (pool for CPU,")
        print("   GPU for the big run). Board frontier rows for anchors:")
        for pr, title in top[:6]:
            print("     • p%d  %s" % (pr, title))
        print("   Holding is NOT earned until this reads SATURATED (>= %d agents AND compute covered)." % AGENT_FLOOR)
    elif not have_ready:
        print("✓ SATURATED (no ready board tasks — restock the board or hold).")
    else:
        print("✓ SATURATED (%d agents + %d compute lanes cover the frontier)." % (agents, lanes_local + lanes_pool))

    # COST-ROUTING — agent tokens count toward the Claude usage limit; mechanical work must go to non-Claude
    # machinery. Fires whenever cheap idle compute exists, so the routing is enforced every cycle, not remembered.
    if idle_pool > 10 or idle_local > 6:
        print("   💸 COST-ROUTING (agent tokens burn the usage limit): put MECHANICAL work on non-Claude machinery —")
        print("      • CPU param grids / TUNING → `tools/sweep_pool.sh` (headless on the %d idle pool cores, 0 tokens)"
              % idle_pool)
        print("      • GPU sweeps/tuning → `tools/gpu_queue.sh add '<cmd>'` (headless, sequential, VRAM-contention-safe,")
        print("        pausable for gaming); multi-SEED of one config → controller fans out `--seeds` directly")
        print("      • reserve AGENTS for genuine BUILDS/integration (new runner, wiring) that need judgment.")
    return 0


def _under_decision(have_ready, dedicated_idle, agents, n_open, total_lanes, agent_floor=3):
    """Pure copy of main()'s under-parallelization decision, for the selftest. Keep in sync with main().
    n_open/total_lanes are accepted for signature stability but NO LONGER gate the compute branch — gating idle
    dedicated compute on the board count was the exact bug (the backlog is always bigger than the board)."""
    under_agents = have_ready and (agents < agent_floor)
    under_compute = have_ready and dedicated_idle
    return under_agents or under_compute


def _selftest():
    """The 2026-08-26 root cause was that this check had shipped UNABLE TO FIRE (the old bar needed idle compute
    AND a board-count that was structurally ~1). A check that cannot fail is the bug. This selftest asserts the
    fixed decision FIRES in its failing direction (few agents / idle dedicated lane) and stays quiet when saturated."""
    # (have_ready, dedicated_idle, agents, n_open, total_lanes) -> expected_under
    cases = [
        (True,  False, 1, 1, 13, True),   # THE REGRESSION: 1 agent, tiny board count, no idle lane -> agent-floor fires
        (True,  False, 2, 1,  6, True),   # the exact hold I was in -> agent-floor fires
        (True,  False, 5, 1,  8, False),  # 5 agents, no idle dedicated lane -> saturated, must NOT fire
        (True,  True,  4, 3,  8, True),   # THE 2nd REGRESSION: agents>=floor, board count LOW, but a POOL NODE idle
                                          #   -> must fire (this is the false-SATURATED-while-pool-idle case)
        (True,  True,  3, 2, 10, True),   # at the agent floor BUT a dedicated lane idle with ready work -> fires
        (False, True,  0, 0,  0, False),  # no ready work -> must NOT fire (idle lane but nothing to run)
        (True,  True,  5, 20, 8, True),   # idle dedicated lane + backlog -> fires (compute branch)
    ]
    bad = [(c, _under_decision(*c[:5])) for c in cases if _under_decision(*c[:5]) != c[5]]
    if bad:
        print("PARALLEL_AUDIT SELFTEST FAILED (the check is unable to fire correctly):")
        for c, got in bad:
            print("   case %s -> got under=%s, expected %s" % (c[:5], got, c[5]))
        sys.exit(1)
    print("parallel_audit selftest OK — fires on agents<floor OR an idle dedicated lane with ready work; quiet when saturated.")
    sys.exit(0)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    sys.exit(main())
