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
    # actionable = open, priority 1-4 (priority-5 north-star kept out), and NOT labeled epic(12)/blocked(11) —
    # those are mission-framing or production-blocked-upstream, not launchable independent work (2026-08-19: the
    # audit was counting epics + blocked-on-no-consumer tasks as "ready", firing a false UNDER-PARALLELIZED).
    _NOT_READY = {11, 12}  # blocked, epic
    act = [t for t in ts if not t.get("done") and 1 <= (t.get("priority") or 0) <= 4
           and not (_NOT_READY & {l.get("id") for l in (t.get("labels") or [])})]
    act.sort(key=lambda t: -(t.get("priority") or 0))
    return len(act), [(t.get("priority") or 0, t.get("title", "")) for t in act]


def mem_state():
    """(avail_mb, total_mb, pct_avail) from /proc/meminfo. THE CEILING THE CPU/GPU AUDIT IS BLIND TO: on
    2026-08-21 ~5 agents + 2 workflows + 3 sim-builds ran at once; each sim-loading job holds ~1-4 GB (bridges +
    Qwen mouth + co-resident organs), the 46 GB box ran OUT OF RAM, and the whole Claude process CRASHED and took
    the background work down. So RAM gates 'launch more' — an idle core is cheap, an OOM crash is not."""
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.split()[0])  # kB
        avail = info.get("MemAvailable", 0) // 1024
        total = info.get("MemTotal", 1) // 1024
        return avail, total, (100 * avail // total if total else 0)
    except Exception:
        return -1, -1, -1


def main():
    nproc, load1, idle_local, lanes_local = local_idle()
    gpu = gpu_state()
    idle_pool, lanes_pool, pool_up = pool_idle()
    n_open, top = open_tasks()
    agents = active_agents()

    total_lanes = lanes_local + lanes_pool + agents
    gpu_free = (0 <= gpu < 30)
    # idle CAPACITY worth filling: >6 local cores, or any reachable idle pool cores (>10), or a free GPU
    cap = []
    if idle_local > 6: cap.append("%d local cores" % idle_local)
    if idle_pool > 10: cap.append("%d pool cores (%d/3 nodes up)" % (idle_pool, pool_up))
    if gpu_free: cap.append("GPU idle(%d%%)" % gpu)

    avail_mb, total_mb, mem_pct = mem_state()
    # MEMORY CEILING (2026-08-21 crash): RAM gates 'launch more'. Heavy = sim-loading jobs (each ~1-4 GB) +
    # agents + workflow sub-agents. Keep a headroom so a burst of them cannot OOM the box. When constrained,
    # HOLDING IS CORRECT even with idle cores + ready tasks — an OOM crash loses all in-flight work.
    MEM_FLOOR_MB = 8000
    mem_constrained = (avail_mb >= 0 and (avail_mb < MEM_FLOOR_MB or mem_pct < 15))

    have_capacity = bool(cap)
    have_ready = (n_open is not None and n_open > 0)
    # under-parallelized: idle capacity AND ready independent work AND not already many lanes AND RAM headroom
    under = have_capacity and have_ready and (n_open > total_lanes) and not mem_constrained

    print("─ PARALLEL AUDIT ─ lanes=%d (local %d + pool %d + agents %d) | GPU=%s | mem=%s | open-tasks=%s"
          % (total_lanes, lanes_local, lanes_pool, agents, ("%d%%" % gpu if gpu >= 0 else "n/a"),
             ("%dMB/%d%% free" % (avail_mb, mem_pct) if avail_mb >= 0 else "n/a"),
             (str(n_open) if n_open is not None else "?")))
    if mem_constrained:
        print("⛔ MEMORY-CONSTRAINED (only %dMB / %d%% free) — do NOT launch more heavy jobs (sim-builds/agents/"
              "workflows); an OOM crash loses ALL in-flight work + is worse than an idle core. Let running work"
              " drain first, then reassess. (This is the ceiling the CPU/GPU lanes are blind to — 2026-08-21.)"
              % (avail_mb, mem_pct))
    elif under:
        print("⛔ UNDER-PARALLELIZED (a STALL, not a hold) — idle: %s ; %d ready board tasks vs %d lanes."
              % (", ".join(cap), n_open, total_lanes))
        print("   LAUNCH now (independent, from the board — agents for build/research, pool for CPU, GPU for the big run):")
        for pr, title in top[:6]:
            print("     • p%d  %s" % (pr, title))
        print("   Holding is NOT earned until this reads SATURATED (and RAM has headroom).")
    elif not have_ready:
        print("✓ SATURATED (no ready board tasks — restock the board or hold).")
    elif not have_capacity:
        print("✓ SATURATED (compute full — holding for lanes is the async pattern).")
    else:
        print("✓ SATURATED (%d lanes cover the %d ready tasks)." % (total_lanes, n_open))

    # COST-ROUTING ENFORCEMENT (owner-flagged 2026-08-19: ~50% of the weekly limit in 1.5 days). Two halves:
    # (1) the model-tiering check — every workflow agent must declare its model or it inherited OPUS by default;
    # cost_audit scans the session's live workflow scripts + committed workflows and prints a ⛔ verdict here so
    # the leak recurs until fixed (same enforcement philosophy as the parallelization check above).
    try:
        import cost_audit as _CA
        _CA.main()
    except Exception as _e:
        print("─ COST AUDIT ─ (unavailable: %s)" % type(_e).__name__)

    # (2) ENGINE-ROUTING — agent tokens count toward the Claude usage limit; mechanical compute must go to
    # non-Claude machinery. Fires whenever cheap idle compute exists, so the routing is nudged every cycle.
    if idle_pool > 10 or idle_local > 6:
        print("   💸 COST-ROUTING (agent tokens burn the usage limit): put MECHANICAL work on non-Claude machinery —")
        print("      • CPU param grids / TUNING → `tools/sweep_pool.sh` (headless on the %d idle pool cores, 0 tokens)"
              % idle_pool)
        print("      • GPU sweeps/tuning → `tools/gpu_queue.sh add '<cmd>'` (headless, sequential, VRAM-contention-safe,")
        print("        pausable for gaming); multi-SEED of one config → controller fans out `--seeds` directly")
        print("      • reserve AGENTS for genuine BUILDS/integration (new runner, wiring) that need judgment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
