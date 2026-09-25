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

2026-09-08 (owner-caught: a whole session under-parallelized while this printed correctly every cycle and
was read past — "past fixes failed being manual/advisory/passive"). Printing to a 15-minute heartbeat is
ADVISORY on its own; this script now also PERSISTS its verdict (`tools/parallel_state.py`, one record per
cycle in `research/coordination/parallel_audit_state.json`, tracking how long each condition has read true
CONTINUOUSLY) so `tools/gates/compute_idle_persistent.py` can BLOCK a commit once dedicated compute (a pool
node / the GPU) has sat idle with ready work for too long — the same enforcement shape `gates/lane_starvation`
already uses for idle CPU lanes, just fed by this script's own idle-dedicated-compute signal instead. The
agent-floor signal (fewer than AGENT_FLOOR concurrent agents) is reported persistently too
(`gates/agent_floor_persistent`, non-blocking) but NOT escalated to a hard block here — see that gate's
docstring for why forcing a minimum agent count to commit is a judgement call left for owner review.
2026-09-25 (pool_stall_check.py): `lanes_pool` below counted a process the instant `pgrep` saw it, with no
check that it was doing anything the record does not already have or that it was still inside its own
historical running time -- the exact hole that let a live-but-stalled pool (7 of 17 D6 processes DUPLICATES of
cells whose output had already landed at the same pinned revision, running 7-26h) read `✓ SATURATED`.
`pool_stall_summary()` runs `tools/pool_stall_check.py`'s read-only DUP-OF-LANDED / OVERDUE check every cycle
and subtracts flagged lanes from `lanes_pool` before the SATURATED/UNDER-PARALLELIZED decision below, printing
them as a standalone ⚠ line -- never blocking, never killing anything (that tool is read-only by design).
2026-09-25 (same day, one addition): six revision-pinned QUEUED lines sat 7.5h because that revision was never
provisioned where it could fit -- `pool_autodispatch.sh`'s own per-cycle log said so, but nothing outside that
one log line ever surfaced it. `queue_unrunnable_summary()` runs `tools/pool_stall_check.py`'s read-only
UNRUNNABLE / memory-budget check every cycle too, printed the same way (queued lines were never counted in
`lanes_pool` to begin with, so there is nothing to subtract -- this is pure surfacing).
"""
import json, os, re, subprocess, sys, time

import parallel_state
import waiver_history

try:
    import pool_stall_check
except Exception as _e:                # exit-0-always: say so loudly instead of silently losing the signal
    pool_stall_check = None
    print("⚠ parallel_audit: could not import pool_stall_check (%s) -- stall/dup detection disabled" % _e)

ROOT = "/home/dant123/Projects/sim"
POOL = ["pool40", "pool41", "pool42"]
VIK = os.path.join(ROOT, "tools", "vikunja.sh")
# (print label, gate NAME, waiver file, waiver max age h) -- IMPORTED from the gates themselves (fix round 3,
# 2026-09-23), not restated: fix 2 hand-copied the NAME strings, the paths and the 6h cap here and pinned them
# with a test. Reading them off the gate modules makes drift impossible rather than detected. (Budget
# accounting is now keyed by the waiver FILE, not the gate name, so even a wrong label could not double-charge
# -- the heartbeat still records under the gate's own NAME so the history reads cleanly.)
try:
    from gates import compute_idle_persistent as _cip, lane_starvation as _ls
    WAIVER_SOURCES = (
        ("compute", _cip.NAME, _cip.WAIVER_FILE, _cip.WAIVER_MAX_H),
        ("lane", _ls.NAME, _ls._waiver_file(), _ls.LANE_WAIVER_MAX_H),
    )
except Exception as _e:        # the heartbeat is exit-0-always; say so loudly instead of guessing paths
    print("⚠ parallel_audit: could not import the waiver gates (%s) -- waiver surfacing disabled" % _e)
    WAIVER_SOURCES = ()
# Every subagent (however it was spawned) gets a transcript at <session>/subagents/**/agent-<id>.jsonl.
# A standalone Agent-tool call's transcript sits directly under subagents/; a Workflow's agent() call
# (a .claude/workflows/*.js fan-out, or an ad-hoc workflow script) writes its transcript one level deeper,
# under subagents/workflows/wf_<run-id>/. See active_agents() below for why this matters.
CLAUDE_PROJECTS_DIR = "/home/dant123/.claude/projects/-home-dant123-Projects-sim"


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


def gpu_queue_busy():
    # The single-GPU lane is COMMITTED (not idle-wasted) when the queue is running a job, or draining a backlog
    # under a live dispatcher. A CPU-heavy GPU job (e.g. a 60-day develop-loop) legitimately holds the lane at low
    # util%, so nvidia-smi utilization ALONE misreads it as free (the 2026-09-01 recurring false-positive: a healthy
    # queue with 13 queued + a running job read as "GPU idle(4%)" every cycle, costing a re-diagnosis each heartbeat).
    # ONE brain-loading GPU proc at a time => a running or queued job IS saturation of this lane, not waste.
    st = sh("bash %s/tools/gpu_queue.sh status 2>/dev/null" % ROOT, timeout=10)
    if not st:
        return False
    up = "dispatcher: up" in st
    running = ("current:" in st) and ("(idle)" not in st)
    m = re.search(r"queued:\s*(\d+)", st)
    queued = int(m.group(1)) if m else 0
    return up and (running or queued > 0)


def _node_can_take_work(h, min_est):
    # A node's idle CORES only count as unused capacity if the dispatcher would actually hand it the next queued job
    # (2026-09-23): the pool is RAM-bound (15 GB nodes, ~5-6 GB jobs), so "cores idle" stayed true for 15 days while
    # the nodes were full, the under_compute streak never reset, and the idle gate could only be satisfied by filler
    # or waivers. Empty queue -> the idle cores DO count (nothing is staged: that is the real defect to flag).
    if min_est is None:
        return True
    out = sh("bash %s --node-budget %s 2>/dev/null" % (os.path.join(ROOT, "tools", "pool_autodispatch.sh"), h), timeout=40)
    m = re.search(r"idle budget=(-?\d+)GB", out)
    return bool(m) and int(m.group(1)) >= min_est


def pool_idle():
    idle, lanes, up = 0, 0, 0
    me = sh("bash %s --min-queued-est-gb 2>/dev/null" % os.path.join(ROOT, "tools", "pool_autodispatch.sh"), timeout=20).strip()
    min_est = int(me) if me.isdigit() else None
    for h in POOL:
        out = sh("timeout 8 ssh -o BatchMode=yes %s \"nproc; cut -d' ' -f1 /proc/loadavg; pgrep -fc research.runners || echo 0\" 2>/dev/null" % h, timeout=12)
        parts = out.split()
        if len(parts) >= 3:
            up += 1
            n, ld, ln = int(parts[0]), float(parts[1]), int(parts[2])
            if _node_can_take_work(h, min_est):
                idle += max(0, int(n - ld))
            lanes += ln
    return idle, lanes, up


def pool_stall_summary(nodes=None, timeout=12):
    """Read-only DUP-OF-LANDED / OVERDUE check across the pool (tools/pool_stall_check.py). Returns
    (flagged_count, lines) -- lines to print, one summary plus one per flagged job (capped). NEVER raises: this
    runs inside the exit-0-always heartbeat, so a failure here must degrade to "0 flagged, say why" rather than
    take the whole cycle down with it."""
    if pool_stall_check is None:
        return 0, []
    try:
        report = pool_stall_check.check_all(nodes=nodes, timeout=timeout)
    except Exception as e:
        return 0, ["⚠ pool-stall-check failed to run (%s) -- treating as 0 flagged, not silently OK" % e]
    flagged = report.get("flagged", [])
    lines = []
    if flagged:
        lines.append("⚠ %s" % report.get("summary_line", "pool stall check flagged %d job(s)" % len(flagged)))
        for row in flagged[:6]:
            lines.append("   ⚠ %s" % pool_stall_check.format_row(row))
        if len(flagged) > 6:
            lines.append("   ⚠ ... and %d more (run: python -m tools.pool_stall_check)" % (len(flagged) - 6))
    return len(flagged), lines


def queue_unrunnable_summary(nodes=None, timeout=12):
    """Read-only UNRUNNABLE / memory-budget-impossible check on QUEUED (not-yet-dispatched) pool.queue lines
    (tools/pool_stall_check.py:check_queue) -- lines to print, never blocking, never raising (exit-0-always
    heartbeat). 2026-09-25: six revision-pinned queue lines sat 7.5h with the dispatcher logging "revision ...
    not provisioned on pool2" every cycle, and NOTHING outside that one log line ever surfaced it -- this makes
    it a heartbeat line instead, the same fix shape as pool_stall_summary() above for running jobs."""
    if pool_stall_check is None:
        return []
    try:
        report = pool_stall_check.check_queue(nodes=nodes, timeout=timeout)
    except Exception as e:
        return ["⚠ pool-queue-check failed to run (%s) -- treating as clean, not silently OK" % e]
    unrunnable = report.get("unrunnable", [])
    mem_stalled = report.get("memory_budget_stalled", [])
    lines = []
    if unrunnable or mem_stalled:
        lines.append("⚠ %s" % report.get("summary_line", "pool queue check flagged stuck line(s)"))
        for row in unrunnable[:4]:
            lines.append("   ⚠ %s" % pool_stall_check.format_unrunnable_row(row))
        for row in mem_stalled[:4]:
            lines.append("   ⚠ %s" % pool_stall_check.format_membudget_row(row))
    return lines


def active_agents(base=None):
    # Count in-flight Claude subagents by their transcript activity. Agent .output files under a session's
    # tasks/ dir are SYMLINKS to the growing JSONL transcript; backgrounded bash/monitor .output files are
    # REGULAR files. The original version counted only those symlinks (agents, not bash tasks) whose TARGET
    # was written in the last ~15 min — `find -L` follows the link so mtime is the agent's REAL activity,
    # not the symlink's creation time (a prior `-mmin -12` bug on the symlink ITSELF undercounted every
    # agent running longer than 12 min, since the symlink is stamped once at launch and never re-touched).
    #
    # 2026-09-08 BUG (owner-caught): that symlink walk sees ONLY standalone Agent-tool calls. A Workflow
    # fan-out (agent() calls inside a .claude/workflows/*.js script, or an ad-hoc workflow script) NEVER
    # gets a tasks/*.output symlink for its agents — each agent() spawns a transcript one level deeper, at
    # <session>/subagents/workflows/wf_<run-id>/agent-<id>.jsonl, which nothing in tasks/ ever points to.
    # So a live 6-agent workflow fan-out read as agents=0 here, printing UNDER-PARALLELIZED (false positive)
    # for the exact activity the gate exists to reward — the false alarm this session was built to fix.
    #
    # FIX: stop walking the /tmp symlink layer (an indirection that only covers one spawn path) and count
    # the transcripts directly. EVERY subagent, standalone or workflow-spawned, writes an appended
    # subagents/**/agent-<id>.jsonl (agent-<id>.jsonl at the top level for a standalone call, one directory
    # deeper under subagents/workflows/wf_<run-id>/ for a workflow's agent()); both are written/appended
    # identically while the agent runs, so recent mtime is real activity in both cases. `find` recurses by
    # default, so one glob (subagents/*, no depth limit) catches both shapes without special-casing either.
    root = base if base is not None else CLAUDE_PROJECTS_DIR
    out = sh(r"""find %s/*/subagents -name 'agent-*.jsonl' -mmin -15 2>/dev/null | wc -l""" % root, timeout=15)
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
    # DUP-OF-LANDED / OVERDUE lanes are NOT real coverage of the frontier -- a duplicate re-deriving a landed
    # result, or a job stuck well past its own history, occupies a core without doing anything the record does
    # not already have. Excluding them here (not bumping idle_pool -- this tool never kills anything, so the
    # core is not actually free) is what stops them being counted toward SATURATED (2026-09-25).
    flagged_pool, stall_lines = pool_stall_summary(POOL)
    lanes_pool = max(0, lanes_pool - flagged_pool)
    # QUEUED (not yet dispatched) lines are never counted in lanes_pool in the first place, so there is nothing
    # to subtract here -- this is purely a surfaced warning (2026-09-25: the failure was a stuck line nobody
    # SAW, not a lane miscounted as covered).
    queue_stuck_lines = queue_unrunnable_summary(POOL)
    n_open, top = open_tasks()
    agents = active_agents()

    total_lanes = lanes_local + lanes_pool + agents
    gpu_free = (0 <= gpu < 30) and not gpu_queue_busy()   # low util is NOT idle when the queue is draining a backlog
    # GAME PAUSE (tools/game.sh on → GAME_MODE): the local GPU is the owner's game, NOT spare capacity — exclude it
    # from every idle-capacity signal so it cannot trip under-parallelization. The mini-PC POOL + build/research
    # AGENTS are separate hardware / GPU-free and stay enforced (the owner's own game-time plan keeps those busy).
    game_paused = os.path.exists(os.path.join(ROOT, "research", "queue", "GAME_MODE"))
    gpu_free = gpu_free and not game_paused
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

    # PERSIST the verdict (2026-09-08 fix for Defect 2: "past fixes failed being manual/advisory/passive").
    # A printed line is read past; a record on disk lets a commit-time gate ask "how long has this been
    # true, continuously" and BLOCK past a budget instead of relying on someone re-reading the heartbeat.
    # Best-effort only (never raises, never changes this script's exit code — see tools/parallel_state.py).
    now_ts = time.time()
    state = parallel_state.persist(now_ts, under_agents=under_agents, under_compute=under_compute,
                                    agents=agents, idle_pool=idle_pool, gpu_free=gpu_free,
                                    n_open=(n_open if n_open is not None else 0))
    streak_c = state.get("since_under_compute")
    streak_a = state.get("since_under_agents")
    streak_c_min = int((now_ts - streak_c) / 60) if streak_c is not None else 0
    streak_a_min = int((now_ts - streak_a) / 60) if streak_a is not None else 0

    print("─ PARALLEL AUDIT ─ lanes=%d (local %d + pool %d + agents %d) | GPU=%s | open-tasks=%s"
          % (total_lanes, lanes_local, lanes_pool, agents, ("%d%%" % gpu if gpu >= 0 else "n/a"),
             (str(n_open) if n_open is not None else "?")))
    if game_paused:
        print("🎮 GAME PAUSE (GAME_MODE set) — the local GPU is the owner's game (excused from idle-parallelization); "
              "the mini-PC pool + build/research agents are separate/GPU-free and STILL enforced below.")
    for _stall_line in stall_lines:
        print(_stall_line)
    for _q_line in queue_stuck_lines:
        print(_q_line)

    # WAIVER SURFACING (2026-09-23, closes the "printed correctly, read past" shape for the escape hatches
    # THEMSELVES, not just the stall they excuse): a live .parallel_compute_waiver / .lane_waiver is easy to
    # forget is even open once the printed UNDER-PARALLELIZED line goes quiet. Print its CLASS + age every
    # cycle it is active, valid or not, so an open escape hatch stays visible on its own line.
    # Gate NAME, waiver path and max age all come from the gate modules via WAIVER_SOURCES (see its comment).
    # History: fix 2 found this loop passing ad hoc "compute"/"lane" labels as the gate name, which wrote a
    # second history key for the same file and double-charged the shared budget. Fix round 3 closed that
    # twice over: the names are imported, and the budget now charges each waiver EPISODE (file + content +
    # mtime) once for its real elapsed time, whoever reads it (tools/waiver_history.py docstring).
    for _print_label, _gate_name, _path, _max_h in WAIVER_SOURCES:
        try:
            _v = waiver_history.evaluate(_gate_name, _path, _max_h, now_ts=now_ts)
        except Exception:
            _v = {"active": False}
        _d = waiver_history.describe(_v)
        if _d:
            print("   🗒  %s waiver OPEN — %s" % (_print_label, _d))
    if under:
        why = []
        if under_agents:
            why.append("only %d build/research agent(s) running (floor %d, %d min straight) — agent work is "
                       "NOT compute-limited, FAN OUT MORE" % (agents, AGENT_FLOOR, streak_a_min))
        if under_compute:
            why.append("idle %s ; %d ready tasks vs %d lanes (%d min straight)"
                       % (", ".join(cap), n_open, total_lanes, streak_c_min))
        print("⛔ UNDER-PARALLELIZED (a STALL, not a hold) — %s." % " ; ".join(why))
        if under_compute and streak_c_min >= 30:
            print("   ⏱  dedicated compute has read idle-with-ready-work for %d min straight — past the "
                  "point `gates/compute_idle_persistent` blocks a commit on (mirrors gates/lane_starvation)."
                  % streak_c_min)
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


def _selftest_agent_detection():
    """FAILING DIRECTION FIRST (2026-09-08 fix). Before this fix, active_agents() walked ONLY
    tasks/*.output symlinks, which a Workflow fan-out's agent() calls never get (their transcripts land
    one directory deeper, at subagents/workflows/wf_<run-id>/agent-<id>.jsonl). A fixture holding ONLY a
    workflow-nested transcript reproduces that exact bug: the old logic would read agents=0 here. Also
    checks a standalone (top-level) transcript is still counted, and that a stale (>15min) transcript and
    a non-agent file are correctly excluded — so the fix doesn't just widen the glob into a false positive.
    """
    import shutil, tempfile, time as _time
    bad = []

    tmp = tempfile.mkdtemp(prefix="parallel_audit_selftest_")
    try:
        sess = os.path.join(tmp, "sess1", "subagents")
        wf = os.path.join(sess, "workflows", "wf_fake123")
        os.makedirs(sess)
        os.makedirs(wf)
        standalone = os.path.join(sess, "agent-standaloneFAKE.jsonl")
        workflow_agent = os.path.join(wf, "agent-workflowFAKE.jsonl")
        stale = os.path.join(sess, "agent-staleFAKE.jsonl")
        not_agent = os.path.join(sess, "notes.jsonl")  # must NOT match agent-*.jsonl
        for p in (standalone, workflow_agent, stale, not_agent):
            open(p, "w").close()
        old = _time.time() - 30 * 60  # 30 min ago -> outside the 15-min recency window
        os.utime(stale, (old, old))
        got = active_agents(base=tmp)
        if got != 2:
            bad.append("expected 2 recent transcripts (1 standalone + 1 workflow-nested), got %d over a "
                       "fixture with a stale transcript + a non-agent file present (must exclude both)" % got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # THE REGRESSION IN ISOLATION: a fan-out with agents running but NO standalone agent at all -- this is
    # what a pure Workflow run looks like, and it is exactly the case that read as 0 this session.
    tmp2 = tempfile.mkdtemp(prefix="parallel_audit_selftest2_")
    try:
        wf2 = os.path.join(tmp2, "sess1", "subagents", "workflows", "wf_onlyme")
        os.makedirs(wf2)
        open(os.path.join(wf2, "agent-onlyworkflow.jsonl"), "w").close()
        got2 = active_agents(base=tmp2)
        if got2 != 1:
            bad.append("REGRESSION: a workflow-ONLY fan-out (no standalone agent) counted as %d, expected "
                       "1 -- this is the exact false 'agents=0' bug a live workflow fan-out hit" % got2)
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    return bad


def _selftest_persist_state():
    """FAILING DIRECTION FIRST (2026-09-08, Defect 2). A condition true on two CONSECUTIVE cycles must keep
    reporting the streak's ORIGINAL start, not reset it each cycle — a reset-every-cycle bug would make
    "how long has this read true, continuously" always ~0 and the persistence-based gate unable to ever
    fire, the exact "check that cannot fail" failure class (docs/FAILURE_GATE_MATRIX.md class 3)."""
    bad = []
    t0 = 1_000_000.0
    s1 = parallel_state.next_state(None, t0, under_agents=True, under_compute=False,
                                    agents=1, idle_pool=0, gpu_free=False, n_open=5)
    if s1["since_under_agents"] != t0:
        bad.append("first cycle true -> since_under_agents must equal now (%r), got %r" % (t0, s1["since_under_agents"]))
    t1 = t0 + 900.0  # 15 min later, STILL true
    s2 = parallel_state.next_state(s1, t1, under_agents=True, under_compute=False,
                                    agents=1, idle_pool=0, gpu_free=False, n_open=5)
    if s2["since_under_agents"] != t0:
        bad.append("REGRESSION: a condition true on two consecutive cycles reset its streak start (got %r, "
                   "expected the ORIGINAL %r) -- this would make persistence-duration always ~0" % (s2["since_under_agents"], t0))
    t2 = t1 + 900.0  # now healthy
    s3 = parallel_state.next_state(s2, t2, under_agents=False, under_compute=False,
                                    agents=5, idle_pool=0, gpu_free=False, n_open=5)
    if s3["since_under_agents"] is not None:
        bad.append("FALSE POSITIVE: streak start was not cleared once the condition read healthy")
    t3 = t2 + 60.0  # re-triggers after a healthy gap
    s4 = parallel_state.next_state(s3, t3, under_agents=True, under_compute=False,
                                    agents=1, idle_pool=0, gpu_free=False, n_open=5)
    if s4["since_under_agents"] != t3:
        bad.append("a condition that RE-TRIGGERS after a healthy gap must start a NEW streak at the "
                   "current time (%r), not resurrect the old one (got %r)" % (t3, s4["since_under_agents"]))
    # freshness: absent/old state must read as NO SIGNAL, never as "still under-parallelized".
    if parallel_state.is_fresh(None, t3):
        bad.append("FALSE POSITIVE: an absent state was treated as fresh")
    stale_state = {"generated_at": t3 - parallel_state.STALE_S - 1}
    if parallel_state.is_fresh(stale_state, t3):
        bad.append("FALSE POSITIVE: a state older than STALE_S was treated as fresh")
    return bad


def _selftest():
    """The 2026-08-26 root cause was that this check had shipped UNABLE TO FIRE (the old bar needed idle compute
    AND a board-count that was structurally ~1). A check that cannot fail is the bug. This selftest asserts the
    fixed decision FIRES in its failing direction (few agents / idle dedicated lane) and stays quiet when saturated.
    It also runs _selftest_agent_detection() (2026-09-08 workflow-undercount fix) and _selftest_persist_state()
    (2026-09-08 persistence-record fix, Defect 2)."""
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
    agent_bad = _selftest_agent_detection()
    persist_bad = _selftest_persist_state()
    if bad or agent_bad or persist_bad:
        print("PARALLEL_AUDIT SELFTEST FAILED (the check is unable to fire correctly):")
        for c, got in bad:
            print("   case %s -> got under=%s, expected %s" % (c[:5], got, c[5]))
        for msg in agent_bad:
            print("   active_agents(): %s" % msg)
        for msg in persist_bad:
            print("   parallel_state: %s" % msg)
        sys.exit(1)
    print("parallel_audit selftest OK — decision fires on agents<floor OR an idle dedicated lane with ready "
          "work (quiet when saturated); active_agents() counts standalone AND workflow-nested transcripts; "
          "parallel_state tracks continuous-streak duration correctly.")
    sys.exit(0)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    sys.exit(main())
