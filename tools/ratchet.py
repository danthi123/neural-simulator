#!/usr/bin/env python3
"""tools/ratchet.py — the FAN-OUT RATCHET: the dispatch half of the enforced-parallelism engine.

WHY THIS EXISTS. `tools/backlog.py` is the ENUMERATOR: it mechanically lists every independent ready
item, each tagged {id, what, source+anchor, target, verify, deps, lane, leverage, on_board}. Enumeration
alone changes nothing — the work still sits until someone reads the list and launches it, which is exactly
the ad-hoc, remembered step that keeps the machine serial. This tool is the DISPATCHER: it consumes
backlog.json + live capacity and routes each independent ready item to the cheapest fitting lane, so the
"fan out to ALL independent ready work" rule is ENFORCED, not remembered (design doc:
docs/plans/2026-08-21-enforcement-layer-self-maintaining-project-os.md §3).

THE AUTONOMY DESIGN — the auto-vs-confirm question is resolved by COST, not by taste:

  • FREE lanes (pool-cpu, gpu-queue) — ~0 tokens, safe → AUTO-dispatch, UNCONSTRAINED. Fill every idle free
    slot with the top-ranked dep-met dispatchable items, in rank order. `pool-cpu` → tools/pool_queue.sh add
    (or tools/sweep_pool.sh for a grid); `gpu-queue` → tools/gpu_queue.sh add (sequential, VRAM-safe,
    singleton daemon — this tool NEVER starts it, only `add`s). Continuous refill: designed to run every
    heartbeat, so on a completion the freed slot pulls the next item next cycle.

  • AGENT lane — tokens count toward the usage limit → GENERATE-AND-CONFIRM, BUDGETED. This tool does NOT
    auto-spawn agents. It EMITS a ranked launch-list (each entry: the exact agent-prompt seed + a suggested
    model tier + why the item needs a mind), capped to RATCHET_AGENT_BUDGET, for the main session/owner to
    fire. The config flag RATCHET_AUTO_AGENTS (default false) is reserved for a future auto-spawn mode; in
    THIS build auto-spawn is not implemented and any attempt to spawn is a caught invariant violation.

THE BLOCKING RATCHET VERDICT (the enforcement, mirrors tools/parallel_audit.py):
  ⛔ UNDER-PARALLELIZED  iff  idle FREE capacity  AND  a dep-met, dispatchable (command-carrying) free-lane
                             item that is not already in flight.  → AUTO-dispatch fixes it (free = safe).
  ⚠  FREE-LANES-READY-BUT-COMMANDLESS  iff  idle free capacity + dep-met free-lane items that carry NO
                             runnable command. This is NOT a hold and NOT a fabricated dispatch — the honest
                             fix is to AUTHOR a command (cheap). Enqueuing prose would be the exact
                             anti-fabrication violation the whole engine forbids (pool_queue.sh refuses it;
                             the GPU queue would run it and fail). The ratchet surfaces these instead.
  ✓  SATURATED  otherwise (no idle free capacity, or no ready free-lane work).
The agent lane is REPORTED every run (the pending launch-list); it NEVER blocks (tokens are budgeted).

MEASURE WORK, NOT LANE COUNT (folds in the parallel_audit fix that kills the false-SATURATED): a lane counts
as "serving" only by its LIVE/RECENT work — queued jobs + a running job + a recent dispatch — not by the mere
existence of a queue file. Idle free capacity = target depth minus that live work.

SELF-CHECK, THE PART THAT CANNOT SILENTLY BREAK (the gates/ registry philosophy: a check that cannot fail is
no check). Before it acts, the ratchet validates its OWN plan with `plan_invariants()`; `execute()` REFUSES a
plan that violates any invariant. `--selftest` proves BOTH directions: a correct plan passes every invariant
AND deliberately-broken plans (an agent item in the free dispatches, a double-dispatch, a dep-blocked
dispatch, an over-capacity dispatch, a ready item left undispatched with idle capacity, an intended
agent-spawn side-effect) are each CAUGHT.

    tools/ratchet.py                     # DRY-RUN: print the plan it WOULD run (no side effects) — the default
    tools/ratchet.py --dispatch          # LIVE: auto-dispatch free-lane items to the queues (agents still emit-only)
    tools/ratchet.py --json              # emit the full plan as JSON
    tools/ratchet.py --quiet             # one heartbeat-friendly block
    tools/ratchet.py --regen             # regenerate backlog.json first (tools/backlog.py --no-vikunja), then plan
    tools/ratchet.py --probe-pool        # measure real idle pool cores over SSH (default: queue-depth capacity)
    tools/ratchet.py --selftest          # pass + demonstrated failing direction, then exit
    tools/ratchet.py --how               # the heartbeat-integration contract
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKLOG_JSON = os.path.join(ROOT, "research", "coordination", "backlog.json")
DISPATCH_LEDGER = os.path.join(ROOT, "research", "coordination", "ratchet_dispatched.jsonl")
BACKLOG_PY = os.path.join(ROOT, "tools", "backlog.py")
GPU_QUEUE_SH = os.path.join(ROOT, "tools", "gpu_queue.sh")
POOL_QUEUE_SH = os.path.join(ROOT, "tools", "pool_queue.sh")

FREE_LANES = ("gpu-queue", "pool-cpu")
POOL_NODES = ("pool40", "pool41", "pool42")


# ─────────────────────────────────────────────────────────────────────────────
# config — the autonomy knobs (env-overridable; the defaults are the safe ones)
# ─────────────────────────────────────────────────────────────────────────────
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Config:
    def __init__(self):
        # AUTO-spawn agents? Default FALSE (emit-only). Reserved for a future mode; unimplemented in this build.
        self.auto_agents = _env_bool("RATCHET_AUTO_AGENTS", False)
        # how many agent-lane items to emit per cycle (token budget — agents count toward the usage limit)
        self.agent_budget = _env_int("RATCHET_AGENT_BUDGET", 3)
        # target live-work depth per free lane (idle capacity = target - live work). GPU is one sequential
        # card, so a shallow target keeps the pipeline full without dumping the whole backlog on it; the pool
        # is three nodes, so it can hold more staged.
        self.gpu_target = _env_int("RATCHET_GPU_TARGET_DEPTH", 3)
        self.pool_target = _env_int("RATCHET_POOL_TARGET_DEPTH", 6)
        # an item is not re-dispatched if its id was dispatched within this many hours (continuous-refill dedup)
        self.dedup_ttl_h = _env_int("RATCHET_DEDUP_TTL_H", 24)
        # the dispatch dedup-ledger (overridden by main()/selftest with the actual path)
        self._ledger_path = DISPATCH_LEDGER

    def target_for(self, lane):
        return self.gpu_target if lane == "gpu-queue" else self.pool_target


# ─────────────────────────────────────────────────────────────────────────────
# deps interpretation — the backlog's `deps` is PROSE, so readiness is a heuristic
# ─────────────────────────────────────────────────────────────────────────────
# BLOCKING language: a real prerequisite that is not yet met (the item cannot START). Measured against the
# actual deps vocabulary the generator emits: "wire into /api/brain-chat first" (flip needs wiring first) and
# "the spiking replacement must reach parity or an honest negative" (scaffold burn-down needs the replacement
# built first) are the two genuine blockers; "" / "none (...)" / "retire/close at: S2+" (a CLOSURE condition,
# not a start-prerequisite) / "mechanism: <x>" (provenance tag) are all READY.
_BLOCKED_RE = re.compile(
    r"\bfirst\b|must (reach|be|land|pass|exist|complete)|reach parity|\brequires?\b|depends on|"
    r"\bblocked\b|\bpending\b|\bawait|not yet|prerequisite|once .*(lands|exists|passes|is built)|"
    r"after .*(lands|is built|passes)",
    re.I,
)


def is_blocked(deps: str) -> bool:
    return bool(_BLOCKED_RE.search(deps or ""))


# ─────────────────────────────────────────────────────────────────────────────
# command resolution — a free-lane item is only AUTO-dispatchable if it carries a
# runnable command. NEVER fabricate one (pool_queue.sh refuses prose; the GPU
# queue would run prose and fail). Real backlog items today carry none, so they
# surface as NEEDS-COMMAND rather than as a fabricated dispatch.
# ─────────────────────────────────────────────────────────────────────────────
_RUNNER_RE = re.compile(
    r"((?:[A-Z_]+=\S+\s+)*(?:\S*python\S*\s+)?-u?\s*-m\s+research\.runners\.[A-Za-z0-9_.]+[^\n|]*)")


def command_for(item: dict):
    """Return a runnable shell command for a free-lane item, or None if it carries none.

    Priority: an explicit `cmd`/`command` field (the clean contract a future backlog can populate, and what
    the selftest mocks) → then a `-m research.runners.X ...` invocation embedded in the item's prose fields.
    Anything else → None (NEEDS-COMMAND)."""
    for k in ("cmd", "command", "dispatch"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    blob = " ".join(str(item.get(k, "")) for k in ("what", "detail", "target", "verify"))
    m = _RUNNER_RE.search(blob)
    return m.group(1).strip() if m else None


def _has_runner_module(cmd: str) -> bool:
    """Anti-fabrication guard for live GPU dispatch: only enqueue a command that actually names a runner."""
    return bool(cmd) and bool(re.search(r"-m\s+research\.runners\.[A-Za-z0-9_.]+", cmd))


# ─────────────────────────────────────────────────────────────────────────────
# live capacity + in-flight state (measure WORK, not lane existence)
# ─────────────────────────────────────────────────────────────────────────────
def _shared_queue_root() -> str:
    """The checkout whose persistent dispatchers consume the queues (same resolution as lane_check /
    lane_starvation): SIM_QUEUE_ROOT override → the git common-dir's parent → this checkout."""
    override = os.environ.get("SIM_QUEUE_ROOT")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if common:
            return os.path.dirname(common)
    except Exception:
        pass
    return ROOT


def _queue_dir() -> str:
    return os.path.join(_shared_queue_root(), "research", "queue")


def _queue_lines(name: str):
    p = os.path.join(_queue_dir(), name)
    out = []
    if os.path.exists(p):
        for ln in open(p, errors="ignore").read().split("\n"):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "\t" in ln:                       # pool queue is "<epoch>\t<cmd>  #checked:<reason>"
                ln = ln.split("\t", 1)[1]
            out.append(ln.split("#checked:")[0].strip())
    return out


def _running_cmds():
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [l.strip() for l in out.split("\n") if "research.runners" in l and "grep" not in l]


def _gpu_paused() -> bool:
    return os.path.exists(os.path.join(_queue_dir(), "GPU_PAUSE"))


def _pool_idle_cores(timeout=12) -> int:
    """Real idle pool cores over SSH (opt-in via --probe-pool; SSH is slow, so it is not the default)."""
    idle = 0
    for h in POOL_NODES:
        try:
            out = subprocess.run(
                ["bash", "-c",
                 "timeout 8 ssh -o BatchMode=yes %s \"nproc; cut -d' ' -f1 /proc/loadavg\" 2>/dev/null" % h],
                capture_output=True, text=True, timeout=timeout).stdout.split()
            if len(out) >= 2:
                idle += max(0, int(int(out[0]) - float(out[1])))
        except (ValueError, IndexError, subprocess.TimeoutExpired, OSError):
            pass
    return idle


def live_work_depth(lane: str) -> int:
    """LIVE/RECENT work serving a lane = queued jobs + a running job (measure work, not queue existence)."""
    if lane == "gpu-queue":
        depth = len(_queue_lines("gpu.queue"))
        run = os.path.join(_queue_dir(), "gpu.running")
        if os.path.exists(run) and os.path.getsize(run) > 0:
            depth += 1
        return depth
    if lane == "pool-cpu":
        return len(_queue_lines("pool.queue"))
    return 0


def probe_capacity(cfg: Config, probe_pool=False) -> dict:
    """Idle FREE slots per free lane = target depth − live work depth. Pool can additionally be gated by real
    idle cores (--probe-pool). GPU_PAUSE does not zero capacity (staging while paused is harmless — jobs wait
    for resume — and keeps the pipeline full), but it is reported."""
    cap = {}
    for lane in FREE_LANES:
        cap[lane] = max(0, cfg.target_for(lane) - live_work_depth(lane))
    if probe_pool:
        cap["pool-cpu"] = min(cap["pool-cpu"], max(0, _pool_idle_cores()))
    return cap


def read_inflight(cfg: Config):
    """Return (dispatched_ids, inflight_cmds) so nothing is double-dispatched.

    ids: this ratchet's own dispatch-ledger, within the dedup TTL (so a completed-and-refill cycle does not
         re-launch the same backlog item). cmds: whatever is already staged/running (both queues + ps), so an
         item whose command is already in flight — however it got there — is not queued again."""
    ids = set()
    if os.path.exists(cfg._ledger_path):
        cutoff = time.time() - cfg.dedup_ttl_h * 3600
        for ln in open(cfg._ledger_path, errors="ignore").read().split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if float(rec.get("ts", 0)) >= cutoff and rec.get("id"):
                ids.add(rec["id"])
    cmds = set()
    for name in ("gpu.queue", "pool.queue"):
        cmds.update(_norm(c) for c in _queue_lines(name))
    cmds.update(_norm(c) for c in _running_cmds())
    cmds.discard("")
    return ids, cmds


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", (cmd or "").strip())


# ─────────────────────────────────────────────────────────────────────────────
# agent launch-list construction (emit-only) — the exact prompt seed + why-a-mind
# ─────────────────────────────────────────────────────────────────────────────
def _suggest_tier(item: dict) -> str:
    """Model-tier the agent per CLAUDE.md cost-routing: haiku=mechanical · sonnet=moderate · opus=hard."""
    t = (item.get("what", "") + " " + item.get("detail", "") + " " + item.get("source", "")).lower()
    if "walls-ledger" in item.get("source", "") or "wall" in t or "surpass" in t or "honest negative" in t:
        return "opus"        # surpassing a wall / banking an honest negative is hard judgment
    if any(k in t for k in ("flip", "wire", "integrat", "retire host scaffold", "on-by-default")):
        return "sonnet"      # production wiring / flips: moderate, well-scoped judgment
    if "coverage gap" in t or "gates/" in t or item.get("source") == "failure-log":
        return "sonnet"      # a new registry gate: moderate build
    return "sonnet"


def agent_prompt_seed(item: dict) -> str:
    """A self-contained prompt seed for an agent lane item (no dependence on this session's context)."""
    lines = [
        "TASK: %s" % item.get("what", ""),
        "CONTEXT: %s" % item.get("detail", ""),
        "SOURCE ANCHOR (read this first): %s" % item.get("anchor", ""),
        "TARGET (where the work lands): %s" % item.get("target", ""),
        "VERIFY (the acceptance bar): %s" % item.get("verify", ""),
    ]
    deps = item.get("deps", "")
    if deps and deps not in ("", "none"):
        lines.append("DEPENDENCIES / NOTES: %s" % deps)
    rel = item.get("related_anchors") or []
    if rel:
        lines.append("RELATED ANCHORS: %s" % "; ".join(rel[:4]))
    return "\n".join(lines)


def _why_a_mind(item: dict) -> str:
    src = item.get("source", "")
    return {
        "ledger-flip": "flip a de-risked faculty to production-default: needs wiring + a lesion-decisive "
                       "6-seed verification, i.e. integration judgment, not a mechanical sweep",
        "ledger-scaffold": "convert a host shortcut to a spiking/synaptic mechanism (or bank an honest "
                           "negative) while keeping the default answer stable — design work",
        "walls-ledger": "build the named biological surpass for an open wall + adversarially verify it — the "
                        "hardest judgment class",
        "failure-log": "author a new registry gate (check + selftest that fails in its failing direction) for "
                       "an open coverage gap — a build, not a run",
        "finding-residual": "run the finding's named next-lever with anti-cheats + a like-for-like control and "
                            "verify the residual moves the right way",
        "vikunja": "an owner-tracked board task: production wiring / de-risk per the task body, needing design "
                   "judgment",
    }.get(src, "requires design judgment, not a mechanical sweep")


# ─────────────────────────────────────────────────────────────────────────────
# the planner — pure function of (items, capacity, in-flight, config)
# ─────────────────────────────────────────────────────────────────────────────
class Plan:
    def __init__(self):
        self.dispatches = []          # [{id, lane, command, item}] — free-lane, ready, command-carrying
        self.needs_command = []       # free-lane items ready but with NO runnable command (author one)
        self.deferred_free = []       # free-lane dispatchable items over this cycle's capacity (refill later)
        self.agent_launch_list = []   # [{id, tier, why, prompt, item}] — EMIT-ONLY, budgeted
        self.agent_deferred = []      # ready agent items beyond the token budget
        self.skipped_blocked = []     # items whose deps are unmet
        self.skipped_inflight = []    # items already dispatched/queued/running
        self.capacity = {}
        self.verdict = ""
        self.verdict_kind = ""        # UNDER-PARALLELIZED | READY-NO-CMD | SATURATED


def build_plan(items, capacity, inflight_ids, inflight_cmds, cfg: Config) -> Plan:
    p = Plan()
    p.capacity = dict(capacity)
    items = sorted(items, key=lambda it: (-it.get("leverage", 0), it.get("rank", 1_000_000)))

    remaining = dict(capacity)      # free slots left per lane this cycle
    for it in items:
        lane = it.get("lane")
        iid = it.get("id")
        if lane in FREE_LANES:
            if is_blocked(it.get("deps", "")):
                p.skipped_blocked.append(it)
                continue
            cmd = command_for(it)
            if iid in inflight_ids or (cmd and _norm(cmd) in inflight_cmds):
                p.skipped_inflight.append(it)
                continue
            if not cmd:
                p.needs_command.append(it)          # ready, but nothing safe to enqueue → surface it
                continue
            if remaining.get(lane, 0) > 0:
                p.dispatches.append({"id": iid, "lane": lane, "command": cmd, "item": it})
                remaining[lane] -= 1
            else:
                p.deferred_free.append(it)          # dispatchable but no free slot this cycle (refill next)
        elif lane == "agent":
            if is_blocked(it.get("deps", "")):
                p.skipped_blocked.append(it)
                continue
            if iid in inflight_ids:
                p.skipped_inflight.append(it)
                continue
            if len(p.agent_launch_list) < cfg.agent_budget:
                p.agent_launch_list.append({
                    "id": iid, "tier": _suggest_tier(it), "why": _why_a_mind(it),
                    "prompt": agent_prompt_seed(it), "item": it})
            else:
                p.agent_deferred.append(it)

    # verdict — measure whether idle free capacity coexists with actionable free-lane work. "Actionable"
    # command-less work is counted only on lanes that STILL have an idle slot (a command-less GPU item while
    # the GPU queue is full is a real backlog gap, but it is not what makes THIS cycle under-parallelized).
    idle_free = sum(v for v in capacity.values() if v > 0)
    idle_lanes = {lane for lane in FREE_LANES if capacity.get(lane, 0) > 0}
    commandless_on_idle = [it for it in p.needs_command if it.get("lane") in idle_lanes]
    if p.dispatches:
        p.verdict_kind = "UNDER-PARALLELIZED"
        p.verdict = ("⛔ UNDER-PARALLELIZED — %d free slot(s) idle with dep-met, command-carrying work ready; "
                     "AUTO-dispatching %d item(s) (free lanes are safe)."
                     % (idle_free, len(p.dispatches)))
    elif idle_free > 0 and commandless_on_idle:
        p.verdict_kind = "READY-NO-CMD"
        p.verdict = ("⚠ FREE LANES READY BUT COMMAND-LESS — %d free slot(s) idle on lane(s) %s and %d dep-met "
                     "item(s) there are ready but carry NO runnable command. This is NOT a hold and NOT grounds "
                     "to fabricate a command (pool_queue.sh would refuse it; the GPU queue would run it and "
                     "fail). Author a command (cheap) or add a `cmd` field to the backlog item, then re-run."
                     % (idle_free, ", ".join(sorted(idle_lanes)), len(commandless_on_idle)))
    else:
        why = ("no idle free capacity" if idle_free == 0 else "no ready free-lane work")
        p.verdict_kind = "SATURATED"
        p.verdict = "✓ SATURATED (%s)." % why
    return p


# ─────────────────────────────────────────────────────────────────────────────
# the SELF-CHECK — the invariants that must hold, and MUST fail in their failing
# direction (the gates/ registry philosophy). execute() refuses a plan that fails.
# ─────────────────────────────────────────────────────────────────────────────
def plan_invariants(plan: Plan, capacity, inflight_ids, inflight_cmds) -> list:
    """Return a list of violations. Empty == the plan is safe to act on.

    This is the check that makes a broken ratchet LOUD instead of quietly wrong."""
    problems = []

    # (1) FREE dispatches must never contain an agent-lane item (agents are emit-only, budgeted).
    for d in plan.dispatches:
        if d["item"].get("lane") == "agent":
            problems.append("agent-lane item %s is in the FREE dispatches (must be emit-only)" % d["id"])
        if d["lane"] not in FREE_LANES:
            problems.append("dispatch %s targets a non-free lane %r" % (d["id"], d["lane"]))

    # (2) no double-dispatch: nothing already in flight, and no duplicate id/command within the plan.
    seen_ids, seen_cmds = set(), set()
    for d in plan.dispatches:
        if d["id"] in inflight_ids:
            problems.append("double-dispatch: %s id already in the dispatch ledger" % d["id"])
        if _norm(d["command"]) in inflight_cmds:
            problems.append("double-dispatch: %s command already queued/running" % d["id"])
        if d["id"] in seen_ids:
            problems.append("double-dispatch within plan: id %s twice" % d["id"])
        if _norm(d["command"]) in seen_cmds:
            problems.append("double-dispatch within plan: identical command twice (%s)" % d["id"])
        seen_ids.add(d["id"])
        seen_cmds.add(_norm(d["command"]))

    # (3) respect deps: never dispatch a dep-blocked item.
    for d in plan.dispatches:
        if is_blocked(d["item"].get("deps", "")):
            problems.append("dispatched a dep-BLOCKED item: %s (deps=%r)" % (d["id"], d["item"].get("deps")))

    # (4) never exceed a lane's idle capacity.
    for lane in FREE_LANES:
        n = sum(1 for d in plan.dispatches if d["lane"] == lane)
        if n > capacity.get(lane, 0):
            problems.append("lane %s over capacity: dispatched %d > %d idle slots" % (lane, n, capacity.get(lane, 0)))

    # (5) completeness: no ready, dispatchable, command-carrying free-lane item may be left undispatched while
    #     its lane still has an idle slot. (Leaving one is the UNDER-PARALLELIZED failure the tool exists to fix.)
    dispatched_per_lane = {lane: sum(1 for d in plan.dispatches if d["lane"] == lane) for lane in FREE_LANES}
    for it in plan.deferred_free:
        lane = it.get("lane")
        if dispatched_per_lane.get(lane, 0) < capacity.get(lane, 0):
            problems.append("ready free-lane item %s deferred while lane %s still had an idle slot"
                            % (it.get("id"), lane))

    return problems


def assert_no_agent_spawn(actions: list) -> list:
    """The side-effect guard: the concrete actions execute() intends to perform must contain NO agent spawn.
    Returns violations (empty == clean). Proven to fail in its failing direction by the selftest."""
    return ["planned a %r side effect — agents are EMIT-ONLY in this build (confirm-rule violation)" % a["kind"]
            for a in actions if a.get("kind") == "spawn-agent"]


# ─────────────────────────────────────────────────────────────────────────────
# execution — refuses an invalid plan; touches the queues only when live=True
# ─────────────────────────────────────────────────────────────────────────────
def execute(plan: Plan, cfg: Config, capacity, inflight_ids, inflight_cmds, live=False):
    """Turn the plan into actions and (only if live) perform them. Returns (actions, results, refusal).

    refusal is a non-empty list iff the plan failed its own invariants — execution is REFUSED (nothing runs).
    Agents are NEVER spawned here; the launch-list is emit-only regardless of live/auto_agents in this build."""
    refusal = plan_invariants(plan, capacity, inflight_ids, inflight_cmds)
    if refusal:
        return [], [], refusal

    # Build the concrete side-effect list. Free-lane dispatches → queue-add actions. Agents → NOT added
    # (emit-only). If a future build enabled auto-spawn it would append kind="spawn-agent" here; the guard
    # below would then catch it, which is exactly the confirm-rule enforcement.
    actions = [{"kind": "queue-add", "lane": d["lane"], "id": d["id"], "command": d["command"]}
               for d in plan.dispatches]
    if cfg.auto_agents:
        # Reserved flag, unimplemented on purpose: we DO NOT append spawn actions. Announce and stay emit-only.
        actions.append({"kind": "note",
                        "text": "RATCHET_AUTO_AGENTS is set but auto-spawn is not implemented in this build; "
                                "agents remain emit-only (the launch-list stands)."})

    spawn_violations = assert_no_agent_spawn(actions)
    if spawn_violations:
        return actions, [], spawn_violations

    results = []
    for a in actions:
        if a["kind"] != "queue-add":
            continue
        if not live:
            results.append({"id": a["id"], "lane": a["lane"], "status": "DRY-RUN (would enqueue)",
                            "command": a["command"]})
            continue
        results.append(_enqueue(a))
        if results[-1]["status"] == "queued":
            _record_dispatch(cfg, a)
    return actions, results, []


def _enqueue(a: dict) -> dict:
    """Actually stage a free-lane command. Anti-fabrication: only a command that names a runner is enqueued."""
    lane, cmd, iid = a["lane"], a["command"], a["id"]
    if not _has_runner_module(cmd):
        return {"id": iid, "lane": lane, "status": "REFUSED (command names no research.runners module)",
                "command": cmd}
    try:
        if lane == "gpu-queue":
            r = subprocess.run(["bash", GPU_QUEUE_SH, "add", cmd],
                               capture_output=True, text=True, timeout=30)
        else:  # pool-cpu
            checked = "ratchet: backlog item %s (%s)" % (iid, a.get("anchor", "auto-dispatched"))
            r = subprocess.run(["bash", POOL_QUEUE_SH, "add", cmd, "--checked", checked],
                               capture_output=True, text=True, timeout=180)
        ok = (r.returncode == 0)
        return {"id": iid, "lane": lane, "status": "queued" if ok else "FAILED",
                "detail": (r.stdout + r.stderr).strip()[:200], "command": cmd}
    except Exception as e:
        return {"id": iid, "lane": lane, "status": "FAILED", "detail": "%s: %s" % (type(e).__name__, e),
                "command": cmd}


def _record_dispatch(cfg: Config, a: dict):
    os.makedirs(os.path.dirname(cfg._ledger_path), exist_ok=True)
    with open(cfg._ledger_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), "id": a["id"], "lane": a["lane"],
                            "command": a["command"]}) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# rendering
# ─────────────────────────────────────────────────────────────────────────────
def render(plan: Plan, results, refusal, meta, quiet=False) -> str:
    if quiet:
        gp = " (GPU paused)" if meta.get("gpu_paused") else ""
        return ("─ RATCHET ─ %s | free-cap gpu=%d pool=%d%s | dispatch=%d needs-cmd=%d agents-emitted=%d"
                % (plan.verdict_kind, plan.capacity.get("gpu-queue", 0), plan.capacity.get("pool-cpu", 0), gp,
                   len(plan.dispatches), len(plan.needs_command), len(plan.agent_launch_list)))
    o = []
    o.append("=" * 100)
    o.append("FAN-OUT RATCHET  —  backlog: %s (%s items)  |  %s"
             % (os.path.relpath(meta["backlog_path"], ROOT), meta.get("n_items", "?"),
                "LIVE (--dispatch)" if meta["live"] else "DRY-RUN (no side effects)"))
    o.append("free capacity (idle slots): gpu-queue=%d  pool-cpu=%d%s   [targets gpu=%d pool=%d]"
             % (plan.capacity.get("gpu-queue", 0), plan.capacity.get("pool-cpu", 0),
                "  (GPU PAUSED — staged jobs wait for resume)" if meta.get("gpu_paused") else "",
                meta["cfg"].gpu_target, meta["cfg"].pool_target))
    o.append("=" * 100)
    o.append(plan.verdict)
    o.append("")

    if refusal:
        o.append("⛔ EXECUTION REFUSED — the plan failed its own invariants (nothing ran):")
        for r in refusal:
            o.append("   - " + r)
        o.append("")

    o.append("── FREE-LANE DISPATCH (auto, 0 tokens) ──")
    if plan.dispatches:
        for d in plan.dispatches:
            res = next((r for r in results if r["id"] == d["id"]), None)
            st = res["status"] if res else "(planned)"
            o.append("  • [%s] %s  →  %s" % (d["lane"], d["id"], st))
            o.append("      $ %s" % d["command"][:150])
    else:
        o.append("  (none dispatched this cycle)")
    if plan.needs_command:
        o.append("  NEEDS-COMMAND (dep-met, idle capacity, but no runnable command — author one, don't fabricate):")
        for it in plan.needs_command[:12]:
            o.append("    ⚠ [%s] %s — %s" % (it.get("lane"), it.get("id"), it.get("what", "")[:70]))
            o.append("        anchor: %s" % it.get("anchor", ""))
    if plan.deferred_free:
        o.append("  deferred (dispatchable, no free slot this cycle — continuous refill picks these next): %d"
                 % len(plan.deferred_free))

    o.append("")
    o.append("── AGENT LANE (EMIT-ONLY, budget=%d — tokens count; fire from the main session) ──"
             % meta["cfg"].agent_budget)
    if plan.agent_launch_list:
        for i, a in enumerate(plan.agent_launch_list, 1):
            o.append("  %d. [%s | tier=%s] %s" % (i, a["id"], a["tier"], a["item"].get("what", "")[:66]))
            o.append("      why a mind: %s" % a["why"])
            o.append("      prompt seed:")
            for ln in a["prompt"].split("\n"):
                o.append("        " + ln)
    else:
        o.append("  (no ready agent-lane items)")
    if plan.agent_deferred:
        o.append("  + %d more ready agent items beyond the budget (raise RATCHET_AGENT_BUDGET to emit more)."
                 % len(plan.agent_deferred))

    o.append("")
    o.append("── skipped: %d dep-blocked, %d already in-flight ──"
             % (len(plan.skipped_blocked), len(plan.skipped_inflight)))
    return "\n".join(o)


HOW = """\
HOW THE HEARTBEAT / MAIN SESSION SHOULD CONSUME tools/ratchet.py
───────────────────────────────────────────────────────────────────────────
This is the DISPATCH half of the enforced-parallelism engine (tools/backlog.py is the enumerator half).
The heartbeat cycle is: regenerate the backlog → RATCHET dispatches free work → emit the agent launch-list.

  1. Regenerate the backlog (fast, file-only):  tools/ratchet.py --regen --dispatch
     (or run tools/backlog.py yourself first, then tools/ratchet.py --dispatch).
  2. FREE lanes auto-dispatch with NO confirmation (0 tokens, safe): every idle gpu-queue / pool-cpu slot is
     filled with the top-ranked dep-met, command-carrying items. Continuous refill: because it runs each
     heartbeat, a completion frees a slot that the next cycle fills — the pipeline stays full.
  3. The AGENT launch-list is EMITTED, never spawned. Fire the listed agents from the main session at the
     suggested model tier, up to the token budget (RATCHET_AGENT_BUDGET). Each entry is a self-contained
     prompt seed — no dependence on the session that produced it.
  4. The VERDICT is the enforcement: ⛔ UNDER-PARALLELIZED means idle free capacity coexisted with ready
     dispatchable work — and the same run just fixed it by dispatching. ⚠ READY-NO-CMD means free-lane work is
     ready but carries no runnable command: author one (cheap); do NOT fabricate a command and do NOT treat it
     as a hold. ✓ SATURATED is the only state in which holding a free lane is earned.
  5. It NEVER starts the GPU daemon (only `add`s to its queue) and NEVER spawns an agent. Default run is a
     DRY-RUN preview; --dispatch is required to touch the queues. Both are safe to run any time.
"""


# ─────────────────────────────────────────────────────────────────────────────
# selftest — proves BOTH directions (required; mirrors backlog.py + the gates registry)
# ─────────────────────────────────────────────────────────────────────────────
def _mock_cfg(ledger_path):
    cfg = Config()
    cfg.auto_agents = False
    cfg.agent_budget = 2
    cfg.gpu_target = 2
    cfg.pool_target = 2
    cfg.dedup_ttl_h = 24
    cfg._ledger_path = ledger_path
    return cfg


def _mock_backlog():
    """Free-lane items carry a `cmd` (the clean contract); one is dep-blocked; one is already in-flight; plus
    agent-lane items and a command-less free-lane item (the real-world NEEDS-COMMAND case)."""
    def cmd(x):
        return "SIM_BACKEND=cupy .venv/bin/python -u -m research.runners.%s --json raw/o.json" % x
    return [
        {"id": "g1", "lane": "gpu-queue", "leverage": 90, "rank": 1, "deps": "", "cmd": cmd("alpha")},
        {"id": "g2", "lane": "gpu-queue", "leverage": 85, "rank": 2, "deps": "retire/close at: S2+", "cmd": cmd("beta")},
        {"id": "g3", "lane": "gpu-queue", "leverage": 80, "rank": 3,
         "deps": "the spiking replacement must reach parity", "cmd": cmd("blocked")},   # BLOCKED
        {"id": "p1", "lane": "pool-cpu", "leverage": 70, "rank": 4, "deps": "mechanism: x", "cmd": cmd("gamma")},
        {"id": "p2", "lane": "pool-cpu", "leverage": 60, "rank": 5, "deps": "", "cmd": cmd("delta")},
        {"id": "pool_done", "lane": "pool-cpu", "leverage": 55, "rank": 6, "deps": "", "cmd": cmd("already")},  # in-flight
        {"id": "nc1", "lane": "gpu-queue", "leverage": 50, "rank": 7, "deps": "",
         "what": "a ready free-lane item with no command"},                            # NEEDS-COMMAND
        {"id": "a1", "lane": "agent", "leverage": 120, "rank": 8, "deps": "none (de_risked=YES)",
         "what": "Flip faculty X to on-by-default", "source": "ledger-flip"},
        {"id": "a2", "lane": "agent", "leverage": 100, "rank": 9, "deps": "wire into /api/brain-chat first",
         "what": "Flip faculty Y (blocked)", "source": "ledger-flip"},                  # BLOCKED agent
        {"id": "a3", "lane": "agent", "leverage": 95, "rank": 10, "deps": "", "what": "Open wall Z",
         "source": "walls-ledger"},
        {"id": "a4", "lane": "agent", "leverage": 90, "rank": 11, "deps": "", "what": "Coverage gap W",
         "source": "failure-log"},                                                      # beyond budget=2
    ]


def selftest() -> list:
    import tempfile
    problems = []
    tmp = tempfile.mkdtemp(prefix="ratchet_selftest_")
    ledger = os.path.join(tmp, "ledger.jsonl")
    cfg = _mock_cfg(ledger)
    items = _mock_backlog()

    # in-flight: the command of `pool_done` is "already staged", and its id was dispatched earlier.
    already_cmd = _norm(next(i["cmd"] for i in items if i["id"] == "pool_done"))
    inflight_ids = {"pool_done"}
    inflight_cmds = {already_cmd}
    capacity = {"gpu-queue": 2, "pool-cpu": 2}

    plan = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)

    # ---- PASS DIRECTION (a) free-lane items dispatched, deps respected, no double-dispatch ----
    disp_ids = {d["id"] for d in plan.dispatches}
    # g1 + g2 fill the 2 gpu slots (g3 blocked); p1 + p2 fill the 2 pool slots (pool_done in-flight excluded)
    if disp_ids != {"g1", "g2", "p1", "p2"}:
        problems.append("PASS(a): expected dispatch {g1,g2,p1,p2}, got %s" % sorted(disp_ids))
    if "g3" in disp_ids:
        problems.append("PASS(a): dispatched the dep-BLOCKED item g3")
    if "pool_done" in disp_ids:
        problems.append("PASS(a): re-dispatched the in-flight item pool_done (double-dispatch)")
    if not any(it.get("id") == "nc1" for it in plan.needs_command):
        problems.append("PASS(a): the command-less ready free-lane item nc1 was not surfaced as NEEDS-COMMAND")
    if plan.verdict_kind != "UNDER-PARALLELIZED":
        problems.append("PASS(a): verdict should be UNDER-PARALLELIZED with idle capacity + ready work, got %s"
                        % plan.verdict_kind)

    # ---- PASS DIRECTION (b) agent items EMITTED (not spawned), budgeted, blocked excluded ----
    emitted = [a["id"] for a in plan.agent_launch_list]
    if emitted != ["a1", "a3"]:      # a1 (lev120) + a3 (lev95); a2 blocked; budget=2 → a4 deferred
        problems.append("PASS(b): expected agent launch-list [a1,a3], got %s" % emitted)
    if any(a["id"] == "a2" for a in plan.agent_launch_list):
        problems.append("PASS(b): emitted the dep-BLOCKED agent item a2")
    if not any(it.get("id") == "a4" for it in plan.agent_deferred):
        problems.append("PASS(b): over-budget agent item a4 was not deferred")
    for a in plan.agent_launch_list:
        if "TASK:" not in a["prompt"] or "SOURCE ANCHOR" not in a["prompt"]:
            problems.append("PASS(b): agent %s prompt seed is not self-contained" % a["id"])

    # the correct plan must pass its own invariants, and execute() must NOT spawn anything (dry-run)
    if plan_invariants(plan, capacity, inflight_ids, inflight_cmds):
        problems.append("PASS: a correct plan FAILED its own invariants")
    actions, results, refusal = execute(plan, cfg, capacity, inflight_ids, inflight_cmds, live=False)
    if refusal:
        problems.append("PASS: execute() refused a valid plan: %s" % refusal)
    if any(a["kind"] == "spawn-agent" for a in actions):
        problems.append("PASS(b): execute() produced an agent-spawn side effect (must be emit-only)")
    if not all(r["status"].startswith("DRY-RUN") for r in results):
        problems.append("PASS: dry-run execute() reported a non-dry-run status (would have touched the queues)")

    # ---- FAILING DIRECTION: every invariant must CATCH its violation ----
    # (c1) an agent item smuggled into the free dispatches
    bad = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)
    bad.dispatches.append({"id": "a1", "lane": "gpu-queue", "command": "x -m research.runners.z",
                           "item": next(i for i in items if i["id"] == "a1")})
    if not plan_invariants(bad, capacity, inflight_ids, inflight_cmds):
        problems.append("FAIL-DIR: an agent item in the free dispatches was NOT caught")

    # (c2) a double-dispatch (item already in flight)
    bad = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)
    dd = next(i for i in items if i["id"] == "pool_done")
    bad.dispatches.append({"id": "pool_done", "lane": "pool-cpu", "command": dd["cmd"], "item": dd})
    if not plan_invariants(bad, capacity, inflight_ids, inflight_cmds):
        problems.append("FAIL-DIR: a double-dispatch (in-flight id/command) was NOT caught")

    # (c3) a dep-blocked item dispatched
    bad = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)
    g3 = next(i for i in items if i["id"] == "g3")
    bad.dispatches.append({"id": "g3", "lane": "gpu-queue", "command": g3["cmd"], "item": g3})
    if not plan_invariants(bad, capacity, inflight_ids, inflight_cmds):
        problems.append("FAIL-DIR: a dep-blocked dispatch was NOT caught")

    # (c4) an over-capacity dispatch
    bad = build_plan(items, {"gpu-queue": 1, "pool-cpu": 0}, set(), set(), cfg)
    over = build_plan(items, {"gpu-queue": 2, "pool-cpu": 2}, set(), set(), cfg)
    if not plan_invariants(over, {"gpu-queue": 1, "pool-cpu": 0}, set(), set()):
        problems.append("FAIL-DIR: dispatching over a lane's idle capacity was NOT caught")

    # (c5) a ready dispatchable item left undispatched while its lane had an idle slot
    bad = build_plan(items, {"gpu-queue": 2, "pool-cpu": 2}, set(), set(), cfg)
    if bad.dispatches:
        moved = bad.dispatches.pop()          # forcibly leave one dispatchable item behind
        bad.deferred_free.append(moved["item"])
        if not plan_invariants(bad, {"gpu-queue": 2, "pool-cpu": 2}, set(), set()):
            problems.append("FAIL-DIR: a ready item left undispatched with an idle slot was NOT caught")

    # (c6) the confirm-rule guard: an intended agent-SPAWN side effect must be caught
    if not assert_no_agent_spawn([{"kind": "queue-add"}, {"kind": "spawn-agent", "id": "a1"}]):
        problems.append("FAIL-DIR: assert_no_agent_spawn did NOT catch an agent-spawn side effect")
    if assert_no_agent_spawn([{"kind": "queue-add"}, {"kind": "note"}]):
        problems.append("FAIL-DIR: assert_no_agent_spawn FALSE-fired on a clean action list")

    # execute() must REFUSE (run nothing) when handed an invalid plan
    invalid = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)
    invalid.dispatches.append({"id": "g3", "lane": "gpu-queue", "command": g3["cmd"], "item": g3})  # dep-blocked
    _actions, _results, refusal2 = execute(invalid, cfg, capacity, inflight_ids, inflight_cmds, live=False)
    if not refusal2 or _results:
        problems.append("FAIL-DIR: execute() did NOT refuse an invariant-violating plan")

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────────────
def _regen_backlog():
    try:
        subprocess.run([sys.executable, BACKLOG_PY, "--no-vikunja"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        print("⚠ backlog regen failed (%s: %s); using the existing backlog.json" % (type(e).__name__, e))


def main():
    ap = argparse.ArgumentParser(description="Fan-out ratchet — the dispatch half of the parallelism engine.")
    ap.add_argument("--dispatch", action="store_true",
                    help="LIVE: actually enqueue free-lane items (agents stay emit-only). Default is a dry-run.")
    ap.add_argument("--regen", action="store_true", help="regenerate backlog.json (file-only) before planning")
    ap.add_argument("--probe-pool", action="store_true", help="measure real idle pool cores over SSH")
    ap.add_argument("--backlog", default=BACKLOG_JSON, help="backlog.json path")
    ap.add_argument("--ledger", default=DISPATCH_LEDGER, help="dispatched-ledger path (dedup)")
    ap.add_argument("--json", action="store_true", help="emit the plan as JSON")
    ap.add_argument("--quiet", action="store_true", help="one heartbeat-friendly line")
    ap.add_argument("--selftest", action="store_true", help="prove pass + failing direction, then exit")
    ap.add_argument("--how", action="store_true", help="print the heartbeat-integration contract")
    args = ap.parse_args()

    if args.how:
        print(HOW)
        return 0

    if args.selftest:
        probs = selftest()
        if probs:
            print("⛔ ratchet.py SELFTEST FAILED:")
            for p in probs:
                print("   - " + p)
            return 1
        print("✔ ratchet.py selftest PASSED — pass direction (free-lane items dispatched respecting deps + no "
              "double-dispatch; agent items EMITTED not spawned, budgeted) + failing direction (agent-in-free-"
              "dispatch, double-dispatch, dep-blocked dispatch, over-capacity, undispatched-with-idle-slot, and "
              "an intended agent-spawn are each CAUGHT; execute() refuses an invalid plan) both demonstrated.")
        return 0

    if args.regen:
        _regen_backlog()

    cfg = Config()
    cfg._ledger_path = args.ledger

    if not os.path.exists(args.backlog):
        print("⛔ backlog not found: %s\n   generate it first: tools/backlog.py  (or run with --regen)"
              % args.backlog)
        return 1
    payload = json.load(open(args.backlog))
    items = payload.get("items", [])

    capacity = probe_capacity(cfg, probe_pool=args.probe_pool)
    inflight_ids, inflight_cmds = read_inflight(cfg)
    plan = build_plan(items, capacity, inflight_ids, inflight_cmds, cfg)
    actions, results, refusal = execute(plan, cfg, capacity, inflight_ids, inflight_cmds, live=args.dispatch)

    meta = {"backlog_path": args.backlog, "n_items": len(items), "live": args.dispatch,
            "gpu_paused": _gpu_paused(), "cfg": cfg}

    if args.json:
        out = {
            "verdict": plan.verdict, "verdict_kind": plan.verdict_kind, "capacity": plan.capacity,
            "live": args.dispatch, "gpu_paused": meta["gpu_paused"], "refusal": refusal,
            "dispatches": [{"id": d["id"], "lane": d["lane"], "command": d["command"]} for d in plan.dispatches],
            "results": results,
            "needs_command": [{"id": it.get("id"), "lane": it.get("lane"), "what": it.get("what"),
                               "anchor": it.get("anchor")} for it in plan.needs_command],
            "deferred_free": [it.get("id") for it in plan.deferred_free],
            "agent_launch_list": [{"id": a["id"], "tier": a["tier"], "why": a["why"], "prompt": a["prompt"]}
                                  for a in plan.agent_launch_list],
            "agent_deferred": [it.get("id") for it in plan.agent_deferred],
            "skipped_blocked": [it.get("id") for it in plan.skipped_blocked],
            "skipped_inflight": [it.get("id") for it in plan.skipped_inflight],
        }
        print(json.dumps(out, indent=2))
    else:
        print(render(plan, results, refusal, meta, quiet=args.quiet))

    return 0


if __name__ == "__main__":
    sys.exit(main())
