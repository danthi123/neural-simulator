"""CI regression for the fan-out ratchet (tools/ratchet.py) — the dispatch half of the parallelism engine.

Runs the tool's own --selftest logic in-process so a regression breaks the build, not just an interactive
run. Mirrors the gate-registry discipline: a check whose selftest cannot fail is not trusted. Beyond the
selftest, the failing-direction is asserted to have TEETH — neutering the invariant guard must make the
selftest FAIL (a defanged check is detected), and the deps/command heuristics + autonomy rules are pinned.
"""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location("ratchet", os.path.join(_ROOT, "tools", "ratchet.py"))
ratchet = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ratchet)


def _fresh():
    """A fresh module instance so a monkeypatch in one test cannot leak into another."""
    spec = importlib.util.spec_from_file_location("ratchet_x", os.path.join(_ROOT, "tools", "ratchet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ratchet_selftest_passes():
    """Both directions: free-lane dispatch respects deps + no double-dispatch; agents emitted not spawned;
    and every invariant catches its violation; execute() refuses an invalid plan."""
    problems = ratchet.selftest()
    assert problems == [], "ratchet.py selftest FAILED:\n" + "\n".join(problems)


def test_failing_direction_has_teeth_invariants():
    """A neutered invariant guard (one that cannot fail) MUST make the selftest fail — otherwise the check
    is indistinguishable from no check (the four unfailable gates this project has shipped)."""
    mod = _fresh()
    mod.plan_invariants = lambda *a, **k: []          # broken guard: always 'clean'
    assert mod.selftest(), "selftest passed with a neutered invariant guard — the check has no teeth"


def test_failing_direction_has_teeth_spawn_guard():
    """A neutered confirm-rule guard (agent-spawn detector) MUST be detected by the selftest."""
    mod = _fresh()
    mod.assert_no_agent_spawn = lambda actions: []    # broken confirm-rule guard
    assert mod.selftest(), "selftest passed with a neutered agent-spawn guard"


def test_deps_readiness_heuristic():
    """The prose `deps` vocabulary the generator actually emits classifies correctly."""
    assert ratchet.is_blocked("wire into /api/brain-chat first")
    assert ratchet.is_blocked("the spiking replacement must reach parity or an honest negative")
    for ready in ("", "none (de_risked=YES)", "retire/close at: S2+", "mechanism: generative-attractor-wander"):
        assert not ratchet.is_blocked(ready), "misclassified READY deps as blocked: %r" % ready


def test_no_command_is_never_fabricated():
    """A free-lane item with only prose yields no command (surfaced as NEEDS-COMMAND, never invented)."""
    assert ratchet.command_for({"what": "Teach the brain to speak", "verify": "per the task"}) is None
    assert ratchet.command_for({"cmd": "x -m research.runners.y --json o.json"}) is not None
    got = ratchet.command_for({"verify": "run .venv/bin/python -u -m research.runners.z --seed 42"})
    assert got and "research.runners.z" in got


def test_agents_are_emit_only():
    """execute() must never produce an agent-spawn side effect, even with RATCHET_AUTO_AGENTS set."""
    import tempfile
    cfg = ratchet._mock_cfg(os.path.join(tempfile.mkdtemp(), "l.jsonl"))
    cfg.auto_agents = True                            # even with the reserved flag ON
    items = ratchet._mock_backlog()
    cap = {"gpu-queue": 2, "pool-cpu": 2}
    plan = ratchet.build_plan(items, cap, set(), set(), cfg)
    actions, results, refusal = ratchet.execute(plan, cfg, cap, set(), set(), live=False)
    assert not refusal
    assert not any(a["kind"] == "spawn-agent" for a in actions), "auto_agents caused a spawn side effect"
    assert plan.agent_launch_list, "agent lane should still be EMITTED"


def test_cmd_bearing_item_really_dispatches_to_the_SHARED_queue(monkeypatch, tmp_path):
    """The seam this change closes: a cmd-bearing free-lane item triggers a REAL queue-add subprocess (not a
    log-only stub), and it targets the SHARED (main-checkout) queue script — never a per-worktree copy that no
    daemon consumes. Captures the actual subprocess argv."""
    mod = _fresh()
    monkeypatch.setenv("SIM_QUEUE_ROOT", str(tmp_path / "mainroot"))   # deterministic shared-root resolution
    calls = []

    class _R:
        returncode = 0
        stdout = "queued job"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda args, **kw: (calls.append(args), _R())[1])
    cfg = mod._mock_cfg(str(tmp_path / "ledger.jsonl"))
    items = [{"id": "g1", "lane": "gpu-queue", "leverage": 90, "rank": 1, "deps": "", "dependencies": [],
              "cmd": "SIM_BACKEND=cupy .venv/bin/python -u -m research.runners.alpha --out raw/o.json"}]
    cap = {"gpu-queue": 1, "pool-cpu": 0}
    plan = mod.build_plan(items, cap, set(), set(), cfg)
    actions, results, refusal = mod.execute(plan, cfg, cap, set(), set(), live=True)
    assert not refusal
    assert any(r["status"] == "queued" for r in results), "the real dispatch did not report 'queued': %s" % results
    add_calls = [c for c in calls if len(c) >= 3 and c[0] == "bash" and c[2] == "add"]
    assert add_calls, "no queue-add subprocess was invoked — dispatch was a no-op"
    script = add_calls[0][1]
    assert script == str(tmp_path / "mainroot" / "tools" / "gpu_queue.sh"), script
    assert ".claude/worktrees" not in script, "dispatch used a per-worktree queue script: %s" % script
    # idempotent: the dispatched id was recorded so a refill cycle will not re-launch it
    ids, _cmds = mod.read_inflight(cfg)
    assert "g1" in ids, "the dispatched id was not recorded in the dedup ledger (not idempotent)"


def test_pool_dispatch_uses_shared_pool_queue_with_checked(monkeypatch, tmp_path):
    """A pool-cpu cmd dispatches via the SHARED pool_queue.sh with the required --checked record token."""
    mod = _fresh()
    monkeypatch.setenv("SIM_QUEUE_ROOT", str(tmp_path / "mainroot"))
    calls = []

    class _R:
        returncode = 0
        stdout = "queued"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda args, **kw: (calls.append(args), _R())[1])
    cfg = mod._mock_cfg(str(tmp_path / "ledger.jsonl"))
    items = [{"id": "p1", "lane": "pool-cpu", "leverage": 70, "rank": 1, "deps": "", "dependencies": [],
              "anchor": "research/findings/x.md:10",
              "cmd": "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.beta --out raw/b.json"}]
    cap = {"gpu-queue": 0, "pool-cpu": 1}
    plan = mod.build_plan(items, cap, set(), set(), cfg)
    mod.execute(plan, cfg, cap, set(), set(), live=True)
    add = [c for c in calls if len(c) >= 3 and c[2] == "add"][0]
    assert add[1] == str(tmp_path / "mainroot" / "tools" / "pool_queue.sh"), add[1]
    assert "--checked" in add, "pool_queue.sh add was invoked without the required --checked record token"


def test_cmdless_free_item_is_not_dispatched_and_agent_item_is_emitted():
    """A free-lane item with NO cmd is surfaced as NEEDS-COMMAND (never fabricated into a dispatch); a
    cmd-less item that needs a mind routes to the emit-only agent lane, not a free-lane dispatch."""
    import tempfile
    cfg = ratchet._mock_cfg(os.path.join(tempfile.mkdtemp(), "l.jsonl"))
    items = [
        {"id": "nc", "lane": "pool-cpu", "leverage": 50, "rank": 1, "deps": "", "dependencies": [],
         "what": "a ready pool item with only prose, no command"},                 # → NEEDS-COMMAND
        {"id": "ag", "lane": "agent", "leverage": 90, "rank": 2, "deps": "", "dependencies": [],
         "what": "Flip faculty to on-by-default", "source": "ledger-flip"},         # → agent (emit-only)
    ]
    cap = {"gpu-queue": 0, "pool-cpu": 2}
    plan = ratchet.build_plan(items, cap, set(), set(), cfg)
    assert not plan.dispatches, "a cmd-less item was dispatched (must never happen)"
    assert any(it["id"] == "nc" for it in plan.needs_command), "cmd-less free item not surfaced as NEEDS-COMMAND"
    assert any(a["id"] == "ag" for a in plan.agent_launch_list), "the agent item was not emitted"


def test_structured_dep_blocks_until_it_clears():
    """A structured dependency (ids) blocks dispatch until the dep clears — by completion (leaves the
    backlog) or by going in-flight."""
    import tempfile
    cfg = ratchet._mock_cfg(os.path.join(tempfile.mkdtemp(), "l.jsonl"))
    cmd = lambda x: "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.%s --out raw/%s.json" % (x, x)
    Y = {"id": "Y", "lane": "pool-cpu", "leverage": 50, "rank": 1, "deps": "", "dependencies": [], "cmd": cmd("y")}
    X = {"id": "X", "lane": "pool-cpu", "leverage": 60, "rank": 2, "deps": "", "dependencies": ["Y"], "cmd": cmd("x")}
    cap = {"gpu-queue": 0, "pool-cpu": 2}     # capacity is NOT the constraint — only the dep can hold X
    plan = ratchet.build_plan([X, Y], cap, set(), set(), cfg)
    disp = {d["id"] for d in plan.dispatches}
    assert "Y" in disp and "X" not in disp, "X dispatched while its dep Y was still open: %s" % disp
    assert any(it["id"] == "X" for it in plan.skipped_blocked)
    # dep clears by completion (Y no longer in the backlog)
    assert "X" in {d["id"] for d in ratchet.build_plan([X], cap, set(), set(), cfg).dispatches}
    # dep clears by going in-flight (Y already dispatched)
    assert "X" in {d["id"] for d in ratchet.build_plan([X, Y], cap, {"Y"}, set(), cfg).dispatches}


def test_failing_direction_has_teeth_structured_dep_invariant():
    """Neutering the structured-dep resolver MUST make the selftest fail — otherwise the dep check is
    indistinguishable from no check."""
    mod = _fresh()
    mod.structured_deps_unmet = lambda *a, **k: []     # broken: nothing ever looks unmet
    assert mod.selftest(), "selftest passed with a neutered structured-dep resolver — the check has no teeth"
