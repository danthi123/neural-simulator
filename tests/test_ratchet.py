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
