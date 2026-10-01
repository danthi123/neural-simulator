"""tools/model_bakeoff/run.py: the leak audit and the grading helpers. The audit decides which bake-off runs are
disqualified (network use, reading a path that could hold an answer), so it must neither miss those nor flag the
paths a run legitimately uses (its own clone, the read-only .venv and reference library)."""
import importlib.util
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_module():
    spec = importlib.util.spec_from_file_location("bakeoff_run", os.path.join(_ROOT, "tools", "model_bakeoff", "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _transcript(tmp_path, commands):
    p = tmp_path / "transcript.jsonl"
    with open(p, "w") as f:
        for name, inp in commands:
            f.write(json.dumps({"type": "tool_use", "name": name, "input": inp}) + "\n")
        f.write(json.dumps({"type": "result", "tokens": {"input": 1, "output": 1}}) + "\n")
    return str(p)


RUN_DIR = "/bk/runs/screen1/model_x/task-a"


def test_audit_passes_a_clean_run(tmp_path):
    r = _run_module()
    t = _transcript(tmp_path, [
        ("terminal", {"command": "cd %s/repo && .venv/bin/python -m pytest -q tests/test_x.py" % RUN_DIR}),
        ("terminal", {"command": "/home/dant123/Projects/neural-simulator/.venv/bin/python -c 'print(1)'"}),
        ("read_file", {"path": "/home/dant123/Projects/sim-catalog/references/feature-catalog.md"}),
        ("terminal", {"command": "grep -rn 'plateau' research/findings | head; ls /etc /usr/lib"}),
        ("terminal", {"command": "git log --oneline -5 && git commit -m 'fix: x'"}),
    ])
    flags, warns = r.audit(t, RUN_DIR)
    assert flags == [], flags


def test_audit_flags_network_and_leaky_paths(tmp_path):
    r = _run_module()
    t = _transcript(tmp_path, [
        ("terminal", {"command": "curl -s https://example.org/x"}),
        ("terminal", {"command": "git fetch origin main"}),
        ("terminal", {"command": "pip install requests"}),
        ("terminal", {"command": "cat /home/dant123/Projects/neural-simulator/research/findings/x.md"}),
        ("read_file", {"path": "/bk/runs/screen1/other_model/task-a/answer.md"}),
        ("terminal", {"command": "ls /bk/bases"}),
        ("terminal", {"command": "sqlite3 /bk/home/state.db .tables"}),
    ])
    flags, _ = r.audit(t, RUN_DIR)
    joined = "\n".join(flags)
    for needle in ("curl", "git fetch", "pip install", "neural-simulator/research", "other_model", "/bk/bases", "/bk/home/state.db"):
        assert needle in joined, (needle, flags)


def test_audit_allows_hermes_scratch_and_the_empty_opt_data(tmp_path):
    """2026-09-30 screen1: a legitimate run was disqualified for a scratch file Hermes's own write_file tool put under
    /bk/home/cache and a writability probe of /opt/data (empty and untraversable in the bake-off container)."""
    r = _run_module()
    t = _transcript(tmp_path, [
        ("write_file", {"path": "/bk/home/cache/scratch/probe_lane.py", "content": "print(1)"}),
        ("terminal", {"command": "touch /opt/data/.probe_write_test 2>&1; ls -ld /opt/data"}),
        ("terminal", {"command": "ls -ld /bk /bk/runs; readlink -f ."}),   # ancestors of the run's own dir
    ])
    flags, _ = r.audit(t, RUN_DIR)
    assert flags == [], flags


def test_audit_only_warns_on_a_quoted_url(tmp_path):
    r = _run_module()
    t = _transcript(tmp_path, [("write_file", {"path": "notes.md", "content": "see https://doi.org/10.1000/x"})])
    flags, warns = r.audit(t, RUN_DIR)
    assert flags == [] and warns, (flags, warns)


def test_slug_is_path_safe():
    r = _run_module()
    assert r.slug("anthropic/claude-sonnet-5.5") == "anthropic_claude-sonnet-5.5"
    assert "/" not in r.slug("../../etc/passwd")


def test_only_tests_are_protected_by_default_for_implement_tasks():
    """Grading restores what the model must not change (the given tests), never the code it was asked to fix
    (2026-09-30 smoke run: restoring every setup file put the planted bug back and failed a correct fix)."""
    r = _run_module()
    impl = {"category": "implement"}
    assert r.is_protected(impl, "tests/test_x.py")
    assert r.is_protected(impl, "bakeoff_smoke/test_calc.py")
    assert r.is_protected(impl, "sim/tests/conftest.py")
    assert not r.is_protected(impl, "bakeoff_smoke/calc.py")
    assert not r.is_protected(impl, "tools/gpu_queue.sh")
    assert not r.is_protected({"category": "operate"}, "tests/test_x.py")
    explicit = {"category": "operate", "protect": ["research/queue/"]}
    assert r.is_protected(explicit, "research/queue/pool.queue")
    assert not r.is_protected(explicit, "research/queue_other.txt")
