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
        ("terminal", {"command": "ls /opt/data/sessions"}),
    ])
    flags, _ = r.audit(t, RUN_DIR)
    joined = "\n".join(flags)
    for needle in ("curl", "git fetch", "pip install", "neural-simulator/research", "other_model", "/bk/bases", "/opt/data"):
        assert needle in joined, (needle, flags)


def test_audit_only_warns_on_a_quoted_url(tmp_path):
    r = _run_module()
    t = _transcript(tmp_path, [("write_file", {"path": "notes.md", "content": "see https://doi.org/10.1000/x"})])
    flags, warns = r.audit(t, RUN_DIR)
    assert flags == [] and warns, (flags, warns)


def test_slug_is_path_safe():
    r = _run_module()
    assert r.slug("anthropic/claude-sonnet-5.5") == "anthropic_claude-sonnet-5.5"
    assert "/" not in r.slug("../../etc/passwd")
