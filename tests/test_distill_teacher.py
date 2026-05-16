import pathlib
from research.runners import distill_teacher as dt

def test_offline_enforced_constant():
    assert dt.LOCAL_FILES_ONLY is True

def test_runtime_isolation_guard_exists():
    assert hasattr(dt, "assert_training_time_only")

def test_no_runtime_module_imports_teacher():
    root = pathlib.Path(__file__).resolve().parents[1]
    runtime = [
        root/"research/runners/grounded_generative_agent.py",
        root/"research/runners/g20_generative_agent.py",
        root/"research/runners/abstention_gate.py",
        root/"research/runners/concept_grammar.py",
    ]
    for f in runtime:
        if f.exists():
            assert "distill_teacher" not in f.read_text(encoding="utf-8"), f
