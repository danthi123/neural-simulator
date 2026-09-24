"""The production-default battery guard refuses ANY BRAIN_* override, not only the three flipped flags."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import assert_flipped_defaults as guard  # noqa: E402


def test_other_brain_flag_in_env_is_refused():
    assert any("BRAIN_PMEM_OP_STABILIZER" in p for p in guard.problems({"BRAIN_PMEM_OP_STABILIZER": "1"}))


def test_lb_probe_flags_and_clean_env_pass():
    assert guard.problems({}) == []
    assert guard.problems({"LB_EPISODIC_DRIVE_PROBE": "1", "SIM_BACKEND": "numpy"}) == []


def test_guard_reads_constants_without_importing_cupy_modules(monkeypatch):
    # CPU-only nodes have no cupy; the guard must not import the flipped modules (2026-09-24: 156 shards failed).
    import builtins
    real_import = builtins.__import__

    def no_cupy(name, *a, **k):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("No module named 'cupy'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_cupy)
    assert guard.problems({}) == []
