"""The production-default battery guard refuses ANY BRAIN_* override, not only the flipped flags."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import assert_flipped_defaults as guard  # noqa: E402


def test_other_brain_flag_in_env_is_refused():
    assert any("BRAIN_PMEM_OP_STABILIZER" in p for p in guard.problems({"BRAIN_PMEM_OP_STABILIZER": "1"}))


def test_lb_probe_flags_and_clean_env_pass():
    assert guard.problems({}) == []
    assert guard.problems({"LB_EPISODIC_DRIVE_PROBE": "1", "SIM_BACKEND": "numpy"}) == []


def test_settle_flag_flipped_default_present_and_true():
    # S09 AG-FLIP prep (2026-09-24, parked branch): _SETTLE_DEFAULT_ON must exist in source and read True here,
    # so a battery run at THIS branch's revision measures the prepared default, not a silent old-default miss.
    assert "BRAIN_AFFECT_MARKER_SETTLE" in guard.FLIPPED
    assert guard.problems({}) == []


def test_settle_flag_in_env_is_refused():
    assert any("BRAIN_AFFECT_MARKER_SETTLE" in p for p in guard.problems({"BRAIN_AFFECT_MARKER_SETTLE": "1"}))


def test_guard_selftest_proves_it_can_fail():
    # AGFLIP review (2026-09-24): the guard had no self-verifying check that it can actually FAIL, unlike every
    # tools/gates/ module. guard.selftest() must report zero problems -- i.e. it successfully demonstrated the
    # override-detection and pre-flip-revision-detection failing directions for every registered flag, including
    # BRAIN_AFFECT_MARKER_SETTLE, without mutating any file on disk.
    st = guard.selftest()
    assert st == [], st


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
