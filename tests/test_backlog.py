"""CI regression for the mechanical work-backlog generator (tools/backlog.py).

Runs the tool's own --selftest logic in-process so a scanner that silently returns empty (the failing
direction) or stops surfacing the known backlog (the pass direction) breaks the build, not just an
interactive run. Mirrors the gate-registry discipline: a check whose selftest cannot fail is not trusted.
"""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location("backlog", os.path.join(_ROOT, "tools", "backlog.py"))
backlog = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backlog)


def test_backlog_selftest_passes():
    """Both directions (known backlog surfaced + empty-scanner-over-nonempty-source caught)."""
    problems = backlog.selftest()
    assert problems == [], "backlog.py selftest FAILED:\n" + "\n".join(problems)


def test_scanners_are_pure_file_readers():
    """Anti-fabrication: every file scanner returns [] on empty source text (no invented filler)."""
    assert backlog.scan_ledger_flips(ledger_text="") == []
    assert backlog.scan_ledger_scaffolds(ledger_text="") == []
    assert backlog.scan_walls_ledger(roadmap_text="") == []
    assert backlog.scan_failure_log(log_text="") == []


def test_failing_direction_guard_fires():
    """The guard must catch an empty scanner when the source clearly has items, and NOT false-fire."""
    assert backlog._guard_scanner_nonempty("x", lambda: [], source_has_items=True) is True
    assert backlog._guard_scanner_nonempty("x", lambda: [], source_has_items=False) is False


def _a_real_runner_module():
    import os
    for f in sorted(os.listdir(backlog.RUNNERS_DIR)):
        if f.endswith(".py") and not f.startswith("__"):
            return f[:-3]
    return None


def test_extract_runnable_cmd_mints_only_a_real_command():
    """The cmd seam: a genuinely-runnable command is minted; a command naming a NON-existent module, a
    placeholder template, or an already-done run is NEVER minted (anti-fabrication + anti-stale)."""
    mod = _a_real_runner_module()
    assert mod, "no runner module on disk to test against"
    cmd, lane = backlog.extract_runnable_cmd(
        "next: `SIM_BACKEND=numpy .venv/bin/python -m research.runners.%s --seed 1`" % mod)
    assert cmd and ("research.runners." + mod) in cmd and lane == "pool-cpu"
    # cupy → gpu-queue
    assert backlog.extract_runnable_cmd(
        "`SIM_BACKEND=cupy .venv/bin/python -m research.runners.%s --seed 1`" % mod)[1] == "gpu-queue"
    # anti-fabrication: a fabricated module name yields nothing
    assert backlog.extract_runnable_cmd(
        "`.venv/bin/python -m research.runners._NOPE_NOT_REAL_9 --seed 1`") == (None, None)
    # a placeholder template is not runnable as-is
    assert backlog.extract_runnable_cmd(
        "`.venv/bin/python -m research.runners.%s --seed $s`" % mod) == (None, None)
    # anti-stale: a command whose declared artifact already exists is DONE (tools/backlog.py always exists)
    assert backlog.extract_runnable_cmd(
        "`.venv/bin/python -m research.runners.%s --out tools/backlog.py`" % mod) == (None, None)


def test_no_emitted_cmd_names_a_fabricated_module():
    """Over the LIVE record, every cmd a scanner emits names a real research.runners module and a free lane
    (the guarantee the ratchet relies on before it ever runs anything)."""
    items, _meta = backlog.generate(use_vikunja=False)
    for it in items:
        c = it.get("cmd")
        if c:
            assert backlog._runner_exists(backlog._runner_module(c)), "fabricated module in cmd: %s" % c
            assert it["lane"] in ("gpu-queue", "pool-cpu"), "cmd on a non-free lane: %s" % it["id"]


def test_structured_dependencies_are_derived_not_invented():
    """link_dependencies adds a structured dep id ONLY for a prose-blocked item that shares a faculty with
    an open wall; a ready item is never given an invented dependency."""
    wall = backlog._item("walls-ledger", "wall-gap4", "Open wall: gap#4 deep credit", "gap#4 deep-credit",
                         "roadmap:1", "t", "v", "", "gpu-queue")
    blocked = backlog._item("ledger-scaffold", "scaffold-x", "Retire host scaffold for gap#4 deep credit",
                            "gap#4 deep-credit host shortcut", "led:1", "t", "v",
                            "the spiking replacement must reach parity", "agent")
    ready = backlog._item("ledger-flip", "flip-y", "Flip vision to on-by-default", "object-anywhere",
                          "led:2", "t", "v", "none (de_risked=YES)", "agent")
    linked = backlog.link_dependencies([wall, blocked, ready])
    bs = next(i for i in linked if i["id"] == "scaffold-x")
    rf = next(i for i in linked if i["id"] == "flip-y")
    assert "wall-gap4" in bs["dependencies"], "prose-blocked same-faculty item did not get the wall dep"
    assert rf["dependencies"] == [], "a ready item was given an invented dependency"
