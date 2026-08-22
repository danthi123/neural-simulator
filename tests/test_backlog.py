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
