"""CLASS PI trigger hole (2026-09-23): staging a source file that a ledger row anchors must run Check A even when that
file is not in the static ANCHORED_FILES list. a41332aa1 deleted an anchored constant from such a file; Check A never
ran, and the stale anchor then blocked every later ledger commit on main."""
from __future__ import annotations

from tools.gates import production_integration as g

_LEDGER = """headline:
  total_faculties: 1
  scaffold_retired: 0
rows:
  - key: demo
    de_risked: YES
    wired: YES
    on_by_default: YES
    scaffold_retired: NO
    retire_status: ADDITIVE
    default_anchor:
      - file: research/runners/_demo_unlisted.py
        assign: _DEMO_DEFAULT_ON
        off_value: "False"
        on_value: "True"
        count: 1
"""


def _patch(monkeypatch, src, staged=()):
    files = {g.LEDGER_REL: _LEDGER, "research/runners/_demo_unlisted.py": src}
    monkeypatch.setattr(g, "_read", lambda rel: files.get(rel))
    monkeypatch.setattr(g, "_staged_changed", lambda: list(staged))


def test_hook_shape_modified_ledger_with_no_added_files_still_runs_check_a(monkeypatch):
    """The pre-commit hook passes only ADDED paths; a ledger edit is a MODIFICATION. The gate must read the index."""
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = False\n", staged=[g.LEDGER_REL])
    probs = g.check([])
    assert any("[A]" in p and "demo" in p for p in probs), probs


def test_hook_shape_modified_anchored_source_runs_check_a(monkeypatch):
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = False\n", staged=["research/runners/_demo_unlisted.py"])
    assert any("[A]" in p for p in g.check([]))


def test_nothing_staged_nothing_checked(monkeypatch):
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = False\n", staged=[])
    assert g.check([]) == []


def test_staging_an_unlisted_anchored_file_runs_check_a(monkeypatch):
    assert "research/runners/_demo_unlisted.py" not in g.ANCHORED_FILES
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = False\n")   # source flipped OFF, ledger still says YES
    probs = g.check(["research/runners/_demo_unlisted.py"])
    assert any("[A]" in p and "demo" in p for p in probs), probs


def test_consistent_unlisted_anchor_passes(monkeypatch):
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = True\n")
    assert g.check(["research/runners/_demo_unlisted.py"]) == []


def test_unrelated_file_does_not_trigger(monkeypatch):
    _patch(monkeypatch, "_DEMO_DEFAULT_ON = False\n")
    assert g.check(["research/runners/some_other_file.py"]) == []
