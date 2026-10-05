"""tools/needs_line.py: one NEEDS-CLAUDE line in the runbook's format, at most 300 characters, and never filed past an
existing section-6 ruling for the same key without --rulings-read (session 6: T12 re-filed after its 10-01 ruling)."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("needs_line", os.path.join(_ROOT, "tools", "needs_line.py"))
N = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N)

RUNBOOK = """# runbook
## 6. RULINGS (Claude writes)

- T12-seam-map: done. T12 is closed. (2026-10-01)
- other: about T120 and T1-A0, not this key. (2026-10-02)
  continued line
- handled: [T12] (2026-10-02)

## NEEDS CLAUDE

- 2026-10-01 00:00 [x] INFO: seeded
"""


def _rb(tmp_path):
    p = tmp_path / "rb.md"
    p.write_text(RUNBOOK)
    return str(p)


def test_compose_format_and_whitespace():
    import datetime
    line = N.compose("T24", "STOP", "a  b\n c", "#1", datetime.datetime(2026, 10, 4, 4, 0))
    assert line == "- 2026-10-04 04:00 [T24] STOP #1: a b c"


def test_rulings_for_matches_the_key_only():
    found = N.rulings_for("T12", RUNBOOK)
    assert len(found) == 2 and found[0].startswith("- T12-seam-map")       # T12-seam-map and handled: [T12]
    assert N.rulings_for("T1", RUNBOOK) == []                              # T12, T120, T1-A0 are other keys
    assert len(N.rulings_for("T120", RUNBOOK)) == 1


def test_refuses_long_lines_and_unread_rulings(tmp_path):
    rb = _rb(tmp_path)
    assert N.main(["--key", "T9", "--kind", "INFO", "--text", "x" * 400, "--runbook", rb, "--write"]) == 1
    assert N.main(["--key", "T12", "--kind", "STOP", "--text", "file missing", "--runbook", rb, "--write"]) == 1
    assert open(rb).read() == RUNBOOK                                       # nothing written on a refusal
    assert N.main(["--key", "T12", "--kind", "STOP", "--text", "file missing", "--runbook", rb, "--write",
                   "--rulings-read"]) == 0
    assert open(rb).read().rstrip().endswith("[T12] STOP: file missing")
    assert N.main(["--key", "T9", "--kind", "INFO", "--text", "ok", "--runbook", rb]) == 0   # print only
    assert open(rb).read().count("\n- 20") == 2
