"""Tests for sim.progress — universal progress event format."""
from __future__ import annotations

import io
import json

import pytest


def test_emit_progress_writes_progress_prefixed_json():
    """emit_progress writes one line: [PROGRESS] {json} to the given file."""
    from sim.progress import emit_progress

    buf = io.StringIO()
    emit_progress("training", 50, 100, phase="P2", unit="episodes",
                   correct_pct=27.3, file=buf)
    line = buf.getvalue().rstrip("\n")
    assert line.startswith("[PROGRESS] ")
    payload = json.loads(line[len("[PROGRESS] "):])
    assert payload == {
        "kind": "training",
        "current": 50,
        "total": 100,
        "phase": "P2",
        "unit": "episodes",
        "correct_pct": 27.3,
    }


def test_emit_progress_minimal_args():
    """current/total/phase/unit/label are all optional — only kind required."""
    from sim.progress import emit_progress

    buf = io.StringIO()
    emit_progress("complete", file=buf)
    payload = json.loads(buf.getvalue().rstrip("\n")[len("[PROGRESS] "):])
    assert payload == {"kind": "complete"}


def test_parse_progress_line_roundtrip():
    """emit then parse should give back the original payload."""
    from sim.progress import emit_progress, parse_progress_line

    buf = io.StringIO()
    emit_progress("eval", 25, 100, phase="W->A", unit="trials", file=buf)
    line = buf.getvalue().rstrip("\n")
    parsed = parse_progress_line(line)
    assert parsed == {
        "kind": "eval",
        "current": 25,
        "total": 100,
        "phase": "W->A",
        "unit": "trials",
    }


def test_parse_progress_line_returns_none_for_non_progress():
    """Lines without [PROGRESS] prefix return None."""
    from sim.progress import parse_progress_line
    assert parse_progress_line("[INFO] just an info line") is None
    assert parse_progress_line("  [P2 ep 10/100] correct=...") is None
    assert parse_progress_line("") is None


def test_parse_progress_line_returns_none_for_malformed_json():
    """[PROGRESS] prefix but invalid JSON still returns None (no crash)."""
    from sim.progress import parse_progress_line
    assert parse_progress_line("[PROGRESS] {invalid}") is None
    assert parse_progress_line("[PROGRESS] not even a brace") is None


def test_parse_last_progress_picks_most_recent():
    """parse_last_progress on a multi-line buffer returns the LAST event."""
    from sim.progress import parse_last_progress
    log = """
[INFO] starting
[PROGRESS] {"kind":"training","current":10,"total":100}
some other line
[PROGRESS] {"kind":"training","current":50,"total":100}
[PROGRESS] {"kind":"eval","current":5,"total":25}
"""
    last = parse_last_progress(log)
    assert last == {"kind": "eval", "current": 5, "total": 25}


def test_parse_last_progress_by_kind_separates_phases():
    """Different kinds are kept separately so frontend can display
    'training: ep 50/100, currently eval: trial 5/25'."""
    from sim.progress import parse_last_progress_by_kind
    log = """
[PROGRESS] {"kind":"training","current":10,"total":100,"phase":"P2"}
[PROGRESS] {"kind":"training","current":100,"total":100,"phase":"P2"}
[PROGRESS] {"kind":"replay","current":250,"total":1000,"phase":"P3"}
[PROGRESS] {"kind":"eval","current":5,"total":25,"phase":"W->A"}
"""
    by_kind = parse_last_progress_by_kind(log)
    assert by_kind["training"]["current"] == 100  # last training, not first
    assert by_kind["replay"]["current"] == 250
    assert by_kind["eval"]["current"] == 5


def test_parse_last_progress_handles_malformed_json_gracefully():
    """A bad event in the middle doesn't break parsing of good ones."""
    from sim.progress import parse_last_progress
    log = """
[PROGRESS] {"kind":"training","current":10,"total":100}
[PROGRESS] {malformed json
[PROGRESS] {"kind":"training","current":50,"total":100}
"""
    last = parse_last_progress(log)
    assert last == {"kind": "training", "current": 50, "total": 100}


def test_emit_progress_extras_included_in_payload():
    """**extras are passed through verbatim."""
    from sim.progress import emit_progress
    buf = io.StringIO()
    emit_progress("step", 100, 1800,
                   pos=[3, 5], goal=[7, 7], reward=1.0, file=buf)
    payload = json.loads(buf.getvalue().rstrip("\n")[len("[PROGRESS] "):])
    assert payload["pos"] == [3, 5]
    assert payload["goal"] == [7, 7]
    assert payload["reward"] == 1.0


def test_progress_line_re_anchored_correctly():
    """The [PROGRESS] tag in the middle of a line is also matched
    (e.g., logging frameworks may prepend timestamps)."""
    from sim.progress import parse_progress_line
    line = "[2026-05-03 22:00] [PROGRESS] {\"kind\":\"eval\",\"current\":1,\"total\":100}"
    parsed = parse_progress_line(line)
    assert parsed is not None
    assert parsed["kind"] == "eval"
