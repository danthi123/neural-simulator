"""Tests for tools/pool_health.py's revision-marker read (RULING misc-1012b).

pool_health used to check the RETIRED ~/derisk-pool/sim checkout. It now asks the dispatcher's own
question: does the live ~/derisk-pool/revisions/<sha> carry the `.provisioned_ok` completion marker,
read through the ONE shared predicate (tools/pool_revision_marker.sh). These tests pin the parser
(the real one, parse_remote — selftest and check_node share it), the STALE-vs-BROKEN split, and the
banned-fix guard (pool_provision.sh is banned on this host; the fix hint must be provision.req).
"""
from __future__ import annotations

import pathlib
import sys

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import pool_health  # noqa: E402  # type: ignore[import-not-found] -- tools/ is on sys.path at runtime

parse = pool_health.parse_remote


class TestMarkerParsing:
    def test_markered_venv_node_is_healthy(self):
        p = parse("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\n")
        assert p["healthy"] and p["marker"] and p["revision"] == "abc123" and p["runners"] == 1412

    def test_no_marker_anywhere_is_not_healthy(self):
        # The retired-checkout bug, reborn: dirs exist but none completed provisioning.
        p = parse("MARKER_MISSING abc123\nMARKER_MISSING def456\n")
        assert not p["healthy"] and not p["marker"] and p["revision"] is None
        assert p["partial"] == ["abc123", "def456"]

    def test_broken_venv_is_not_healthy(self):
        # The REAL failure mode (venv) must never silently pass.
        assert not parse("MARKER_OK abc123\nCODE_OK\nVENV_BROKEN\n1412\n")["healthy"]

    def test_empty_revisions_root_is_not_healthy(self):
        assert not parse("NO_REVISIONS\n")["healthy"]

    def test_newer_unmarked_dir_does_not_hide_the_live_revision(self):
        # A provisioning-in-progress dir (newest, unmarked) must not mask the live marked one.
        p = parse("MARKER_MISSING new99\nMARKER_OK abc123\nCODE_OK\nVENV_OK\n900\n")
        assert p["healthy"] and p["revision"] == "abc123" and p["partial"] == ["new99"]

    def test_runner_missing_is_stale_not_broken(self):
        p = parse("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\nRUNNER_MISSING", runner="research.runners.new")
        assert p["healthy"] and not p["runner_ok"]
        p2 = parse("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\nRUNNER_OK", runner="research.runners.new")
        assert p2["healthy"] and p2["runner_ok"]

    def test_row_schema_keeps_tool_health_keys(self):
        # tools/tool_health.py parses --json rows for reachable/healthy/code/venv/runners: keep them.
        import subprocess
        from unittest import mock
        canned = "MARKER_OK abc123\nCODE_OK\nVENV_OK\n900\n"
        with mock.patch.object(subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=canned, stderr="")
            row = pool_health.check_node("pool40", None)
        for key in ("node", "reachable", "healthy", "code", "venv", "runners"):
            assert key in row, "tool_health contract key missing: %s" % key
        assert row["healthy"] and row["revision"] == "abc123"


class TestGuards:
    def test_source_has_no_retired_checkout(self):
        src = (_TOOLS / "pool_health.py").read_text()
        assert '= "~/derisk-pool/sim"' not in src, "retired checkout reference returned"
        assert 'REVISIONS_ROOT = "~/derisk-pool/revisions"' in src

    def test_source_never_suggests_the_banned_fix(self):
        src = (_TOOLS / "pool_health.py").read_text()
        assert "bash tools/pool_provision.sh" not in src, "pool_health suggests the banned fix"
        assert "pool_provision.sh is banned" in src

    def test_uses_the_shared_marker_file_constant(self):
        # The marker filename must come from the ONE shared predicate's default, not a local invention.
        marker_sh = (_TOOLS / "pool_revision_marker.sh").read_text()
        assert 'POOL_REVISION_MARKER_FILE=".provisioned_ok"' in marker_sh
        assert "pool_revision_marker.sh" in (_TOOLS / "pool_health.py").read_text()

    def test_selftest_passes(self):
        assert pool_health.selftest() == []


if __name__ == "__main__":
    raise SystemExit(0)