"""Tests for research.runners.morning_briefing — chain status detection.

Mocks the Path / file system minimally to test that _chain_status()
correctly classifies each chain stage.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def _make_log(path: Path, content: str = "(stub)\n",
              mtime_minutes_ago: float = 0):
    """Create a log file with given content and optional staleness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if mtime_minutes_ago > 0:
        old_time = (datetime.now() - timedelta(minutes=mtime_minutes_ago)).timestamp()
        os.utime(path, (old_time, old_time))


def _setup_raw_dir(tmp_path: Path) -> Path:
    """Set up a fake research/findings/raw/g11_bg layout under tmp_path."""
    raw = tmp_path / "research" / "findings" / "raw" / "g11_bg"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def _patch_cwd(monkeypatch, tmp_path: Path):
    """Run morning_briefing from tmp_path so it sees the fake raw dir."""
    monkeypatch.chdir(tmp_path)


def test_unknown_when_no_dir(tmp_path: Path, monkeypatch):
    """No raw dir → stage 'unknown'."""
    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "unknown"


def test_minimal_iso_running(tmp_path: Path, monkeypatch):
    """Some .pid.done files but no biology master log → minimal_iso_running."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "minimal_iso_seed44.log", "stub", mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "minimal_iso_running"
    assert cs["minimal_iso_done"] == 2


def test_biology_sweep_running(tmp_path: Path, monkeypatch):
    """biology master log exists but no COMPLETE marker → biology_sweep_running."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "Launched seed42 fs_only as PID 1234\n", mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "biology_sweep_running"
    assert not cs["biology_complete"]


def test_biology_sweep_done_with_verdict(tmp_path: Path, monkeypatch):
    """COMPLETE marker + waiter log VERDICT → biology_sweep_done + verdict."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "Launched seed42 fs_only as PID 1234\n=== biology-sweep COMPLETE ===\n",
              mtime_minutes_ago=2)
    _make_log(raw / "wait_biology_then_decide.log",
              "Polling...\nVERDICT: A\n",
              mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["biology_complete"] is True
    assert cs["verdict"] == "A"


def test_a1_running_when_minbio_master_exists(tmp_path: Path, monkeypatch):
    """minimum-biology master log exists, no COMPLETE → A1_running."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "=== biology-sweep COMPLETE ===\n", mtime_minutes_ago=10)
    _make_log(raw / "minimum-biology.master.log",
              "Launched seed42 topo_weak\n", mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "A1_running"


def test_a1_done_when_minbio_complete(tmp_path: Path, monkeypatch):
    """minimum-biology master log has COMPLETE marker → A1_done."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "=== biology-sweep COMPLETE ===\n", mtime_minutes_ago=60)
    _make_log(raw / "minimum-biology.master.log",
              "...\n=== minimum-biology COMPLETE ===\n", mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "A1_done"


def test_b1_running_when_sanity_log_exists(tmp_path: Path, monkeypatch):
    """Sanity check log exists → B1_running."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "=== biology-sweep COMPLETE ===\n", mtime_minutes_ago=60)
    _make_log(raw / "eval_sanity_check.log",
              "starting...\n", mtime_minutes_ago=2)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "B1_running"


def test_stall_warning_fires_when_log_old(tmp_path: Path, monkeypatch):
    """If chain is running but newest log update is > 30 min old, warn."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    # 45 min stale log
    _make_log(raw / "minimal_iso_seed44.log", "(stub)",
              mtime_minutes_ago=45)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "minimal_iso_running"
    assert cs["stall_warning"] is not None
    assert "stalled" in cs["stall_warning"].lower()


def test_no_stall_warning_when_log_fresh(tmp_path: Path, monkeypatch):
    """Fresh log update → no stall warning."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42,):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "minimal_iso_seed44.log", "(stub)", mtime_minutes_ago=5)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    assert cs["stage"] == "minimal_iso_running"
    assert cs["stall_warning"] is None


def test_read_waiter_text_handles_utf16_le_with_bom(tmp_path: Path, monkeypatch):
    """Waiter log written by PowerShell Out-File defaults to UTF-16 LE.
    The reader must decode it correctly even though our other code uses
    UTF-8."""
    raw = _setup_raw_dir(tmp_path)
    text_content = "=== Wait-biology-then-decide started 2026-05-04 ===\nVERDICT: A\n"
    # PowerShell-style: UTF-16 LE with BOM
    raw_bytes = b"\xff\xfe" + text_content.encode("utf-16-le")
    (raw / "wait_biology_then_decide.log").write_bytes(raw_bytes)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _read_waiter_text
    out = _read_waiter_text()
    # Should be cleanly decoded (no replacement chars, no null bytes)
    assert "Wait-biology-then-decide" in out
    assert "VERDICT: A" in out
    assert "\x00" not in out
    assert "�" not in out


def test_read_waiter_text_handles_utf8(tmp_path: Path, monkeypatch):
    """If the log is UTF-8 (e.g. someone manually edited it or
    Out-File was overridden), the reader should still work."""
    raw = _setup_raw_dir(tmp_path)
    text_content = "=== Wait-biology-then-decide started ===\nVERDICT: B\n"
    (raw / "wait_biology_then_decide.log").write_text(text_content,
                                                       encoding="utf-8")

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _read_waiter_text
    out = _read_waiter_text()
    assert "Wait-biology-then-decide" in out
    assert "VERDICT: B" in out


def test_chain_status_parses_verdict_from_utf16_le_log(tmp_path: Path,
                                                        monkeypatch):
    """End-to-end: waiter writes UTF-16 LE log with VERDICT line; chain
    status correctly parses it."""
    raw = _setup_raw_dir(tmp_path)
    for seed in (42, 43, 44, 100, 101, 102):
        (raw / f"minimal_iso_seed{seed}.pid.done").write_text(f"{seed}")
    _make_log(raw / "biology-sweep.master.log",
              "=== biology-sweep COMPLETE ===\n", mtime_minutes_ago=5)
    # UTF-16 LE waiter log with VERDICT
    text = "Polling complete\nVERDICT: A\nLaunching follow-up\n"
    raw_bytes = b"\xff\xfe" + text.encode("utf-16-le")
    (raw / "wait_biology_then_decide.log").write_bytes(raw_bytes)

    _patch_cwd(monkeypatch, tmp_path)
    from research.runners.morning_briefing import _chain_status
    cs = _chain_status()
    # Verdict should be parsed despite UTF-16 LE encoding
    assert cs["verdict"] == "A"
