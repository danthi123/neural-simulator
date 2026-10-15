"""tree-c5-offcurve / RULING c5-probe-merged: the battery worker's flag-independent per-turn RSS record.

battery_turn record: gated ONLY on BRAIN_IDLE_REPLAY_LEAKLOG (never on BRAIN_IDLE_REPLAY, so the OFF arm
logs too); unset leaves the run byte-identical (no file, no write); set appends one jsonl row per turn with
{ts, turn label, rss_kb, hwm_kb, event=battery_turn} read from /proc/self/status; a probe never raises.
"""
import json
import os

import pytest

from research.runners.onebrain_regression_battery import _battery_turn_record


def test_unset_env_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_IDLE_REPLAY_LEAKLOG", raising=False)
    side = tmp_path / "side.jsonl"
    _battery_turn_record("emo")            # must be a no-op: one environ lookup, byte-identical, no file
    assert not side.exists()


def test_set_env_appends_one_record(tmp_path, monkeypatch):
    side = tmp_path / "side.jsonl"
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_LEAKLOG", str(side))
    _battery_turn_record("emo")
    _battery_turn_record("bc_a")
    rows = [json.loads(line) for line in side.read_text().splitlines()]
    assert [r["turn"] for r in rows] == ["emo", "bc_a"]
    for r in rows:
        assert r["event"] == "battery_turn"
        assert isinstance(r["ts"], (int, float)) and r["ts"] > 0
        assert isinstance(r["pid"], int) and r["pid"] > 0
        assert r["rss_kb"] >= 0            # kB from /proc/self/status VmRSS
        assert r["hwm_kb"] >= r["rss_kb"]  # VmHWM is the high-water mark of VmRSS


def test_unwritable_path_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_LEAKLOG", str(tmp_path))   # a directory: open() for append fails
    _battery_turn_record("emo")            # swallow-and-continue is the documented contract


def test_record_is_independent_of_replay_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_IDLE_REPLAY", raising=False)           # the OFF arm: replay flag entirely unset
    side = tmp_path / "off_arm.jsonl"
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_LEAKLOG", str(side))
    _battery_turn_record("rich_well")
    rows = [json.loads(line) for line in side.read_text().splitlines()]
    assert rows and rows[0]["event"] == "battery_turn"          # OFF arm still gets its curve