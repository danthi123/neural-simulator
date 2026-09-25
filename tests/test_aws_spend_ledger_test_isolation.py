"""Pins the 2026-09-25 guard: a test process never writes the PRODUCTION AWS spend ledger.

A test helper that forgot AWS_SPEND_LEDGER wrote a stub instance ("i-existing", launch 2020-01-01) into
research/queue/.aws_spend_ledger.jsonl on 2026-09-24 and 2026-09-25, adding ~$14.8 of phantom spend to "today"
(aws-guard would have stopped both real pool nodes at the $50 cap mid-job). These tests point the module's
notion of "production" at a tmp file, so a broken guard can never touch the real ledger while being tested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import aws_spend_ledger as L  # noqa: E402

ROW = [{"id": "i-guardtest", "type": "r7i.4xlarge", "state": "running", "cost_today_usd": 1.0}]


def test_refuses_production_path_under_pytest(tmp_path, monkeypatch):
    fake_prod = tmp_path / "prod_ledger.jsonl"
    monkeypatch.setattr(L, "PRODUCTION_LEDGER_FILE", str(fake_prod))
    assert os.environ.get("PYTEST_CURRENT_TEST")          # the guard's trigger is present under pytest
    assert L.record(ROW, ledger_file=str(fake_prod)) is False
    assert not fake_prod.exists()                           # nothing written


def test_refuses_default_path_when_it_is_production(tmp_path, monkeypatch):
    fake_prod = tmp_path / "prod_ledger.jsonl"
    monkeypatch.setattr(L, "PRODUCTION_LEDGER_FILE", str(fake_prod))
    monkeypatch.setattr(L, "LEDGER_FILE", str(fake_prod))  # env var unset -> default resolves to production
    assert L.record(ROW) is False
    assert not fake_prod.exists()


def test_writes_an_isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PRODUCTION_LEDGER_FILE", str(tmp_path / "prod_ledger.jsonl"))
    iso = tmp_path / "iso_ledger.jsonl"
    assert L.record(ROW, ledger_file=str(iso)) is True
    assert "i-guardtest" in iso.read_text()


def test_guard_is_off_outside_tests(tmp_path, monkeypatch):
    fake_prod = tmp_path / "prod_ledger.jsonl"
    monkeypatch.setattr(L, "PRODUCTION_LEDGER_FILE", str(fake_prod))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert L.record(ROW, ledger_file=str(fake_prod)) is True   # production callers are unaffected
    assert fake_prod.exists()
