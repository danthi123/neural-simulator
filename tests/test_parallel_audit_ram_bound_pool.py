"""A RAM-bound pool node's idle cores are not unused capacity (2026-09-23: the under_compute streak never reset for
15 days because 15 GB nodes holding two ~6 GB jobs always showed idle cores)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import parallel_audit as pa  # noqa: E402


def test_empty_queue_counts_idle_cores(monkeypatch):
    monkeypatch.setattr(pa, "sh", lambda *a, **k: "should not be called")
    assert pa._node_can_take_work("poolX", None) is True


def test_node_budget_below_smallest_job_is_not_idle(monkeypatch):
    monkeypatch.setattr(pa, "sh", lambda *a, **k: "idle budget=1GB")
    assert pa._node_can_take_work("poolX", 6) is False


def test_node_that_fits_the_next_job_counts(monkeypatch):
    monkeypatch.setattr(pa, "sh", lambda *a, **k: "idle budget=7GB")
    assert pa._node_can_take_work("poolX", 6) is True


def test_busy_node_is_not_idle(monkeypatch):
    monkeypatch.setattr(pa, "sh", lambda *a, **k: "busy/unreachable (budget=9GB)")
    assert pa._node_can_take_work("poolX", 1) is False
