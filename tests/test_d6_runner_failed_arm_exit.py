"""A dead D6 arm worker must surface as a failed arm, not a silent rc=0 (2026-09-23: 12 of 35 v3 arms vanished)."""
from research.runners import d6_learn_through_use_lb as d6


def test_dead_worker_is_reported_as_failed_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(d6, "_spawn", lambda *a, **k: None)
    arm = sorted(d6.arms_for("capability"))[0]
    res = d6.run([42], str(tmp_path), variant="capability", only_arms=[arm])
    assert res["failed_arms"] == ["s42_%s" % arm]


def test_no_failed_arms_when_workers_write(tmp_path, monkeypatch):
    monkeypatch.setattr(d6, "_spawn", lambda *a, **k: {"turns": {}})
    arm = sorted(d6.arms_for("capability"))[0]
    res = d6.run([42], str(tmp_path), variant="capability", only_arms=[arm])
    assert res["failed_arms"] == []
