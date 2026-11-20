# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the split switch)
"""The life script runs on the TEST split only while its sha256 equals the recorded hash; DEV is unchanged."""
import pytest

from brain.eval import life, life_bank, life_bank_test


def test_dev_schedule_unchanged_and_bank_restored():
    s1, l1 = life.build_life(311, 7, "dev")
    life.build_life(311, 7, "test")
    s2, l2 = life.build_life(311, 7, "dev")
    assert (s1, l1) == (s2, l2) and life.LB is life_bank


def test_test_split_uses_test_items():
    sched, ledger = life.build_life(42, 15, "test")
    assert life.LB is life_bank_test
    ab = [i for i, e in ledger.items() if e["cls"] == "AB"]
    assert len(ab) == 16 and all(i in life_bank_test.ALL for i in ab)
    texts = {t["text"] for d in sched for t in d["turns"]}
    assert not texts & {x.telling for x in life_bank.AB + life_bank.F + life_bank.E + life_bank.D}
    assert len([q for q in life.battery(42, 5, ledger) if q["kind"].startswith("open-")]) == 6
    life.use_split("dev")


def test_edited_test_split_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "h.sha256"
    f.write_text("0" * 64 + "  x\n")
    monkeypatch.setattr(life, "TEST_HASH_FILE", str(f))
    with pytest.raises(RuntimeError):
        life.use_split("test")
    life.use_split("dev")
