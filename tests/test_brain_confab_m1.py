"""M1 comparator harness: set hygiene and the fit/AND helpers (no model needed)."""
from brain.experiments import confab_m1 as m


def test_v3_sets_shape_and_disjoint():
    assert len(m.FRESH_TOLD3) == 12 and len(m.FRESH_NEVER3) == 24
    assert sum(1 for _, _, al in m.FRESH_NEVER3 if al) == 8
    assert all(k for _, k, _ in m.FRESH_TOLD3)


def test_pct_and_and():
    assert m.pct([1, 2, 3, 4], 2) == 0.5 and m.pct([1, 2, 3, 4], 0) == 0.0
    ref = [{"C_cmean": float(i), "A_lp": float(-i)} for i in range(10)]
    rows = [{"C_cmean": 9.0, "A_lp": -9.0}]
    m.add_and(rows, ref, "C_cmean")
    assert rows[0]["AND"] == min(1.0, 0.1)


def test_kind():
    assert m.kind({"set": "v3_told"}) == "told" and m.kind({"set": "dev_never"}) == "never"
    assert m.kind({"set": "v2_general"}) == "other"
