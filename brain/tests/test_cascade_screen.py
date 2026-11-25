# Capability: Self and identity
# Record: brain/research/2026-11-25-cascade-screen-prereg.md (prereg)
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""cascade_screen on a stub life: condition A reproduces the banked battery; the decay conditions run; the table reads."""
import json

from brain.experiments import cascade_screen as CS
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)


def test_cascade_screen_on_a_stub_life(tmp_path):
    """On a stub life with night snapshots, condition A reproduces the banked intact and hippocampus-off replies and
    fetches row for row; the decay conditions run through the same battery; the table reads."""
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    out = CS.run(res, str(snaps), str(tmp_path / "h" / "trunk"), StubCortex(), str(tmp_path / "w"), weekend=False,
                 nights=(1, 2, 3), log=lambda *a: None)
    for n in out["nights"]:
        for arm, a in n["arms"].items():
            assert not a["void"]
            assert a["A"]["match_banked"] == len(a["A"]["rows"]), (n["night"], arm)
            assert a["A"]["V"] == a["banked"]["V"] and a["A"]["ok"] == a["banked"]["ok"]
            assert all(len(a[c]["rows"]) == len(a["A"]["rows"]) for c in CS.CONDS)
    v = CS.table([dict(out, seed=s) for s in (1, 2, 3)], None)
    assert set(v["per_beta"]) == {"D50", "D25"} and all(x["status"] in ("KILLED", "HOLD", "UNDEFINED") for x in v["per_beta"].values())


def test_k5_from_probe_counts():
    def run(g, echo, qs):
        return {"gain": g, "echo": echo, "onset_repeat": 0, "n_free": 14,
                "turns": [{"forced": True, "brain": "x?"}] * 2 + [{"forced": False, "brain": "a?" if i < qs else "a."}
                                                                  for i in range(14)]}
    e = {"runs": [run("d0.5", 14, 0), run("d0.25", 3, 1)]}
    b = {"runs": [run("d0.5", 0, 0), run("d0.25", 0, 5)]}
    k = CS.k5_from_probe(e, b)
    assert k["D50"]["fail"] and k["D50"]["E"] == [14]
    assert not k["D25"]["fail"] and k["D25"]["B_q"] == [5]        # one loop sample of two: below the >= 2 bar
