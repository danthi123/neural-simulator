# Capability: Self and identity
# Record: brain/research/2026-11-26-voice-split-prereg.md (prereg)
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""voice_split on a stub life: every planned arm/condition runs the full battery; its fresh-state rows reproduce the
cascade screen's B rows; knows() and the table read."""
import json

from brain.experiments import cascade_screen as CS
from brain.experiments import voice_split as VS
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)


def test_voice_split_on_a_stub_life(tmp_path):
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    trunk = str(tmp_path / "h" / "trunk")
    out = VS.run(res, str(snaps), trunk, StubCortex(), str(tmp_path / "w"), weekend=False, nights=(1, 2, 3),
                 log=lambda *a: None)
    casc = CS.run(res, str(snaps), trunk, StubCortex(), str(tmp_path / "w2"), weekend=False, nights=(1, 2, 3),
                  log=lambda *a: None)
    for n, cn in zip(out["nights"], casc["nights"]):
        n_q = len(cn["arms"]["intact"]["A"]["rows"])
        assert {(a, c) for a, cs in n["arms"].items() for c in cs} == set(VS.PLAN)
        assert all(len(v["rows"]) == n_q for cs in n["arms"].values() for v in cs.values())
    same, tot = VS.reproduction(out, casc)
    assert tot > 0 and same == tot
    v = VS.table([dict(out, seed=s) for s in (1, 2)], [dict(casc, seed=s) for s in (1, 2)],
                 {s: res["ledger"] for s in (1, 2)})
    assert v["verdict"] in ("SLEEP-LEARNED", "NOT-SLEEP", "FETCH-PRIMED", "UNDEFINED")
    ctx = [r for n in out["nights"] for r in n["arms"]["both-off"]["ctx"]["rows"]]
    told = [r for r in ctx if (res["ledger"].get(r["item"] or "") or {}).get("telling")]
    assert all(r["told_first"] for r in told) and not any(r["told_first"] for r in ctx if r not in told)


def test_table_rule_on_synthetic_cells():
    """SLEEP-LEARNED when hippocampus-off speaks the known fact in the user's voice far more than the in-context
    pretrained arm, on items across seeds; NOT-SLEEP when they match."""
    led = {f"i{k}": {"keys": [f"k{k}"], "telling": "x"} for k in range(8)}
    def night(uv_h, uv_c):
        def rows(uv):
            return [{"kind": "H", "item": f"i{k}", "brain": (f"My k{k}." if uv else f"Your k{k}."), "you": "q"}
                    for k in range(8)]
        arms = {}
        for a, c in VS.PLAN:
            arms.setdefault(a, {})[c] = {"rows": rows(uv_h if a in ("hippocampus-off", "intact") else uv_c),
                                         "right_episode": 0, "hedge_told": 0}
        return {"night": 2, "arms": arms}
    def outs(uv_h, uv_c):
        return [{"seed": s, "brain_kw": {}, "nights": [night(uv_h, uv_c) for _ in range(3)]} for s in (1, 2, 3)]
    casc = lambda o: {"seed": o["seed"], "brain_kw": {}, "nights": [{"night": 2, "arms": {
        a: {"B": {"rows": o["nights"][0]["arms"][a]["B"]["rows"]}} for a in ("intact", "hippocampus-off")}}]}
    o = outs(True, False)
    assert VS.table(o, [casc(x) for x in o], {s: led for s in (1, 2, 3)})["verdict"] == "SLEEP-LEARNED"
    o = outs(True, True)
    assert VS.table(o, [casc(x) for x in o], {s: led for s in (1, 2, 3)})["verdict"] == "NOT-SLEEP"


def test_knows_on_a_banked_row():
    import os
    p = "brain/experiments/results/2026-11-25-cascade-screen/casc_s320.json"
    lp = "_scratch/stage2/read/lifeKP_s320.json"
    if not (os.path.exists(p) and os.path.exists(lp)):
        import pytest
        pytest.skip("banked files absent")
    led = json.load(open(lp))["ledger"]
    rows = [r for n in json.load(open(p))["nights"] for r in n["arms"]["intact"]["B"]["rows"]]
    assert sum(VS.knows(r, led) for r in rows if CS.is_told(r, led)) > 20


def test_knows_whole_word():
    led = {"volvo": {"keys": ["volvo"]}}
    assert VS.knows({"item": "volvo", "brain": "You drive a red Volvo."}, led)
    assert not VS.knows({"item": "volvo", "brain": "You drive a Volvoish car."}, led)
    assert not VS.knows({"item": None, "brain": "Volvo"}, led)


def test_outputs_record_the_real_device_and_strategy(tmp_path):
    """The Cortex has no .device; the headers must carry cortex.strategy (e.g. 'cuda fp16'), not '?'/'cpu'."""
    import inspect

    from brain.experiments import life_kw
    from brain.experiments import voice_tag as VT

    class CudaStub(StubCortex):
        strategy = "cuda fp32"

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    trunk = str(tmp_path / "h" / "trunk")
    for mod, w in ((VS, "w"), (CS, "w2")):
        out = mod.run(res, str(snaps), trunk, CudaStub(), str(tmp_path / w), weekend=False, nights=(1,),
                      log=lambda *a: None)
        assert out["device"] == "cuda" and out["strategy"] == "cuda fp32"
    src = inspect.getsource(VT.run)  # full voice_tag.run needs sleeps; the header expression is checked in source
    assert '"strategy": str(getattr(cortex, "strategy"' in src and 'getattr(cortex, "device"' not in src
